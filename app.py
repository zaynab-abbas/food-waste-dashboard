# app.py
# Import necessary libraries
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# --- Page Configuration ---
# Set the layout and title for the web page. This should be the first Streamlit command.
st.set_page_config(
    page_title="Global Food Waste Dashboard",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Caching Data ---
# Use st.cache_data to load data only once, speeding up the app.
@st.cache_data
def load_data(filepath):
    """
    Loads and cleans the food waste data from a CSV file.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        st.error(f"Error: The file '{filepath}' was not found. Please make sure it's in the same directory as app.py.")
        # Provide a fallback empty dataframe to prevent the app from crashing
        return pd.DataFrame()

    # --- Data Cleaning ---
    # Identify columns that should be numeric
    numeric_cols = [
        'combined figures (kg/capita/year)',
        'Household estimate (kg/capita/year)',
        'Retail estimate (kg/capita/year)',
        'Food service estimate (kg/capita/year)'
    ]
    # Convert these columns to numeric, coercing errors to NaN
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows where essential data is missing
    df.dropna(subset=['Country', 'combined figures (kg/capita/year)'], inplace=True)
    return df

# --- Main Application ---
# Load the data using the cached function
df = load_data('food_waste_data.csv')

# Check if the dataframe is empty after loading
if df.empty:
    st.stop() # Stop the app if data loading failed

# --- Sidebar ---
# Create a sidebar for user inputs and navigation
st.sidebar.title("Dashboard Options")
st.sidebar.markdown("Use the options below to explore the data.")

# --- Main Title ---
st.title("♻️ Global Food Waste Dashboard")
st.markdown("An interactive dashboard to explore food waste estimates across the globe.")

# --- 1. Global Waste Map ---
st.header("Global Food Waste Map")
st.markdown("Hover over a country to see its combined food waste per capita.")

# Create the choropleth map
fig_map = px.choropleth(
    df,
    locations="Country",
    locationmode="country names",
    color="combined figures (kg/capita/year)",
    hover_name="Country",
    color_continuous_scale=px.colors.sequential.YlOrRd,
    title="Combined Food Waste (kg/capita/year)"
)
fig_map.update_layout(
    geo=dict(showframe=False, showcoastlines=False),
    margin={"r":0,"t":40,"l":0,"b":0}
)
st.plotly_chart(fig_map, use_container_width=True)


# --- 2. Country Rankings ---
st.header("Country Rankings")
st.markdown("Discover the countries with the highest and lowest food waste per capita.")

# Use columns to display charts side-by-side
col1, col2 = st.columns(2)

with col1:
    st.subheader("Top 10 Highest Waste Countries")
    top_10_waste = df.sort_values(by='combined figures (kg/capita/year)', ascending=False).head(10)
    fig_top = px.bar(
        top_10_waste,
        x='combined figures (kg/capita/year)',
        y='Country',
        orientation='h',
        color='combined figures (kg/capita/year)',
        color_continuous_scale='Reds'
    )
    fig_top.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_top, use_container_width=True)

with col2:
    st.subheader("Top 10 Lowest Waste Countries")
    bottom_10_waste = df.sort_values(by='combined figures (kg/capita/year)', ascending=True).head(10)
    fig_bottom = px.bar(
        bottom_10_waste,
        x='combined figures (kg/capita/year)',
        y='Country',
        orientation='h',
        color='combined figures (kg/capita/year)',
        color_continuous_scale='Greens_r'
    )
    fig_bottom.update_layout(yaxis={'categoryorder':'total descending'})
    st.plotly_chart(fig_bottom, use_container_width=True)


# --- 3. Country Specific Breakdown ---
st.header("Waste Breakdown by Country")
st.markdown("Select a country to see its food waste distribution by sector.")

# Dropdown for country selection
country_list = sorted(df['Country'].unique())
selected_country = st.selectbox("Select a Country", country_list)

if selected_country:
    country_data = df[df['Country'] == selected_country].iloc[0]
    sector_data = {
        'Sector': ['Household', 'Retail', 'Food Service'],
        'Waste (kg/capita/year)': [
            country_data['Household estimate (kg/capita/year)'],
            country_data['Retail estimate (kg/capita/year)'],
            country_data['Food service estimate (kg/capita/year)']
        ]
    }
    sector_df = pd.DataFrame(sector_data).dropna()

    if not sector_df.empty:
        fig_pie = px.pie(
            sector_df,
            names='Sector',
            values='Waste (kg/capita/year)',
            title=f"Food Waste Sectors in {selected_country}",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.warning(f"No detailed sector data available for {selected_country}.")


# --- 4. Clustering Analysis ---
st.header("Country Clustering Analysis")
st.markdown("This analysis groups countries with similar waste profiles. We use K-Means clustering with k=3 based on our EDA.")

# Prepare data for clustering
features = ['Household estimate (kg/capita/year)', 'Retail estimate (kg/capita/year)', 'Food service estimate (kg/capita/year)']
cluster_data = df[features].copy()

# Fill missing values with the median for robust clustering
for feature in features:
    median_val = cluster_data[feature].median()
    cluster_data[feature].fillna(median_val, inplace=True)

# Scale the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(cluster_data)

# Apply K-Means
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(scaled_data)

# Add cluster labels to a copy of the dataframe for plotting
plot_df = df.copy()
plot_df['Cluster'] = kmeans.labels_
plot_df['Cluster'] = plot_df['Cluster'].astype(str) # Convert to string for discrete colors

# Create scatter plot
fig_cluster = px.scatter(
    plot_df,
    x='Household estimate (kg/capita/year)',
    y='Retail estimate (kg/capita/year)',
    color='Cluster',
    hover_name='Country',
    title="Country Clusters based on Waste Profiles"
)
st.plotly_chart(fig_cluster, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.info("This dashboard was created by Zaynab Abbas as a final data science project at The Developer Academy to analyze global food waste patterns.")
