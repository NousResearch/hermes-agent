# Sarasota Water Atlas rainfall map rendering

Session-derived workflow for creating Manasota Key / Englewood rainfall maps from the Sarasota Water Atlas near-real-time rainfall page.

## Data source

- Public page: `https://sarasota.wateratlas.usf.edu/rainfall/latest/`
- The page fetches JSON from: `https://api.wateratlas.usf.edu/rainfall/latest/?s=8`
- Useful fields per station:
  - `name`, `id`, `lastUpdated`
  - `location.latitude`, `location.longitude`
  - `total24h`, `total7d`, `total31d`
- Page radio options correspond to API fields:
  - 24 Hour Total Rainfall → `total24h`
  - 7 Day Total Rainfall → `total7d`
  - 31 Day Total Rainfall → `total31d`

## Map extent that captures lower Sarasota / Manasota Key / Englewood

A good working extent for the requested lower stations is:

- lon min/max: `-82.56, -82.16`
- lat min/max: `26.90, 27.25`

This captures the lower coastal / Englewood-area stations, including:

- `CST-3 Indian Mound Park` — label as `Englewood / Indian Mound Park`
- `Lemon Bay Park`
- `Lemon Bay Canal`
- `AL-1  Jacaranda Bridge`
- `FRK-1 Donavan Rd`
- `FRK-2 Stoner Road`
- `GOT-1 Tangerine Wds`
- `GOT-2 Park Forest`
- `HC-1  Venice Ave E`
- `SO-1 Oscar Scherer Park`
- `CUR-2 Capri Isle`

## 3 km rainfall grid technique

For a 3 km resolution visualization:

1. Fetch all Sarasota ARMS stations from the API.
2. Convert lon/lat to approximate km coordinates using:
   - `km_per_deg_lat = 110.574`
   - `km_per_deg_lon = 111.320 * cos(mid_lat)`
3. Create grid cell centers every 3 km over the map extent.
4. Optionally clip grid cells to county polygons so cells do not cover the Gulf. A simple source for county outlines is Plotly's public county GeoJSON: `https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json`; use Florida FIPS for Sarasota (`12115`), Charlotte (`12015`), Manatee (`12081`), and DeSoto (`12027`) as needed.
5. Interpolate each cell using inverse-distance weighting from all available stations:
   - radius: about 45 km
   - power: `2.0`
6. Draw cells as semi-transparent rectangles with subtle white outlines so the 3 km resolution is visible.
7. Overlay station dots and numeric labels using the selected field's station total.

Important disclaimer: label the result as a 3 km visualization/interpolation from station totals, not an official gridded rainfall product.

## Water Atlas-style screenshot recreation

When Ron references the prior screenshot/map and asks to “go back to this map,” he likely wants the page-like Water Atlas visual, not the polished custom 3 km interpolation map.

Use these visual targets:

- Output aspect close to the screenshot: portrait image around `760 x 1230`.
- Header: white band with `Sarasota County Water Atlas` in teal and `Near Real-time Rainfall Map` underneath, plus a dark teal separator line.
- Basemap: pale road/topographic map resembling the Water Atlas Leaflet/Esri view. OSM tiles can be used if desaturated, brightened, and low-contrast.
- Controls: left zoom +/- box and left white radio panel; select `31 Day Total Rainfall` with blue checked radio and bold label.
- Station style: draw blue-gray rectangles with dark outline and large numeric rainfall totals, mimicking Leaflet marker labels.
- Extent: keep the same south Sarasota / Venice / Manasota Key / Englewood area, with lower stations visible rather than hidden by the radio panel.
- Include a bottom-left `3 km` scale marker.

Useful approximate viewport:

- lon min/max: about `-82.555, -82.125`
- lat min/max: about `26.835, 27.245`
- portrait canvas: `760 x 1230`
- map area below header: header about `112 px`; map height about `1065 px`

Implementation notes:

1. Fetch current station values from `https://api.wateratlas.usf.edu/rainfall/latest/?s=8` and draw `total31d` labels.
2. Use Web Mercator math to crop tiles to the viewport, typically at zoom 12.
3. Cache public OSM tiles under `~/.hermes/cache/osm_tiles/` to avoid repeated downloads.
4. Apply `Color ~0.58`, `Contrast ~0.78`, `Brightness ~1.18` to make OSM tiles look closer to the pale Water Atlas basemap.
5. Add small manual nudges for congested labels so Englewood/Manasota-area values remain readable:
   - `CST-3 Indian Mound Park`
   - `Lemon Bay Park`
   - `Lemon Bay Canal`
   - `FRK-1 Donavan Rd`
   - `FRK-2 Stoner Road`
   - `GOT-1 Tangerine Wds`
   - `GOT-2 Park Forest`
6. QA with vision before delivery: header not cropped, radio panel readable, 31-day option selected, lower station labels visible, 3 km scale present.

## Styling notes for Ron's Florida email

- Use a clean no-label basemap or simple county/coastline geometry; avoid cluttered road/city labels since custom callouts are added.
- Keep Manasota Key and Englewood explicitly labeled.
- Add callouts for lower coastal stations so they remain readable when the map is tiled or emailed.
- Include source URL and latest update timestamp from the station data.
- For an email tile, create the three panels in this order: `24-Hour`, `7-Day`, `31-Day`.
- Use a concise header such as: `Manasota Key / Englewood Rainfall Totals`.
- In the tile subtitle, include: `24-hour, 7-day, and 31-day totals • 3 km grid • Sarasota Water Atlas ARMS • latest <timestamp>`.
- Visually QA the tile before delivery: check spelling, all three panel labels, no severe cropping, source/footer readable, and lower station callouts still visible enough at the tile scale.
