# Sarasota Water Atlas rainfall map workflow

Use this when Ron asks for rainfall maps from Sarasota County Water Atlas, especially Manasota Key / Englewood / lower Sarasota County.

## Source page and API

- User-facing page: `https://sarasota.wateratlas.usf.edu/rainfall/latest/`
- The page uses Leaflet and fetches rainfall station totals from:
  - `https://api.wateratlas.usf.edu/rainfall/latest/?s=8`
- API records include:
  - `name`, `id`, `lastUpdated`
  - `location.latitude`, `location.longitude`
  - `total24h`, `total7d`, `total31d`

## User-requested map style

For a 31-day Manasota Key / Englewood rainfall map:

1. Query the API and use `total31d` values.
2. Focus the extent around south Sarasota / Manasota / Englewood, roughly:
   - longitude `-82.56` to `-82.16`
   - latitude `26.90` to `27.25`
3. Use a **3 km grid** when requested. Treat this as a visualization/interpolation from point stations, not an official gridded rainfall product.
4. Use all available Sarasota ARMS stations for interpolation, but label the lower/coastal stations near Manasota Key / Englewood prominently.
5. Include a footer note similar to:
   - `Values are station-based 31-day totals from the Water Atlas API; grid is a 3 km visualization/interpolation, not an official gridded rainfall product.`

## Lower/coastal station callouts to include when in extent

These names were present in the API and are useful around Manasota Key / Englewood:

- `CST-3 Indian Mound Park` — label as `Englewood / Indian Mound Park`
- `Lemon Bay Park`
- `Lemon Bay Canal`
- `AL-1  Jacaranda Bridge` — label as `Jacaranda Bridge`
- `FRK-1 Donavan Rd`
- `FRK-2 Stoner Road`
- `GOT-1 Tangerine Wds`
- `GOT-2 Park Forest`
- `HC-1  Venice Ave E` — label as `Venice Ave E`
- `SO-1 Oscar Scherer Park` — label as `Oscar Scherer`
- `CUR-2 Capri Isle` — label as `Capri Isle`

## Rendering notes

- A simple no-label county outline basemap works better than a label-heavy tile map because custom station callouts are dense.
- Plot county polygons underneath the interpolated cells; do not accidentally redraw land polygons with transparent fill on an RGB image, which can make the land render black. Draw transparent overlays on an RGBA layer and alpha-composite, then redraw boundaries only.
- Use a readable green precipitation color ramp, station dots, numeric station values, and a separate legend card.
- Visually QA before delivery: title visible, land not black, 3 km grid visible, Manasota/Englewood station labels readable, source/footer not cropped.
