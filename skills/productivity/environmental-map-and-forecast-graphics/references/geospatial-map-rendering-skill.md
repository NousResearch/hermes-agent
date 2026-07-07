---
name: geospatial-map-rendering
description: "Render custom geospatial maps from KML/KMZ/ZIP files and public map/data layers with clean basemaps, overlays, labels, legends, and visual QA."
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [maps, geospatial, kml, kmz, zip, overlays, cartography, visualization]
    created_by: agent
---

# Geospatial Map Rendering

## Trigger

Use this skill when the user asks to turn geospatial files or map layers into a polished custom map image, especially when working with:

- `.kml`, `.kmz`, or renamed `.zip` KMZ archives
- Public GIS/ArcGIS REST layers that can be queried directly
- Raster map overlays such as density/current PNGs
- Risk/contour lines, polygons, or point sample layers
- City/location labels, legends, basemaps, and iterative visual cleanup

## Core workflow

1. **Inspect the archive/layers first.**
   - `.kmz` is ZIP-compatible; copy/rename to `.zip` if the file is already accessible locally.
   - If the chat/document ingestion layer rejects `.kmz` before saving it, the agent cannot rename it yet; ask the user to upload the same KMZ renamed to `.zip`, upload a ZIP containing the KMZ, or extract/send the `.kml` plus assets.
   - List archive contents and identify KML plus any raster overlays.
   - Read KML metadata, `GroundOverlay` bounds, feature counts, and data dates.

2. **Parse geospatial data directly.**
   - KML coordinates are `longitude,latitude[,altitude]`.
   - Extract `Placemark`, `LineString`, `Polygon`, `Point`, `ExtendedData`, and risk/date fields as needed.
   - Do not infer risk classes from the picture alone when KML attributes exist.

3. **Build the map in layers.**
   Recommended order:
   - Basemap.
   - Georeferenced raster overlays, e.g. density/current PNGs.
   - Vector features, e.g. KML risk contours/lines/polygons.
   - Custom city/place labels and markers.
   - Legend/title/source/date.

4. **Choose basemaps for the labeling strategy.**
   - If adding custom city labels, avoid basemaps with built-in city/road labels; duplicate labels look messy.
   - Prefer no-label physical/topographic/terrain basemaps when the user wants custom callouts.
      - Esri World Physical Map worked well for a more colorful Florida physical/topographic look while avoiding duplicate city callouts.
      - Esri World Terrain Base avoided labels but looked too plain/gray for this use case.
      - OpenTopoMap looked colorful but included built-in city labels, causing duplicate callouts.
   - If a colorful topo basemap includes labels, switch to a no-label physical/terrain base and add only the desired labels manually.

5. **Iterate visually.**
   - Generate a PNG.
   - Inspect it visually before sending.
   - Fix overlap, muddy overlays, legend size/placement, missing labels, and color contrast.

## Cartographic style preferences learned

- For custom city labels, direct-on-map text with a white halo can look cleaner than offshore boxed callouts.
- Include both sides of a region when context matters; for Florida sargassum maps, include Gulf coast cities and major Atlantic-side cities.
- For Florida coast context, a good label set is: Pensacola, Destin, Panama City, Apalachicola, Cedar Key, Crystal River, Tampa, Clearwater, St. Petersburg, Sarasota, Venice, Manasota Key, Punta Gorda, Fort Myers, Naples, Marco Island, Key West, Fernandina Beach, Jacksonville, St. Augustine, Daytona Beach, Cape Canaveral, Melbourne, Orlando, Vero Beach, Port St. Lucie, West Palm Beach, Boca Raton, Fort Lauderdale, Miami.
- Add user-requested local places explicitly, even if small, e.g. Manasota Key.
- Legends should be large enough to read, near the mapped area, and should not cover key labels or risk areas.
- Put the data date in the legend, not only in the title.
- Use same-size legend swatches/rectangles for comparable risk classes.

## Florida red tide maps

For Ron's Florida daily email, use the FWC current-status map he specifically requested, not the USF trajectory composite:

- Script: `~/.hermes/scripts/fwc-red-tide-current-status-map.py`
- Public FWC app: `https://gis.myfwc.com/redtidecurrentstatus/`
- ArcGIS web map item: `032578e992e04012a25db77499318ff9`
- Feature layer: `https://services2.arcgis.com/z6TmTIyYXEYhuNM0/arcgis/rest/services/HAB_Current_Web_Layer/FeatureServer/0`
- Output directory: `~/.hermes/cache/red_tide/`

This renders a polished PNG from the FWC `HAB_Current_Web_Layer` sampling points, including city labels, Manasota Key marker, abundance legend, source/footer, and a clear note that the FWC app is a current-status sampling map for the most recent 8 days, not a forward trajectory prediction model.

If the user explicitly asks for a red tide *prediction/forecast* map instead of the FWC current-status map, use the USF–FWC short-term HAB trajectory workflow in `references/florida-red-tide-prediction-map.md`:

- Pull current USF OCL WFCOM HAB Tracking images: `upper_R0.gif` for surface trajectories and `lower_R0.gif` for subsurface / near-bottom trajectories.
- Pull the current short-term forecast summary sentence from FWC Red Tide Current Status.
- Compose a polished Telegram-ready PNG with two side-by-side panels, a FWC forecast summary card, a “how to read the forecast” legend, source footer, and a clear note that it is model-based transport guidance, not a health advisory.
- Visually inspect the composite before delivery, especially footer/source wrapping and summary/legend overlap.

## Sarasota Water Atlas rainfall maps

For Ron's rainfall maps from `https://sarasota.wateratlas.usf.edu/rainfall/latest/`, use the workflow in `references/sarasota-water-atlas-rainfall-map.md`.

Key defaults:

- Query `https://api.wateratlas.usf.edu/rainfall/latest/?s=8` directly instead of scraping marker labels from the page.
- For Manasota Key / Englewood 31-day maps, use `total31d`, a 3 km IDW visualization grid when requested, and prominent callouts for lower/coastal stations near Manasota Key including the Englewood / Indian Mound Park station.
- If Ron asks to “go back to this map” or provides a screenshot of the Water Atlas page, recreate the **Water Atlas-style screenshot** rather than substituting the polished custom grid artifact: white Sarasota County Water Atlas header, pale OSM/Esri-like basemap, blue boxed station totals, left radio panel with “31 Day Total Rainfall” selected, and bottom-left 3 km scale. Match the screenshot’s viewport/feel and keep lower Manasota Key / Englewood station boxes visible.
- Include a clear note that any custom 3 km grid is an interpolation/visualization from station totals, not an official gridded rainfall product. This disclaimer is not needed for a pure Water Atlas-style station screenshot recreation unless a custom grid is overlaid.
- Visually QA that the land is not black, station labels do not overlap/crop badly, the radio panel does not hide the lower stations, and the title/header plus 3 km scale are readable before sending.

## Sarasota Water Atlas rainfall maps

When Ron asks for Manasota Key / Englewood rainfall maps from the Sarasota Water Atlas near-real-time rainfall page, use the API behind the page rather than screenshotting markers:

- Page: `https://sarasota.wateratlas.usf.edu/rainfall/latest/`
- API: `https://api.wateratlas.usf.edu/rainfall/latest/?s=8`
- Fields: `total24h`, `total7d`, `total31d`, with station name/id/location/lastUpdated.
- For a 3 km resolution map, build a 3 km IDW interpolation grid from all Sarasota ARMS stations, overlay station dots/values, and label lower coastal stations near Manasota Key / Englewood.
- Always include a disclaimer that the 3 km grid is an interpolated visualization from station totals, not an official gridded rainfall product.
- For Florida email use, produce a 3-panel tile in this order: **24-Hour**, **7-Day**, **31-Day**, and visually QA spelling, labels, footer/source, and lower station readability before sending.

Detailed notes and station/extent defaults: `references/wateratlas-rainfall-map-rendering.md`.

## Sargassum/KMZ map defaults

Use these defaults for NOAA/USF-style sargassum risk/current files unless the user asks otherwise:

- Low risk: bright light blue.
- Risk 1: bright yellow.
- Risk 2: vivid orange.
- Risk 3: bright red.
- Current raster layer: keep visible but subtle; original current rasters can look like thick smudges if opacity is too high.
- Density raster layer: transparent enough to preserve basemap context.
- Risk contours/features: bright and high contrast against the basemap.

## Daily NOAA/AOML sargassum cron automation

When the user wants a recurring Florida sargassum map:

1. Store reusable scripts under `~/.hermes/scripts/` and reference them in cron by filename only, e.g. `daily-sargassum-map.py`.
2. Build the KMZ URL from a `YYYYMMDD` product date:
   - `https://cwcgom.aoml.noaa.gov/SIR/KMZ/Sargassum_risk_currents_FA_<YYYYMMDD>.kmz`
3. For morning cron runs, try today's local date first, then walk backward several days because the newest NOAA/AOML KMZ may not be posted yet at 6am.
4. Treat the downloaded `.kmz` as a ZIP archive in Python (`zipfile.ZipFile`) and validate that it contains KML before rendering.
5. Read the KML `GroundOverlay` bounds and KML `SimpleData name="date"` instead of hardcoding dates/bounds.
- For Ron's daily Florida packet, include:
   - the rendered sargassum map,
   - NHC 7-day Atlantic tropical outlook: `https://www.nhc.noaa.gov/xgtwo/resize/xgtwo_atl_7d0_w1920.png`,
   - a public NOAA/NWS weather/radar image that includes Florida, currently `https://radar.weather.gov/ridge/standard/SOUTHEAST_0.gif` converted to PNG for Telegram delivery,
   - Sarasota Water Atlas-style Manasota Key / Englewood rainfall maps for 24-hour, 7-day, and 31-day totals when Ron asks to include local rainfall; use `~/.hermes/scripts/sarasota-rainfall-wateratlas-maps.py --period all` and see `references/florida-email-rainfall-integration.md`.
7. Test with a known historical date before scheduling, e.g. `python3 ~/.hermes/scripts/daily-sargassum-map.py --date 20260601 --force`.
8. Schedule with Hermes cron at `0 6 * * *`, `script="daily-sargassum-map.py"`, and `no_agent=True` for a low-token daily watchdog-style image delivery.

## Florida Email Task

## Sarasota Water Atlas rainfall maps

When Ron asks for rainfall maps from `https://sarasota.wateratlas.usf.edu/rainfall/latest/`, default to Water Atlas-style station amount maps rather than a polished interpolated thematic map unless he explicitly requests interpolation/contours.

- Use the public page API directly: `https://api.wateratlas.usf.edu/rainfall/latest/?s=8`.
- Render 24-hour, 7-day, and 31-day totals from `total24h`, `total7d`, and `total31d`.
- Focus the view on Manasota Key / Englewood and preserve lower station coverage, including CST-3 Indian Mound Park / Englewood, Lemon Bay Park, Lemon Bay Canal, Jacaranda Bridge, Venice Ave E, Capri Isle, and Forked/Gottfried Creek area stations.
- If the user asks for “3 km resolution” in the Water Atlas-style version, include a 3 km scale marker; do not imply the station-box map is an official gridded product.
- Accepted local renderer: `~/.hermes/scripts/sarasota-rainfall-wateratlas-maps.py`.
- Session-specific implementation details and GitHub preservation notes: `references/sarasota-wateratlas-rainfall-email-maps.md`.

## Florida Email Task

## Sarasota Water Atlas rainfall maps

When Ron asks for Sarasota/Manasota/Englewood rainfall maps from the Water Atlas page, prefer a Water Atlas-style station-box map rather than a polished interpolated surface unless he explicitly asks for interpolation.

- Source page: `https://sarasota.wateratlas.usf.edu/rainfall/latest/`
- Public API used by the page: `https://api.wateratlas.usf.edu/rainfall/latest/?s=8`
- Render 24-hour, 7-day, and 31-day variants from `total24h`, `total7d`, and `total31d`.
- Include a 3 km scale marker and keep lower Manasota Key / Englewood stations visible.
- Use the session reference for exact viewport, label nudges, and Florida Email Task integration: `references/sarasota-wateratlas-rainfall-maps.md`.

## Florida Email Task

For Ron's daily emailed Florida packet, use `~/.hermes/scripts/florida-email-task.py` plus the cron wrapper `~/.hermes/scripts/florida-email-task-send.sh`.

- The email task is authorized to send only to `reisworth@gmail.com` by default.
- It calls `daily-sargassum-map.py` to generate the 3 images: sargassum map, NHC 7-day Atlantic tropical outlook, and NOAA/NWS Southeast radar image including Florida.
- It calls `sarasota-rainfall-wateratlas-maps.py --period all` to generate Water Atlas-style 24-hour, 7-day, and 31-day rainfall maps for the Manasota Key / Englewood lower stations, embedded with CIDs `rainfall_24h`, `rainfall_7d`, and `rainfall_31d`.
- It calls `manasota-tide-graph.py` for a one-day Manasota Key tide graph with daylight shading.
- It calls `fwc-red-tide-current-status-map.py` for the FWC ArcGIS current-status sampling map and embeds it inline with CID `red_tide_map`; see `references/fwc-red-tide-current-status-map.md`.
- It fetches SandBar Tiki & Grille live music from `https://www.sandbartikigrille.com/events/` and includes events for the next 5 days from the Florida email date.
- The SandBar page uses event archive `<article>` cards with date blocks, `post-title`, and `span.sr` labels for `Time` and `Genre`; parse those HTML fields directly rather than relying on loose visible-text regex.
- Build a fun, lively HTML email with the images inline via CIDs and a 5-day music lineup section.
- Send via Gmail SMTP using `EMAIL_ADDRESS`, `EMAIL_PASSWORD`, and `EMAIL_SMTP_HOST` from `~/.hermes/.env`.
- The send wrapper stays silent on success so cron does not send Telegram status pings; it prints the captured log only on failure.
- Cron job name: `Florida Email Task`; script: `florida-email-task-send.sh`; schedule: `20 6 * * *`.

## Verification checklist

Before final delivery:

- Basemap has no duplicate built-in city callouts if custom city labels are present.
- Custom city labels are readable and not badly overlapping.
- User-requested places are present.
- Current layer is visible but not dominant/smudgy.
- Risk colors are bright enough to see.
- Legend includes data date, density, currents, low risk, and higher risk classes.
- Final image file exists and was visually inspected.

## References

- `references/sargassum-kmz-map-rendering.md` — session-derived details for NOAA/USF-style sargassum KMZ/ZIP map rendering.
- `references/florida-red-tide-prediction-map.md` — USF/FWC HAB trajectory forecast composite map workflow.
- `references/fwc-red-tide-current-status-map.md` — FWC ArcGIS FeatureServer current-status red tide sampling map workflow and Florida email integration notes.
- `references/sarasota-water-atlas-rainfall-map.md` — Sarasota Water Atlas rainfall API, 31-day/3 km Manasota Key-Englewood map workflow, and lower-station callout notes.
- `references/wateratlas-rainfall-map-rendering.md` — Water Atlas-style screenshot recreation details, viewport defaults, and 24h/7d/31d email tile guidance.
- `references/florida-email-rainfall-integration.md` — Florida Email Task integration for Water Atlas-style 24h/7d/31d rainfall maps and reusable renderer details.
