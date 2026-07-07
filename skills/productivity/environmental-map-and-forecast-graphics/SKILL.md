---
name: environmental-map-and-forecast-graphics
description: "Use when creating polished environmental/geospatial graphics: KML/KMZ maps, GIS overlays, Florida sargassum/red tide/rainfall products, tide charts, and recurring coastal forecast image/email automation."
version: 1.1.0
author: Hermes Agent
license: MIT
platforms: [linux]
metadata:
  hermes:
    tags: [maps, geospatial, kml, kmz, noaa, tides, rainfall, red-tide, sargassum, coastal, visualization]
---

# Environmental Map and Forecast Graphics

## Overview

Use this skill for user-facing environmental visualizations: static geospatial maps, GIS/KML/KMZ overlays, NOAA/FWC/Water Atlas coastal products, tide graphs, rainfall panels, and scheduled Florida/coastal forecast image packets. The class-level pattern is the same across these jobs: fetch authoritative data, render a polished visual artifact, visually QA it, then deliver the image or email.

## When to Use

- The user provides `.kml`, `.kmz`, `.zip`, Google Earth files, GIS layers, raster overlays, risk contours, or public ArcGIS/NOAA/FWC map layers.
- The user asks for Florida/coastal maps such as sargassum, red tide, rainfall totals, radar, tropical outlooks, or local station maps.
- The user asks for tide charts/graphs with daylight, high/low labels, or Telegram-ready delivery.
- The user asks to automate daily/recurring environmental image packets or email reports.

## Shared Workflow

1. **Identify the source class**
   - KML/KMZ/ZIP: inspect archive contents, parse KML, read `GroundOverlay` bounds and feature attributes.
   - Public GIS/ArcGIS: query the REST layer directly when possible instead of scraping rendered map markers.
   - NOAA/FWC/API graphics: fetch both image/API data and source metadata, including dates and station names.
   - Tide charts: resolve the nearest relevant NOAA CO-OPS station and fetch both smooth predictions and high/low predictions.

2. **Preserve data provenance**
   - Put source name, data date/window, station/layer name, and important caveats in the title, legend, footer, or email copy.
   - For modeled products, distinguish current-status sampling maps from forecasts/trajectory guidance and from health advisories.

3. **Render in layers**
   - Basemap or chart background.
   - Raster overlays or filled regions.
   - Vector features/contours/points.
   - Custom labels, city markers, high/low tide callouts, or station values.
   - Legend, title, source/date footer, disclaimers.

4. **Use the right implementation level**
   - Python + Pillow + NumPy is usually enough for custom static map/chart artifacts.
   - Heavy GIS stacks are optional; do not require `geopandas`/`cartopy` for simple KML/raster or station-map rendering.
   - For recurring work, place reusable scripts under `~/.hermes/scripts/` and schedule with Hermes cron using `no_agent=True` when the script emits the final fixed message/media.

5. **Visually QA before delivery**
   - Open or inspect the generated image before sending.
   - Check label readability, legend size/placement, source/date text, overlap, color contrast, cropped footer/title, and whether all user-requested locations are present.
   - If the visual artifact is clearly flawed, regenerate it before delivering.

## KML/KMZ and GIS Overlay Maps

- Treat `.kmz` as a ZIP archive; if upload ingestion rejects `.kmz`, ask for a `.zip` wrapper or extracted `.kml` plus assets.
- KML coordinates are `longitude,latitude[,altitude]`.
- Parse `Placemark`, `LineString`, `Polygon`, `Point`, `ExtendedData`, `GroundOverlay`, and style colors before deciding how to draw.
- KML color strings are AABBGGRR, not RRGGBBAA.
- For georeferenced rasters, map output pixels back to lon/lat and then into source-image coordinates from the overlay bounds.
- Make near-white empty raster backgrounds transparent and keep current/vector rasters subtle enough that they do not become smudges.
- See `references/sargassum-kmz-map-rendering.md` and `references/florida-sargassum-map-20260611.md` for concrete sargassum/KMZ lessons.

## Basemaps, Labels, and Legends

- If adding custom city labels, avoid basemaps with prominent built-in labels; duplicate labels look messy.
- Use direct on-map text with white halos when the user wants labels on the map; use callout boxes only when collisions are otherwise unavoidable.
- Legends should be large, readable, close to the mapped area, and not covering key labels/data.
- Include the data date in the legend or footer.
- For comparable risk categories, use same-size legend swatches rather than mixed line/box sizes.
- If the user says the map is boring, try a more colorful topo/terrain/physical basemap while preserving label readability.

## Florida/Coastal Product Patterns

### Sargassum

- Build NOAA/AOML KMZ URLs from `YYYYMMDD`: `https://cwcgom.aoml.noaa.gov/SIR/KMZ/Sargassum_risk_currents_FA_<YYYYMMDD>.kmz`.
- For morning automation, try today then walk backward because newest KMZs may not be posted yet.
- Use KML `GroundOverlay` bounds and `SimpleData name="date"` rather than hardcoding.
- Default risk palette: low risk bright light blue/cyan, risk 1 yellow, risk 2 orange, risk 3 red.

### Red tide

- For Ron's Florida daily email, use the FWC current-status sampling map by default, not the USF trajectory composite.
- If the user asks for prediction/forecast, use the USF/FWC short-term HAB trajectory workflow and clearly label it as model-based transport guidance.
- See `references/fwc-red-tide-current-status-map.md` and `references/florida-red-tide-prediction-map.md`.

### Sarasota/Manasota rainfall

- Use `https://api.wateratlas.usf.edu/rainfall/latest/?s=8` behind the Sarasota Water Atlas rainfall page.
- Render 24-hour, 7-day, and 31-day values from `total24h`, `total7d`, `total31d`.
- Prefer Water Atlas-style station-box maps for user requests to match that page; use interpolated 3 km grids only when requested and label them as interpolated visualizations.
- Preserve lower Manasota Key / Englewood station visibility.
- See the Water Atlas references copied into this skill.

### Tide charts

- Resolve the NOAA CO-OPS station, fetch `product=predictions` with `interval=6` for the smooth curve and `interval=hilo` for high/low labels.
- For one-day charts, set the x-axis from local midnight to next local midnight even if the final NOAA point is 23:54.
- Shade daylight using the station/location latitude/longitude and local timezone; guard against UTC conversion making sunset appear before sunrise.
- Render a high-resolution PNG with source station, datum, local time, high/low callouts, and footer/source text.
- See `references/noaa-tide-graph-pattern.md`.

## Automation and Delivery

- Reusable scripts belong in `~/.hermes/scripts/`, not inside a one-off chat transcript.
- Cron jobs that generate fixed image/email outputs should generally use `no_agent=True`; empty stdout should mean silent success when appropriate.
- For Telegram/media delivery, include `MEDIA:/absolute/path/to/file.png` in the message.
- For email packets, keep scripts quiet on success and print logs only on failure so cron does not spam status pings.

### Polished environmental email packets

- If a daily email includes multiple maps/charts, avoid a plain stack of images. Use a wide hero image plus clean individual card sections with concise human summary, then the authoritative map/chart.
- For Ron's Florida daily email, do **not** use decorative photo headers inside each map/chart card; he redlined those out. Keep only the top hero image and the actual chart/map images.
- Do not prefix Florida email section titles with ordering numbers like `1)` / `2)`; preserve order through layout only.
- Source maps/charts should remain prominent and preserve provenance/date labels.
- For Ron's Florida daily email, current added local forecast charts are generated inside `~/.hermes/scripts/florida-email-task.py` using Open-Meteo forecast and marine APIs for Manasota Key (`26.9241,-82.3600`): 7-day weather, 7-day wind, and 7-day swell. Treat them as daily-updating NOAA/NCEP model-derived guidance and label that source in chart footers. Render these charts at high resolution (currently 2200×1360) and keep rain/condition/date/legend zones physically separated; Ron specifically noticed rain percentages overlapping lows and high temps crowding condition callouts.
- For Ron's Manasota Key Fun Pack email music section, use SWFLLive Englewood events (`https://swfllive.com/events?city=Englewood`) for a compact 5-day lineup organized by day/venue. Filter to Beachcomber Trading Post, SandBar Tiki & Grille, and White Elephant Pub. SWFLLive is a Next.js/RSC page; event rows can be parsed from the embedded escaped `tableRows` JSON payload. The email From/subject branding should be `Manasota Key Fun Pack`, not `Ara1bot Florida Fun Pack`.
- When maintaining Ron's Manasota Key Fun Pack, load `references/manasota-key-fun-pack-email.md` for the durable workflow: branding, Gmail clipping/collapsing safeguards, forecast chart spacing pitfalls, SWFLLive entertainment parsing, Brave Search secret handling, and verification checklist.
- Combine Sarasota Water Atlas 24-hour, 7-day, and 31-day rainfall maps into one side-by-side panel for email scanning; keep the individual map images available for source verification.
- Optimize inline email image bytes before attaching (resize large maps/charts/photos and encode as progressive JPEG around quality 82–86). Verify total `.eml` image payload and make sure the large sargassum/rainfall products are no longer multi-MB attachments.
- Verify generated `.eml` files by counting expected inline images/cards and visually previewing extracted HTML or a contact sheet; file existence is not enough.
- See `references/florida-email-photo-led-map-cards.md` for the Florida daily email pattern and verification recipe.

## Common Pitfalls

1. **Skipping visual QA.** These tasks are visual deliverables; file existence is not enough.
2. **Flattening KML/KMZ packages.** A KMZ may include rasters/assets that must move with the KML.
3. **Wrong coordinate order.** KML is lon/lat, not lat/lon.
4. **Duplicate labels.** Built-in basemap labels plus custom labels usually look bad.
5. **Unclear product semantics.** Current-status sampling, model forecasts, radar, and health advisories are different; label them plainly.
6. **Date/time mistakes.** Use tools for current dates, tide dates, and timezone-aware sunrise/sunset calculations.
7. **Overcrowded title/legend/footer.** Move legends to footer or resize/regenerate when overlap appears.

## Verification Checklist

- [ ] Source/layer/station and date/window are visible.
- [ ] Requested locations and labels are present and readable.
- [ ] Legend is readable and not hiding important data.
- [ ] Colors/line weights are visible against the basemap.
- [ ] Footer/source/caveat text is not cropped.
- [ ] Final image was visually inspected before delivery.

## References

This umbrella absorbed the previous `geospatial-map-rendering`, `geospatial-map-artifacts`, and `tide-forecast-graphics` skills. Their full prior SKILL.md files and supporting references/templates were preserved under this skill's `references/` directory.
