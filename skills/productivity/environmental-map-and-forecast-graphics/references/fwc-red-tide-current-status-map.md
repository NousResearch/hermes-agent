# FWC Red Tide Current Status Map

Use this reference when the user asks to use the FWC interactive red tide map at `https://gis.myfwc.com/redtidecurrentstatus/`, or asks to include that current-status map in the Florida daily email.

## Source

- Public web app: `https://gis.myfwc.com/redtidecurrentstatus/`
- ArcGIS web map item discovered from the app: `032578e992e04012a25db77499318ff9` — "Florida Red Tide Current Status Map"
- Public feature layer:
  `https://services2.arcgis.com/z6TmTIyYXEYhuNM0/arcgis/rest/services/HAB_Current_Web_Layer/FeatureServer/0`

No credentials are required for the REST layer.

## Query pattern

Use ArcGIS REST `query` with JSON output and geometries:

```text
https://services2.arcgis.com/z6TmTIyYXEYhuNM0/arcgis/rest/services/HAB_Current_Web_Layer/FeatureServer/0/query
  ?where=1%3D1
  &outFields=*
  &returnGeometry=true
  &f=json
  &returnExceededLimitFeatures=true
```

Useful fields observed:

- `SampleDate_t` / `SAMPLE_DATE` — sample date fields
- `LOCATION`
- `COUNTY`
- `LATITUDE`, `LONGITUDE`
- `Abundance`
- `SAMPLE_DEPTH`

Abundance categories:

- `not present/background (0-1,000)`
- `very low (>1,000-10,000)`
- `low (>10,000-100,000)`
- `medium (>100,000-1,000,000)`
- `high (>1,000,000)`

## Rendering guidance

- Treat this as a current-status sampling map, not a forward trajectory prediction model.
- Query the layer directly instead of screenshotting the app; direct data access gives cleaner styling, reliable legends, and attachment-ready images.
- Cache generated images under `~/.hermes/cache/red_tide/` for cron/email workflows.
- A Pillow-only renderer is sufficient when geopandas/pyshp are unavailable:
  - Use public state/county outline GeoJSON for context.
  - Project lon/lat to the image with fixed Florida bounds, e.g. roughly `x=-87.9..-79.6`, `y=24.0..31.4`.
  - Draw a light water/land basemap, state outline, and county outlines.
  - Render sample points by abundance with readable dot sizes. For `not present/background`, mimic the FWC legend with a white/empty dot and dark outline.
  - Add a local highlight ring for Manasota Key when the map is for Ron's Florida packet.
- Include a date range in the subtitle or legend derived from the returned sample dates. The FWC app describes the data as the most recent eight days of sampling.
- Add a clear note: "current-status sampling map, not a forward trajectory prediction model." This avoids conflating it with the USF/FWC HAB trajectory forecast.

## Visual QA checklist

- Title and subtitle are not cropped.
- Dots are visible at email/Telegram size, including when all points are `not present/background`.
- Legend includes all five abundance classes and counts.
- Source/footer is readable and includes the FWC web map URL.
- No important labels or west/east coast clusters are hidden by cards.
- If all current points are background/no elevated samples, state that explicitly in the summary card so the mostly white/outlined markers do not look like missing data.

## Florida daily email integration

When integrating into `~/.hermes/scripts/florida-email-task.py`:

1. Put the reusable renderer at `~/.hermes/scripts/florida-red-tide-map.py` with an absolute `/usr/bin/python3` shebang if it depends on system packages.
2. Have the email task call the renderer and attach the output inline with a stable CID such as `red_tide_map`.
3. Add an HTML section after the existing weather/coastal images with a short explanation and source link.
4. Keep success silent in the cron wrapper; only print logs on failure.
5. Dry-run by generating an `.eml` or sending a controlled test and verify the CID is present and the image displays inline.
