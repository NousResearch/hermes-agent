# Sargassum KMZ/ZIP map rendering notes

These notes came from an iterative Florida Gulf Coast sargassum map task using an uploaded ZIP renamed from/standing in for KMZ:

- Archive: `Sargassum_risk_currents_FA_20260611.zip`
- Main KML: `risk_currents_FA.kml`
- Raster overlays: `FA_density.png`, `currents.png`
- KML date field: `20260611` / displayed as `2026-06-11`

## Upload/ingestion workaround

KMZ files are ZIP containers, but the document-ingestion layer may reject the `.kmz` extension before the agent can access the file. When that happens, do not imply a local rename will fix the already-rejected upload. Ask the user to upload one of these instead:

- the same file renamed from `.kmz` to `.zip`
- a `.zip` containing the original `.kmz`
- extracted `.kml` plus referenced raster/image assets

Once saved locally, inspect it as a normal ZIP/KMZ archive and extract KML/raster assets from the package.

## File structure encountered

Typical contents:

- `risk_currents_FA.kml`
- `currents.png`
- `FA_density.png`
- `images/*Logo*.png`
- `images/risk_colors.png`

The KML contained thousands of `Placemark` features with `LineString` geometries and `ExtendedData` like:

- `SimpleData name="risk"`: classes `0`, `1`, `2`, `3`
- `SimpleData name="date"`: `YYYYMMDD`

The raster overlays were intended to be georeferenced with KML `GroundOverlay` bounds. In this session the overlay bounds were:

- west: `-100`
- east: `-50`
- south: `2`
- north: `35`

Always read bounds from the actual KML rather than hardcoding them.

## Rendering decisions that worked

Layer order:

1. No-label physical/topographic basemap.
2. `FA_density.png`, with near-white/empty pixels transparent.
3. `currents.png`, with opacity reduced heavily to avoid broad smudges.
4. KML risk lines/contours, recolored brightly.
5. Custom labels and legend.

Risk colors that were easier to see:

- risk `0`: bright light blue / low risk
- risk `1`: yellow
- risk `2`: orange
- risk `3`: red

Legend preferences:

- Include the data date inside the legend.
- Use same-size rectangles for density, currents, and all risk classes.
- Show a small arrow cue for currents if possible.

City/context preferences for Florida maps:

- Include Gulf cities: Pensacola, Destin, Panama City, Apalachicola, Cedar Key, Crystal River, Tampa, Clearwater, St. Petersburg, Sarasota, Venice, Manasota Key, Punta Gorda, Fort Myers, Naples, Marco Island, Key West.
- Include Atlantic/interior context cities: Fernandina Beach, Jacksonville, St. Augustine, Daytona Beach, Cape Canaveral, Melbourne, Orlando, Vero Beach, Port St. Lucie, West Palm Beach, Boca Raton, Fort Lauderdale, Miami.
- If the basemap already has labels, do not add duplicate callouts. Switch basemap instead.

## Basemap lesson

OpenTopoMap looked better/colorful but included built-in city labels, causing duplicate city callouts. Esri World Terrain Base avoided duplicate labels but looked too gray/plain. Esri World Physical Map was the best compromise in this task: more color/topographic texture with no obvious duplicate city callouts at the Florida Gulf/Atlantic scale. Always visually verify because tile-provider styling can change.

## Daily cron workflow

Use `/home/ara1/.hermes/scripts/daily-sargassum-map.py` for the recurring NOAA/AOML Florida sargassum map job.

- The source URL pattern is `https://cwcgom.aoml.noaa.gov/SIR/KMZ/Sargassum_risk_currents_FA_<YYYYMMDD>.kmz`.
- The script accepts `--date YYYYMMDD` for tests and otherwise tries today's local date, then walks backward through recent dates so a 6am run still succeeds if the newest product has not posted yet.
- It downloads the `.kmz`, validates it as a ZIP/KMZ with KML inside, reads KML `GroundOverlay` bounds and `SimpleData name="date"`, renders the final sargassum map, then also downloads supporting public weather images and prints multiple `MEDIA:/absolute/path.png` lines for direct Telegram delivery.
- Supporting images currently included:
  - NHC 7-day Atlantic tropical outlook: `https://www.nhc.noaa.gov/xgtwo/resize/xgtwo_atl_7d0_w1920.png`
  - Public NOAA/NWS Southeast radar mosaic including Florida: `https://radar.weather.gov/ridge/standard/SOUTHEAST_0.gif` converted to PNG for delivery.
- Historical sargassum test that succeeded: `python3 ~/.hermes/scripts/daily-sargassum-map.py --date 20260601 --force` rendered `/home/ara1/.hermes/cache/sargassum_maps/florida_sargassum_risk_density_currents_20260601.png` with 1673 risk features drawn.
- Full packet trial that succeeded on 2026-06-12 produced the sargassum map, NHC tropical outlook image, and NOAA/NWS Southeast radar image.
- Cron should be script-only/no-agent at `0 6 * * *`, delivered to the origin/home chat.
- Hermes cron runs `.py` scripts with the scheduler's `sys.executable`, not the file shebang. On Ara1bot, the sargassum script needs system Pillow/NumPy, so schedule the bash wrapper `daily-sargassum-map.sh`, which execs `/usr/bin/python3 /home/ara1/.hermes/scripts/daily-sargassum-map.py`, rather than scheduling `daily-sargassum-map.py` directly.

## Florida Email Task

Daily email variant:

- Script: `/home/ara1/.hermes/scripts/florida-email-task.py`
- Cron wrapper: `/home/ara1/.hermes/scripts/florida-email-task-send.sh`
- Cron job name: `Florida Email Task`
- Schedule: `20 6 * * *`
- Recipient safety default: only `reisworth@gmail.com`
- Email contents:
  - inline sargassum map from `daily-sargassum-map.py`
  - inline NHC 7-day Atlantic tropical outlook
  - inline NOAA/NWS Southeast radar map including Florida
  - SandBar Tiki & Grille next-5-day live music lineup from `https://www.sandbartikigrille.com/events/`
- SandBar parsing note: the page has event cards with `<div class="day/month/date">`, `<h2 class="post-title">`, and meta `<span class="sr">Time</span>` / `<span class="sr">Genre</span>`; parse those fields directly to avoid truncating genres such as `Rock, Pop, Country`.
- Email send uses Python `smtplib` with `EMAIL_ADDRESS`, `EMAIL_PASSWORD`, and `EMAIL_SMTP_HOST` from `~/.hermes/.env`.

## Common pitfalls

- Current rasters can look like thick gray/yellow smudges when too opaque. Start with low opacity and inspect.
- A plain no-label basemap can look boring; prefer no-label physical/topographic layers when available.
- Offshore boxed callouts can be legible but may feel less polished; direct map labels with a white halo often look cleaner.
- User may iterate on cartographic aesthetics several times; treat visual QA as part of the task, not an optional extra.
