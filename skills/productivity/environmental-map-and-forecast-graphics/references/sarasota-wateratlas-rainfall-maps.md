# Sarasota Water Atlas rainfall maps — Manasota Key / Englewood

Session-derived workflow for rendering Water Atlas-style rainfall station maps for Ron's Florida packet.

## Source

- Page: `https://sarasota.wateratlas.usf.edu/rainfall/latest/`
- Public API used by the page: `https://api.wateratlas.usf.edu/rainfall/latest/?s=8`

The page exposes three station total fields:

- `total24h` → 24 Hour Total Rainfall
- `total7d` → 7 Day Total Rainfall
- `total31d` → 31 Day Total Rainfall

Station objects include `name`, `id`, `lastUpdated`, `stationUrl`, `location.latitude`, and `location.longitude`.

## User-preferred output

Ron preferred a screenshot-like Water Atlas/Leaflet presentation over a polished interpolated GIS surface for this use case:

- White Water Atlas-style header.
- Left control panel with the rainfall radio options and the active period selected.
- Pale basemap with station total boxes, not city callout-heavy cartography.
- A **3 km scale marker**.
- Focus zoomed to the **Manasota Key / Englewood / lower Sarasota** area.
- Lower stations must remain visible, including Englewood / Indian Mound Park, Lemon Bay Park, Lemon Bay Canal, Jacaranda Bridge, Venice Ave E, Capri Isle, Forked/Gottfried Creek-area stations.

## Working implementation on Ara1bot

Script now used by the Florida Email Task:

```text
~/.hermes/scripts/sarasota-rainfall-wateratlas-maps.py
```

Run all three periods:

```bash
/usr/bin/python3 ~/.hermes/scripts/sarasota-rainfall-wateratlas-maps.py --period all
```

Expected output directory:

```text
~/.hermes/cache/sarasota_rainfall/
```

## Viewport/settings that worked

```python
XMIN, XMAX = -82.555, -82.125
YMIN, YMAX = 26.835, 27.245
W, H = 760, 1230
ZOOM = 12
```

Rendering approach:

1. Fetch OSM tiles for the viewport.
2. Desaturate, reduce contrast, and brighten the basemap to resemble the Water Atlas map style.
3. Draw the Water Atlas-style header and selector panel.
4. Draw blue rainfall boxes at station locations.
5. Apply hand-tuned nudges for congested lower stations.
6. Add a 3 km scale marker.
7. QA visually for clipped labels, especially Lemon Bay Canal and lower Englewood-area stations.

## Integration notes

The Florida Email Task embeds the three rainfall maps inline with CIDs:

- `rainfall_24h`
- `rainfall_7d`
- `rainfall_31d`

The main email script calls the rainfall renderer before tide/red-tide generation and attaches all three PNGs.

## GitHub backup

The Florida email/maps automation was backed up to private GitHub repo:

```text
https://github.com/Araeisgit/FloridaEmail
```

Do not commit `.env` or credentials. A token may be configured in `gh`; verify access with `gh auth status` before pushing.
