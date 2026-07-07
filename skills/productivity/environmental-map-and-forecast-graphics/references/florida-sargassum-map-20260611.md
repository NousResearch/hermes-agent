# Florida sargassum map session notes — 2026-06-11 source data

## Source archive shape

The user uploaded `Sargassum_risk_currents_FA_20260611.zip`, originally discussed as a `.kmz`-style Google Earth package. Contents included:

- `risk_currents_FA.kml` — main KML, about 12.5 MB
- `currents.png` — current data raster overlay
- `FA_density.png` — sargassum/floating algae density raster overlay
- logo and legend image assets under `images/`

KML findings:

- `GroundOverlay` bounds for both rasters were `west=-100`, `east=-50`, `south=2`, `north=35`.
- The KML had many `LineString` placemarks with `ExtendedData` fields: `risk` and `date`.
- Date field was `20260611`, rendered as `2026-06-11`.
- Risk values observed: `0`, `1`, `2`, `3`.
- KML style colors are AABBGGRR; `ffffffb2` corresponds to light blue/cyan and should be treated as low risk in this source family.

## Map rendering technique that worked

A static PNG was generated with Python, PIL, and NumPy:

1. Fetch slippy-map tiles for the bounding box around Florida and the Gulf.
2. Build a tile mosaic and crop it to the desired Web Mercator bounding box.
3. For raster overlays, convert each output pixel to lon/lat and sample the source raster using the KML `GroundOverlay` bounds.
4. Make near-white raster backgrounds transparent.
5. Composite layers in this order:
   - basemap
   - density overlay, semi-transparent
   - current overlay, very subtle
   - risk lines/contours, bright high-contrast
   - city labels, title, legend

Useful bounding box for Florida + Gulf context:

```python
W, S, E, N = -88.8, 23.7, -79.6, 31.2
ZOOM = 7
```

## User-driven visual preferences from the session

The map improved through several corrections:

- Initial city labels conflicted with built-in basemap labels. Fix: use a cleaner basemap or custom labels with halos/callouts.
- The legend was too small/far away. Fix: make it large, place it near Florida but not over key data, and include the data date.
- The user wanted `Manasota Key` added.
- The user wanted a basemap that was less boring and showed Florida topology/terrain in different colors. Fix: switch from a no-label neutral basemap to a more colorful topo/terrain/physical basemap when visual richness is requested.
- The user wanted city callouts directly on the map, not offshore callout boxes. Fix: place text near real city points using white halo text instead of opaque boxes.
- Add major Atlantic-side cities for context: Jacksonville, St. Augustine, Daytona Beach, Cape Canaveral, Orlando, West Palm Beach, Fort Lauderdale, Miami.
- Current lines/raster were too thick and smudgy. Fix: reduce current overlay opacity aggressively; keep it subtle.
- Risk colors needed to be brighter. Fix: use bright cyan/yellow/orange/red and draw risk contours after current/density overlays.
- Legend risk/category swatches should be same-size rectangles, not mixed line samples and different-sized blocks.

## Final legend semantics used

- Currents — yellow rectangle with arrow cue
- Density — green rectangle
- Low risk — bright light blue/cyan
- Risk 1 — yellow
- Risk 2 — orange
- Risk 3 — red / highest

## Verification checklist

Before delivering a revised map:

- Does the basemap match the requested style: clean/no-label vs topo/terrain?
- Are requested Gulf and Atlantic cities present?
- Is Manasota Key included when relevant?
- Are labels readable without overwhelming the map?
- Does the legend include the data date?
- Are legend swatches uniform where categories are comparable?
- Are current overlays subtle enough not to look like smudges?
- Are risk colors bright and visible on top of the basemap?
