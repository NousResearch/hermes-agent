---
name: geospatial-map-artifacts
description: "Create polished geospatial map artifacts from KML/KMZ/ZIP/raster overlays, with clear legends, labels, and visual verification."
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [maps, geospatial, kml, kmz, zip, raster-overlays, cartography, legends, visualization]
    created_by: agent
---

# Geospatial Map Artifacts

## Trigger

Use this skill when the user asks to process, visualize, improve, or create a map from:

- `.kml`, `.kmz`, `.zip`, or Google Earth files
- geospatial raster overlays such as `.png` images bundled with KML/KMZ
- map layers with risk zones, currents, density, routes, city labels, placemarks, or polygons
- iterative visual/cartographic edits: cleaner basemap, larger legend, label placement, colors, date/source attribution, etc.

## Core workflow

1. **Inspect the archive/layers first.**
   - Treat `.kmz` as a ZIP archive.
   - If the attachment was rejected before Hermes saved it, ask the user to resend as `.zip` or put the `.kmz` inside a `.zip`.
   - If accessible, copy rather than destructively rename: `file.kmz` → sibling `file.zip`.
   - Verify with Python `zipfile.is_zipfile()` and list archive contents.

2. **Extract geospatial structure.**
   - KML coordinate order is `longitude,latitude,altitude`, not `latitude,longitude`.
   - Parse `GroundOverlay` bounds (`north/south/east/west`) for raster images.
   - Count and classify KML elements: `Placemark`, `LineString`, `Polygon`, `Point`, `GroundOverlay`.
   - Extract metadata such as date, risk fields, layer names, and style colors.

3. **Choose a basemap appropriate to the requested visual style.**
   - For clean custom labeling: use a no-label basemap such as CartoDB Positron no-label tiles.
   - For a more colorful/interesting map: use topo/terrain tiles such as OpenTopoMap or another terrain/physical basemap.
   - If the user complains the map is boring, switch to a more visually textured topo/terrain basemap.
   - If the user wants the agent to add all labels manually, avoid basemaps with prominent built-in city labels when possible.

4. **Render overlays carefully.**
   - Raster overlays in KML may be geographic/equirectangular while web-map tiles are Web Mercator; resample by converting output pixels back to lon/lat before sampling source image pixels.
   - Make near-white or empty raster backgrounds transparent.
   - Keep density overlays semi-transparent so the basemap remains readable.
   - Current/vector raster layers can look like broad smudges; reduce opacity aggressively and/or show them as subtle directional texture rather than dominant marks.
   - Draw risk contours/lines after density/current overlays so they remain visible.

5. **Legend and labels.**
   - Legends should be large, readable, and placed near the mapped area without covering important data/city labels.
   - Include the data date in the legend when the source contains a date.
   - Use same-size legend swatches/rectangles for comparable categories.
   - If low risk is represented by light blue/cyan in the source, explicitly include “Low risk = light blue” in the legend.
   - Use bright, high-contrast risk colors when the user says risk colors are hard to see.
   - For city labels, use either:
     - callout boxes with leader lines when labels would collide with basemap text, or
     - direct on-map text with a white halo when the user wants labels directly on the map.
   - Add relevant contextual cities beyond the initial coast if requested (e.g., Atlantic-side cities for Florida maps).

6. **Verify the generated artifact visually before delivery.**
   - Use vision analysis or a screenshot/image review.
   - Check: legend readability, data date, labels visible, overlays not obscuring basemap, requested cities included, color/line-size issues fixed.
   - If a revision is clearly needed, fix and re-render before sending.

## Implementation notes

Python + PIL + NumPy is sufficient for many static map artifacts:

- Fetch slippy map tiles by z/x/y.
- Convert lon/lat to Web Mercator global pixels.
- Crop the tile mosaic to the requested bounding box.
- Resample KML `GroundOverlay` rasters by mapping each output pixel to lon/lat and then into the source raster.
- Draw KML `LineString` risk contours with `ImageDraw.line`.
- Draw city markers/text and legend on a final upscaled image.

Prefer using existing installed libraries when available, but do not require heavy GIS stacks (`geopandas`, `cartopy`) for simple KML/raster overlay maps.

## Pitfalls

- Do not claim a `.kmz` was renamed if Hermes never received/saved the file.
- Do not assume a KMZ has only `doc.kml`; list archive contents and find all `.kml` files.
- Do not leave legends tiny or far from the relevant geography; map artifacts are visual deliverables, not just data dumps.
- Do not let a basemap’s built-in city labels fight with custom labels. Switch basemap or use halos/callouts.
- Do not make current/vector raster layers too opaque; they can read as smudges.
- Do not make all risk categories line-width-coded if the user asked for same-size legend rectangles or equal visual weight.

## Reference examples

- `references/florida-sargassum-map-20260611.md` — session notes and implementation lessons from a Florida Gulf Coast sargassum KMZ/ZIP map, including legend/color/label iterations.
