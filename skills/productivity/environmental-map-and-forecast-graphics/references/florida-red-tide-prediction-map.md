# Florida red tide prediction map workflow

Session-derived workflow for making a Telegram-ready Florida red tide prediction map.

## Best current data sources

- USF Ocean Circulation Lab WFCOM HAB Tracking landing page:
  - `https://ocl.marine.usf.edu/Models/WFCOM4/HAB_tracking/`
- Current surface trajectories image:
  - `https://ocl.marine.usf.edu/Models/WFCOM4/HAB_tracking/upper_R0.gif`
- Current subsurface / near-bottom trajectories image:
  - `https://ocl.marine.usf.edu/Models/WFCOM4/HAB_tracking/lower_R0.gif`
- Current coastal concentration/forecast image:
  - `https://ocl.marine.usf.edu/Models/WFCOM4/HAB_tracking/hab_coastal_WFS_R0.png`
- FWC red tide statewide status page:
  - `https://myfwc.com/research/redtide/statewide/`

The FWC page includes current summary text such as:

```text
Short-term (3.5 day) forecasts provided by the USF-FWC Collaboration for Prediction of Red Tides predict ...
```

It also links current statewide maps/status files with dated filenames, e.g. `statewidemap0612.jpg` and tables/maps by region.

## Recommended deliverable

For a user asking “create a map showing the red tide prediction for Florida,” produce a polished composite rather than forwarding a raw source image:

1. Download the current `upper_R0.gif` and `lower_R0.gif` images.
2. Extract the current FWC forecast-summary sentence from the FWC page.
3. Compose a single PNG with:
   - title: “Florida Red Tide Prediction Map”
   - subtitle: “USF–FWC short-term HAB trajectory forecast • surface and subsurface transport”
   - two panels: “Surface water trajectories” and “Subsurface / near-bottom trajectories”
   - forecast summary card using the FWC short-term forecast text
   - “How to read the forecast” legend:
     - X marks: sample/starting locations
     - Colored lines: forecast transport by *K. brevis* category
     - White line: present/drifter trajectory
   - source footer naming USF OCL WFCOM HAB Tracking and FWC Red Tide Current Status
   - note that this is model-based transport guidance, not a health advisory
4. Visually inspect the final PNG before sending; ensure footer/source text is not cropped and summary text does not overlap the legend.

## Implementation notes

- Use Pillow directly; it worked reliably for composing the source GIF/PNG images into a clean Telegram-ready composite.
- Source GIFs are 757×800 and can be resized into side-by-side panels.
- Keep the raw map legends inside each source panel; add a simplified external legend only to explain the main visual cues.
- The raw USF trajectory panels may show a time span like `From 2026-06-11 20 ET To 2026-06-11 20 ET`; preserve the source image as-is rather than rewriting its internal annotations.
- The final composite size that worked well: about `1800×1360` pixels.

## Pitfalls

- Do not present this as a beach-health advisory. It is a model-based HAB transport forecast and local conditions can change quickly.
- The FWC statewide status map is useful context, but the USF `upper_R0.gif`/`lower_R0.gif` trajectory images are better for a “prediction” map because they show forecast transport.
- If using footer text from live pages, wrap it into a footer box; long source URLs can get cropped on the right edge if drawn as one line.
