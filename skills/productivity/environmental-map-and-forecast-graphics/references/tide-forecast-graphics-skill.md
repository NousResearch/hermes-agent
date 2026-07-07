---
name: tide-forecast-graphics
description: Create NOAA tide forecast graphs with daylight shading, high/low labels, and Telegram-ready image delivery.
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux]
metadata:
  hermes:
    tags: [tides, noaa, coops, forecast, chart, daylight, sunrise, sunset, telegram, visualization]
    category: productivity
    requires_toolsets: [terminal, vision]
---

# Tide Forecast Graphics

Use this skill when the user asks for a tide chart/graph/forecast for a coastal location, especially if they want it pasted/sent to Telegram as an image.

## Workflow

1. **Resolve the tide station**
   - Search NOAA CO-OPS / Tides & Currents for the nearest station.
   - If the user names a beach/island rather than a station, use the nearest relevant NOAA station and state it in the footer.
   - For Manasota Key, FL, a good station is `8725809` — Manasota, FL.

2. **Fetch NOAA tide predictions**
   - Use NOAA CO-OPS datagetter API with `product=predictions`, `datum=MLLW`, `time_zone=lst_ldt`, `units=english`, `format=json`.
   - Use `interval=6` for a smooth curve.
   - Use `interval=hilo` for high/low marker labels.
   - For a one-day chart, make the x-axis exactly local `12:00 AM` to next-day `12:00 AM`, not just the API's last data point.

3. **Compute daylight accurately**
   - Use the station or location coordinates and local timezone.
   - Shade daylight hours with a light yellow background from sunrise to sunset.
   - Label sunrise and sunset times on the chart and repeat them in the footer.
   - Watch for sunset calculations returning the previous local date when converting from UTC; if sunset is earlier than sunrise for the target date, add one day.

4. **Render a polished graph**
   - Prefer a high-resolution PNG, e.g. 1600px wide.
   - Use a readable title, subtitle, y-axis in feet above MLLW, local-time x-axis, high/low callouts, and a source footer.
   - Use light yellow for daylight, pale blue/gray for night, blue tide fill/line, and orange low-tide markers.
   - Keep legends out of the title/subtitle area; footer legends are safer for Telegram image previews.

5. **Verify visually before sending**
   - Use vision QA before final delivery.
   - Check that the chart is the requested span, daylight shading starts/ends at the labeled sunrise/sunset, labels are legible, and no legend/title/footer overlaps.
   - If there is overlap, patch/regenerate the image before sending.

6. **Deliver to Telegram**
   - Include `MEDIA:/absolute/path/to/file.png` in the final response.
   - Briefly mention the source station and sunrise/sunset basis.

## Pitfalls

- Do not use mental date/time math; use tools for current date and timezone-aware calculations.
- Do not rely on `matplotlib` being installed. Pillow (`PIL`) can draw a complete chart with lines, filled polygons, labels, and legends.
- NOAA `end_date` for a single date may return points through 23:54; set the x-axis to next midnight manually for a true full-day chart.
- Visual overlap is common when placing legends at the top right; after QA, move legends to the footer if they cover the subtitle.
- If the user asks for daylight shading, calculate sunrise/sunset for the actual location/date rather than approximating by fixed hours.

## Reference

See `references/noaa-tide-graph-pattern.md` for API endpoints, rendering details, and the Manasota Key example from the session that created this skill.
