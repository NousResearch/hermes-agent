# NOAA Tide Graph Pattern

Session-derived pattern for producing Telegram-ready tide charts.

## NOAA CO-OPS API

Smooth tide curve:

```text
https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?begin_date=YYYYMMDD&end_date=YYYYMMDD&station=STATION_ID&product=predictions&datum=MLLW&time_zone=lst_ldt&units=english&interval=6&format=json
```

High/low labels:

```text
https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?begin_date=YYYYMMDD&end_date=YYYYMMDD&station=STATION_ID&product=predictions&datum=MLLW&time_zone=lst_ldt&units=english&interval=hilo&format=json
```

Station metadata:

```text
https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations/STATION_ID.json
```

Manasota Key example:

- Nearby station: `8725809`
- Station name from metadata: `MANASOTA`
- Coordinates: `27.0117, -82.41`
- Timezone for rendering: `America/New_York`
- Source label: `NOAA CO-OPS tide predictions API; station Manasota, FL`

## One-day x-axis

For a full-day chart, set:

```python
t0 = datetime.combine(date, time(0), tzinfo=tz)
t1 = t0 + timedelta(days=1)
```

Map x positions against `t0..t1`, even when the final NOAA 6-minute point is 23:54. This prevents the chart from visually ending early.

## Daylight shading

Use NOAA-style solar calculation or an equivalent trustworthy library using the station/location latitude and longitude. The implementation used in-session exposed a date-boundary pitfall: sunset converted from UTC appeared as the previous local date. Guard with:

```python
if sunset < sunrise:
    sunset += timedelta(days=1)
```

Shade only `sunrise..sunset` with light yellow. Label sunrise/sunset on the chart and in the footer.

## Pillow rendering notes

`matplotlib` may not be installed on the Raspberry Pi environment. Pillow is sufficient:

- `Image.new('RGB', (1600, 1000), background)`
- Draw grid lines and labels with `ImageDraw`.
- Draw daylight rectangle before grid and tide curve.
- Draw tide fill with `polygon([(x0, baseline)] + curve_points + [(xN, baseline)])`.
- Draw tide line with `line(points, width=5 or 6)`.
- Use DejaVu fonts from `/usr/share/fonts/truetype/dejavu/` when present.

## Visual QA checklist

Before sending to Telegram, inspect the rendered PNG and verify:

- Requested time range is visible, e.g. 12 AM through evening and next midnight implied.
- Daylight yellow starts at sunrise and ends at sunset.
- Title and subtitle are not covered by the legend.
- High/low callouts do not hide the main curve too badly.
- Footer/source text is not cut off.
- Legend is preferably in the footer if the top area is crowded.

## Telegram delivery

Final response can be short:

```text
Updated tide graph with one full day and daylight shading:

MEDIA:/absolute/path/to/chart.png
```

Mention station/source only if useful; avoid long explanations unless the user asks.
