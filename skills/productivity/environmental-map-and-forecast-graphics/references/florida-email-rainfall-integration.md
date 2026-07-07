# Florida Email Task rainfall-map integration

Session-derived notes for adding Sarasota Water Atlas rainfall maps to Ron's daily Florida email packet.

## Goal

Include Water Atlas-style rainfall visuals for Manasota Key / Englewood in the Florida Email Task, matching the page-like station-box map Ron preferred:

- 24 Hour Total Rainfall
- 7 Day Total Rainfall
- 31 Day Total Rainfall

Use the same screenshot-like visual language as the Sarasota Water Atlas rainfall page: white/teal header, pale basemap, blue station value boxes, left radio panel with the active period selected, and a bottom-left `3 km` scale marker.

## Reusable renderer

Script path:

- `~/.hermes/scripts/sarasota-rainfall-wateratlas-maps.py`

Data source:

- API: `https://api.wateratlas.usf.edu/rainfall/latest/?s=8`
- Page/source label: `https://sarasota.wateratlas.usf.edu/rainfall/latest/`

Usage:

```bash
/usr/bin/python3 ~/.hermes/scripts/sarasota-rainfall-wateratlas-maps.py --period all
```

The script prints three `MEDIA:` paths in this order:

1. `24h` map using `total24h`
2. `7d` map using `total7d`
3. `31d` map using `total31d`

It caches OSM basemap tiles under `~/.hermes/cache/osm_tiles/` and writes generated maps under `~/.hermes/cache/sarasota_rainfall/`.

## Viewport/style defaults

Good screenshot-like viewport for lower Sarasota / Manasota Key / Englewood:

- lon min/max: `-82.555, -82.125`
- lat min/max: `26.835, 27.245`
- canvas: `760 x 1230`
- header: `112 px`
- map area height: `1065 px`
- Web Mercator tile zoom: `12`

Basemap styling used to mimic Water Atlas:

- OSM tiles, then `Color ~= 0.58`, `Contrast ~= 0.78`, `Brightness ~= 1.18`
- station total boxes: pale blue fill, dark outline, large values
- radio panel selects the requested period, with inactive options still visible

Manual nudges are important for lower/congested labels such as Indian Mound Park, Lemon Bay Park, Lemon Bay Canal, FRK-1/FRK-2, GOT-1/GOT-2, and Jacaranda Bridge so the Englewood / Manasota lower stations stay visible.

## Florida Email Task integration points

Main script:

- `~/.hermes/scripts/florida-email-task.py`

Add or preserve:

```python
RAINFALL_SCRIPT = Path.home() / ".hermes" / "scripts" / "sarasota-rainfall-wateratlas-maps.py"
```

Add helper:

```python
def run_rainfall_maps():
    cmd = [str(RAINFALL_SCRIPT), "--period", "all"]
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=240)
    if proc.returncode != 0:
        raise RuntimeError(f"Rainfall maps failed\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    media = [Path(line.replace("MEDIA:", "", 1).strip()) for line in proc.stdout.splitlines() if line.startswith("MEDIA:")]
    if len(media) < 3:
        raise RuntimeError(f"Expected 3 MEDIA paths from rainfall script, got {len(media)}\n{proc.stdout}")
    for p in media[:3]:
        if not p.exists() or p.stat().st_size < 1024:
            raise RuntimeError(f"Missing/empty rainfall image: {p}")
    return {"stdout": proc.stdout, "rainfall_24h": media[0], "rainfall_7d": media[1], "rainfall_31d": media[2]}
```

Then call it after the sargassum packet and before email construction:

```python
packet = run_sargassum_packet(args.date)
packet.update(run_rainfall_maps())
```

Embed the three inline images in the HTML email with CIDs:

- `rainfall_24h`
- `rainfall_7d`
- `rainfall_31d`

Add them to the related image loop with filenames:

- `sarasota-wateratlas-rainfall-24h.png`
- `sarasota-wateratlas-rainfall-7d.png`
- `sarasota-wateratlas-rainfall-31d.png`

## Verification

Run a dry build before relying on the cron job:

```bash
/usr/bin/python3 ~/.hermes/scripts/florida-email-task.py --to reisworth@gmail.com --dry-run
```

Verify the dry run prints all three rainfall image paths and that the preview `.eml` contains CIDs for `rainfall_24h`, `rainfall_7d`, and `rainfall_31d`.

Visually QA at least the 24-hour and 7-day maps after creating the renderer or changing the viewport:

- correct radio option selected
- lower Manasota Key / Englewood station boxes visible
- no severe crop or hidden station boxes
- `3 km` scale marker visible
- footer/source timestamp readable
