# Sarasota Water Atlas rainfall maps for Florida Email Task

Session-derived workflow for rendering Sarasota Water Atlas-style rainfall maps focused on Manasota Key / Englewood and preserving the workflow in GitHub.

## Data source

- Page: `https://sarasota.wateratlas.usf.edu/rainfall/latest/`
- API used by the page: `https://api.wateratlas.usf.edu/rainfall/latest/?s=8`
- Key fields:
  - `total24h`
  - `total7d`
  - `total31d`
  - `name`, `id`, `lastUpdated`, `location.latitude`, `location.longitude`

## User preference / expected visual style

Ron preferred the Water Atlas-style station map over a polished interpolated thematic map for the email. When he asks for rainfall maps from this source:

1. Default to Water Atlas-style station amount boxes unless he explicitly asks for interpolation/contours.
2. Keep the map zoomed into Manasota Key / Englewood, not all of Sarasota County.
3. Include the lower stations around Manasota Key / Englewood, especially:
   - CST-3 Indian Mound Park / Englewood area
   - Lemon Bay Park
   - Lemon Bay Canal
   - Jacaranda Bridge
   - Venice Ave E
   - Capri Isle
   - Forked Creek / Gottfried Creek area stations
4. Include a 3 km scale marker when requested. In the Water Atlas-style version, this is a scale marker, not a 3 km gridded interpolation.
5. For the daily Florida email, include all three period maps: 24-hour, 7-day, and 31-day totals.

## Known-good renderer on Ara1bot

Script:

```text
~/.hermes/scripts/sarasota-rainfall-wateratlas-maps.py
```

Typical commands:

```bash
/usr/bin/python3 ~/.hermes/scripts/sarasota-rainfall-wateratlas-maps.py --period all
/usr/bin/python3 ~/.hermes/scripts/sarasota-rainfall-wateratlas-maps.py --period 24h
/usr/bin/python3 ~/.hermes/scripts/sarasota-rainfall-wateratlas-maps.py --period 7d
/usr/bin/python3 ~/.hermes/scripts/sarasota-rainfall-wateratlas-maps.py --period 31d
```

Output directory:

```text
~/.hermes/cache/sarasota_rainfall/
```

Important viewport settings from the accepted version:

```python
XMIN, XMAX = -82.555, -82.125
YMIN, YMAX = 26.835, 27.245
W, H = 760, 1230
ZOOM = 12
```

Rendering approach:

- Fetch public Water Atlas API JSON directly rather than scraping the rendered page.
- Use OpenStreetMap tiles, then desaturate/brighten them to resemble the Water Atlas Leaflet/Esri style.
- Draw the white Water Atlas-style header and gray angled corner.
- Draw the radio selector panel and select the requested period.
- Draw station totals as pale-blue boxes similar to the Water Atlas site.
- Use hand-tuned nudges for congested lower-station labels so they do not crop at the image edge.

## Florida Email Task integration

Main script:

```text
~/.hermes/scripts/florida-email-task.py
```

Wrapper:

```text
~/.hermes/scripts/florida-email-task-send.sh
```

The full email workflow now includes:

1. Sargassum packet.
2. Sarasota Water Atlas rainfall maps for 24h, 7d, 31d.
3. Manasota Key tide graph.
4. FWC red tide current-status map.
5. SWFLLive Englewood live music lineup filtered to Beachcomber Trading Post, SandBar Tiki & Grille, and White Elephant Pub.
6. HTML email with CID inline images.

Dry-run verification:

```bash
/usr/bin/python3 ~/.hermes/scripts/florida-email-task.py --to reisworth@gmail.com --dry-run
```

The dry run should produce an `.eml` preview under:

```text
~/.hermes/cache/florida_email_task/
```

## GitHub preservation

Local repo created to preserve scripts and instructions:

```text
/home/ara1/florida-email-maps
```

It includes:

- `scripts/florida-email-task.py`
- `scripts/florida-email-task-send.sh`
- `scripts/daily-sargassum-map.py`
- `scripts/sarasota-rainfall-wateratlas-maps.py`
- `scripts/manasota-tide-graph.py`
- `scripts/fwc-red-tide-current-status-map.py`
- `README.md`
- `docs/map-workflows.md`

Before pushing, scan for actual secrets, run Python syntax checks, and commit locally.

GitHub repo created/pushed after token access was fixed:

```text
https://github.com/Araeisgit/FloridaEmail
```

Observed auth pitfall: a GitHub token can be valid for `gh repo list`/repo reads but still fail repository creation with:

```text
GraphQL: Resource not accessible by personal access token (createRepository)
```

When that happens, either create the empty private repo manually and grant the token access, or re-authenticate `gh` with a classic PAT that includes `repo` scope and repository-creation permission. After access is fixed, merge any GitHub-created initial README commit with `--allow-unrelated-histories` if needed, keep the fuller local README, then push `main`.
