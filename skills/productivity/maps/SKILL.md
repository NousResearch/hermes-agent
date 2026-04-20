---
name: maps
description: >
  Location intelligence — geocode a place, reverse-geocode coordinates,
  find nearby places (44 POI categories), driving/walking/cycling
  distance + time, turn-by-turn directions, timezone lookup, bounding
  box + area for a named place, and POI search within a rectangle.
  Uses OpenStreetMap + Overpass + OSRM. Free, no API key.
version: 1.2.0
author: Mibayy
license: MIT
metadata:
  hermes:
    tags: [maps, geocoding, places, routing, distance, directions, nearby, location, openstreetmap, nominatim, overpass, osrm]
    category: productivity
    requires_toolsets: [terminal]
    supersedes: [find-nearby]
---

# Maps Skill

Location intelligence using free, open data sources. 8 commands, 44 POI
categories, zero dependencies (Python stdlib only), no API key required.

Data sources: OpenStreetMap/Nominatim, Overpass API, OSRM, TimeAPI.io.

This skill supersedes the old `find-nearby` skill — all of find-nearby's
functionality is covered by the `nearby` command below, with the same
`--near "<place>"` shortcut and multi-category support.

## When to Use

- User sends a Telegram location pin (latitude/longitude in the message) → `nearby`
- User wants coordinates for a place name → `search`
- User has coordinates and wants the address → `reverse`
- User asks for nearby restaurants, hospitals, pharmacies, hotels, etc. → `nearby`
- User wants driving/walking/cycling distance or travel time → `distance`
- User wants turn-by-turn directions between two places → `directions`
- User wants timezone information for a location → `timezone`
- User wants to search for POIs within a geographic area → `area` + `bbox`

## Prerequisites

Python 3.8+ (stdlib only — no pip installs needed).

Script path: `~/.hermes/skills/maps/scripts/maps_client.py`

## Commands

```bash
MAPS=~/.hermes/skills/maps/scripts/maps_client.py
```

### search — Geocode a place name

```bash
python3 $MAPS search "Eiffel Tower"
python3 $MAPS search "1600 Pennsylvania Ave, Washington DC"
```

Returns: lat, lon, display name, type, bounding box, importance score.

### reverse — Coordinates to address

```bash
python3 $MAPS reverse 48.8584 2.2945
```

Returns: full address breakdown (street, city, state, country, postcode).

### nearby — Find places by category

```bash
# By coordinates (from a Telegram location pin, for example)
python3 $MAPS nearby 48.8584 2.2945 restaurant --limit 10
python3 $MAPS nearby 40.7128 -74.0060 hospital --radius 2000

# By address / city / zip / landmark — --near auto-geocodes
python3 $MAPS nearby --near "Times Square, New York" --category cafe
python3 $MAPS nearby --near "90210" --category pharmacy

# Multiple categories merged into one query
python3 $MAPS nearby --near "downtown austin" --category restaurant --category bar --limit 10
```

44 categories: restaurant, cafe, bar, hospital, pharmacy, hotel, supermarket,
atm, gas_station, parking, museum, park, school, university, bank, police,
fire_station, library, airport, train_station, bus_stop, church, mosque,
synagogue, dentist, doctor, cinema, theatre, gym, swimming_pool, post_office,
convenience_store, bakery, bookshop, laundry, car_wash, car_rental,
bicycle_rental, taxi, veterinary, zoo, playground, stadium, nightclub.

Each result includes: `name`, `address`, `lat`/`lon`, `distance_m`,
`maps_url` (clickable Google Maps link), `directions_url` (Google Maps
directions from the search point), and promoted tags when available —
`cuisine`, `hours` (opening_hours), `phone`, `website`.

### distance — Travel distance and time

```bash
python3 $MAPS distance "Paris" --to "Lyon"
python3 $MAPS distance "New York" --to "Boston" --mode driving
python3 $MAPS distance "Big Ben" --to "Tower Bridge" --mode walking
```

Modes: driving (default), walking, cycling. Returns road distance, duration,
and straight-line distance for comparison.

### directions — Turn-by-turn navigation

```bash
python3 $MAPS directions "Eiffel Tower" --to "Louvre Museum" --mode walking
python3 $MAPS directions "JFK Airport" --to "Times Square" --mode driving
```

Returns numbered steps with instruction, distance, duration, road name, and
maneuver type (turn, depart, arrive, etc.).

### timezone — Timezone for coordinates

```bash
python3 $MAPS timezone 48.8584 2.2945
python3 $MAPS timezone 35.6762 139.6503
```

Returns timezone name, UTC offset, and current local time.

### area — Bounding box and area for a place

```bash
python3 $MAPS area "Manhattan, New York"
python3 $MAPS area "London"
```

Returns bounding box coordinates, width/height in km, and approximate area.
Useful as input for the bbox command.

### bbox — Search within a bounding box

```bash
python3 $MAPS bbox 40.75 -74.00 40.77 -73.98 restaurant --limit 20
```

Finds POIs within a geographic rectangle. Use `area` first to get the
bounding box coordinates for a named place.

## Working With Telegram Location Pins

When a user sends a location pin, the message contains `latitude:` and
`longitude:` fields. Extract those and pass them straight to `nearby`:

```bash
# User sent a pin at 36.17, -115.14 and asked "find cafes nearby"
python3 $MAPS nearby 36.17 -115.14 cafe --radius 1500
```

Present results as a numbered list with names, distances, and the
`maps_url` field so the user gets a tap-to-open link in chat. For "open
now?" questions, check the `hours` field; if missing or unclear, verify
with `web_search` since OSM hours are community-maintained and not always
current.

## Workflow Examples

**"Find Italian restaurants near the Colosseum":**
1. `nearby --near "Colosseum Rome" --category restaurant --radius 500`
   — one command, auto-geocoded

**"What's near this location pin they sent?":**
1. Extract lat/lon from the Telegram message
2. `nearby LAT LON cafe --radius 1500`

**"How do I walk from hotel to conference center?":**
1. `directions "Hotel Name" --to "Conference Center" --mode walking`

**"What restaurants are in downtown Seattle?":**
1. `area "Downtown Seattle"` → get bounding box
2. `bbox S W N E restaurant --limit 30`

## Pitfalls

- Nominatim ToS: max 1 req/s (handled automatically by the script)
- `nearby` requires lat/lon OR `--near "<address>"` — one of the two is needed
- OSRM routing coverage is best for Europe and North America
- Overpass API can be slow during peak hours; the script automatically
  falls back between mirrors (overpass-api.de → overpass.kumi.systems)
- `distance` and `directions` use `--to` flag for the destination (not positional)
- If a zip code alone gives ambiguous results globally, include country/state

## Verification

```bash
python3 ~/.hermes/skills/maps/scripts/maps_client.py search "Statue of Liberty"
# Should return lat ~40.689, lon ~-74.044

python3 ~/.hermes/skills/maps/scripts/maps_client.py nearby --near "Times Square" --category restaurant --limit 3
# Should return a list of restaurants within ~500m of Times Square
```
