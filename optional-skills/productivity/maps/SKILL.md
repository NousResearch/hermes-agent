---
name: maps
description: >
  Geocoding, reverse geocoding, nearby POI search (44 categories),
  distance/routing, turn-by-turn directions, timezone lookup, bounding box
  search, and area info. Uses OpenStreetMap + Overpass + OSRM. Free, no API key.
version: 1.1.0
author: Mibayy
license: MIT
metadata:
  hermes:
    tags: [maps, geocoding, places, routing, distance, directions, openstreetmap, nominatim, overpass, osrm]
    category: productivity
    requires_toolsets: [terminal]
---

# Maps Skill

Location intelligence using free, open data sources. 8 commands, 44 POI
categories, zero dependencies (Python stdlib only), no API key required.

Data sources: OpenStreetMap/Nominatim, Overpass API, OSRM, TimeAPI.io.

## When to Use

- User wants coordinates for a place name
- User has coordinates and wants the address
- User asks for nearby restaurants, hospitals, pharmacies, hotels, etc.
- User wants driving/walking/cycling distance or travel time
- User wants turn-by-turn directions between two places
- User wants timezone information for a location
- User wants to search for POIs within a geographic area

## Prerequisites

Python 3.8+ (stdlib only — no pip installs needed).

Script path after install: `~/.hermes/skills/maps/scripts/maps_client.py`

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
python3 $MAPS nearby 48.8584 2.2945 restaurant --limit 10
python3 $MAPS nearby 40.7128 -74.0060 hospital --radius 2000
python3 $MAPS nearby 51.5074 -0.1278 cafe --limit 5 --radius 300
```

44 categories: restaurant, cafe, bar, hospital, pharmacy, hotel, supermarket,
atm, gas_station, parking, museum, park, school, university, bank, police,
fire_station, library, airport, train_station, bus_stop, church, mosque,
synagogue, dentist, doctor, cinema, theatre, gym, swimming_pool, post_office,
convenience_store, bakery, bookshop, laundry, car_wash, car_rental,
bicycle_rental, taxi, veterinary, zoo, playground, stadium, nightclub.

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

## Workflow Examples

**"Find Italian restaurants near the Colosseum":**
1. `search "Colosseum Rome"` → get lat/lon
2. `nearby LAT LON restaurant --radius 500`

**"How do I walk from hotel to conference center?":**
1. `directions "Hotel Name" --to "Conference Center" --mode walking`

**"What restaurants are in downtown Seattle?":**
1. `area "Downtown Seattle"` → get bounding box
2. `bbox S W N E restaurant --limit 30`

## Pitfalls

- Nominatim ToS: max 1 req/s (handled automatically by the script)
- `nearby` requires lat/lon — use `search` first to get coordinates
- OSRM routing coverage is best for Europe and North America
- Overpass API can be slow during peak hours (script retries automatically)
- `distance` and `directions` use `--to` flag for the destination (not positional)

## Verification

```bash
python3 ~/.hermes/skills/maps/scripts/maps_client.py search "Statue of Liberty"
# Should return lat ~40.689, lon ~-74.044
```
