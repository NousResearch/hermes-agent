---
name: maps
description: Geocoding, reverse geocoding, nearby places (restaurants, hospitals, cafes...), distance and routing between locations, and timezone lookup. Uses OpenStreetMap/Nominatim + Overpass API + OSRM. No API key required.
version: 1.0.0
author: Mibayy
license: MIT
metadata:
  hermes:
    tags: [maps, geocoding, places, routing, distance, openstreetmap, nominatim, overpass, osm]
    category: utilities
    requires_toolsets: [terminal]
---

# Maps Skill

Location intelligence using open data sources.
5 commands: geocode, reverse geocode, nearby POI search, distance/routing, timezone.

Free, no API key. Uses OpenStreetMap, Nominatim, Overpass API, OSRM. Zero dependencies.

---

## When to Use
- User wants to find coordinates for a place
- User has coordinates and wants the address
- User asks "find restaurants near me" or nearby places of any category
- User wants distance or travel time between two locations
- User wants to know what timezone a location is in
- User asks about hospitals, pharmacies, hotels, parks, ATMs near a location

---

## Prerequisites
Python 3.8+ stdlib only. No pip installs.

Script path: `~/.hermes/skills/maps/scripts/maps_client.py`

---

## Quick Reference

```
SCRIPT=~/.hermes/skills/maps/scripts/maps_client.py

python3 $SCRIPT search "Eiffel Tower"
python3 $SCRIPT reverse 48.8584 2.2945
python3 $SCRIPT nearby 48.8584 2.2945 restaurant --limit 5
python3 $SCRIPT nearby 48.8584 2.2945 hospital --radius 1000
python3 $SCRIPT distance "Paris" "Lyon"
python3 $SCRIPT distance "Paris" "Lyon" --mode driving
python3 $SCRIPT timezone 48.8584 2.2945
```

---

## Commands

### search QUERY
Geocode a place name — get coordinates and details.
```bash
python3 $SCRIPT search "Notre-Dame de Paris"
python3 $SCRIPT search "Times Square New York"
```
Returns top 5 results: name, lat, lon, type, bounding_box.

### reverse LAT LON
Convert coordinates to a human-readable address.
```bash
python3 $SCRIPT reverse 48.8584 2.2945
python3 $SCRIPT reverse 40.7128 -74.0060
```

### nearby LAT LON CATEGORY [--limit N] [--radius M]
Find nearby places. Default radius: 500m, limit: 10.

Supported categories: restaurant, cafe, bar, hospital, pharmacy, hotel, supermarket, atm, gas_station, parking, museum, park
```bash
python3 $SCRIPT nearby 43.2965 5.3698 restaurant --limit 10
python3 $SCRIPT nearby 43.2965 5.3698 pharmacy --radius 1000
```
Results sorted by distance. Includes name, address, distance_m.

### distance ORIGIN DESTINATION [--mode MODE]
Driving distance and travel time between two places.
Modes: driving (default), walking, cycling
```bash
python3 $SCRIPT distance "Paris" "Marseille"
python3 $SCRIPT distance "Lyon" "Geneva" --mode driving
```
Returns: distance_km, duration_minutes, straight_line_km.

### timezone LAT LON
Get timezone for coordinates.
```bash
python3 $SCRIPT timezone 48.8584 2.2945
python3 $SCRIPT timezone 35.6762 139.6503
```

---

## Pitfalls
- Nominatim ToS: max 1 request/second. The script handles this automatically.
- Overpass API may timeout for very large radius searches.
- OSRM routing works for Europe/North America. For other regions results may be incomplete.
- nearby requires lat/lon, not address — use search first to get coordinates.

---

## Verification
```bash
python3 ~/.hermes/skills/maps/scripts/maps_client.py search "Eiffel Tower"
# Should return lat ~48.858, lon ~2.294
```
