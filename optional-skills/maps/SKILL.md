---
name: maps
description: Geocoding, reverse geocoding, nearby places (restaurants, hospitals, cafes...), distance and routing between locations, and timezone lookup. Uses OpenStreetMap/Nominatim + Overpass API + OSRM. No API key required.
version: 1.0.0
author: Mibayy
license: MIT
metadata:
  hermes:
    tags: [maps, geocoding, places, routing, distance, openstreetmap, nominatim, overpass]
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
- User asks for nearby restaurants, hospitals, pharmacies, hotels, etc.
- User wants distance or travel time between two locations
- User wants to know the timezone of a location

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
python3 $SCRIPT nearby 48.8584 2.2945 restaurant --limit 10
python3 $SCRIPT nearby 48.8584 2.2945 hospital --radius 1000
python3 $SCRIPT distance "Paris" "Lyon" --mode driving
python3 $SCRIPT timezone 48.8584 2.2945
```

---

## Commands

### search QUERY
Geocode a place name. Returns top 5 with lat, lon, type, bounding_box.

### reverse LAT LON
Coordinates to full address (street, city, country, postcode).

### nearby LAT LON CATEGORY [--limit N] [--radius M]
Find nearby places via Overpass API. Default radius 500m, limit 10.
Categories: restaurant, cafe, bar, hospital, pharmacy, hotel, supermarket, atm, gas_station, parking, museum, park

### distance ORIGIN DESTINATION [--mode driving|walking|cycling]
Route distance + travel time via OSRM. Geocodes both places automatically.

### timezone LAT LON
Get timezone identifier for any coordinates.

---

## Pitfalls
- Nominatim ToS: max 1 req/s (handled automatically by the script).
- nearby requires lat/lon — use search first to get coordinates.
- OSRM routing is best-effort for non-European regions.

---

## Verification
```bash
python3 ~/.hermes/skills/maps/scripts/maps_client.py search "Eiffel Tower"
# Should return lat ~48.858, lon ~2.294
```
