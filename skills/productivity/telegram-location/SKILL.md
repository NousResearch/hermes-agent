---
name: telegram-location
description: Use shared coordinates (Telegram location/venue pins, or pasted lat/lon) to find nearby places (restaurants/cafes/etc.), optionally "open now", and return coordinate-anchored map links.
version: 1.0.0
author: community
license: MIT
metadata:
  hermes:
    tags: [telegram, location, nearby, restaurants, open-now, maps]
    related_skills: [duckduckgo-search]
---

# Telegram Location (Nearby Places)

Handle Telegram shared locations/venues by extracting `latitude`/`longitude`, asking for intent + constraints, and returning nearby recommendations with reliable, coordinate-anchored map links.

## When to Use

Use this skill when:

- The user shares a Telegram **location pin**, **live location**, or **venue** (Hermes injects `latitude:` / `longitude:` lines).
- The user says "near me" and you need them to share their location.
- The user provides exact `lat, lon` and asks for nearby places.

## Quick Reference

`SKILL_DIR` is the directory containing this `SKILL.md`.

```bash
# Nearby restaurants within 1500m (no open-now filtering)
python3 SKILL_DIR/scripts/nearby_places.py --lat <LAT> --lon <LON> \
  --amenity restaurant \
  --radius-m 1500 \
  --limit 10

# "Open now" (best-effort). Keep unknown-hours results so you can verify via web search.
python3 SKILL_DIR/scripts/nearby_places.py --lat <LAT> --lon <LON> \
  --amenity restaurant \
  --radius-m 2000 \
  --open-now --include-unknown-hours \
  --limit 15

# Multiple amenities (repeat --amenity)
python3 SKILL_DIR/scripts/nearby_places.py --lat <LAT> --lon <LON> \
  --amenity cafe --amenity bar \
  --radius-m 1500 \
  --limit 15
```

## Procedure

1. Extract coordinates.
- Look for `latitude: ...` / `longitude: ...` in the conversation.
- If missing, ask the user to share a location pin (Telegram: Attach -> Location -> Send).

2. Ask only for missing constraints.
- What place type: `restaurant`, `cafe`, `bar`, `pharmacy`, etc.
- Radius (meters) or a human constraint ("10 min walk") you convert to `radius_m`.
- Whether "open now" is required.
- Optional preferences: cuisine, budget, dietary, vibe.

3. Fetch candidates via the helper script.
- Use `--open-now --include-unknown-hours` when the user wants "open now".
- If results are sparse, widen the radius (e.g., 1500 -> 3000m).

4. Verify hours/details for high confidence (recommended).
- The script uses OSM `opening_hours` and only handles simple formats.
- For top candidates with `open_now=null` (unknown) or where hours matter, verify with `web_search` (or the `duckduckgo-search` skill if Firecrawl is unavailable).

5. Respond with a short list + reliable links.
- Include: name, distance, open status (open/closed/unknown).
- Prefer **coordinate-anchored** links from the script output:
  - `google_maps_url` (pin by coordinates)
  - `google_maps_directions_url` (directions using exact coordinates)
- Avoid name-only Google Maps searches, and never round/truncate coordinates.

6. Ask one follow-up question.
- Example: "Want me to filter by cuisine/price, or verify hours for any of these?"

## Pitfalls

- **Lat/lon mistakes**: swapped coordinates or rounded values can move results tens of kilometers.
- **Google Maps name searches**: same-named places can resolve far away; use coordinate links.
- **Overpass flakiness**: Overpass can rate limit or fail; the script falls back to Nominatim.
- **"Open now" uncertainty**: OSM `opening_hours` is often missing or complex; treat `open_now=null` as unknown and verify via web search.

## Verification

- Confirm returned `distance_m` values are plausible for the chosen radius.
- Open one of the `google_maps_url` links and confirm it lands in the same neighborhood as the shared pin.
- If the user reports wrong locations, re-check that you used the exact shared coordinates and did not swap/round them.
