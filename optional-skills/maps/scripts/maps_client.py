#!/usr/bin/env python3
"""
maps_client.py - CLI tool for maps, geocoding, routing, and POI search.
Uses only Python stdlib. Data from OpenStreetMap/Nominatim, Overpass, OSRM.
"""

import argparse
import json
import math
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

USER_AGENT = "HermesAgent/1.0 (contact: hermes@agent.ai)"
DATA_SOURCE = "OpenStreetMap/Nominatim"

NOMINATIM_SEARCH  = "https://nominatim.openstreetmap.org/search"
NOMINATIM_REVERSE = "https://nominatim.openstreetmap.org/reverse"
OVERPASS_API      = "https://overpass-api.de/api/interpreter"
OSRM_BASE         = "https://router.project-osrm.org/route/v1"

# Seconds to sleep between Nominatim requests (ToS requirement)
NOMINATIM_RATE_LIMIT = 1.0

# Maximum retries for HTTP errors
MAX_RETRIES = 3
RETRY_DELAY = 2.0  # seconds

# Category -> OSM tag mapping
CATEGORY_TAGS = {
    "restaurant":  ("amenity", "restaurant"),
    "cafe":        ("amenity", "cafe"),
    "bar":         ("amenity", "bar"),
    "hospital":    ("amenity", "hospital"),
    "pharmacy":    ("amenity", "pharmacy"),
    "hotel":       ("tourism", "hotel"),
    "supermarket": ("shop",    "supermarket"),
    "atm":         ("amenity", "atm"),
    "gas_station": ("amenity", "fuel"),
    "parking":     ("amenity", "parking"),
    "museum":      ("tourism", "museum"),
    "park":        ("leisure", "park"),
}

VALID_CATEGORIES = sorted(CATEGORY_TAGS.keys())

OSRM_PROFILES = {
    "driving": "driving",
    "walking": "foot",
    "cycling": "bike",
}

# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def print_json(data):
    """Print data as pretty-printed JSON to stdout."""
    print(json.dumps(data, indent=2, ensure_ascii=False))


def error_exit(message, code=1):
    """Print an error result and exit."""
    print_json({"error": message, "status": "error"})
    sys.exit(code)


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def http_get(url, params=None, retries=MAX_RETRIES, silent=False):
    """
    Perform an HTTP GET request, returning parsed JSON.
    Adds the required User-Agent header.
    Retries on transient errors.
    If silent=True, raises RuntimeError instead of calling error_exit on failure.
    """
    if params:
        url = url + "?" + urllib.parse.urlencode(params)

    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

    last_error = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw)
        except urllib.error.HTTPError as exc:
            last_error = f"HTTP {exc.code}: {exc.reason} for {url}"
            if exc.code in (429, 503, 502, 504):
                # Rate-limited or server busy — back off
                time.sleep(RETRY_DELAY * attempt)
            else:
                # Non-retryable HTTP error
                if silent:
                    raise RuntimeError(last_error)
                error_exit(last_error)
        except urllib.error.URLError as exc:
            last_error = f"URL error: {exc.reason}"
            time.sleep(RETRY_DELAY * attempt)
        except json.JSONDecodeError as exc:
            last_error = f"JSON parse error: {exc}"
            time.sleep(RETRY_DELAY * attempt)

    msg = f"Request failed after {retries} attempts. Last error: {last_error}"
    if silent:
        raise RuntimeError(msg)
    error_exit(msg)


def http_post(url, data_str, retries=MAX_RETRIES):
    """
    Perform an HTTP POST with a plain-text body (for Overpass QL).
    Returns parsed JSON.
    """
    encoded = data_str.encode("utf-8")
    req = urllib.request.Request(
        url,
        data=encoded,
        headers={
            "User-Agent": USER_AGENT,
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )

    last_error = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw)
        except urllib.error.HTTPError as exc:
            last_error = f"HTTP {exc.code}: {exc.reason}"
            if exc.code in (429, 503, 502, 504):
                time.sleep(RETRY_DELAY * attempt)
            else:
                error_exit(last_error)
        except urllib.error.URLError as exc:
            last_error = f"URL error: {exc.reason}"
            time.sleep(RETRY_DELAY * attempt)
        except json.JSONDecodeError as exc:
            last_error = f"JSON parse error: {exc}"
            time.sleep(RETRY_DELAY * attempt)

    error_exit(f"POST failed after {retries} attempts. Last error: {last_error}")


# ---------------------------------------------------------------------------
# Geo math
# ---------------------------------------------------------------------------

def haversine_m(lat1, lon1, lat2, lon2):
    """Return distance in metres between two lat/lon points."""
    R = 6_371_000  # Earth radius in metres
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi  = math.radians(lat2 - lat1)
    dlam  = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


# ---------------------------------------------------------------------------
# Nominatim helpers
# ---------------------------------------------------------------------------

def nominatim_search(query, limit=5):
    """Geocode a free-text query. Returns list of result dicts."""
    params = {
        "q":              query,
        "format":         "json",
        "limit":          limit,
        "addressdetails": 1,
    }
    time.sleep(NOMINATIM_RATE_LIMIT)
    return http_get(NOMINATIM_SEARCH, params=params)


def nominatim_reverse(lat, lon):
    """Reverse geocode lat/lon. Returns a single result dict."""
    params = {
        "lat":            lat,
        "lon":            lon,
        "format":         "json",
        "addressdetails": 1,
    }
    time.sleep(NOMINATIM_RATE_LIMIT)
    return http_get(NOMINATIM_REVERSE, params=params)


def geocode_single(query):
    """
    Geocode a query and return (lat, lon, display_name).
    Exits with error if nothing found.
    """
    results = nominatim_search(query, limit=1)
    if not results:
        error_exit(f"Could not geocode: {query}")
    r = results[0]
    return float(r["lat"]), float(r["lon"]), r.get("display_name", query)


# ---------------------------------------------------------------------------
# Command: search
# ---------------------------------------------------------------------------

def cmd_search(args):
    """Geocode a place name and return top results."""
    query = " ".join(args.query)
    raw   = nominatim_search(query, limit=5)

    if not raw:
        print_json({
            "query":       query,
            "results":     [],
            "count":       0,
            "data_source": DATA_SOURCE,
        })
        return

    results = []
    for item in raw:
        bb = item.get("boundingbox", [])
        results.append({
            "name":         item.get("name") or item.get("display_name", ""),
            "display_name": item.get("display_name", ""),
            "lat":          float(item["lat"]),
            "lon":          float(item["lon"]),
            "type":         item.get("type", ""),
            "category":     item.get("category", ""),
            "osm_type":     item.get("osm_type", ""),
            "osm_id":       item.get("osm_id", ""),
            "bounding_box": {
                "min_lat": float(bb[0]) if len(bb) > 0 else None,
                "max_lat": float(bb[1]) if len(bb) > 1 else None,
                "min_lon": float(bb[2]) if len(bb) > 2 else None,
                "max_lon": float(bb[3]) if len(bb) > 3 else None,
            },
            "importance":   item.get("importance"),
        })

    print_json({
        "query":       query,
        "results":     results,
        "count":       len(results),
        "data_source": DATA_SOURCE,
    })


# ---------------------------------------------------------------------------
# Command: reverse
# ---------------------------------------------------------------------------

def cmd_reverse(args):
    """Reverse geocode coordinates to a human-readable address."""
    try:
        lat = float(args.lat)
        lon = float(args.lon)
    except ValueError:
        error_exit("LAT and LON must be numeric values.")

    if not (-90 <= lat <= 90):
        error_exit("Latitude must be between -90 and 90.")
    if not (-180 <= lon <= 180):
        error_exit("Longitude must be between -180 and 180.")

    data = nominatim_reverse(lat, lon)

    if "error" in data:
        error_exit(f"Reverse geocode failed: {data['error']}")

    address = data.get("address", {})

    print_json({
        "lat":          lat,
        "lon":          lon,
        "display_name": data.get("display_name", ""),
        "address": {
            "house_number":  address.get("house_number", ""),
            "road":          address.get("road", ""),
            "neighbourhood": address.get("neighbourhood", ""),
            "suburb":        address.get("suburb", ""),
            "city":          address.get("city") or address.get("town") or address.get("village", ""),
            "county":        address.get("county", ""),
            "state":         address.get("state", ""),
            "postcode":      address.get("postcode", ""),
            "country":       address.get("country", ""),
            "country_code":  address.get("country_code", ""),
        },
        "osm_type":    data.get("osm_type", ""),
        "osm_id":      data.get("osm_id", ""),
        "data_source": DATA_SOURCE,
    })


# ---------------------------------------------------------------------------
# Command: nearby
# ---------------------------------------------------------------------------

def build_overpass_query(tag_key, tag_val, lat, lon, radius, limit):
    """Build an Overpass QL query string."""
    return (
        f'[out:json][timeout:25];\n'
        f'(\n'
        f'  node["{tag_key}"="{tag_val}"](around:{radius},{lat},{lon});\n'
        f'  way["{tag_key}"="{tag_val}"](around:{radius},{lat},{lon});\n'
        f');\n'
        f'out center {limit};\n'
    )


def cmd_nearby(args):
    """Find nearby POIs using the Overpass API."""
    try:
        lat = float(args.lat)
        lon = float(args.lon)
    except ValueError:
        error_exit("LAT and LON must be numeric values.")

    category = args.category.lower()
    if category not in CATEGORY_TAGS:
        error_exit(
            f"Unknown category '{category}'. "
            f"Valid categories: {', '.join(VALID_CATEGORIES)}"
        )

    radius = int(args.radius)
    limit  = int(args.limit)

    if radius <= 0:
        error_exit("Radius must be a positive integer (metres).")
    if limit <= 0:
        error_exit("Limit must be a positive integer.")

    tag_key, tag_val = CATEGORY_TAGS[category]
    query = build_overpass_query(tag_key, tag_val, lat, lon, radius, limit)

    # POST the query as form-encoded data
    post_data = "data=" + urllib.parse.quote(query)
    raw = http_post(OVERPASS_API, post_data)

    elements = raw.get("elements", [])
    places = []

    for el in elements:
        # Ways have a "center" sub-dict; nodes have lat/lon directly
        if el["type"] == "way":
            center = el.get("center", {})
            el_lat = center.get("lat")
            el_lon = center.get("lon")
        else:
            el_lat = el.get("lat")
            el_lon = el.get("lon")

        if el_lat is None or el_lon is None:
            continue

        tags = el.get("tags", {})
        name = tags.get("name") or tags.get("name:en") or ""

        # Build a short address from available tags
        addr_parts = []
        for part_key in ("addr:housenumber", "addr:street", "addr:city"):
            val = tags.get(part_key)
            if val:
                addr_parts.append(val)
        address_str = ", ".join(addr_parts) if addr_parts else ""

        dist_m = haversine_m(lat, lon, el_lat, el_lon)

        places.append({
            "name":        name,
            "category":    category,
            "address":     address_str,
            "lat":         el_lat,
            "lon":         el_lon,
            "distance_m":  round(dist_m, 1),
            "osm_type":    el.get("type", ""),
            "osm_id":      el.get("id", ""),
            "tags":        {
                k: v for k, v in tags.items()
                if k not in ("name", "addr:housenumber", "addr:street", "addr:city")
            },
        })

    # Sort by distance
    places.sort(key=lambda p: p["distance_m"])

    print_json({
        "center_lat":  lat,
        "center_lon":  lon,
        "category":    category,
        "radius_m":    radius,
        "count":       len(places),
        "results":     places,
        "data_source": DATA_SOURCE,
    })


# ---------------------------------------------------------------------------
# Command: distance
# ---------------------------------------------------------------------------

def cmd_distance(args):
    """Calculate road distance and travel time between two places."""
    origin_query      = " ".join(args.origin)
    destination_query = " ".join(args.destination)
    mode              = args.mode.lower()

    if mode not in OSRM_PROFILES:
        error_exit(f"Invalid mode '{mode}'. Choose from: {', '.join(OSRM_PROFILES)}")

    # Geocode origin
    o_lat, o_lon, o_name = geocode_single(origin_query)

    # Geocode destination (with rate-limit sleep; geocode_single calls nominatim which sleeps)
    d_lat, d_lon, d_name = geocode_single(destination_query)

    profile = OSRM_PROFILES[mode]
    url = (
        f"{OSRM_BASE}/{profile}/"
        f"{o_lon},{o_lat};{d_lon},{d_lat}"
        f"?overview=false&steps=false"
    )

    osrm_data = http_get(url)

    if osrm_data.get("code") != "Ok":
        error_exit(
            f"OSRM routing failed: {osrm_data.get('message', osrm_data.get('code', 'unknown error'))}"
        )

    routes = osrm_data.get("routes", [])
    if not routes:
        error_exit("No route found between the two locations.")

    route         = routes[0]
    distance_m    = route.get("distance", 0)
    duration_s    = route.get("duration", 0)
    distance_km   = round(distance_m / 1000, 3)
    duration_min  = round(duration_s / 60, 2)

    # Straight-line distance for reference
    straight_m    = haversine_m(o_lat, o_lon, d_lat, d_lon)

    print_json({
        "origin": {
            "query":        origin_query,
            "display_name": o_name,
            "lat":          o_lat,
            "lon":          o_lon,
        },
        "destination": {
            "query":        destination_query,
            "display_name": d_name,
            "lat":          d_lat,
            "lon":          d_lon,
        },
        "mode":               mode,
        "distance_km":        distance_km,
        "distance_m":         round(distance_m, 1),
        "duration_minutes":   duration_min,
        "duration_seconds":   round(duration_s, 1),
        "straight_line_km":   round(straight_m / 1000, 3),
        "data_source":        DATA_SOURCE,
    })


# ---------------------------------------------------------------------------
# Command: timezone
# ---------------------------------------------------------------------------

def cmd_timezone(args):
    """
    Get timezone information for a lat/lon coordinate.

    Strategy:
      1. Try worldtimeapi.org (uses IP geolocation — works best for your own IP,
         but we ask for lat/lon-based lookup).
      2. Fall back to timezone string embedded in Nominatim reverse-geocode result
         (present in the extratags field when available).
      3. Final fallback: derive UTC offset approximation from longitude.
    """
    try:
        lat = float(args.lat)
        lon = float(args.lon)
    except ValueError:
        error_exit("LAT and LON must be numeric values.")

    timezone_str  = None
    timezone_src  = None
    current_time  = None
    utc_offset    = None

    # --- Strategy 1: worldtimeapi by lat/lon (timezone lookup) ---
    # worldtimeapi doesn't accept lat/lon directly but we can try
    # fetching the list of timezones and find the best by UTC offset approximation.
    # A better endpoint: https://worldtimeapi.org/api/timezone/<zone>
    # Since worldtimeapi has no lat/lon endpoint, we use the longitude to estimate
    # the UTC offset and then look up that offset from their list.

    # Approximate UTC offset from longitude
    approx_offset_h = round(lon / 15)

    try:
        tz_list_data = http_get("https://worldtimeapi.org/api/timezone", silent=True)
        if isinstance(tz_list_data, list):
            # Look for a timezone that matches the approximate region
            # Try to narrow by continent using lat/lon
            region_hints = []
            if -170 <= lon <= -25 and lat >= 15:
                region_hints = ["America"]
            elif -25 <= lon <= 50 and lat >= 35:
                region_hints = ["Europe"]
            elif 25 <= lon <= 60 and lat < 35:
                region_hints = ["Africa", "Asia"]
            elif lon > 60:
                region_hints = ["Asia", "Australia", "Pacific"]
            elif lat < -10:
                region_hints = ["Australia", "Pacific", "America"]

            candidate = None
            if region_hints:
                for hint in region_hints:
                    matches = [z for z in tz_list_data if z.startswith(hint + "/")]
                    if matches:
                        candidate = matches[0]
                        break

            if not candidate and tz_list_data:
                candidate = tz_list_data[0]

            if candidate:
                tz_data = http_get(
                    f"https://worldtimeapi.org/api/timezone/{candidate}", silent=True
                )
                timezone_str = tz_data.get("timezone")
                current_time = tz_data.get("datetime")
                utc_offset   = tz_data.get("utc_offset")
                timezone_src = "worldtimeapi.org (region approximation)"

    except (SystemExit, RuntimeError):
        pass  # API may be down; continue to fallbacks

    # --- Strategy 2: Nominatim reverse geocode extratags ---
    if not timezone_str:
        try:
            params = {
                "lat":            lat,
                "lon":            lon,
                "format":         "json",
                "addressdetails": 1,
                "extratags":      1,
            }
            time.sleep(NOMINATIM_RATE_LIMIT)
            rev_data = http_get(NOMINATIM_REVERSE, params=params, silent=True)
            extratags = rev_data.get("extratags") or {}
            tz_from_osm = extratags.get("timezone")
            if tz_from_osm:
                timezone_str = tz_from_osm
                timezone_src = "OpenStreetMap/Nominatim extratags"
                # Fetch live time from worldtimeapi for this zone
                try:
                    tz_data = http_get(
                        f"https://worldtimeapi.org/api/timezone/{tz_from_osm}",
                        silent=True,
                    )
                    current_time = tz_data.get("datetime")
                    utc_offset   = tz_data.get("utc_offset")
                except (SystemExit, RuntimeError):
                    pass
        except (SystemExit, RuntimeError):
            pass

    # --- Strategy 3: longitude-based UTC offset approximation ---
    if not timezone_str:
        utc_offset = f"+{approx_offset_h:02d}:00" if approx_offset_h >= 0 else f"{approx_offset_h:03d}:00"
        timezone_str = f"UTC{utc_offset}"
        timezone_src = "longitude approximation (±longitude/15)"

    print_json({
        "lat":          lat,
        "lon":          lon,
        "timezone":     timezone_str,
        "utc_offset":   utc_offset,
        "current_time": current_time,
        "source":       timezone_src,
        "data_source":  DATA_SOURCE,
    })


# ---------------------------------------------------------------------------
# CLI setup
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        prog="maps_client.py",
        description=(
            "CLI maps tool: geocoding, reverse geocoding, POI search, "
            "routing, and timezone lookup. Powered by OpenStreetMap data."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- search --
    p_search = sub.add_parser("search", help="Geocode a place name to coordinates.")
    p_search.add_argument("query", nargs="+", help="Place name or address to search.")

    # -- reverse --
    p_reverse = sub.add_parser(
        "reverse", help="Reverse geocode coordinates to an address."
    )
    p_reverse.add_argument("lat", help="Latitude (decimal degrees).")
    p_reverse.add_argument("lon", help="Longitude (decimal degrees).")

    # -- nearby --
    p_nearby = sub.add_parser(
        "nearby", help="Find nearby places of a given category."
    )
    p_nearby.add_argument("lat",      help="Center latitude (decimal degrees).")
    p_nearby.add_argument("lon",      help="Center longitude (decimal degrees).")
    p_nearby.add_argument(
        "category",
        help=f"POI category. Options: {', '.join(VALID_CATEGORIES)}",
    )
    p_nearby.add_argument(
        "--radius", "-r",
        default=500,
        type=int,
        metavar="METRES",
        help="Search radius in metres (default: 500).",
    )
    p_nearby.add_argument(
        "--limit", "-n",
        default=10,
        type=int,
        metavar="N",
        help="Maximum number of results (default: 10).",
    )

    # -- distance --
    p_dist = sub.add_parser(
        "distance", help="Calculate road distance and travel time between two places."
    )
    p_dist.add_argument(
        "origin",
        nargs="+",
        help='Origin address or place name. Wrap in quotes if multi-word.',
    )
    p_dist.add_argument(
        "destination",
        nargs="+",
        help='Destination address or place name.',
    )
    p_dist.add_argument(
        "--mode", "-m",
        default="driving",
        choices=list(OSRM_PROFILES.keys()),
        help="Travel mode (default: driving).",
    )

    # -- timezone --
    p_tz = sub.add_parser(
        "timezone", help="Get timezone information for coordinates."
    )
    p_tz.add_argument("lat", help="Latitude (decimal degrees).")
    p_tz.add_argument("lon", help="Longitude (decimal degrees).")

    return parser


def main():
    parser = build_parser()
    args   = parser.parse_args()

    dispatch = {
        "search":   cmd_search,
        "reverse":  cmd_reverse,
        "nearby":   cmd_nearby,
        "distance": cmd_distance,
        "timezone": cmd_timezone,
    }

    handler = dispatch.get(args.command)
    if handler is None:
        error_exit(f"Unknown command: {args.command}")

    # distance uses two nargs="+" positional args; argparse lumps them together.
    # We need to split origin and destination on "--" separator when both are plain text.
    # Since argparse can't cleanly separate two nargs="+" positionals, we handle
    # this via a pre-processing step: if the command is "distance", the first
    # nargs="+" eats everything. Re-parse with a sentinel if needed.
    if args.command == "distance":
        # argparse gives all words to 'origin' when two nargs='+' are used.
        # Detect the separator "--" or rely on the convention that the user
        # quotes each argument. If destination is empty, try splitting origin.
        if not args.destination and len(args.origin) > 1:
            # Heuristic: split on "--" if present, otherwise error.
            combined = args.origin
            if "--" in combined:
                idx           = combined.index("--")
                args.origin      = combined[:idx]
                args.destination = combined[idx + 1:]
            else:
                error_exit(
                    "For 'distance', provide ORIGIN and DESTINATION as separate quoted "
                    "strings, e.g.:\n"
                    '  maps_client.py distance "New York" "Los Angeles"\n'
                    "Or separate them with --:\n"
                    "  maps_client.py distance New York -- Los Angeles"
                )

    handler(args)


if __name__ == "__main__":
    main()
