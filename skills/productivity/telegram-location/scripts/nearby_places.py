#!/usr/bin/env python3
"""Nearby places (OSM Overpass) helper for the "telegram-location" skill.

- Dependency-free (Python stdlib only)
- Prints JSON to stdout
- Supports basic "open now" filtering via OSM `opening_hours` (simple cases only)

Notes:
- Overpass can be flaky; this tool falls back to Nominatim.
- "Open now" requires local wall-clock time; we try timeapi.io. If that fails,
  results are returned without filtering and `open_now` will be null.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import sys
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Sequence, Tuple


OVERPASS_ENDPOINTS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.nchc.org.tw/api/interpreter",
)
NOMINATIM_SEARCH_URL = "https://nominatim.openstreetmap.org/search"
TIMEAPI_COORD_URL = "https://timeapi.io/api/Time/current/coordinate"
USER_AGENT = "hermes-skill-telegram-location/1.0"

_KEEP_TAG_PREFIXES = ("addr:", "contact:")
_KEEP_TAG_KEYS = {
    "name",
    "amenity",
    "cuisine",
    "opening_hours",
    "website",
    "phone",
    "nominatim:category",
    "nominatim:type",
    "wheelchair",
    "delivery",
    "takeaway",
    "outdoor_seating",
    "reservation",
}

_DAY_TO_IDX = {"Mo": 0, "Tu": 1, "We": 2, "Th": 3, "Fr": 4, "Sa": 5, "Su": 6}


def _die(msg: str, *, code: int = 2, extra: Optional[dict] = None) -> None:
    payload: Dict[str, Any] = {"success": False, "error": msg}
    if extra:
        payload.update(extra)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    raise SystemExit(code)


def _print(data: Any) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


def _http_get_json(url: str, *, timeout: float) -> Dict[str, Any]:
    value = _http_get_json_value(url, timeout=timeout)
    if not isinstance(value, dict):
        raise RuntimeError(f"Unexpected JSON type: {type(value).__name__}")
    return value


def _http_get_json_value(url: str, *, timeout: float) -> Any:
    req = urllib.request.Request(
        url,
        method="GET",
        headers={
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    text = raw.decode("utf-8", errors="replace")
    try:
        parsed = json.loads(text)
    except Exception as e:
        raise RuntimeError(f"Invalid JSON response: {e}: {text[:300]!r}")
    return parsed


def _http_post_form_json(url: str, *, form: Dict[str, str], timeout: float) -> Dict[str, Any]:
    body = urllib.parse.urlencode(form).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded; charset=utf-8",
            "User-Agent": USER_AGENT,
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    text = raw.decode("utf-8", errors="replace")
    try:
        parsed = json.loads(text)
    except Exception as e:
        raise RuntimeError(f"Invalid JSON response: {e}: {text[:300]!r}")
    if not isinstance(parsed, dict):
        raise RuntimeError(f"Unexpected JSON type: {type(parsed).__name__}")
    return parsed


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return r * c


def _sanitize_amenities(values: Sequence[str]) -> List[str]:
    out: List[str] = []
    for v in values:
        s = v.strip().lower().replace("-", "_")
        if not s:
            continue
        if not all(ch.isalnum() or ch == "_" for ch in s):
            _die(f"Invalid amenity value: {v!r} (allowed: letters/digits/underscore only)")
        out.append(s)
    return sorted(set(out))


def _overpass_query(lat: float, lon: float, *, radius_m: int, amenities: Sequence[str], require_name: bool) -> str:
    # Use a strict regex to avoid OSM query injection (amenities are sanitized).
    regex = "|".join(amenities) if amenities else "restaurant"
    name_filter = '["name"]' if require_name else ""
    # nwr = node/way/relation
    return (
        f"[out:json][timeout:25];"
        f"(nwr[\"amenity\"~\"^({regex})$\"]{name_filter}(around:{radius_m},{lat},{lon}););"
        f"out center tags;"
    )


def _fetch_overpass(query: str, *, timeout: float) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    errors: List[str] = []
    for ep in OVERPASS_ENDPOINTS:
        try:
            data = _http_post_form_json(ep, form={"data": query}, timeout=timeout)
            return data, {"endpoint": ep}
        except Exception as e:
            errors.append(f"{ep}: {type(e).__name__}: {e}")
            continue
    raise RuntimeError("All Overpass endpoints failed: " + " | ".join(errors))


def _viewbox_for_radius(lat: float, lon: float, *, radius_m: int) -> Tuple[float, float, float, float]:
    # Rough meters-to-degrees conversion for small radii.
    lat_delta = radius_m / 111_320.0
    cos_lat = math.cos(math.radians(lat))
    lon_delta = radius_m / (111_320.0 * cos_lat) if abs(cos_lat) > 1e-6 else lat_delta
    return (lon - lon_delta, lat - lat_delta, lon + lon_delta, lat + lat_delta)


def _fetch_nominatim(
    *,
    lat: float,
    lon: float,
    radius_m: int,
    amenities: Sequence[str],
    limit: int,
    timeout: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    # Nominatim doesn't support amenity filters the same way Overpass does; we query per amenity keyword
    # and bound to a viewbox approximating the radius, then filter results client-side.
    vb = _viewbox_for_radius(lat, lon, radius_m=radius_m)
    viewbox = ",".join(str(x) for x in vb)

    max_amenities = 3
    if len(amenities) > max_amenities:
        amenities = list(amenities)[:max_amenities]

    per_query_limit = min(50, max(15, limit * 3))
    all_results: List[Dict[str, Any]] = []
    errors: List[str] = []

    for a in amenities or ["restaurant"]:
        params = {
            "format": "jsonv2",
            "q": a,
            "bounded": "1",
            "viewbox": viewbox,
            "limit": str(per_query_limit),
            "extratags": "1",
        }
        url = NOMINATIM_SEARCH_URL + "?" + urllib.parse.urlencode(params)
        try:
            value = _http_get_json_value(url, timeout=timeout)
            if not isinstance(value, list):
                raise RuntimeError(f"Unexpected JSON type: {type(value).__name__}")
            for item in value:
                if isinstance(item, dict):
                    all_results.append(item)
        except Exception as e:
            errors.append(f"{a}: {type(e).__name__}: {e}")

    if not all_results:
        raise RuntimeError("Nominatim returned no results" + (f" ({' | '.join(errors)})" if errors else ""))

    return all_results, {"provider": "nominatim", "url": NOMINATIM_SEARCH_URL, "errors": errors or None}


def _nominatim_item_matches_amenities(item: Dict[str, Any], amenities: Sequence[str]) -> bool:
    if not amenities:
        return True

    cat = item.get("category")
    typ = item.get("type")
    if isinstance(cat, str) and isinstance(typ, str):
        if cat == "amenity" and typ in amenities:
            return True

    extratags = item.get("extratags")
    if isinstance(extratags, dict):
        a = extratags.get("amenity")
        if isinstance(a, str) and a in amenities:
            return True

    return False


def _nominatim_to_elements(items: Sequence[Dict[str, Any]], *, amenities: Sequence[str]) -> List[Dict[str, Any]]:
    """Convert Nominatim search results into an Overpass-like element list.

    Output element format: {type, id, lat, lon, tags}.
    """
    elements: List[Dict[str, Any]] = []
    for item in items:
        if not _nominatim_item_matches_amenities(item, amenities):
            continue

        osm_type = item.get("osm_type")
        if osm_type not in ("node", "way", "relation"):
            continue
        try:
            osm_id = int(item.get("osm_id"))
            lat = float(item.get("lat"))
            lon = float(item.get("lon"))
        except Exception:
            continue

        tags: Dict[str, Any] = {}
        if isinstance(item.get("name"), str) and item.get("name"):
            tags["name"] = item["name"]
        if isinstance(item.get("extratags"), dict):
            tags.update(item["extratags"])
        # Surface a hint about what Nominatim classified this as.
        if isinstance(item.get("category"), str):
            tags["nominatim:category"] = item["category"]
        if isinstance(item.get("type"), str):
            tags["nominatim:type"] = item["type"]

        elements.append(
            {
                "type": osm_type,
                "id": osm_id,
                "lat": lat,
                "lon": lon,
                "tags": tags,
            }
        )
    return elements


def _osm_url(osm_type: str, osm_id: int) -> str:
    return f"https://www.openstreetmap.org/{osm_type}/{osm_id}"


def _google_maps_url(lat: float, lon: float) -> str:
    # Unambiguous coordinate pin.
    return f"https://www.google.com/maps/search/?api=1&query={lat},{lon}"


def _google_maps_directions_url(
    origin_lat: float,
    origin_lon: float,
    dest_lat: float,
    dest_lon: float,
    *,
    travelmode: str = "walking",
) -> str:
    mode = (travelmode or "").strip().lower()
    if mode not in ("walking", "driving", "transit", "bicycling"):
        mode = "walking"
    return (
        "https://www.google.com/maps/dir/?api=1"
        f"&origin={origin_lat},{origin_lon}"
        f"&destination={dest_lat},{dest_lon}"
        f"&travelmode={mode}"
    )


def _google_maps_place_search_url(query: str, *, near_lat: float, near_lon: float, zoom: int = 16) -> str:
    """Build a Google Maps search URL that is biased to a nearby viewport.

    This is a convenience link only. Google may still resolve to a same-named
    business elsewhere. Prefer coordinate pins/directions for reliability.
    """
    q = (query or "").strip()
    if not q:
        return _google_maps_url(near_lat, near_lon)
    encoded = urllib.parse.quote(q, safe="")
    z = int(zoom)
    if z < 10:
        z = 10
    if z > 20:
        z = 20
    return f"https://www.google.com/maps/search/{encoded}/@{near_lat},{near_lon},{z}z"


def _build_maps_query(name: Optional[str], tags_raw: Dict[str, Any]) -> Optional[str]:
    if not name:
        return None
    parts: List[str] = [str(name).strip()]

    # Prefer a single full address if present.
    addr_full = tags_raw.get("addr:full")
    if isinstance(addr_full, str) and addr_full.strip():
        parts.append(addr_full.strip())
        return ", ".join([p for p in parts if p])

    # Otherwise compose a minimal address.
    housenum = tags_raw.get("addr:housenumber")
    street = tags_raw.get("addr:street")
    city = tags_raw.get("addr:city")
    postcode = tags_raw.get("addr:postcode")

    street_line = ""
    if isinstance(street, str) and street.strip():
        if isinstance(housenum, str) and housenum.strip():
            street_line = f"{housenum.strip()} {street.strip()}"
        else:
            street_line = street.strip()
    if street_line:
        parts.append(street_line)
    if isinstance(city, str) and city.strip():
        parts.append(city.strip())
    if isinstance(postcode, str) and postcode.strip():
        parts.append(postcode.strip())

    return ", ".join([p for p in parts if p])


def _get_local_time(lat: float, lon: float, *, timeout: float) -> Dict[str, Any]:
    url = TIMEAPI_COORD_URL + "?" + urllib.parse.urlencode({"latitude": str(lat), "longitude": str(lon)})
    return _http_get_json(url, timeout=timeout)


def _parse_time_hhmm(s: str) -> Optional[int]:
    s = s.strip()
    if not s:
        return None
    if s == "24:00":
        return 24 * 60
    parts = s.split(":")
    if len(parts) != 2:
        return None
    try:
        hh = int(parts[0])
        mm = int(parts[1])
    except ValueError:
        return None
    if not (0 <= hh <= 24 and 0 <= mm <= 59):
        return None
    if hh == 24 and mm != 0:
        return None
    return hh * 60 + mm


def _expand_days(expr: str) -> Optional[List[int]]:
    expr = expr.strip()
    if not expr:
        return None
    if "PH" in expr or "SH" in expr:
        return None

    days: set[int] = set()
    for chunk in expr.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start, end = [p.strip() for p in chunk.split("-", 1)]
            if start not in _DAY_TO_IDX or end not in _DAY_TO_IDX:
                return None
            sidx = _DAY_TO_IDX[start]
            eidx = _DAY_TO_IDX[end]
            if sidx <= eidx:
                for i in range(sidx, eidx + 1):
                    days.add(i)
            else:
                # Wrap-around (e.g., Fr-Mo)
                for i in range(sidx, 7):
                    days.add(i)
                for i in range(0, eidx + 1):
                    days.add(i)
        else:
            if chunk not in _DAY_TO_IDX:
                return None
            days.add(_DAY_TO_IDX[chunk])

    return sorted(days)


def _parse_opening_hours_simple(oh: str) -> Optional[Dict[int, Optional[List[Tuple[int, int]]]]]:
    """Parse a subset of OSM opening_hours into a weekly schedule.

    Returns:
      dict weekday_idx -> None (unknown) | [] (closed) | [(start_min, end_min), ...]

    Notes:
    - Supports day ranges/list + one or more HH:MM-HH:MM ranges.
    - Does not support holidays (PH), date ranges, week numbers, "sunrise-sunset", etc.
    """
    oh = (oh or "").strip()
    if not oh:
        return None

    if oh.lower() == "24/7":
        return {i: [(0, 24 * 60)] for i in range(7)}

    schedule: Dict[int, Optional[List[Tuple[int, int]]]] = {i: None for i in range(7)}
    rules = [r.strip() for r in oh.split(";") if r.strip()]
    if not rules:
        return None

    for rule in rules:
        if "PH" in rule or "SH" in rule:
            return None
        if "sunrise" in rule or "sunset" in rule:
            return None

        if " " not in rule:
            return None
        day_part, time_part = rule.split(None, 1)
        days = _expand_days(day_part)
        if days is None:
            return None

        tp = time_part.strip()
        if not tp:
            return None
        if tp.lower() in ("off", "closed"):
            for d in days:
                schedule[d] = []
            continue
        if tp.lower() == "24/7":
            for d in days:
                schedule[d] = [(0, 24 * 60)]
            continue

        ranges: List[Tuple[int, int]] = []
        for part in [p.strip() for p in tp.split(",") if p.strip()]:
            if "-" not in part:
                return None
            a, b = [x.strip() for x in part.split("-", 1)]
            start = _parse_time_hhmm(a)
            end = _parse_time_hhmm(b)
            if start is None or end is None:
                return None
            ranges.append((start, end))

        for d in days:
            schedule[d] = ranges

    return schedule


def _opening_hours_open_now(oh: str, *, local_dt: _dt.datetime) -> Optional[bool]:
    """Return True/False if parseable, otherwise None."""
    sched = _parse_opening_hours_simple(oh)
    if sched is None:
        return None

    wd = local_dt.weekday()
    mins = local_dt.hour * 60 + local_dt.minute

    todays = sched.get(wd)
    if todays is None:
        return None
    if todays == []:
        # Still may be open due to a crossing-midnight interval from yesterday.
        pass
    else:
        for start, end in todays:
            if start == end:
                continue
            if start < end:
                if start <= mins < end:
                    return True
            else:
                # Cross-midnight interval (e.g., 18:00-02:00)
                if mins >= start or mins < end:
                    return True

    # Check crossing-midnight intervals from previous day that extend into today.
    yday = (wd - 1) % 7
    y = sched.get(yday)
    if y:
        for start, end in y:
            if start > end and mins < end:
                return True

    return False


def _pick_tags(raw: Dict[str, Any]) -> Dict[str, str]:
    tags = raw or {}
    out: Dict[str, str] = {}
    for k, v in tags.items():
        if not isinstance(k, str):
            continue
        if k in _KEEP_TAG_KEYS or k.startswith(_KEEP_TAG_PREFIXES):
            out[k] = str(v)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="nearby_places.py",
        description="Find nearby OSM places via Overpass API (optionally filter open-now). Prints JSON.",
    )
    p.add_argument("--lat", type=float, required=True, help="Latitude")
    p.add_argument("--lon", type=float, required=True, help="Longitude")
    p.add_argument("--radius-m", type=int, default=1500, help="Search radius in meters (default: 1500)")
    p.add_argument("--amenity", action="append", default=["restaurant"], help="OSM amenity value (repeatable)")
    p.add_argument("--limit", type=int, default=15, help="Max results to return (default: 15)")
    p.add_argument("--no-require-name", action="store_true", help="Include items without a name tag")

    p.add_argument("--travelmode", default="walking", help="Google Maps directions mode (walking|driving|transit|bicycling)")

    p.add_argument("--open-now", action="store_true", help="Compute/filter open-now using `opening_hours` when possible")
    p.add_argument(
        "--include-unknown-hours",
        action="store_true",
        help="When --open-now is set and parsing is possible, keep items with unparseable/missing hours (open_now=null).",
    )

    p.add_argument("--http-timeout-seconds", type=float, default=30.0, help="Per-request timeout (default: 30)")
    args = p.parse_args(argv)

    if not (-90.0 <= args.lat <= 90.0) or not (-180.0 <= args.lon <= 180.0):
        _die("Invalid coordinates")

    radius_m = int(args.radius_m)
    if radius_m <= 0:
        _die("--radius-m must be > 0")
    if radius_m > 20_000:
        _die("--radius-m too large (max 20000)")

    limit = int(args.limit)
    if limit <= 0:
        _die("--limit must be > 0")
    if limit > 200:
        _die("--limit too large (max 200)")

    amenities = _sanitize_amenities(args.amenity or [])
    require_name = not args.no_require_name

    warnings: List[str] = []

    local_time: Optional[dict] = None
    local_dt: Optional[_dt.datetime] = None
    local_time_error: Optional[str] = None

    if args.open_now:
        try:
            local_time = _get_local_time(args.lat, args.lon, timeout=float(args.http_timeout_seconds))
            # timeapi returns local wall-clock components; build a naive datetime.
            local_dt = _dt.datetime(
                int(local_time["year"]),
                int(local_time["month"]),
                int(local_time["day"]),
                int(local_time["hour"]),
                int(local_time["minute"]),
                int(local_time.get("seconds", 0)),
            )
        except Exception as e:
            local_time_error = f"{type(e).__name__}: {e}"
            warnings.append("Failed to fetch local time from timeapi.io; open-now filtering was not applied.")

    apply_open_now_filter = bool(args.open_now and local_dt is not None)

    query = _overpass_query(args.lat, args.lon, radius_m=radius_m, amenities=amenities, require_name=require_name)

    elements: List[Dict[str, Any]] = []
    osm_search: Dict[str, Any] = {
        "provider": "overpass",
        "query": query,
    }
    overpass_error: Optional[str] = None
    try:
        overpass, overpass_meta = _fetch_overpass(query, timeout=float(args.http_timeout_seconds))
        elems = overpass.get("elements")
        if not isinstance(elems, list):
            raise RuntimeError("Unexpected Overpass response (no elements list)")
        elements = elems
        osm_search.update({"provider": "overpass", **overpass_meta})
    except Exception as e:
        overpass_error = f"{type(e).__name__}: {e}"
        try:
            nomi_items, nomi_meta = _fetch_nominatim(
                lat=args.lat,
                lon=args.lon,
                radius_m=radius_m,
                amenities=amenities,
                limit=limit,
                timeout=float(args.http_timeout_seconds),
            )
            elements = _nominatim_to_elements(nomi_items, amenities=amenities)
            if not elements:
                raise RuntimeError("Nominatim returned no usable elements")
            osm_search = {
                "provider": "nominatim",
                "overpass_error": overpass_error,
                "viewbox": _viewbox_for_radius(args.lat, args.lon, radius_m=radius_m),
                **nomi_meta,
            }
            warnings.append("Overpass failed; fell back to Nominatim (results may be less precise).")
        except Exception as e2:
            _die(
                "Nearby search failed (Overpass and Nominatim)",
                extra={
                    "overpass_error": overpass_error,
                    "nominatim_error": f"{type(e2).__name__}: {e2}",
                },
            )

    results: List[Dict[str, Any]] = []
    seen: set[tuple] = set()
    for el in elements:
        if not isinstance(el, dict):
            continue
        osm_type = el.get("type")
        osm_id = el.get("id")
        if osm_type not in ("node", "way", "relation"):
            continue
        if not isinstance(osm_id, int):
            continue

        lat = el.get("lat")
        lon = el.get("lon")
        if (lat is None or lon is None) and isinstance(el.get("center"), dict):
            lat = el["center"].get("lat")
            lon = el["center"].get("lon")
        if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
            continue

        tags_raw = el.get("tags") if isinstance(el.get("tags"), dict) else {}
        name = tags_raw.get("name")
        if require_name and not name:
            continue

        dist_m = _haversine_m(args.lat, args.lon, float(lat), float(lon))
        key = (name or "", round(float(lat), 6), round(float(lon), 6))
        if key in seen:
            continue
        seen.add(key)

        tags = _pick_tags(tags_raw)
        opening_hours = tags_raw.get("opening_hours") if isinstance(tags_raw.get("opening_hours"), str) else None

        open_now: Optional[bool] = None
        if apply_open_now_filter and opening_hours:
            open_now = _opening_hours_open_now(opening_hours, local_dt=local_dt)  # type: ignore[arg-type]

        if apply_open_now_filter:
            if open_now is True:
                pass
            elif open_now is None and args.include_unknown_hours:
                pass
            else:
                continue

        maps_query = _build_maps_query(str(name) if name else None, tags_raw) or (str(name) if name else "")

        results.append(
            {
                "name": str(name) if name else None,
                "distance_m": int(round(dist_m)),
                "open_now": open_now,
                "opening_hours": opening_hours,
                "lat": float(lat),
                "lon": float(lon),
                "osm_type": osm_type,
                "osm_id": osm_id,
                "osm_url": _osm_url(osm_type, osm_id),
                "google_maps_url": _google_maps_url(float(lat), float(lon)),
                "google_maps_pin_url": _google_maps_url(float(lat), float(lon)),
                "google_maps_directions_url": _google_maps_directions_url(
                    float(args.lat),
                    float(args.lon),
                    float(lat),
                    float(lon),
                    travelmode=str(args.travelmode or "walking"),
                ),
                "google_maps_place_search_query": maps_query or None,
                "google_maps_place_search_url": _google_maps_place_search_url(
                    maps_query,
                    near_lat=float(lat),
                    near_lon=float(lon),
                    zoom=17,
                ),
                "tags": tags,
            }
        )

    results.sort(key=lambda r: (r.get("distance_m", 10**12), (r.get("name") or "")))
    results = results[:limit]

    out: Dict[str, Any] = {
        "success": True,
        "input": {
            "lat": args.lat,
            "lon": args.lon,
            "radius_m": radius_m,
            "amenity": amenities,
            "limit": limit,
            "require_name": require_name,
            "travelmode": str(args.travelmode or "walking"),
            "open_now_requested": bool(args.open_now),
            "open_now_filter_applied": bool(apply_open_now_filter),
            "include_unknown_hours": bool(args.include_unknown_hours),
        },
        "osm_search": osm_search,
        "local_time": local_time if local_time else None,
        "local_time_error": local_time_error,
        "warnings": warnings or None,
        "results": results,
    }
    _print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
