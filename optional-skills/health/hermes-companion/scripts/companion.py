#!/usr/bin/env python3
"""Read Hermes Companion iPhone telemetry from the local iCloud folder.

The iOS app writes latest_location.json, location_history.json, and latest_health.json
into its iCloud container. macOS materialises those files on disk. This script turns
them into the place, motion, sleep, and workout facts the Hermes skill speaks
from. It uses the Python standard library only.

A location file exists only after GPSPersistDecision accepts a move. Its
timestamp is the last accepted move. A growing age at a known place means the
person is still there.

Usage:
  python3 companion.py
  python3 companion.py --json
  python3 companion.py --timeline
  python3 companion.py --health
  python3 companion.py --context
  python3 companion.py --list
  python3 companion.py --add --name Home --category home --lat 52.52 --lon 13.40
  python3 companion.py --remove home
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import shutil
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import (
    Final,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypedDict,
    Union,
    cast,
)

MOTION_FRESH_SECONDS: Final[int] = 300
LONG_STAY_SECONDS: Final[int] = 600
MOVING_ACTIVITIES: Final[Tuple[str, ...]] = ("walking", "running", "cycling", "automotive")
MOTION_LABELS: Final[Mapping[str, str]] = {
    "walking": "walking",
    "running": "running",
    "cycling": "cycling",
    "automotive": "driving",
}
CATEGORIES: Final[Tuple[str, ...]] = ("home", "work", "gym", "cafe", "outdoors", "general")
PROSE_HEALTH_KEYS: Final[Tuple[str, ...]] = (
    "sleep_insight",
    "workout_insight",
    "nutrition_reminder",
    "suggested_openers",
    "suggestedOpeners",
)

PLACE_CITY_KEYS: Final[Tuple[str, ...]] = ("city", "locality", "placemark_locality")

ICLOUD_DIRS: Final[Tuple[Path, ...]] = (
    Path.home() / "Library/Mobile Documents/iCloud~com~hermes~HermesCompanion/Documents",
    Path.home() / "Library/Mobile Documents/iCloud~com~hermes~HermesCompanion",
    Path.home() / "Library/Mobile Documents/com~apple~CloudDocs/HermesCompanion",
    Path.home() / "Library/Mobile Documents/com~apple~CloudDocs/Hermes Companion",
    Path.home() / ".hermes/hermes-companion/data",
)


# ==============================================================================
# Typed Data Transfer Objects (no Any)
# ==============================================================================

class PlaceRecord(TypedDict, total=False):
    city: str
    locality: str
    id: str
    name: str
    category: str
    activity: str
    latitude: float
    longitude: float
    radius_meters: float
    notes: str
    distance_meters: float


class SleepFacts(TypedDict, total=False):
    formatted_duration: Optional[str]
    total_sleep_minutes: Optional[Union[int, float]]
    quality_rating: Optional[str]


class WorkoutFacts(TypedDict, total=False):
    workout_type: Optional[str]
    is_currently_active: Optional[bool]
    phase: Optional[str]
    duration_minutes: Optional[Union[int, float]]
    minutes_since_completion: Optional[Union[int, float]]
    active_calories: Optional[Union[int, float]]


class HealthFacts(TypedDict):
    sleep: Optional[SleepFacts]
    workout: Optional[WorkoutFacts]


class ResolvedHealth(TypedDict):
    status: str
    source_file: str
    recorded_at: Optional[str]
    age_seconds: Optional[int]
    age_human: str
    recovery_status: str
    step_count_today: Union[int, float]
    active_calories_today: Union[int, float]
    resting_heart_rate_bpm: Optional[Union[int, float]]
    current_heart_rate_bpm: Optional[Union[int, float]]
    heart_rate_variability_sdnn: Optional[Union[int, float]]
    sleep: Optional[SleepFacts]
    workout: Optional[WorkoutFacts]


class ResolvedLocation(TypedDict, total=False):
    status: str
    source_file: str
    latitude: float
    longitude: float
    coordinates: str
    accuracy_meters: Optional[float]
    recorded_at: Optional[str]
    age_seconds: Optional[int]
    age_human: str
    minutes_since_last_move: Optional[int]
    movement_reason: Optional[str]
    is_stale: bool
    still_there: bool
    still_there_status: str
    place_name: str
    place_category: str
    activity: str
    is_at_known_place: bool
    known_place_id: Optional[str]
    distance_to_center_meters: Optional[float]
    motion_activity: str
    motion_confidence: Optional[str]
    motion_age_seconds: Optional[int]
    motion_fresh: bool
    is_moving_now: bool
    trigger_source: Optional[str]
    app_state: Optional[str]
    arrived_at: str
    dwell_time: str
    previous_place: str
    telemetry_gap: str
    placemark_name: Optional[str]
    placemark_locality: Optional[str]
    placemark_thoroughfare: Optional[str]


class TimelineStayEvent(TypedDict, total=False):
    type: str
    place_name: str
    place_category: str
    arrived_at: Optional[str]
    departed_at: Optional[str]
    dwell_minutes: int
    dwell_human: str
    records_count: int
    is_current: bool
    telemetry_gap_before: Optional[str]
    start_timestamp: datetime
    last_timestamp: datetime


class TimelineTransitEvent(TypedDict, total=False):
    type: str
    from_place: str
    to_place: str
    place_name: str
    place_category: str
    started_at: Optional[str]
    ended_at: Optional[str]
    distance_meters: float
    duration_minutes: int
    activity: str
    is_current: bool
    telemetry_gap: Optional[str]
    start_timestamp: datetime
    last_timestamp: datetime


TimelineEvent = Union[TimelineStayEvent, TimelineTransitEvent]


class TimelinePayload(TypedDict, total=False):
    status: str
    total_records: int
    events_count: int
    current_place: str
    current_category: str
    current_motion: str
    latest_recorded_at: Optional[str]
    latest_age_seconds: Optional[int]
    latest_age_human: str
    events: List[TimelineEvent]
    source_file: str


class ProfileRecord(TypedDict):
    id: str
    name: str
    last_active: str


class ProfilesConfig(TypedDict):
    updated_at: str
    profiles: List[ProfileRecord]


class DispatchThreadMeta(TypedDict, total=False):
    thread_id: str
    subject: str
    target_profile: str
    status: str
    created_at: str
    updated_at: str
    message_count: int
    last_snippet: Optional[str]
    last_error: Optional[str]
    dispatch_retries: Optional[int]
    failed_at: Optional[str]


@dataclass(frozen=True)
class JsonRead:
    """Outcome of reading a JSON object from the container. Immutable.

    `reason` is empty when the object was read, and otherwise names the failure:
    missing, empty, invalid JSON, not a JSON object, or unreadable with the OS error.
    """
    data: Optional[dict[str, object]]
    reason: str


class ThreadScan(TypedDict):
    threads: List[DispatchThreadMeta]
    unreadable: List[str]


class SweepReport(TypedDict):
    replies: int
    pending: int
    unreadable: int


class DispatchMessageRecord(TypedDict):
    id: str
    thread_id: str
    sender: str
    timestamp: str
    body: str


# ==============================================================================
# Pure Functional Helpers & Utilities
# ==============================================================================

def log(level: str, message: str) -> None:
    """Structured single line on stderr, flushed and timestamped.

    launchd captures stderr into companion-inbox.error.log; an unflushed buffer dies
    with the process, and without a timestamp a stall cannot be placed on a timeline.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"{stamp} {level} hermes-companion: {message}", file=sys.stderr, flush=True)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _to_float(val: object, default: float = 0.0) -> float:
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        try:
            return float(val)
        except ValueError:
            pass
    return default


def _to_int(val: object, default: int = 0) -> int:
    if isinstance(val, (int, float)):
        return int(val)
    if isinstance(val, str):
        try:
            return int(float(val))
        except ValueError:
            pass
    return default


def parse_utc(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def age_seconds(value: Optional[str], now: datetime) -> Optional[int]:
    recorded = parse_utc(value)
    if recorded is None:
        return None
    return max(0, int((now - recorded).total_seconds()))


def human_age(seconds: Optional[int]) -> str:
    if seconds is None:
        return "unknown"
    if seconds < 60:
        return f"{seconds}s"
    hours, minutes = divmod(seconds // 60, 60)
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


def haversine_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2.0) ** 2
    return radius * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def slug(name: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in name.strip())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "place"


def expand(path: str) -> Path:
    return Path(path).expanduser()


def icloud_directories(override: Optional[str]) -> List[Path]:
    if override:
        return [expand(override)]
    env = os.environ.get("HERMES_COMPANION_ICLOUD_DIR", "").strip()
    if env:
        return [expand(env)]
    storage_env = os.environ.get("HERMES_COMPANION_STORAGE_DIR", "").strip()
    if storage_env:
        return [expand(storage_env)]
    return list(ICLOUD_DIRS)


def read_json_file(path: Path) -> Optional[Mapping[str, object]]:
    return safe_read_json(path)


def first_reading(directories: Sequence[Path], filename: str) -> Tuple[Optional[Mapping[str, object]], Optional[Path]]:
    for directory in directories:
        path = directory / filename
        if not safe_is_file(path):
            continue
        data = read_json_file(path)
        if data is not None:
            return data, path
    return None, None


def default_places_file() -> Path:
    """Places file for this machine, respecting active profile with no branch leak."""
    env = os.environ.get("HERMES_COMPANION_PLACES", "").strip()
    if env:
        return expand(env)
    hermes_home = os.environ.get("HERMES_HOME", "").strip()
    if hermes_home:
        return expand(hermes_home) / "state" / "places.json"
    profiles = Path.home() / ".hermes" / "profiles"
    found = sorted(profiles.glob("*/state/places.json")) if profiles.is_dir() else []
    if len(found) == 1:
        return found[0]
    return Path.home() / ".hermes" / "hermes-companion" / "places.json"


def place_from_record(item: Mapping[str, object]) -> PlaceRecord:
    """Normalise one raw place object. Pure.

    The optional city/locality keys are kept: they are the only city name available when
    the iPhone has not reverse-geocoded the current fix, and losing them is what lets the
    agent guess a city out of thin air.
    """
    city = text_field(item, "city")
    locality = text_field(item, "locality")
    record = PlaceRecord(
        id=str(item.get("id", "")),
        name=str(item.get("name", "")),
        category=str(item.get("category", "general")),
        activity=str(item.get("activity", f"at {item.get('name', 'place')}")),
        latitude=_to_float(item.get("latitude")),
        longitude=_to_float(item.get("longitude")),
        radius_meters=_to_float(item.get("radius_meters"), 150.0),
        notes=str(item.get("notes", "")),
    )
    if city:
        record["city"] = city
    if locality:
        record["locality"] = locality
    return record


def load_places(path: Path) -> List[PlaceRecord]:
    records: List[PlaceRecord] = []
    if safe_is_file(path):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                records = [
                    place_from_record(item)
                    for item in data
                    if isinstance(item, dict) and "latitude" in item and "longitude" in item
                ]
            else:
                log("ERROR", f"places file {path} must be a JSON list")
        except (OSError, json.JSONDecodeError) as exc:
            log("ERROR", f"could not read places {path}: {exc}")

    # Seamlessly merge places configured in the iOS app via iCloud container
    for directory in ICLOUD_DIRS:
        icloud_file = directory / "places.json"
        if safe_is_file(icloud_file) and (not safe_is_file(path) or icloud_file.resolve() != path.resolve()):
            try:
                ic_data = json.loads(icloud_file.read_text(encoding="utf-8"))
                if isinstance(ic_data, list):
                    existing_names = {r["name"].strip().lower() for r in records}
                    existing_ids = {r["id"].strip().lower() for r in records if r["id"]}
                    for item in ic_data:
                        if not isinstance(item, dict) or "latitude" not in item or "longitude" not in item:
                            continue
                        p_name = str(item.get("name", "")).strip().lower()
                        p_id = str(item.get("id", "")).strip().lower()
                        if p_id in existing_ids or (p_name and p_name in existing_names):
                            continue
                        records.append(place_from_record(item))
                        if p_name:
                            existing_names.add(p_name)
                        if p_id:
                            existing_ids.add(p_id)
            except Exception:
                pass
            break

    return records


def save_places(path: Path, places: Sequence[PlaceRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(list(places), indent=2, ensure_ascii=False) + "\n"
    path.write_text(content, encoding="utf-8")
    log("INFO", f"wrote {len(places)} place(s) to {path}")
    path_str = str(path.resolve())
    is_temp = (
        path_str.startswith(tempfile.gettempdir())
        or path_str.startswith("/tmp")
        or path_str.startswith("/var/folders")
        or path_str.startswith("/private/var")
        or path_str.startswith("/private/tmp")
    )
    if not is_temp:
        for directory in ICLOUD_DIRS:
            if directory.is_dir():
                icloud_file = directory / "places.json"
                if icloud_file.resolve() != path.resolve():
                    try:
                        icloud_file.write_text(content, encoding="utf-8")
                        log("INFO", f"mirrored {len(places)} place(s) to iCloud: {icloud_file}")
                    except Exception:
                        pass
                break




def match_place(latitude: float, longitude: float, places: Sequence[PlaceRecord]) -> Optional[PlaceRecord]:
    valid_matches = [
        (dist, place)
        for place in places
        for dist in [haversine_meters(latitude, longitude, float(place["latitude"]), float(place["longitude"]))]
        if dist <= float(place.get("radius_meters", 150.0))
    ]

    if not valid_matches:
        return None

    best_dist, best_place = min(valid_matches, key=lambda item: item[0])
    return {**best_place, "distance_meters": round(best_dist, 1)}


def determine_still_there(
    at_known: bool,
    moving_now: bool,
    is_transit: bool,
    gps_age: Optional[int],
    motion_fresh: bool,
    motion_moving: bool,
) -> Tuple[bool, str]:
    """Pure functional resolution of whether the user is still at their last fix."""
    if moving_now or is_transit:
        return False, "no"
    if gps_age is None:
        return False, "unconfirmed (missing or invalid timestamp)"
    if motion_fresh and not motion_moving:
        return True, "yes"
    if gps_age <= 1800:
        return True, "yes"
    return False, f"unconfirmed (no fresh ping in {human_age(gps_age)})"


def resolve_location(
    raw: Mapping[str, object],
    places: Sequence[PlaceRecord],
    now: datetime,
    source: str,
) -> ResolvedLocation:
    latitude = _to_float(raw.get("latitude"))
    longitude = _to_float(raw.get("longitude"))
    recorded_at = str(raw.get("timestamp") or raw.get("recorded_at") or "") or None
    gps_age = age_seconds(recorded_at, now)

    motion = str(raw.get("motion_activity") or "unknown")
    motion_ts_val = raw.get("motion_timestamp")
    motion_age = age_seconds(str(motion_ts_val), now) if motion_ts_val else None
    motion_age_value = motion_age if motion_age is not None else 999_999

    valid_motion = motion in ("stationary", *MOVING_ACTIVITIES)
    motion_fresh = valid_motion and motion_age_value <= MOTION_FRESH_SECONDS
    motion_moving = motion in MOVING_ACTIVITIES
    long_stay = gps_age is None or gps_age > LONG_STAY_SECONDS
    moving_now = motion_moving if motion_fresh else False
    known = match_place(latitude, longitude, places)
    label = MOTION_LABELS.get(motion)
    minutes = (gps_age // 60) if gps_age is not None else None

    if moving_now and known and label in ("walking", "running", "cycling"):
        place_name = str(known.get("name") or "Known place")
        category = str(known.get("category") or "general")
        activity = label
        at_known = True
        is_transit = False
    elif moving_now:
        place_name = "In Transit"
        category = "transit"
        activity = motion if motion in MOVING_ACTIVITIES else "moving"
        at_known = False
        is_transit = True
        known = None
    elif known:
        place_name = str(known.get("name") or "Known place")
        category = str(known.get("category") or "general")
        activity = str(known.get("activity") or category)
        at_known = True
        is_transit = False
    elif raw.get("placemark_name"):
        pm_name = str(raw["placemark_name"]).strip()
        pm_loc = str(raw.get("placemark_locality") or "").strip()
        place_name = f"{pm_name}, {pm_loc}" if pm_loc and pm_loc not in pm_name else pm_name
        category = "apple_maps"
        activity = f"at {pm_name}"
        at_known = False
        is_transit = False
    else:
        place_name = "Unlisted place"
        category = "unlisted"
        activity = "unlisted"
        at_known = False
        is_transit = False

    raw_acc = raw.get("horizontal_accuracy", raw.get("accuracy"))
    accuracy = _to_float(raw_acc) if raw_acc is not None else None

    still_there, still_there_txt = determine_still_there(
        at_known=at_known,
        moving_now=moving_now,
        is_transit=is_transit,
        gps_age=gps_age,
        motion_fresh=motion_fresh,
        motion_moving=motion_moving,
    )

    reason = str(raw["movement_reason"]) if raw.get("movement_reason") else None
    motion_conf = str(raw["motion_confidence"]) if raw.get("motion_confidence") else None
    trig_src = str(raw["source"]) if raw.get("source") else None
    app_st = str(raw["app_state"]) if raw.get("app_state") else None

    return {
        "status": "ok",
        "source_file": source,
        "latitude": latitude,
        "longitude": longitude,
        "coordinates": f"{latitude:.6f}, {longitude:.6f}",
        "accuracy_meters": accuracy,
        "recorded_at": recorded_at,
        "age_seconds": gps_age,
        "age_human": human_age(gps_age),
        "minutes_since_last_move": minutes,
        "movement_reason": reason,
        "is_stale": long_stay,
        "still_there": still_there,
        "still_there_status": still_there_txt,
        "place_name": place_name,
        "place_category": category,
        "activity": activity,
        "is_at_known_place": at_known,
        "known_place_id": known.get("id") if known else None,
        "distance_to_center_meters": known.get("distance_meters") if known else None,
        "motion_activity": motion if valid_motion else "unknown",
        "motion_confidence": motion_conf,
        "motion_age_seconds": motion_age,
        "motion_fresh": motion_fresh,
        "is_moving_now": moving_now,
        "trigger_source": trig_src,
        "app_state": app_st,
        "placemark_name": str(raw["placemark_name"]) if raw.get("placemark_name") else None,
        "placemark_locality": str(raw["placemark_locality"]) if raw.get("placemark_locality") else None,
        "placemark_thoroughfare": str(raw["placemark_thoroughfare"]) if raw.get("placemark_thoroughfare") else None,
    }


def health_facts(raw: Mapping[str, object]) -> HealthFacts:
    """Numbers and codes only. English prose fields are filtered out."""
    raw_ctx = raw.get("conversational_context")
    ctx: Mapping[str, object] = raw_ctx if isinstance(raw_ctx, Mapping) else {}
    leftover = [key for key in PROSE_HEALTH_KEYS if ctx.get(key) or raw.get(key)]
    if leftover:
        log("INFO", "ignoring health prose fields: " + ", ".join(leftover))

    raw_sleep = raw.get("sleep")
    sleep: Optional[Mapping[str, object]] = raw_sleep if isinstance(raw_sleep, Mapping) else None

    raw_workout = raw.get("workout")
    workout: Optional[Mapping[str, object]] = raw_workout if isinstance(raw_workout, Mapping) else None

    sleep_facts: Optional[SleepFacts] = None
    if sleep is not None:
        sleep_facts = {
            "formatted_duration": str(sleep["formatted_duration"]) if sleep.get("formatted_duration") else None,
            "total_sleep_minutes": _to_float(sleep.get("total_sleep_minutes")),
            "quality_rating": str(sleep["quality_rating"]) if sleep.get("quality_rating") else None,
        }

    workout_facts: Optional[WorkoutFacts] = None
    if workout is not None:
        workout_facts = {
            "workout_type": str(workout["workout_type"]) if workout.get("workout_type") else None,
            "is_currently_active": bool(workout["is_currently_active"]) if "is_currently_active" in workout else None,
            "phase": str(workout["phase"]) if workout.get("phase") else None,
            "duration_minutes": _to_float(workout.get("duration_minutes")),
            "minutes_since_completion": _to_float(workout.get("minutes_since_completion")),
            "active_calories": _to_float(workout.get("active_calories")),
        }

    return {"sleep": sleep_facts, "workout": workout_facts}


def resolve_health(raw: Mapping[str, object], now: datetime, source: str) -> ResolvedHealth:
    recorded_at = str(raw.get("timestamp") or raw.get("recorded_at") or "") or None
    seconds = age_seconds(recorded_at, now)
    facts = health_facts(raw)
    return {
        "status": "ok",
        "source_file": source,
        "recorded_at": recorded_at,
        "age_seconds": seconds,
        "age_human": human_age(seconds),
        "recovery_status": str(raw.get("recovery_status", "unknown")),
        "step_count_today": _to_float(raw.get("step_count_today")),
        "active_calories_today": _to_float(raw.get("active_calories_today")),
        "resting_heart_rate_bpm": _to_float(raw["resting_heart_rate_bpm"]) if raw.get("resting_heart_rate_bpm") is not None else None,
        "current_heart_rate_bpm": _to_float(raw["current_heart_rate_bpm"]) if raw.get("current_heart_rate_bpm") is not None else None,
        "heart_rate_variability_sdnn": _to_float(raw["heart_rate_variability_sdnn"]) if raw.get("heart_rate_variability_sdnn") is not None else None,
        "sleep": facts["sleep"],
        "workout": facts["workout"],
    }


def format_location(loc: ResolvedLocation) -> str:
    motion_age = loc.get("motion_age_seconds")
    motion_age_text = "none" if motion_age is None else str(motion_age)
    still_there_txt = loc.get("still_there_status") or ("yes" if loc.get("still_there") else "no")
    minutes_text = "unknown" if loc.get("minutes_since_last_move") is None else str(loc["minutes_since_last_move"])
    age_seconds_text = "unknown" if loc.get("age_seconds") is None else str(loc["age_seconds"])

    lines = [
        "facts_only: reply in the user's language; do not quote this block",
        f"place_name: {loc.get('place_name', 'unknown')}",
        f"place_category: {loc.get('place_category', 'general')}",
        f"still_there: {still_there_txt}",
        f"minutes_since_last_move: {minutes_text}",
        f"movement_reason: {loc.get('movement_reason') or 'absent'}",
        f"motion_activity: {loc.get('motion_activity', 'unknown')}",
        f"motion_fresh: {'yes' if loc.get('motion_fresh') else 'no'}",
        f"motion_age_seconds: {motion_age_text}",
        f"is_moving_now: {'yes' if loc.get('is_moving_now') else 'no'}",
        f"recorded_at: {loc.get('recorded_at') or 'unknown'}",
        f"age_seconds: {age_seconds_text}",
        f"coordinates: {loc.get('coordinates', 'unknown')}",
        f"source: {loc.get('source_file', 'unknown')}",
    ]
    if loc.get("placemark_name"):
        lines.append(f"apple_maps_placemark: {loc['placemark_name']}")
    if loc.get("arrived_at"):
        lines.append(f"arrived_at: {loc['arrived_at']}")
    if loc.get("dwell_time"):
        lines.append(f"dwell_time: {loc['dwell_time']}")
    if loc.get("previous_place"):
        lines.append(f"previous_place: {loc['previous_place']}")
    if loc.get("telemetry_gap"):
        lines.append(f"telemetry_gap: {loc['telemetry_gap']}")
    return "\n".join(lines)


def format_health(health: ResolvedHealth) -> str:
    lines = [
        "facts_only: reply in the user's language; do not quote this block",
        f"recovery_status: {health.get('recovery_status')}",
        f"recorded_at: {health.get('recorded_at') or 'unknown'}",
        f"age_seconds: {'unknown' if health.get('age_seconds') is None else health['age_seconds']}",
    ]
    sleep = health.get("sleep")
    if sleep:
        dur = sleep.get("formatted_duration") or sleep.get("total_sleep_minutes") or "unknown"
        lines.append(f"sleep_duration: {dur}")
        lines.append(f"sleep_quality: {sleep.get('quality_rating') or 'unknown'}")
    else:
        lines.append("sleep_duration: none")

    workout = health.get("workout")
    if workout:
        lines.append(f"workout_type: {workout.get('workout_type') or 'unknown'}")
        lines.append(f"workout_active: {'yes' if workout.get('is_currently_active') else 'no'}")
        lines.append(f"workout_phase: {workout.get('phase') or 'unknown'}")
        lines.append(f"workout_duration_minutes: {workout.get('duration_minutes')}")
        if workout.get("minutes_since_completion") is not None:
            lines.append(f"minutes_since_workout: {workout.get('minutes_since_completion')}")
    else:
        lines.append("workout_type: none")

    lines.append(f"steps_today: {_to_int(health.get('step_count_today'))}")
    lines.append(f"active_calories_today: {_to_int(health.get('active_calories_today'))}")
    if health.get("resting_heart_rate_bpm") is not None:
        lines.append(f"resting_heart_rate_bpm: {_to_int(health['resting_heart_rate_bpm'])}")
    if health.get("heart_rate_variability_sdnn") is not None:
        lines.append(f"hrv_sdnn_ms: {_to_int(health['heart_rate_variability_sdnn'])}")
    lines.append(f"source: {health['source_file']}")
    return "\n".join(lines)


def read_history(directories: Sequence[Path]) -> Tuple[List[Mapping[str, object]], Optional[Path]]:
    raw, path = first_reading(directories, "location_history.json")
    if raw is None or path is None:
        return [], None
    records = raw.get("records") if isinstance(raw, dict) else raw
    if not isinstance(records, list):
        return [], path
    valid: List[Mapping[str, object]] = [
        item for item in records if isinstance(item, dict) and "latitude" in item and "longitude" in item
    ]
    valid.sort(key=lambda r: str(r.get("timestamp") or r.get("recorded_at") or ""))
    return valid, path


def parse_since(since_str: Optional[str], now: datetime) -> Optional[datetime]:
    if not since_str:
        return None
    val = since_str.strip().lower()
    if val == "today":
        return datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
    if val.endswith("h"):
        try:
            return now - timedelta(hours=float(val[:-1]))
        except ValueError:
            pass
    if val.endswith("d"):
        try:
            return now - timedelta(days=float(val[:-1]))
        except ValueError:
            pass
    if val.endswith("m"):
        try:
            return now - timedelta(minutes=float(val[:-1]))
        except ValueError:
            pass
    return parse_utc(since_str)


def build_timeline(
    records: Sequence[Mapping[str, object]],
    places: Sequence[PlaceRecord],
    now: datetime,
    since: Optional[datetime] = None,
) -> TimelinePayload:
    filtered_records = [
        r
        for r in records
        if not since or (parse_utc(str(r.get("timestamp") or r.get("recorded_at") or "")) or now) >= since
    ]

    if not filtered_records:
        return {"events": [], "events_count": 0, "status": "no_records", "total_records": 0}

    events: List[TimelineEvent] = []
    current_stay: Optional[TimelineStayEvent] = None
    current_transit: Optional[TimelineTransitEvent] = None
    last_record: Optional[Mapping[str, object]] = None
    last_place_name: Optional[str] = None

    for r in filtered_records:
        lat = _to_float(r.get("latitude"))
        lon = _to_float(r.get("longitude"))
        ts_str = str(r.get("timestamp") or r.get("recorded_at") or "")
        ts = parse_utc(ts_str) or now
        motion = str(r.get("motion_activity") or "unknown")
        known = match_place(lat, lon, places)

        gap_detected = False
        gap_minutes = 0
        if last_record:
            last_ts = parse_utc(str(last_record.get("timestamp") or last_record.get("recorded_at") or "")) or now
            delta_sec = (ts - last_ts).total_seconds()
            dist = haversine_meters(_to_float(last_record.get("latitude")), _to_float(last_record.get("longitude")), lat, lon)
            if delta_sec > 1800 and dist > 100:
                gap_detected = True
                gap_minutes = int(delta_sec // 60)

        pm = str(r.get("placemark_name") or "").strip()
        pm_loc = str(r.get("placemark_locality") or "").strip()
        pm_display = f"{pm}, {pm_loc}" if (pm_loc and pm and pm_loc not in pm) else pm

        is_place = known is not None or bool(pm)
        if known:
            place_name = known["name"]
            place_cat = known["category"]
        elif pm:
            place_name = pm_display
            place_cat = "apple_maps"
        else:
            place_name = "Unlisted place"
            place_cat = "unlisted"

        if is_place:
            if current_transit is not None:
                events.append(current_transit)
                current_transit = None

            if current_stay and current_stay["place_name"] == place_name:
                dwell_sec = max(0, int((ts - current_stay["start_timestamp"]).total_seconds()))
                current_stay = {
                    **current_stay,
                    "departed_at": ts_str,
                    "records_count": current_stay["records_count"] + 1,
                    "last_timestamp": ts,
                    "dwell_minutes": dwell_sec // 60,
                    "dwell_human": human_age(dwell_sec),
                    "telemetry_gap_before": f"silent for {gap_minutes}m before this point" if gap_detected else current_stay.get("telemetry_gap_before"),
                }
            else:
                if current_stay:
                    events.append(current_stay)
                if last_record:
                    last_known = match_place(_to_float(last_record.get("latitude")), _to_float(last_record.get("longitude")), places)
                    from_name = last_known["name"] if last_known else (last_place_name or "Transit")
                    dist = haversine_meters(_to_float(last_record.get("latitude")), _to_float(last_record.get("longitude")), lat, lon)
                    if from_name != place_name and dist > 50:
                        transit_event: TimelineTransitEvent = {
                            "type": "transit",
                            "from_place": from_name,
                            "to_place": place_name,
                            "started_at": str(last_record.get("timestamp") or last_record.get("recorded_at")),
                            "ended_at": ts_str,
                            "distance_meters": round(dist, 1),
                            "duration_minutes": max(1, int((ts - (parse_utc(str(last_record.get("timestamp") or last_record.get("recorded_at"))) or ts)).total_seconds() // 60)),
                            "activity": motion if motion in MOVING_ACTIVITIES else "moving",
                            "telemetry_gap": f"silent for {gap_minutes}m during transit" if gap_detected else None,
                        }
                        events.append(transit_event)

                current_stay = {
                    "type": "stay",
                    "place_name": place_name,
                    "place_category": place_cat,
                    "arrived_at": ts_str,
                    "departed_at": ts_str,
                    "start_timestamp": ts,
                    "last_timestamp": ts,
                    "dwell_minutes": 0,
                    "dwell_human": "just arrived",
                    "records_count": 1,
                    "is_current": False,
                    "telemetry_gap_before": f"silent for {gap_minutes}m before arrival" if gap_detected else None,
                }
        else:
            if current_stay:
                events.append(current_stay)
                current_stay = None

            dist = haversine_meters(_to_float(last_record.get("latitude")), _to_float(last_record.get("longitude")), lat, lon) if last_record else 0.0
            if current_transit is None:
                current_transit = {
                    "type": "transit",
                    "place_name": "In Transit",
                    "place_category": "transit",
                    "from_place": last_place_name or "Unknown",
                    "to_place": "Moving",
                    "started_at": ts_str,
                    "ended_at": ts_str,
                    "start_timestamp": ts,
                    "last_timestamp": ts,
                    "distance_meters": round(dist, 1),
                    "duration_minutes": 0,
                    "activity": motion if motion in MOVING_ACTIVITIES else "moving",
                    "is_current": False,
                    "telemetry_gap": f"silent for {gap_minutes}m" if gap_detected else None,
                }
            else:
                updated_dist = round(current_transit.get("distance_meters", 0.0) + dist, 1)
                duration_m = max(1, int((ts - current_transit["start_timestamp"]).total_seconds() // 60))
                current_transit = {
                    **current_transit,
                    "ended_at": ts_str,
                    "last_timestamp": ts,
                    "distance_meters": updated_dist,
                    "duration_minutes": duration_m,
                }

        last_record = r
        last_place_name = place_name

    final_events = list(events)
    if current_stay:
        dwell_now = max(0, int((now - current_stay["start_timestamp"]).total_seconds()))
        active_stay: TimelineStayEvent = {
            **current_stay,
            "is_current": True,
            "dwell_minutes": dwell_now // 60,
            "dwell_human": human_age(dwell_now),
            "departed_at": None,
        }
        final_events.append(active_stay)
    elif current_transit:
        active_transit: TimelineTransitEvent = {
            **current_transit,
            "is_current": True,
        }
        final_events.append(active_transit)

    clean_events: List[TimelineEvent] = []
    for ev in final_events:
        clean_ev = dict(ev)
        clean_ev.pop("start_timestamp", None)
        clean_ev.pop("last_timestamp", None)
        clean_events.append(cast(TimelineEvent, clean_ev))

    latest = filtered_records[-1]
    latest_lat = _to_float(latest.get("latitude"))
    latest_lon = _to_float(latest.get("longitude"))
    latest_known = match_place(latest_lat, latest_lon, places)
    latest_pm = str(latest.get("placemark_name") or "").strip()
    latest_pm_loc = str(latest.get("placemark_locality") or "").strip()
    latest_pm_display = f"{latest_pm}, {latest_pm_loc}" if (latest_pm_loc and latest_pm and latest_pm_loc not in latest_pm) else latest_pm

    if latest_known:
        curr_place = latest_known["name"]
        curr_cat = latest_known["category"]
    elif latest_pm:
        curr_place = latest_pm_display
        curr_cat = "apple_maps"
    else:
        curr_place = "Unlisted place"
        curr_cat = "unlisted"

    latest_ts_str = str(latest.get("timestamp") or latest.get("recorded_at") or "")
    latest_ts = parse_utc(latest_ts_str) or now
    age_now = int((now - latest_ts).total_seconds())

    return {
        "status": "ok",
        "total_records": len(filtered_records),
        "events_count": len(clean_events),
        "current_place": curr_place,
        "current_category": curr_cat,
        "current_motion": str(latest.get("motion_activity") or "unknown"),
        "latest_recorded_at": latest_ts_str or None,
        "latest_age_seconds": age_now,
        "latest_age_human": human_age(age_now),
        "events": clean_events,
    }


def query_history_sqlite(
    records: Sequence[Mapping[str, object]],
    places: Sequence[PlaceRecord],
    sql_query: str,
) -> List[Mapping[str, object]]:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE locations (
            id TEXT,
            timestamp TEXT,
            latitude REAL,
            longitude REAL,
            altitude REAL,
            horizontal_accuracy REAL,
            course REAL,
            source TEXT,
            app_state TEXT,
            motion_activity TEXT,
            motion_confidence TEXT,
            motion_timestamp TEXT,
            movement_reason TEXT,
            device_name TEXT,
            place_name TEXT,
            place_category TEXT,
            placemark_name TEXT,
            placemark_locality TEXT
        )
        """
    )
    for r in records:
        lat = _to_float(r.get("latitude"))
        lon = _to_float(r.get("longitude"))
        known = match_place(lat, lon, places)
        pm_name = str(r.get("placemark_name") or "") or None
        pm_loc = str(r.get("placemark_locality") or "") or None
        if known:
            p_name = known["name"]
            p_cat = known["category"]
        elif pm_name:
            p_name = f"{pm_name}, {pm_loc}" if (pm_loc and pm_loc not in pm_name) else pm_name
            p_cat = "apple_maps"
        else:
            p_name = None
            p_cat = None
        cur.execute(
            """
            INSERT INTO locations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(r.get("id", "")),
                str(r.get("timestamp") or r.get("recorded_at") or ""),
                lat,
                lon,
                _to_float(r.get("altitude")),
                _to_float(r.get("horizontal_accuracy") or r.get("accuracy")),
                _to_float(r.get("course"), -1.0),
                str(r.get("source") or ""),
                str(r.get("app_state") or ""),
                str(r.get("motion_activity") or "unknown"),
                str(r.get("motion_confidence") or ""),
                str(r.get("motion_timestamp") or ""),
                str(r.get("movement_reason") or ""),
                str(r.get("device_name") or ""),
                p_name,
                p_cat,
                pm_name,
                pm_loc,
            ),
        )
    conn.commit()
    cur.execute(sql_query)
    rows = cur.fetchall()
    results: List[Mapping[str, object]] = [dict(row) for row in rows]
    conn.close()
    return results


def format_timeline(timeline: TimelinePayload) -> str:
    lines = [
        "facts_only: reply in the user's language; do not quote this block",
        f"current_place: {timeline.get('current_place', 'unknown')}",
        f"current_motion: {timeline.get('current_motion', 'unknown')}",
        f"latest_fix_age: {timeline.get('latest_age_human', 'unknown')}",
        f"events_count: {timeline.get('events_count', 0)}",
        "timeline_events:",
    ]
    events = timeline.get("events", [])
    if not events:
        lines.append("  (no events in this time window)")
    for ev in events:
        ev_type = ev.get("type", "event")
        if ev_type == "stay":
            stay_ev = cast(TimelineStayEvent, ev)
            curr_tag = " [current]" if stay_ev.get("is_current") else ""
            dep = f" -> departed {stay_ev.get('departed_at')}" if stay_ev.get("departed_at") else " -> now"
            gap = f" [{stay_ev['telemetry_gap_before']}]" if stay_ev.get("telemetry_gap_before") else ""
            lines.append(
                f"  • stay: {stay_ev.get('place_name', 'unknown')} ({stay_ev.get('place_category', 'general')}) | arrived {stay_ev.get('arrived_at')}{dep} | dwell: {stay_ev.get('dwell_human')}{curr_tag}{gap}"
            )
        elif ev_type == "transit":
            transit_ev = cast(TimelineTransitEvent, ev)
            gap = f" [{transit_ev['telemetry_gap']}]" if transit_ev.get("telemetry_gap") else ""
            dist = f"~{int(_to_float(transit_ev.get('distance_meters')))}m" if transit_ev.get("distance_meters") else "transit"
            lines.append(
                f"  • transit: {transit_ev.get('activity', 'moving')} {dist} from {transit_ev.get('from_place')} to {transit_ev.get('to_place')} | {transit_ev.get('started_at')} -> {transit_ev.get('ended_at')} ({transit_ev.get('duration_minutes')}m){gap}"
            )
        else:
            lines.append(f"  • {ev_type}: {ev}")
    return "\n".join(lines)


def emit(payload: Mapping[str, object], as_json: bool, text: str) -> int:
    if as_json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(text)
    return 0


# ==============================================================================
# CLI Commands
# ==============================================================================

def missing_icloud_explanation(directories: Sequence[Path], item_type: str = "location") -> str:
    """Generate helpful diagnostics message when an expected telemetry file is missing."""
    if sys.platform != "darwin":
        return (
            f"No accepted {item_type} file from the iPhone yet.\n"
            "Notice: Hermes Companion strictly requires macOS with native iCloud Drive access. "
            "Non-macOS environments (Linux, Docker containers, Windows) cannot sync local iCloud containers."
        )
    container_exists = any(d.is_dir() for d in directories)
    if not container_exists:
        return (
            f"No accepted {item_type} file from the iPhone yet.\n"
            "iCloud container has not synced to this Mac yet. Open Hermes Companion on your iPhone to initialize the container.\n"
            "Run 'companion.py --doctor' to diagnose system requirements."
        )
    return f"No accepted {item_type} file from the iPhone yet."


def command_where(args: argparse.Namespace, now: datetime) -> int:
    directories = icloud_directories(args.icloud_dir)
    raw, path = first_reading(directories, "latest_location.json")
    if raw is None or path is None or "latitude" not in raw or "longitude" not in raw:
        print(missing_icloud_explanation(directories, "location"))
        return 0
    places = load_places(args.places)
    loc = resolve_location(raw, places, now, str(path))

    history_records, _ = read_history(directories)
    if history_records:
        timeline = build_timeline(history_records, places, now)
        events = timeline.get("events", [])
        if events:
            current_ev = events[-1] if events[-1].get("is_current") else None
            enrichments: dict[str, str] = {}
            if current_ev and current_ev.get("type") == "stay":
                stay_item = cast(TimelineStayEvent, current_ev)
                if stay_item.get("arrived_at"):
                    enrichments["arrived_at"] = str(stay_item["arrived_at"])
                if stay_item.get("dwell_human"):
                    enrichments["dwell_time"] = str(stay_item["dwell_human"])
                if stay_item.get("telemetry_gap_before"):
                    enrichments["telemetry_gap"] = str(stay_item["telemetry_gap_before"])
            stays = [cast(TimelineStayEvent, ev) for ev in events if ev.get("type") == "stay"]
            if len(stays) >= 2 and stays[-1].get("is_current"):
                prev = stays[-2]
                enrichments["previous_place"] = f"{prev.get('place_name')} (departed {prev.get('departed_at')})"
            if enrichments:
                for k, v in enrichments.items():
                    loc[k] = v  # type: ignore[literal-required]

    return emit(loc, args.json, format_location(loc))


def command_timeline(args: argparse.Namespace, now: datetime) -> int:
    directories = icloud_directories(args.icloud_dir)
    history_records, path = read_history(directories)
    if not history_records or path is None:
        print(missing_icloud_explanation(directories, "location history"))
        return 0
    places = load_places(args.places)
    since_dt = parse_since(args.since, now)

    if args.sql:
        results = query_history_sqlite(history_records, places, args.sql)
        payload = {"count": len(results), "query": args.sql, "results": results}
        if args.json:
            print(json.dumps(payload, indent=2, ensure_ascii=False))
            return 0
        if not results:
            print("No rows returned by query.")
            return 0
        headers = list(results[0].keys())
        print(f"SQL Results ({len(results)} rows):")
        print(" | ".join(headers))
        for row in results:
            print(" | ".join(str(row.get(h, "")) for h in headers))
        return 0

    timeline = build_timeline(history_records, places, now, since=since_dt)
    timeline_with_source: TimelinePayload = {**timeline, "source_file": str(path)}
    return emit(timeline_with_source, args.json, format_timeline(timeline_with_source))


def command_health(args: argparse.Namespace, now: datetime) -> int:
    directories = icloud_directories(args.icloud_dir)
    raw, path = first_reading(directories, "latest_health.json")
    if raw is None or path is None:
        print(missing_icloud_explanation(directories, "health snapshot"))
        return 0
    health = resolve_health(raw, now, str(path))
    return emit(health, args.json, format_health(health))


def command_context(args: argparse.Namespace, now: datetime) -> int:
    directories = icloud_directories(args.icloud_dir)
    loc_raw, loc_path = first_reading(directories, "latest_location.json")
    health_raw, health_path = first_reading(directories, "latest_health.json")
    loc = None
    if loc_raw and loc_path and "latitude" in loc_raw and "longitude" in loc_raw:
        loc = resolve_location(loc_raw, load_places(args.places), now, str(loc_path))
    health = resolve_health(health_raw, now, str(health_path)) if health_raw and health_path else None
    payload = {"location": loc, "health": health}
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0
    parts = [
        format_location(loc) if loc else "facts_only: reply in the user's language; do not quote this block\nplace_name: unavailable",
        format_health(health) if health else "facts_only: reply in the user's language; do not quote this block\nrecovery_status: unavailable",
    ]
    print("\n\n".join(parts))
    return 0


def command_list(args: argparse.Namespace) -> int:
    places = load_places(args.places)
    if args.json:
        print(json.dumps(places, indent=2, ensure_ascii=False))
        return 0
    if not places:
        print(f"No places in {args.places}")
        return 0
    print(f"Known places ({len(places)}) in {args.places}:")
    for place in places:
        print(
            "• [{id}] {name} ({category}) — {activity} | {lat:.5f}, {lon:.5f} (±{radius:.0f} m)".format(
                id=place.get("id", "?"),
                name=place.get("name", "Unnamed"),
                category=place.get("category", "general"),
                activity=place.get("activity") or f"at {place.get('name', 'place')}",
                lat=float(place.get("latitude", 0.0)),
                lon=float(place.get("longitude", 0.0)),
                radius=float(place.get("radius_meters", 150.0)),
            )
        )
    return 0


def command_add(args: argparse.Namespace) -> int:
    if not args.name or args.lat is None or args.lon is None:
        log("ERROR", "--add requires --name, --lat, and --lon")
        return 2
    category = (args.category or "general").lower()
    if category not in CATEGORIES:
        log("ERROR", f"category must be one of: {', '.join(CATEGORIES)}")
        return 2
    places = load_places(args.places)
    place_id = slug(args.name)
    record: PlaceRecord = {
        "id": place_id,
        "name": args.name,
        "category": category,
        "activity": args.activity or f"at {args.name}",
        "latitude": args.lat,
        "longitude": args.lon,
        "radius_meters": args.radius,
        "notes": args.notes or "",
    }
    updated = [
        record if (p.get("id") == place_id or p.get("name") == args.name) else p
        for p in places
    ]
    replaced = any(p.get("id") == place_id or p.get("name") == args.name for p in places)
    if not replaced:
        updated.append(record)
    save_places(args.places, updated)
    print(f"{'Updated' if replaced else 'Added'} {args.name} [{place_id}]")
    return 0


def command_remove(args: argparse.Namespace) -> int:
    if not args.remove:
        log("ERROR", "--remove requires a place id or name")
        return 2
    target = args.remove.lower()
    places = load_places(args.places)
    kept = [place for place in places if str(place.get("id", "")).lower() != target and str(place.get("name", "")).lower() != target]
    if len(kept) == len(places):
        log("WARN", f"no place named {args.remove}")
        return 0
    save_places(args.places, kept)
    print(f"Removed {args.remove}")
    return 0


def get_threads_dir(override: Optional[str] = None) -> Optional[Path]:
    for directory in icloud_directories(override):
        cand = directory / "threads"
        if cand.is_dir():
            return cand
    dirs = icloud_directories(override)
    if dirs:
        cand = dirs[0] / "threads"
        cand.mkdir(parents=True, exist_ok=True)
        return cand
    return None


def get_archive_dir(override: Optional[str] = None) -> Optional[Path]:
    for directory in icloud_directories(override):
        cand = directory / "archive"
        if cand.is_dir():
            return cand
    dirs = icloud_directories(override)
    if dirs:
        cand = dirs[0] / "archive"
        cand.mkdir(parents=True, exist_ok=True)
        return cand
    return None


def safe_is_dir(path: Path) -> bool:
    try:
        return path.is_dir()
    except OSError:
        return False


def safe_is_file(path: Path) -> bool:
    try:
        return path.is_file()
    except OSError:
        return False


def warn_once(path: Path, message: str) -> None:
    """Warn about a path once per process, so a corrupt file cannot flood the log."""
    key = str(path)
    if key in _WARNED_PATHS:
        return
    _WARNED_PATHS.add(key)
    log("WARN", message)


def log_skip_throttled(key: str, level: str, message: str) -> bool:
    """Log a recurring condition at most once per SKIP_LOG_INTERVAL_SECONDS. Returns True when logged.

    A sweep runs every few seconds: an unreadable meta, or a thread whose newest readable
    letter is the agent's own, would otherwise repeat the same line twenty times a minute.
    """
    now_mono = time.monotonic()
    last = _SKIP_LOG_AT.get(key)
    if last is not None and now_mono - last < SKIP_LOG_INTERVAL_SECONDS:
        return False
    _SKIP_LOG_AT[key] = now_mono
    log(level, message)
    return True


def read_json_checked(path: Path, retries: int = 4) -> JsonRead:
    """Read a JSON object from the container, reporting why it could not be read.

    The reason is the point. A meta.json whose *content* CloudDocs has not materialised yet
    reads as an ordinary failure, and for thirty minutes a pending letter was invisible to a
    sweep that kept cycling because nothing recorded that it had skipped it.
    """
    last_reason = "unreadable"
    for attempt in range(retries):
        if not safe_is_file(path):
            return JsonRead(None, "missing")
        try:
            content = path.read_text(encoding="utf-8")
        except OSError as exc:
            # Darwin EDEADLK (Resource deadlock avoided), EAGAIN, mid-sync EIO, permissions.
            last_reason = f"unreadable ({exc.strerror or exc})"
        except Exception as exc:
            last_reason = f"unreadable ({exc})"
        else:
            if not content.strip():
                last_reason = "empty"
            else:
                try:
                    parsed = json.loads(content)
                except json.JSONDecodeError as exc:
                    last_reason = f"invalid JSON ({exc})"
                else:
                    if isinstance(parsed, dict):
                        return JsonRead(cast(dict[str, object], parsed), "")
                    last_reason = "not a JSON object"

        if attempt < retries - 1:
            time.sleep(0.15 * (attempt + 1))
            continue
        return JsonRead(None, last_reason)
    return JsonRead(None, last_reason)


def safe_read_json(path: Path, retries: int = 4) -> Optional[dict[str, object]]:
    """Value or None. Use read_json_checked when the reason matters."""
    return read_json_checked(path, retries).data


def safe_write_text(path: Path, content: str, retries: int = 4) -> bool:
    """Write atomically: hidden temp file in the same directory, then os.replace.

    A partial write into a synced iCloud container is exactly what produces corrupt
    payloads on the iPhone and `[Errno 11] Resource deadlock avoided` on this side, so
    the rename is not optional. The temp name is dot-prefixed, and the iOS client skips
    hidden files, so it never shows up as a half-written letter.
    """
    for attempt in range(retries):
        tmp_path: Optional[Path] = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
            tmp_path = Path(tmp_name)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(content)
                handle.flush()
                try:
                    os.fsync(handle.fileno())
                except OSError:
                    pass
            os.replace(tmp_path, path)
            return True
        except OSError:
            if tmp_path is not None:
                try:
                    tmp_path.unlink(missing_ok=True)
                except OSError:
                    pass
            if attempt < retries - 1:
                time.sleep(0.15 * (attempt + 1))
                continue
            return False
        except Exception:
            if tmp_path is not None:
                try:
                    tmp_path.unlink(missing_ok=True)
                except OSError:
                    pass
            return False
    return False


def trigger_icloud_download(path: Path) -> None:
    """Request immediate download of a dataless .icloud placeholder file on macOS."""
    if sys.platform != "darwin":
        return
    escaped = str(path.resolve()).replace('"', '\\"')
    try:
        # Fast path (~30ms, no compiler/toolchain overhead): JXA Foundation
        cmd = [
            "osascript", "-l", "JavaScript", "-e",
            f'ObjC.import("Foundation"); try {{ $.NSFileManager.defaultManager.startDownloadingUbiquitousItemAtURLError($.NSURL.fileURLWithPath("{escaped}"), null); }} catch(e) {{}}'
        ]
        res = subprocess.run(cmd, capture_output=True, timeout=2)
        if res.returncode == 0:
            return
    except Exception:
        pass
    try:
        # Fallback path: swift invocation
        cmd = [
            "swift", "-e",
            f'import Foundation; try? FileManager.default.startDownloadingUbiquitousItem(at: URL(fileURLWithPath: "{escaped}"))'
        ]
        subprocess.run(cmd, capture_output=True, timeout=3)
    except Exception:
        pass


# In-memory performance cache for parsed thread metadata
_DISPATCH_THREAD_CACHE: Dict[str, Tuple[float, DispatchThreadMeta]] = {}
_DISPATCH_MESSAGE_CACHE: Dict[str, Dict[str, DispatchMessageRecord]] = {}
_LAST_SYNC_MARKER_STATE: Dict[str, Tuple[float, Optional[str]]] = {}
_LAST_SWEEP_REPORT: Dict[str, SweepReport] = {}


def reset_dispatch_cache() -> None:
    """Clear in-memory caches for dispatch threads, messages, and sync markers (used in tests)."""
    _DISPATCH_THREAD_CACHE.clear()
    _DISPATCH_MESSAGE_CACHE.clear()
    _LAST_SYNC_MARKER_STATE.clear()
    _LAST_SWEEP_REPORT.clear()


def scan_dispatch_threads(threads_dir: Path) -> ThreadScan:
    """Read every thread's meta.json, naming the ones that could not be read.

    A meta that exists but cannot be read — CloudDocs has not materialised its content, or
    Darwin hands us EDEADLK mid-sync — is a pending letter nobody can see. It is reported,
    its download is requested, and it is never silently dropped.
    """
    blank: ThreadScan = {"threads": [], "unreadable": []}
    if not safe_is_dir(threads_dir):
        return blank
    try:
        entries = list(threads_dir.iterdir())
    except (OSError, Exception):
        return blank

    results: List[DispatchThreadMeta] = []
    unreadable: List[str] = []
    needs_cooldown = False
    is_active_threads_dir = (threads_dir.name != "archive")
    for entry in entries:
        try:
            if not safe_is_dir(entry):
                continue
            meta_file = entry / "meta.json"
            icloud_meta = entry / ".meta.json.icloud"
            has_placeholder = False
            if safe_is_file(icloud_meta) and not safe_is_file(meta_file):
                trigger_icloud_download(icloud_meta)
                needs_cooldown = True
                has_placeholder = True

            cache_key = str(meta_file.resolve())
            if safe_is_file(meta_file):
                try:
                    mtime = meta_file.stat().st_mtime
                    if cache_key in _DISPATCH_THREAD_CACHE:
                        cached_mtime, cached_meta = _DISPATCH_THREAD_CACHE[cache_key]
                        if cached_mtime == mtime:
                            if is_active_threads_dir and cached_meta.get("status") == "archived":
                                archive_dir = threads_dir.parent / "archive"
                                archive_dir.mkdir(parents=True, exist_ok=True)
                                dest = archive_dir / entry.name
                                try:
                                    if dest.exists():
                                        shutil.rmtree(dest)
                                    shutil.move(str(entry), str(dest))
                                    _DISPATCH_THREAD_CACHE.pop(cache_key, None)
                                except Exception:
                                    pass
                                continue
                            results.append(cached_meta)
                            continue
                except Exception:
                    pass
            else:
                if has_placeholder and cache_key in _DISPATCH_THREAD_CACHE:
                    # Graceful cache fallback: retain thread during CloudDocs background download
                    _, cached_meta = _DISPATCH_THREAD_CACHE[cache_key]
                    results.append(cached_meta)
                    continue
                _DISPATCH_THREAD_CACHE.pop(cache_key, None)

            read = read_json_checked(meta_file)
            if read.data:
                meta_dict = cast(DispatchThreadMeta, read.data)
                if is_active_threads_dir and meta_dict.get("status") == "archived":
                    archive_dir = threads_dir.parent / "archive"
                    archive_dir.mkdir(parents=True, exist_ok=True)
                    dest = archive_dir / entry.name
                    try:
                        if dest.exists():
                            shutil.rmtree(dest)
                        shutil.move(str(entry), str(dest))
                        _DISPATCH_THREAD_CACHE.pop(cache_key, None)
                        log("INFO", f"auto-migrated archived thread {entry.name} to {dest}")
                    except Exception as exc:
                        log("WARN", f"failed to migrate archived thread {entry.name}: {exc}")
                    continue
                try:
                    _DISPATCH_THREAD_CACHE[cache_key] = (meta_file.stat().st_mtime, meta_dict)
                except Exception:
                    pass
                results.append(meta_dict)
            elif read.reason != "missing":
                if cache_key in _DISPATCH_THREAD_CACHE:
                    # Graceful cache fallback: retain thread during dataless sync
                    _, cached_meta = _DISPATCH_THREAD_CACHE[cache_key]
                    results.append(cached_meta)
                else:
                    unreadable.append(entry.name)
                if log_skip_throttled(
                    f"meta:{meta_file}",
                    "WARN",
                    f"thread '{entry.name}' meta.json is {read.reason}; asked iCloud for it and skipped this sweep",
                ):
                    trigger_icloud_download(meta_file)
        except (OSError, Exception):
            continue

    if needs_cooldown:
        # Brief slice to give macOS bird time to materialize the placeholder if download finishes quickly
        time.sleep(0.15)
        for entry in entries:
            try:
                meta_file = entry / "meta.json"
                if safe_is_file(meta_file) and not any(r.get("thread_id") == entry.name for r in results):
                    data = safe_read_json(meta_file)
                    if data:
                        meta_dict = cast(DispatchThreadMeta, data)
                        if is_active_threads_dir and meta_dict.get("status") == "archived":
                            archive_dir = threads_dir.parent / "archive"
                            archive_dir.mkdir(parents=True, exist_ok=True)
                            dest = archive_dir / entry.name
                            try:
                                if dest.exists():
                                    shutil.rmtree(dest)
                                shutil.move(str(entry), str(dest))
                                _DISPATCH_THREAD_CACHE.pop(str(meta_file.resolve()), None)
                            except Exception:
                                pass
                            continue
                        try:
                            _DISPATCH_THREAD_CACHE[str(meta_file.resolve())] = (meta_file.stat().st_mtime, meta_dict)
                        except Exception:
                            pass
                        results.append(meta_dict)
            except Exception:
                continue

    return {
        "threads": sorted(results, key=lambda t: str(t.get("updated_at", "")), reverse=True),
        "unreadable": unreadable,
    }


def list_dispatch_threads(threads_dir: Path) -> List[DispatchThreadMeta]:
    """Threads whose metadata could be read. For callers that do not track skipped ones."""
    return scan_dispatch_threads(threads_dir)["threads"]


def list_thread_messages(threads_dir: Path, thread_id: str) -> List[DispatchMessageRecord]:
    thread_folder = threads_dir / thread_id
    if not safe_is_dir(thread_folder):
        arch_folder = (threads_dir.parent / "archive") / thread_id
        if safe_is_dir(arch_folder):
            thread_folder = arch_folder
    msgs_dir = thread_folder / "messages"
    if not safe_is_dir(msgs_dir):
        return []
    try:
        has_placeholders = False
        if safe_is_dir(msgs_dir):
            for item in msgs_dir.iterdir():
                if item.name.startswith(".") and item.name.endswith(".icloud"):
                    trigger_icloud_download(item)
                    has_placeholders = True
        if has_placeholders:
            time.sleep(0.15)
        files = sorted(msgs_dir.glob("*.json"))
    except (OSError, Exception):
        return []
    seen_ids: Set[str] = set()
    found_placeholders = False
    messages: List[DispatchMessageRecord] = []
    for file in files:
        read = read_json_checked(file)
        if read.data:
            rec = cast(DispatchMessageRecord, read.data)
            msg_id = str(rec.get("id", ""))
            if msg_id:
                seen_ids.add(msg_id)
            messages.append(rec)
        elif read.reason != "missing":
            found_placeholders = True
            if log_skip_throttled(
                f"message:{file}",
                "WARN",
                f"letter file {file.name} in thread '{thread_id}' is {read.reason}; asked iCloud for it and skipped it",
            ):
                trigger_icloud_download(file)

    if thread_id in _DISPATCH_MESSAGE_CACHE and (has_placeholders or found_placeholders):
        cached_msgs = _DISPATCH_MESSAGE_CACHE[thread_id]
        for mid, mrec in cached_msgs.items():
            if mid not in seen_ids:
                seen_ids.add(mid)
                messages.append(mrec)

    if messages:
        _DISPATCH_MESSAGE_CACHE[thread_id] = {
            str(m.get("id", "")): m for m in messages if m.get("id")
        }

    return sorted(messages, key=lambda m: str(m.get("timestamp", "")))


def active_profile_name(cli_profile: Optional[str] = None) -> str:
    if cli_profile and cli_profile.strip():
        return cli_profile.strip()
    env_prof = os.environ.get("HERMES_PROFILE", "").strip()
    if env_prof:
        return env_prof
    hermes_home = os.environ.get("HERMES_HOME", "").strip()
    if hermes_home:
        path = Path(hermes_home)
        if "profiles" in path.parts:
            idx = path.parts.index("profiles")
            if idx + 1 < len(path.parts):
                return path.parts[idx + 1]
    return "default"


def discover_installed_hermes_profiles(hermes_base: Optional[Path] = None) -> List[str]:
    """Scan Hermes profiles directory for profiles with hermes-companion installed."""
    discovered: List[str] = []
    base_dirs: List[Path] = []
    if hermes_base:
        base_dirs.append(hermes_base)
    else:
        env_home = os.environ.get("HERMES_HOME", "").strip()
        if env_home:
            base_dirs.append(Path(env_home).expanduser())
        base_dirs.append(Path.home() / ".hermes")

    for base in base_dirs:
        profiles_dir = base / "profiles"
        if not profiles_dir.is_dir():
            continue
        try:
            for item in sorted(profiles_dir.iterdir(), key=lambda p: p.name):
                if item.is_dir() and not item.name.startswith("."):
                    skill_link = item / "skills" / "hermes-companion"
                    if skill_link.exists() and item.name not in discovered:
                        discovered.append(item.name)
        except Exception:
            continue
    return discovered


def register_active_profile(
    profile_id: str,
    icloud_dir: Optional[str] = None,
    now: Optional[datetime] = None,
    auto_discover: bool = False,
    hermes_base: Optional[Path] = None,
) -> None:
    directories = icloud_directories(icloud_dir)
    if not directories:
        return
    primary = directories[0]
    try:
        primary.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    profiles_file = primary / "profiles.json"

    moment = now or utcnow()
    timestamp_str = moment.strftime("%Y-%m-%dT%H:%M:%SZ")

    records: List[ProfileRecord] = []
    if profiles_file.is_file():
        try:
            data = json.loads(profiles_file.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "profiles" in data and isinstance(data["profiles"], list):
                for item in data["profiles"]:
                    if isinstance(item, dict) and "id" in item:
                        records.append(cast(ProfileRecord, item))
        except Exception:
            records = []

    display_name = "Default Agent" if profile_id == "default" else f"{profile_id.capitalize()} Agent"
    found = False
    for r in records:
        if r.get("id") == profile_id:
            r["last_active"] = timestamp_str
            if not r.get("name"):
                r["name"] = display_name
            found = True
            break
    if not found:
        records.append({
            "id": profile_id,
            "name": display_name,
            "last_active": timestamp_str,
        })

    if auto_discover:
        discovered = discover_installed_hermes_profiles(hermes_base)
        for disc_id in discovered:
            if not any(r.get("id") == disc_id for r in records):
                disc_name = "Default Agent" if disc_id == "default" else f"{disc_id.capitalize()} Agent"
                records.append({
                    "id": disc_id,
                    "name": disc_name,
                    "last_active": "2026-01-01T00:00:00Z",
                })

    records = sorted(records, key=lambda p: str(p.get("last_active", "")), reverse=True)
    payload: ProfilesConfig = {
        "updated_at": timestamp_str,
        "profiles": records,
    }
    if safe_write_text(profiles_file, json.dumps(payload, indent=2, ensure_ascii=False)):
        log("INFO", f"registered profile '{profile_id}' in {profiles_file}")
    else:
        log("WARN", f"could not write {profiles_file}")


def create_dispatch_thread(
    threads_dir: Path,
    subject: str,
    initial_body: str,
    target_profile: str = "default",
    sender: str = "agent",
    now: Optional[datetime] = None,
) -> DispatchThreadMeta:
    moment = now or utcnow()
    timestamp_str = moment.strftime("%Y-%m-%dT%H:%M:%SZ")
    date_part = moment.strftime("%Y%m%d%H%M%S")
    rand_part = uuid.uuid4().hex[:6]
    thread_id = f"th_{date_part}_{rand_part}"
    thread_folder = threads_dir / thread_id
    thread_folder.mkdir(parents=True, exist_ok=True)
    msgs_dir = thread_folder / "messages"
    msgs_dir.mkdir(parents=True, exist_ok=True)

    meta: DispatchThreadMeta = {
        "thread_id": thread_id,
        "subject": subject,
        "target_profile": target_profile,
        "status": "replied" if sender == "agent" else "pending_agent",
        "created_at": timestamp_str,
        "updated_at": timestamp_str,
        "message_count": 1,
        "last_snippet": initial_body[:120],
    }

    meta_file = thread_folder / "meta.json"
    safe_write_text(meta_file, json.dumps(meta, indent=2, ensure_ascii=False))

    msg_id = f"msg_{uuid.uuid4().hex[:8]}"
    msg_filename = f"001_{sender}_{date_part}_{msg_id}.json"
    message: DispatchMessageRecord = {
        "id": msg_id,
        "thread_id": thread_id,
        "sender": sender,
        "timestamp": timestamp_str,
        "body": initial_body,
    }
    safe_write_text(msgs_dir / msg_filename, json.dumps(message, indent=2, ensure_ascii=False))
    touch_sync_marker(threads_dir, thread_id, now=moment)

    return meta


def touch_sync_marker(
    threads_dir: Path,
    thread_id: str,
    now: Optional[datetime] = None,
) -> None:
    """Touch the root .sync_marker.json file to trigger macOS launchd WatchPaths immediately."""
    try:
        marker_file = threads_dir / ".sync_marker.json"
        moment = now or utcnow()
        data = {
            "updated_at": moment.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "thread_id": thread_id,
        }
        safe_write_text(marker_file, json.dumps(data, indent=2, ensure_ascii=False))
    except Exception:
        pass


def update_thread_retry(
    threads_dir: Path,
    thread_id: str,
    retries: int,
    error_message: str,
    now: Optional[datetime] = None,
) -> None:
    """Record a transient dispatch attempt failure without setting permanent failed status."""
    thread_folder = threads_dir / thread_id
    meta_file = thread_folder / "meta.json"
    meta = cast(Optional[DispatchThreadMeta], safe_read_json(meta_file))
    if not meta:
        return
    moment = now or utcnow()
    meta["dispatch_retries"] = retries
    meta["last_error"] = f"[Attempt {retries}/3] {error_message}"
    meta["updated_at"] = moment.strftime("%Y-%m-%dT%H:%M:%SZ")
    safe_write_text(meta_file, json.dumps(meta, indent=2, ensure_ascii=False))


def mark_thread_failed(
    threads_dir: Path,
    thread_id: str,
    error_message: str,
    now: Optional[datetime] = None,
) -> None:
    """Mark a thread as failed with an error message to prevent infinite retry loops."""
    thread_folder = threads_dir / thread_id
    meta_file = thread_folder / "meta.json"
    meta = cast(Optional[DispatchThreadMeta], safe_read_json(meta_file))
    if not meta:
        return
    moment = now or utcnow()
    timestamp_str = moment.strftime("%Y-%m-%dT%H:%M:%SZ")
    meta["status"] = "failed"
    meta["last_error"] = error_message
    meta["failed_at"] = timestamp_str
    meta["updated_at"] = timestamp_str
    safe_write_text(meta_file, json.dumps(meta, indent=2, ensure_ascii=False))
    touch_sync_marker(threads_dir, thread_id, now=moment)


def reply_to_thread(
    threads_dir: Path,
    thread_id: str,
    body: str,
    sender: str = "agent",
    now: Optional[datetime] = None,
) -> Optional[DispatchMessageRecord]:
    thread_folder = threads_dir / thread_id
    meta_file = thread_folder / "meta.json"
    meta = cast(Optional[DispatchThreadMeta], safe_read_json(meta_file))
    if not meta:
        return None

    # Turn-based status validation & optimistic concurrency guard
    current_status = meta.get("status", "")
    if sender == "agent":
        if current_status not in ("pending_agent", "failed"):
            log("WARN", f"turn-based collision guard: agent cannot reply to thread '{thread_id}' with status '{current_status}'")
            return None
    elif sender == "user":
        if current_status != "replied":
            log("WARN", f"turn-based collision guard: user cannot reply to thread '{thread_id}' with status '{current_status}'")
            return None

    moment = now or utcnow()
    timestamp_str = moment.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Determine highest existing sequence index on disk to guard against collisions
    disk_seq = 0
    msgs_dir = thread_folder / "messages"
    if safe_is_dir(msgs_dir):
        try:
            for item in msgs_dir.iterdir():
                name = item.name
                if name.startswith("."):
                    name = name[1:]
                if name.endswith(".icloud"):
                    name = name[:-7]
                if name.endswith(".json"):
                    prefix = name[:-5].split("_")[0]
                    if prefix.isdigit():
                        disk_seq = max(disk_seq, int(prefix))
        except OSError:
            pass

    seq = max(meta.get("message_count", 0), disk_seq) + 1
    msg_id = f"msg_{uuid.uuid4().hex[:8]}"
    date_part = moment.strftime("%Y%m%d%H%M%S")
    filename = f"{seq:03d}_{sender}_{date_part}_{msg_id}.json"

    message: DispatchMessageRecord = {
        "id": msg_id,
        "thread_id": thread_id,
        "sender": sender,
        "timestamp": timestamp_str,
        "body": body,
    }

    msgs_dir = thread_folder / "messages"
    try:
        msgs_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    msg_file = msgs_dir / filename
    safe_write_text(msg_file, json.dumps(message, indent=2, ensure_ascii=False))

    meta["message_count"] = seq
    meta["status"] = "replied" if sender == "agent" else "pending_agent"
    meta["updated_at"] = timestamp_str
    meta["last_snippet"] = body[:120]
    meta.pop("last_error", None)
    meta.pop("dispatch_retries", None)
    meta.pop("failed_at", None)
    safe_write_text(meta_file, json.dumps(meta, indent=2, ensure_ascii=False))
    touch_sync_marker(threads_dir, thread_id, now=moment)

    return message


def archive_dispatch_thread(
    threads_dir: Path,
    thread_id: str,
    now: Optional[datetime] = None,
) -> Optional[DispatchThreadMeta]:
    """Archive a thread: mark status as archived and move folder to Documents/archive/{id}."""
    archive_dir = threads_dir.parent / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    source = threads_dir / thread_id
    dest = archive_dir / thread_id

    if not safe_is_dir(source) and not safe_is_dir(dest):
        return None

    folder = dest if safe_is_dir(dest) else source
    meta_file = folder / "meta.json"
    meta = cast(Optional[DispatchThreadMeta], safe_read_json(meta_file))
    if not meta:
        return None

    moment = now or utcnow()
    meta["status"] = "archived"
    meta["updated_at"] = moment.strftime("%Y-%m-%dT%H:%M:%SZ")
    safe_write_text(meta_file, json.dumps(meta, indent=2, ensure_ascii=False))

    if safe_is_dir(source) and source != dest:
        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(source), str(dest))

    _DISPATCH_THREAD_CACHE.pop(str((source / "meta.json").resolve()), None)
    _DISPATCH_THREAD_CACHE.pop(str((dest / "meta.json").resolve()), None)

    touch_sync_marker(threads_dir, thread_id, now=moment)
    return meta


def unarchive_dispatch_thread(
    threads_dir: Path,
    thread_id: str,
    now: Optional[datetime] = None,
) -> Optional[DispatchThreadMeta]:
    """Unarchive a thread: mark status as replied and move folder back to Documents/threads/{id}."""
    archive_dir = threads_dir.parent / "archive"
    source = archive_dir / thread_id
    dest = threads_dir / thread_id

    if not safe_is_dir(source) and not safe_is_dir(dest):
        return None

    folder = dest if safe_is_dir(dest) else source
    meta_file = folder / "meta.json"
    meta = cast(Optional[DispatchThreadMeta], safe_read_json(meta_file))
    if not meta:
        return None

    moment = now or utcnow()
    meta["status"] = "replied"
    meta["updated_at"] = moment.strftime("%Y-%m-%dT%H:%M:%SZ")
    safe_write_text(meta_file, json.dumps(meta, indent=2, ensure_ascii=False))

    if safe_is_dir(source) and source != dest:
        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(source), str(dest))

    _DISPATCH_THREAD_CACHE.pop(str((source / "meta.json").resolve()), None)
    _DISPATCH_THREAD_CACHE.pop(str((dest / "meta.json").resolve()), None)

    touch_sync_marker(threads_dir, thread_id, now=moment)
    return meta


def command_archive_thread(args: argparse.Namespace, now: datetime) -> int:
    threads_dir = get_threads_dir(args.icloud_dir)
    if not threads_dir:
        log("ERROR", "cannot resolve iCloud threads directory")
        return 1

    meta = archive_dispatch_thread(threads_dir, args.archive, now=now)
    if not meta:
        log("ERROR", f"thread {args.archive} not found or unreadable")
        return 1

    if args.json:
        print(json.dumps(meta, indent=2, ensure_ascii=False))
        return 0

    print(f"Archived thread [{args.archive}] and moved to Documents/archive/")
    return 0


def command_unarchive_thread(args: argparse.Namespace, now: datetime) -> int:
    threads_dir = get_threads_dir(args.icloud_dir)
    if not threads_dir:
        log("ERROR", "cannot resolve iCloud threads directory")
        return 1

    meta = unarchive_dispatch_thread(threads_dir, args.unarchive, now=now)
    if not meta:
        log("ERROR", f"thread {args.unarchive} not found or unreadable")
        return 1

    if args.json:
        print(json.dumps(meta, indent=2, ensure_ascii=False))
        return 0

    print(f"Unarchived thread [{args.unarchive}] and moved to Documents/threads/")
    return 0


def delete_dispatch_thread(
    threads_dir: Path,
    thread_id: str,
    now: Optional[datetime] = None,
) -> bool:
    """Delete a thread: permanently remove folder from Documents/threads/{id} and/or Documents/archive/{id}."""
    archive_dir = threads_dir.parent / "archive"
    source = threads_dir / thread_id
    arch_source = archive_dir / thread_id

    deleted = False
    if safe_is_dir(source):
        shutil.rmtree(source, ignore_errors=True)
        deleted = True
    if safe_is_dir(arch_source):
        shutil.rmtree(arch_source, ignore_errors=True)
        deleted = True

    _DISPATCH_THREAD_CACHE.pop(str((source / "meta.json").resolve()), None)
    _DISPATCH_THREAD_CACHE.pop(str((arch_source / "meta.json").resolve()), None)
    _DISPATCH_MESSAGE_CACHE.pop(thread_id, None)

    if deleted:
        moment = now or utcnow()
        touch_sync_marker(threads_dir, thread_id, now=moment)

    return deleted


def command_delete_thread(args: argparse.Namespace, now: datetime) -> int:
    threads_dir = get_threads_dir(args.icloud_dir)
    if not threads_dir:
        log("ERROR", "cannot resolve iCloud threads directory")
        return 1

    thread_id = getattr(args, "delete_thread", None)
    if not thread_id:
        log("ERROR", "no thread ID provided to delete")
        return 1

    success = delete_dispatch_thread(threads_dir, thread_id, now=now)
    if not success:
        log("ERROR", f"thread {thread_id} not found")
        return 1

    if getattr(args, "json", False):
        print(json.dumps({"deleted": True, "thread_id": thread_id}, indent=2))
        return 0

    print(f"Deleted thread [{thread_id}] and removed from disk")
    return 0


def command_threads(args: argparse.Namespace) -> int:
    threads_dir = get_threads_dir(args.icloud_dir)
    if not threads_dir or not threads_dir.is_dir():
        log("WARN", "threads directory is not in the iCloud container yet")
        if args.json:
            print("[]")
        else:
            print("No dispatch threads found in iCloud container.")
        return 0

    if getattr(args, "archived", False):
        archive_dir = threads_dir.parent / "archive"
        threads = list_dispatch_threads(archive_dir) if safe_is_dir(archive_dir) else []
    else:
        threads = list_dispatch_threads(threads_dir)

    if args.pending:
        threads = [t for t in threads if t.get("status") == "pending_agent"]
    if args.profile:
        target = args.profile.strip().lower()
        threads = [
            t for t in threads
            if str(t.get("target_profile", "default")).lower() in (target, "all")
        ]

    if args.json:
        print(json.dumps(threads, indent=2, ensure_ascii=False))
        return 0

    if not threads:
        print("No dispatch threads found.")
        return 0

    print("facts_only: reply in the user's language; do not quote this block")
    print(f"thread_count: {len(threads)}")
    print("threads:")
    for t in threads:
        status_label = "Awaiting" if t.get("status") == "pending_agent" else str(t.get("status", "unknown"))
        count = t.get("message_count", 0)
        memo_str = "1 memo" if count == 1 else f"{count} memos"
        target_prof = t.get("target_profile", "default")
        print(f"• [{t.get('thread_id', '?')}] {t.get('subject', 'Untitled')} [to: {target_prof}] ({status_label}, {memo_str})")
        if t.get("last_snippet"):
            print(f"  snippet: {t['last_snippet']}")
        print(f"  updated_at: {t.get('updated_at', 'unknown')}")
    return 0


def command_thread_detail(args: argparse.Namespace) -> int:
    threads_dir = get_threads_dir(args.icloud_dir)
    if not threads_dir:
        log("ERROR", "cannot resolve iCloud threads directory")
        return 1

    thread_id = args.thread
    thread_folder = threads_dir / thread_id
    if not safe_is_dir(thread_folder):
        arch_folder = (threads_dir.parent / "archive") / thread_id
        if safe_is_dir(arch_folder):
            thread_folder = arch_folder
    meta_file = thread_folder / "meta.json"
    if not safe_is_file(meta_file):
        icloud_meta = thread_folder / ".meta.json.icloud"
        if safe_is_file(icloud_meta):
            trigger_icloud_download(icloud_meta)
            time.sleep(0.2)

    meta = cast(Optional[DispatchThreadMeta], safe_read_json(meta_file))
    if not meta:
        log("ERROR", f"thread {thread_id} not found or unreadable")
        return 1

    messages = list_thread_messages(threads_dir, thread_id)
    if args.json:
        payload = {"meta": meta, "messages": messages}
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    print("facts_only: reply in the user's language; do not quote this block")
    print(f"thread_id: {meta.get('thread_id')}")
    print(f"subject: {meta.get('subject')}")
    print(f"target_profile: {meta.get('target_profile', 'default')}")
    print(f"status: {meta.get('status')}")
    print(f"created_at: {meta.get('created_at')}")
    print(f"updated_at: {meta.get('updated_at')}")
    print(f"message_count: {len(messages)}")
    print("chronology:")
    for m in messages:
        sender_label = "User" if m.get("sender") == "user" else "Hermes Agent"
        print(f"\n--- [{m.get('timestamp')}] {sender_label} ({m.get('id')}) ---")
        print(m.get("body", ""))
    return 0


def command_reply(args: argparse.Namespace, now: datetime) -> int:
    if not args.message:
        log("ERROR", "--reply requires --message")
        return 2

    threads_dir = get_threads_dir(args.icloud_dir)
    if not threads_dir:
        log("ERROR", "cannot resolve iCloud threads directory")
        return 1

    msg = reply_to_thread(threads_dir, args.reply, args.message, sender="agent", now=now)
    if not msg:
        log("ERROR", f"failed to append reply to thread {args.reply}")
        return 1

    if args.json:
        print(json.dumps(msg, indent=2, ensure_ascii=False))
        return 0

    print(f"Replied to thread [{args.reply}] ({msg['id']}) at {msg['timestamp']}")
    return 0


def command_new_thread(args: argparse.Namespace, now: datetime) -> int:
    if not args.subject or not args.message:
        log("ERROR", "--new-thread requires --subject and --message")
        return 2

    threads_dir = get_threads_dir(args.icloud_dir)
    if not threads_dir:
        log("ERROR", "cannot resolve iCloud threads directory")
        return 1

    target_profile = (args.to_profile or args.profile or "default").strip()
    thread = create_dispatch_thread(
        threads_dir,
        args.subject,
        args.message,
        target_profile=target_profile,
        sender="agent",
        now=now,
    )
    if args.json:
        print(json.dumps(thread, indent=2, ensure_ascii=False))
        return 0

    print(f"Created thread [{thread['thread_id']}] '{thread['subject']}' [to: {target_profile}]")
    return 0


# MARK: - Autonomous Background Dispatch & LaunchAgent Service

LAUNCH_AGENT_LABEL = "ai.hermes.companion-inbox"
DAEMON_DEFAULT_INTERVAL: Final[float] = 3.0
DAEMON_MIN_INTERVAL: Final[float] = 0.1
DAEMON_HEARTBEAT_SECONDS: Final[float] = 300.0
_STOP_REQUESTED: bool = False
_WARNED_PATHS: set[str] = set()
_SKIP_LOG_AT: dict[str, float] = {}
SKIP_LOG_INTERVAL_SECONDS: Final[float] = 300.0


def get_dispatch_lock_path() -> Path:
    lock_dir = Path.home() / ".hermes"
    try:
        lock_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        lock_dir = Path(tempfile.gettempdir())
    return lock_dir / "companion_dispatch.lock"


def acquire_dispatch_lock(lock_file: Path, stale_seconds: int = 180) -> bool:
    """Acquire local exclusion lock with stale process and age detection."""
    try:
        if lock_file.exists():
            try:
                mtime = lock_file.stat().st_mtime
                age = time.time() - mtime
                pid_str = lock_file.read_text(encoding="utf-8").strip()
                pid = int(pid_str) if pid_str.isdigit() else None
                is_alive = False
                if pid and pid > 0:
                    try:
                        os.kill(pid, 0)
                        is_alive = True
                    except OSError:
                        is_alive = False
                if not is_alive or age > stale_seconds:
                    lock_file.unlink(missing_ok=True)
                else:
                    return False
            except Exception:
                lock_file.unlink(missing_ok=True)

        fd = os.open(str(lock_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode("utf-8"))
        os.close(fd)
        return True
    except (FileExistsError, OSError):
        return False


def release_dispatch_lock(lock_file: Path) -> None:
    try:
        if lock_file.exists():
            lock_file.unlink(missing_ok=True)
    except Exception:
        pass


def find_hermes_executable() -> Optional[str]:
    """Locate the hermes CLI executable across PATH, virtualenvs, homebrew, and Mac app bundles."""
    candidates = [
        shutil.which("hermes"),
        str(Path.home() / ".local" / "bin" / "hermes"),
        str(Path.home() / ".hermes" / "hermes-agent" / ".hermes" / "bin" / "hermes"),
        str(Path.home() / ".hermes" / "bin" / "hermes"),
        str(Path.home() / ".cargo" / "bin" / "hermes"),
        "/opt/homebrew/bin/hermes",
        "/usr/local/bin/hermes",
        "/Applications/Hermes.app/Contents/MacOS/hermes",
        "/Applications/Hermes.app/Contents/Resources/hermes",
        "/Applications/Hermes Agent.app/Contents/MacOS/hermes",
        str(Path.home() / "Applications" / "Hermes.app" / "Contents" / "MacOS" / "hermes"),
        str(Path.home() / "Applications" / "Hermes.app" / "Contents" / "Resources" / "hermes"),
    ]
    for c in candidates:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


def get_launch_agent_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{LAUNCH_AGENT_LABEL}.plist"


def render_launch_agent_plist(
    python_bin: str,
    script_path: str,
    threads_dir: Optional[Path],
    home: str,
    interval: float = DAEMON_DEFAULT_INTERVAL,
) -> str:
    """Render the LaunchAgent XML. Pure: same inputs, same bytes.

    KeepAlive plus a resident --daemon replaces the transient StartInterval sweep. A job
    that exits in milliseconds, every 15s, is what launchd throttles; a job that never
    exits is never throttled. WatchPaths still kicks the agent when the container changes,
    and PYTHONUNBUFFERED keeps a dying process from taking its last log lines with it.
    """
    watch_xml = ""
    if threads_dir:
        watch_xml = f"""
    <key>WatchPaths</key>
    <array>
        <string>{threads_dir}</string>
    </array>"""

    system_path = (
        f"{home}/.local/bin:{home}/.cargo/bin:{home}/.hermes/bin:{home}/.hermes/hermes-agent/.hermes/bin:"
        "/Applications/Hermes.app/Contents/MacOS:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
    )

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{LAUNCH_AGENT_LABEL}</string>
    <key>ProgramArguments</key>
    <array>
        <string>{python_bin}</string>
        <string>{script_path}</string>
        <string>--daemon</string>
        <string>--interval</string>
        <string>{interval:g}</string>
    </array>
    <key>KeepAlive</key>
    <true/>
    <key>ThrottleInterval</key>
    <integer>10</integer>{watch_xml}
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>{system_path}</string>
        <key>HERMES_HOME</key>
        <string>{home}/.hermes</string>
        <key>PYTHONUNBUFFERED</key>
        <string>1</string>
    </dict>
    <key>RunAtLoad</key>
    <true/>
    <key>StandardOutPath</key>
    <string>{home}/.hermes/logs/companion-inbox.log</string>
    <key>StandardErrorPath</key>
    <string>{home}/.hermes/logs/companion-inbox.error.log</string>
</dict>
</plist>"""


def dispatch_python_bin() -> str:
    """Interpreter to pin in the LaunchAgent. Deliberately not plain sys.executable.

    companion.py is executed by whatever interpreter its caller happens to run under:
    a Hermes profile agent runs it with the Hermes venv Python, a profile on the local
    model with its own, and the plist was rendered with `sys.executable`. Each caller
    therefore wrote itself into the plist and relaunched the job, so the agent flapped
    (four daemon generations in a minute, each killed by the next caller's launchctl).
    The interpreter this machine already uses for the agent wins, whatever calls us.
    """
    override = os.environ.get("HERMES_COMPANION_PYTHON", "").strip()
    if override and os.access(override, os.X_OK):
        return override

    uv_root = Path.home() / ".local" / "share" / "uv" / "python"
    for pattern in ("cpython-3.12-*/bin/python3", "cpython-3.*/bin/python3"):
        for candidate in sorted(uv_root.glob(pattern), reverse=True):
            if os.access(candidate, os.X_OK):
                return str(candidate)

    for fallback in (shutil.which("python3") or "", sys.executable or "", "/usr/bin/python3"):
        if fallback and os.access(fallback, os.X_OK):
            return fallback
    return "/usr/bin/python3"


def is_daemon_plist(text: str) -> bool:
    """True when a plist already describes the resident daemon. Pure."""
    return "--daemon" in text and "<key>KeepAlive</key>" in text and "<key>StartInterval</key>" not in text


def install_launch_agent(icloud_dir: Optional[str] = None) -> bool:
    """Install and (re)load the dispatch LaunchAgent."""
    if sys.platform != "darwin":
        log("WARN", "LaunchAgent requires macOS; cannot install on non-macOS system")
        return False

    hermes_bin = find_hermes_executable()
    if not hermes_bin:
        log("WARN", "hermes executable not found in PATH or standard locations; auto-dispatch of pending letters will fail until hermes is installed")

    plist_path = get_launch_agent_path()
    plist_path.parent.mkdir(parents=True, exist_ok=True)

    plist_content = render_launch_agent_plist(
        dispatch_python_bin(),
        str(Path(__file__).resolve()),
        get_threads_dir(icloud_dir),
        str(Path.home()),
    )

    try:
        domain = f"gui/{os.getuid()}"
        subprocess.run(["launchctl", "bootout", f"{domain}/{LAUNCH_AGENT_LABEL}"], capture_output=True)
        subprocess.run(["launchctl", "unload", str(plist_path)], capture_output=True)
        plist_path.write_text(plist_content, encoding="utf-8")
        res = subprocess.run(["launchctl", "bootstrap", domain, str(plist_path)], capture_output=True, text=True)
        if res.returncode != 0:
            res = subprocess.run(["launchctl", "load", str(plist_path)], capture_output=True, text=True)
        log("INFO", f"installed background dispatch daemon: {plist_path} (rc={res.returncode})")
        return res.returncode == 0
    except Exception as exc:
        log("WARN", f"could not install LaunchAgent: {exc}")
        return False


def check_environment_diagnostics(icloud_dir: Optional[str] = None) -> List[Tuple[str, str, str, str]]:
    """Perform pre-flight diagnostics across OS, iCloud, permissions, and Hermes executable.

    Returns a list of tuples: (status, category, message, remedy)
    status: 'PASS', 'WARN', 'FAIL', 'INFO'
    """
    results: List[Tuple[str, str, str, str]] = []

    # 1. Operating System & Container Check
    is_darwin = sys.platform == "darwin"
    is_container = (
        Path("/.dockerenv").exists()
        or os.environ.get("CONTAINER") in ("docker", "podman")
        or os.environ.get("DOCKER_CONTAINER") == "1"
        or Path("/run/.containerenv").exists()
    )
    if not is_darwin or is_container:
        os_desc = "Container (Docker/Podman)" if is_container else sys.platform
        results.append((
            "FAIL",
            "Operating System",
            f"{os_desc} is not supported. Hermes Companion strictly requires a native macOS host.",
            "Run Hermes directly on macOS under your user account. Containers and Linux/Windows cannot access native iCloud Drive synchronization (~/Library/Mobile Documents)."
        ))
    else:
        mac_ver = platform.mac_ver()[0] or platform.release()
        results.append((
            "PASS",
            "Operating System",
            f"macOS {mac_ver} (native host)",
            ""
        ))

    # 2. iCloud Drive Root Directory
    mobile_docs = Path.home() / "Library" / "Mobile Documents"
    if not is_darwin:
        pass
    elif not mobile_docs.is_dir():
        results.append((
            "FAIL",
            "iCloud Drive",
            f"Directory '{mobile_docs}' not found.",
            "Sign in to your Apple Account and enable iCloud Drive in System Settings -> Apple Account -> iCloud."
        ))
    else:
        results.append((
            "PASS",
            "iCloud Drive",
            f"Active at {mobile_docs}",
            ""
        ))

    # 3. macOS Privacy / Files and Folders Permissions (TCC)
    if is_darwin and mobile_docs.is_dir():
        try:
            _ = list(mobile_docs.iterdir())
            results.append((
                "PASS",
                "Privacy Permissions",
                "Files and Folders access to iCloud Drive verified.",
                ""
            ))
        except (PermissionError, OSError) as exc:
            results.append((
                "FAIL",
                "Privacy Permissions",
                f"Permission denied accessing iCloud Drive: {exc}",
                "Grant Terminal or Python access in System Settings -> Privacy & Security -> Files and Folders (or Full Disk Access)."
            ))

    # 4. Hermes Companion App Container & Files
    dirs = icloud_directories(icloud_dir)
    found_container: Optional[Path] = None
    for d in dirs:
        if d.is_dir():
            found_container = d
            break

    if found_container:
        can_write = False
        test_probe = found_container / ".probe_tmp"
        try:
            test_probe.write_text("probe", encoding="utf-8")
            test_probe.unlink(missing_ok=True)
            can_write = True
        except OSError:
            can_write = False

        status = "PASS" if can_write else "FAIL"
        msg = f"Found at {found_container} (write access: {'yes' if can_write else 'no'})"
        remedy = "" if can_write else f"Ensure write permissions on {found_container}"
        results.append((status, "App Container", msg, remedy))

        loc_file = found_container / "latest_location.json"
        health_file = found_container / "latest_health.json"
        threads_dir = found_container / "threads"
        places_file = found_container / "places.json"
        marker_file = found_container / ".sync_marker.json"

        inventory: List[str] = []
        if loc_file.is_file():
            inventory.append("location")
        if health_file.is_file():
            inventory.append("health")
        if threads_dir.is_dir():
            inventory.append(f"threads ({len(list(threads_dir.iterdir()))})")
        if places_file.is_file():
            inventory.append("places")
        if marker_file.is_file():
            inventory.append("sync_marker")

        inv_msg = ", ".join(inventory) if inventory else "empty (awaiting first sync)"
        results.append(("INFO", "Container Files", f"Synced files: {inv_msg}", ""))
    else:
        results.append((
            "WARN",
            "App Container",
            "iCloud~com~hermes~HermesCompanion container directory not found yet.",
            "Apple's CloudDocs only creates this folder after the iPhone app writes its first record. Open the Hermes Companion iOS app, allow permissions, and wait 1–2 minutes for iCloud sync."
        ))

    # 5. Hermes Executable Discovery
    hermes_bin = find_hermes_executable()
    if hermes_bin:
        results.append((
            "PASS",
            "Hermes CLI",
            f"Executable located at {hermes_bin}",
            ""
        ))
    else:
        results.append((
            "FAIL",
            "Hermes CLI",
            "'hermes' executable not found in PATH or standard application bundles.",
            "Install Hermes CLI (brew install / pipx / uv) or ensure its binary directory is added to PATH. If using the Hermes macOS app, symlink its binary (e.g. /Applications/Hermes.app/Contents/MacOS/hermes) to /opt/homebrew/bin/hermes or ~/.local/bin/hermes."
        ))

    # 6. LaunchAgent Background Inbox Service
    if is_darwin:
        plist_path = get_launch_agent_path()
        if plist_path.is_file():
            res = subprocess.run(["launchctl", "list", LAUNCH_AGENT_LABEL], capture_output=True, text=True)
            if res.returncode == 0:
                results.append((
                    "PASS",
                    "Background Service",
                    f"LaunchAgent '{LAUNCH_AGENT_LABEL}' is active.",
                    ""
                ))
            else:
                results.append((
                    "WARN",
                    "Background Service",
                    "LaunchAgent plist exists but is not running in launchctl.",
                    "Run 'companion.py --install-service' to reload the background daemon."
                ))
        else:
            results.append((
                "INFO",
                "Background Service",
                "Not installed (run 'companion.py --install-service' for automatic background letter replies).",
                ""
            ))

    return results


def command_doctor(args: argparse.Namespace) -> int:
    """Print system diagnostics report and return 0 if all required checks pass."""
    results = check_environment_diagnostics(args.icloud_dir)
    print("Hermes Companion Pre-Flight Diagnostics:")
    print("=" * 60)
    has_fail = False
    for status, category, msg, remedy in results:
        badge = f"[{status}]"
        print(f"{badge:<7} {category:<22} {msg}")
        if remedy:
            print(f"        -> Remedy: {remedy}")
        if status == "FAIL":
            has_fail = True
    print("=" * 60)
    if has_fail:
        print("Diagnostic summary: Required dependencies or permissions are missing.")
        return 1
    print("Diagnostic summary: All critical checks passed.")
    return 0


def uninstall_launch_agent() -> bool:
    """Unload and remove macOS LaunchAgent."""
    plist_path = get_launch_agent_path()
    if not plist_path.is_file():
        return True
    try:
        subprocess.run(["launchctl", "unload", str(plist_path)], capture_output=True)
        plist_path.unlink()
        log("INFO", f"uninstalled background service: {plist_path}")
        return True
    except Exception as exc:
        log("WARN", f"could not uninstall LaunchAgent: {exc}")
        return False


def ensure_launch_agent_installed(icloud_dir: Optional[str] = None) -> None:
    """Install the LaunchAgent, or upgrade a transient one, then leave it alone.

    Every companion command from every profile lands here, so this must be a no-op once
    the resident daemon is installed: rewriting the plist means unloading the job, and a
    caller that rewrites it on every run keeps killing the agent it just started.
    """
    if sys.platform != "darwin":
        return
    plist_path = get_launch_agent_path()
    try:
        current = plist_path.read_text(encoding="utf-8")
    except OSError:
        current = ""
    if is_daemon_plist(current):
        return
    install_launch_agent(icloud_dir)


def clean_hermes_chat_output(raw: str) -> str:
    """Filter out CLI banners, session markers, and reasoning blocks to extract pure LLM response."""
    # 1. Strip closed reasoning blocks like <think>...</think> or <|thought...|>
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    text = re.sub(r"<\|thought.*?\|>", "", text, flags=re.DOTALL)

    # 2. If unclosed <|thought... remains, strip the thought preamble
    if "<|thought" in text:
        # Check if thought block ends before a response
        match = re.search(r"\n\n([A-Z¿¡\"'].*)", text, flags=re.DOTALL)
        if match:
            text = match.group(1).strip()
        else:
            text = re.sub(r"<\|thought.*", "", text, flags=re.DOTALL).strip()

    lines = text.strip().splitlines()
    cleaned: List[str] = []
    in_banner = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("Warning:") or stripped.startswith("⚠️"):
            in_banner = True
            continue
        if in_banner:
            if "for custom." in stripped or stripped.startswith("'"):
                continue
            in_banner = False
        if stripped.startswith("Session ") or stripped.startswith("↻ Resumed session"):
            continue
        if stripped.startswith("session_id:"):
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip()


def place_city(place: Optional[Mapping[str, object]]) -> Optional[str]:
    """City of a saved place, when the user's places file names one. Pure."""
    if not place:
        return None
    for key in PLACE_CITY_KEYS:
        value = place.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def text_field(record: Mapping[str, object], key: str) -> Optional[str]:
    """Trimmed string field, or None. Pure."""
    value = record.get(key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def resolve_city_and_area(
    raw: Mapping[str, object],
    location: Mapping[str, object],
    places: Sequence[PlaceRecord],
    history_records: Sequence[Mapping[str, object]],
) -> Tuple[Optional[str], Optional[str]]:
    """City and street for the live fix, with a last-known fallback. Pure.

    The iPhone only reverse-geocodes some fixes, so the newest reading often carries no
    placemark at all. Without a fallback the agent is handed a place name and no city,
    and it fills the blank from its own imagination (a letter sent from Stuttgart was
    answered as if Daniel were in Paris). Order: live fix placemark, the city saved with
    the matched place, then the newest placemark anywhere in the local history.
    """
    area = text_field(location, "placemark_name") or text_field(location, "placemark_thoroughfare")
    city = text_field(raw, "placemark_locality")

    if not city:
        city = place_city(match_place(_to_float(raw.get("latitude")), _to_float(raw.get("longitude")), places))

    if not city or not area:
        for record in reversed(list(history_records)):
            record_city = text_field(record, "placemark_locality")
            record_area = text_field(record, "placemark_name") or text_field(record, "placemark_thoroughfare")
            if not city and record_city:
                city = record_city
            if not area and record_area:
                area = record_area
            if city and area:
                break

    return city, area


def get_physical_context_summary(
    icloud_dir: Optional[str] = None,
    places_file: Optional[Path] = None,
    now: Optional[datetime] = None,
) -> str:
    """Compile a concise, factual physical telemetry briefing to inform the Hermes agent."""
    moment = now or utcnow()
    directories = icloud_directories(icloud_dir)
    loc_raw, loc_path = first_reading(directories, "latest_location.json")
    health_raw, health_path = first_reading(directories, "latest_health.json")

    parts: List[str] = []
    if loc_raw and loc_path:
        places_path = places_file or default_places_file()
        places = load_places(places_path)
        loc = resolve_location(loc_raw, places, moment, str(loc_path))

        loc_lines = [
            f"Place: {loc.get('place_name', 'Unknown')}",
            f"Coordinates: {loc.get('latitude')}, {loc.get('longitude')}",
        ]
        history_records, _ = read_history(directories)
        city, area = resolve_city_and_area(loc_raw, loc, places, history_records)
        if city:
            loc_lines.append(f"City: {city}")
        if area and area != city:
            loc_lines.append(f"Area/Street: {area}")
        if loc.get("motion_activity") and loc["motion_activity"] != "unknown":
            loc_lines.append(f"Motion: {loc['motion_activity']}")
        if loc.get("still_there") is True:
            loc_lines.append("Still at this place: Yes")

        if history_records:
            timeline = build_timeline(history_records, places, moment)
            events = timeline.get("events", [])
            if events and events[-1].get("is_current") and events[-1].get("type") == "stay":
                stay = cast(TimelineStayEvent, events[-1])
                if stay.get("dwell_human"):
                    loc_lines.append(f"Dwell time: {stay['dwell_human']}")

        parts.append("Current Location & Movement:\n" + "\n".join(f"  • {l}" for l in loc_lines))

    if health_raw and health_path:
        health = resolve_health(health_raw, moment, str(health_path))
        health_lines: List[str] = []
        if health.get("sleep_hours") is not None:
            health_lines.append(f"Last sleep: {health['sleep_hours']:.1f} hours ({health.get('sleep_quality', '')})")
        if health.get("step_count") is not None:
            health_lines.append(f"Steps today: {health['step_count']:,}")
        if health.get("last_workout_activity"):
            health_lines.append(f"Recent workout: {health['last_workout_activity']} ({health.get('last_workout_minutes', 0)} min)")
        if health_lines:
            parts.append("Today's Health & Activity:\n" + "\n".join(f"  • {l}" for l in health_lines))

    return "\n\n".join(parts)


def dispatch_pending_once(
    args: argparse.Namespace,
    now: datetime,
    threads_dir: Path,
    force: bool = False,
) -> SweepReport:
    """One sweep: answer every letter waiting for the agent. Reports what it did.

    Every phase is guarded: a dataless CloudDocs placeholder, an unreadable meta, a corrupt
    payload, a busy session or a timed-out agent must cost that one thread, never the sweep.
    A sweep that saw a pending letter and wrote nothing says why, at least once every
    SKIP_LOG_INTERVAL_SECONDS.
    """
    zero: SweepReport = {"replies": 0, "pending": 0, "unreadable": 0}
    try:
        marker_key = str(threads_dir.resolve())
        marker_file = threads_dir / ".sync_marker.json"
        marker_state: Optional[Tuple[float, Optional[str]]] = None
        if safe_is_file(marker_file):
            try:
                st = marker_file.stat()
                m_updated = None
                m_data = safe_read_json(marker_file)
                if isinstance(m_data, dict):
                    m_updated = m_data.get("updated_at")
                marker_state = (st.st_mtime, m_updated)
            except Exception:
                marker_state = None

        last_marker = _LAST_SYNC_MARKER_STATE.get(marker_key)
        last_report = _LAST_SWEEP_REPORT.get(marker_key)

        if not force and marker_state is not None and last_marker is not None and marker_state == last_marker:
            if last_report and last_report.get("pending", 0) == 0 and last_report.get("unreadable", 0) == 0:
                return zero

        scan = scan_dispatch_threads(threads_dir)
        pending = [t for t in scan["threads"] if t.get("status") == "pending_agent"]
        report: SweepReport = {"replies": 0, "pending": len(pending), "unreadable": len(scan["unreadable"])}
        if marker_state is not None:
            _LAST_SYNC_MARKER_STATE[marker_key] = marker_state
        _LAST_SWEEP_REPORT[marker_key] = report

        if not pending:
            # Zero cost / zero tokens / zero CPU
            return report

        hermes_bin = find_hermes_executable()
        if not hermes_bin:
            log("WARN", "hermes executable not found, cannot auto-dispatch pending letters")
            return report

        lock_file = get_dispatch_lock_path()
        if not acquire_dispatch_lock(lock_file):
            log_skip_throttled("lock-busy", "INFO", f"another run holds {lock_file}; skipping this sweep")
            return report

        try:
            for thread in pending:
                thread_id = thread["thread_id"]
                target_profile = (thread.get("target_profile") or "default").strip()
                subject = thread.get("subject", "Letter")

                messages = list_thread_messages(threads_dir, thread_id)
                if not messages:
                    log_skip_throttled(
                        f"nomsg:{thread_id}",
                        "WARN",
                        f"thread '{thread_id}' is pending but no readable letter was found in it",
                    )
                    continue

                last_msg = messages[-1]
                if last_msg.get("sender") != "user":
                    log_skip_throttled(
                        f"agentlast:{thread_id}",
                        "INFO",
                        f"thread '{thread_id}' is pending but its newest readable letter is the agent's; waiting for the rest to sync",
                    )
                    continue

                # Check if this is the initial letter or a follow-up ("repregunta")
                is_initial_turn = len(messages) <= 1
                user_body = last_msg.get("body", "").strip()

                # Build physical telemetry briefing so the agent has Daniel's live context
                telemetry_briefing = get_physical_context_summary(args.icloud_dir, getattr(args, "places", None), now=now)
                context_section = f"\nDaniel's live physical telemetry (from Hermes Companion):\n{telemetry_briefing}\n" if telemetry_briefing else ""

                if is_initial_turn:
                    prompt = (
                        f"You have received a new asynchronous letter in Hermes Post from Daniel.\n"
                        f"Subject: {subject}\n\n"
                        f"Letter content:\n{user_body}\n"
                        f"{context_section}\n"
                        f"Instruction:\n"
                        f"Reply thoughtfully to Daniel's letter in your own natural persona and voice. "
                        f"Reply in the language Daniel used (Spanish if the letter is in Spanish). "
                        f"You have access to Daniel's live physical telemetry above—use it naturally if relevant to his letter (e.g. location, place, city, weather, physical state). "
                        f"Write ONLY the exact text of your letter reply without meta preamble or quotes."
                    )
                else:
                    prompt = (
                        f"Daniel replied to the letter thread (Subject: '{subject}'):\n\n"
                        f"{user_body}\n"
                        f"{context_section}\n"
                        f"Instruction:\n"
                        f"Continue this thread conversation in your natural voice. "
                        f"Reply in the language Daniel used (Spanish if the letter is in Spanish). "
                        f"You have access to Daniel's live physical telemetry above—use it naturally if relevant. "
                        f"Write ONLY the exact text of your reply letter."
                    )

                # Isolate each thread into its own dedicated, persistent Hermes session
                session_title = f"post_{thread_id}"
                cmd = [hermes_bin]
                if target_profile and target_profile != "default":
                    cmd.extend(["-p", target_profile])
                cmd.extend(["chat", "-c", session_title, "--create-if-missing", "-Q", "-q", prompt])

                log("INFO", f"auto-replying to thread '{thread_id}' with profile '{target_profile}' in session '{session_title}'...")
                started = time.monotonic()
                letter_age = human_age(age_seconds(str(last_msg.get("timestamp") or ""), now))
                try:
                    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                    cleaned_reply = clean_hermes_chat_output(proc.stdout)

                    # Check for session lock contention (e.g. session open in GUI or CLI)
                    if proc.returncode != 0 and ("SESSION_NOT_OWNED" in proc.stderr or "SESSION_NOT_OWNED" in proc.stdout):
                        log("WARN", f"session '{session_title}' is locked/busy; falling back to isolated one-shot execution...")
                        fallback_cmd = [hermes_bin]
                        if target_profile and target_profile != "default":
                            fallback_cmd.extend(["-p", target_profile])
                        fallback_cmd.extend(["chat", "--oneshot", "-Q", "-q", prompt])
                        fallback_proc = subprocess.run(fallback_cmd, capture_output=True, text=True, timeout=120)
                        cleaned_fallback = clean_hermes_chat_output(fallback_proc.stdout)
                        if fallback_proc.returncode == 0 and cleaned_fallback:
                            reply_to_thread(threads_dir, thread_id, cleaned_fallback, sender="agent", now=now)
                            log("INFO", f"successfully auto-replied to thread '{thread_id}' via fallback session")
                            report["replies"] += 1
                            continue

                    if proc.returncode == 0 and cleaned_reply:
                        reply_to_thread(threads_dir, thread_id, cleaned_reply, sender="agent", now=now)
                        log("INFO", f"answered thread '{thread_id}' in {time.monotonic() - started:.1f}s (letter was {letter_age} old when the sweep started)")
                        report["replies"] += 1
                    else:
                        err = proc.stderr.strip() or f"exit code {proc.returncode}"
                        log("ERROR", f"hermes exit {proc.returncode} for thread '{thread_id}'; stderr tail: {err[-500:]}")
                        retries = int(thread.get("dispatch_retries", 0)) + 1
                        if retries < 3:
                            log("WARN", f"hermes execution attempt {retries}/3 failed for thread '{thread_id}': {err}; will retry")
                            update_thread_retry(threads_dir, thread_id, retries, err[:140], now=now)
                        else:
                            log("ERROR", f"hermes execution failed after 3 attempts for thread '{thread_id}': {err}")
                            mark_thread_failed(threads_dir, thread_id, f"Agent error: {err[:140]}", now=now)
                except subprocess.TimeoutExpired:
                    log("WARN", f"hermes execution timed out for thread '{thread_id}' after 120s")
                    retries = int(thread.get("dispatch_retries", 0)) + 1
                    if retries < 3:
                        update_thread_retry(threads_dir, thread_id, retries, "Timeout after 120s", now=now)
                    else:
                        mark_thread_failed(threads_dir, thread_id, "Hermes execution timed out after 120s", now=now)
                except Exception as exc:
                    log("WARN", f"hermes execution error for thread '{thread_id}': {exc}")
                    retries = int(thread.get("dispatch_retries", 0)) + 1
                    if retries < 3:
                        update_thread_retry(threads_dir, thread_id, retries, str(exc)[:140], now=now)
                    else:
                        mark_thread_failed(threads_dir, thread_id, f"Execution error: {str(exc)[:140]}", now=now)

            new_marker_state = None
            if safe_is_file(marker_file):
                try:
                    st = marker_file.stat()
                    m_updated = None
                    m_data = safe_read_json(marker_file)
                    if isinstance(m_data, dict):
                        m_updated = m_data.get("updated_at")
                    new_marker_state = (st.st_mtime, m_updated)
                except Exception:
                    pass
            if new_marker_state is not None:
                _LAST_SYNC_MARKER_STATE[marker_key] = new_marker_state
            remaining_pending = max(0, len(pending) - report["replies"])
            _LAST_SWEEP_REPORT[marker_key] = {
                "replies": report["replies"],
                "pending": remaining_pending,
                "unreadable": len(scan["unreadable"]),
            }
            return report
        finally:
            release_dispatch_lock(lock_file)
    except Exception as exc:
        log("ERROR", f"unexpected error during dispatch sweep: {exc}\n{traceback.format_exc()}")
        return zero


def command_dispatch_pending(args: argparse.Namespace, now: datetime) -> int:
    """Transient one-shot sweep, for a manual run or a cron entry.

    Always exits 0: a reply count is not an error code, and launchd/cron read a nonzero
    exit as a failed job.
    """
    try:
        threads_dir = get_threads_dir(args.icloud_dir)
        if not threads_dir:
            return 0
        report = dispatch_pending_once(args, now, threads_dir, force=True)
        if report["replies"]:
            log("INFO", f"one-shot sweep wrote {report['replies']} agent reply(ies)")
        if report["unreadable"]:
            log("WARN", f"one-shot sweep could not read {report['unreadable']} thread meta(s)")
        return 0
    except Exception as exc:
        log("ERROR", f"unexpected error during command_dispatch_pending: {exc}\n{traceback.format_exc()}")
        return 0


def command_daemon(args: argparse.Namespace) -> int:
    """Long-running dispatch loop for the KeepAlive LaunchAgent. Returns the exit code.

    A sweep that finds nothing costs one directory listing, so the loop is idle-cheap,
    and because the process never exits launchd has nothing to throttle: a letter that
    lands in the container is picked up on the next tick instead of minutes later.
    SIGTERM unwinds in well under the sweep interval (launchd's exit timeout is 5s).
    """
    interval = max(DAEMON_MIN_INTERVAL, float(getattr(args, "interval", DAEMON_DEFAULT_INTERVAL) or DAEMON_DEFAULT_INTERVAL))
    max_cycles: Optional[int] = getattr(args, "max_cycles", None)
    for handled in (signal.SIGTERM, signal.SIGINT):
        signal.signal(handled, request_daemon_stop)

    log("INFO", f"dispatch daemon online (pid {os.getpid()}, interval {interval:g}s, script {os.path.abspath(__file__)})")
    cycles = 0
    last_heartbeat = time.monotonic()
    last_report: SweepReport = {"replies": 0, "pending": 0, "unreadable": 0}
    while not daemon_stop_requested():
        cycles += 1
        began = time.monotonic()
        try:
            threads_dir = get_threads_dir(args.icloud_dir)
            if threads_dir:
                report = dispatch_pending_once(args, utcnow(), threads_dir)
                last_report = report
                if report["replies"]:
                    log("INFO", f"cycle {cycles}: wrote {report['replies']} agent reply(ies)")
        except Exception as exc:
            log("ERROR", f"cycle {cycles} failed: {exc}\n{traceback.format_exc()}")

        moment = time.monotonic()
        if moment - last_heartbeat >= DAEMON_HEARTBEAT_SECONDS:
            last_heartbeat = moment
            unreadable_note = f", {last_report['unreadable']} unreadable" if last_report["unreadable"] else ""
            pending_note = f", {last_report['pending']} pending" if last_report["pending"] else ""
            log("INFO", f"heartbeat: {cycles} cycles, last sweep took {moment - began:.2f}s{unreadable_note}{pending_note}")

        if max_cycles is not None and cycles >= max_cycles:
            log("INFO", f"daemon reached --max-cycles {max_cycles}; exiting")
            break
        sleep_until_deadline(began + interval)

    log("INFO", f"dispatch daemon stopped after {cycles} cycles")
    return 0


def request_daemon_stop(signum: int, frame: object) -> None:
    """SIGTERM/SIGINT handler: ask the loop to finish the current sweep and exit."""
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    log("INFO", f"daemon received signal {signum}; unwinding")


def daemon_stop_requested() -> bool:
    return _STOP_REQUESTED


def sleep_until_deadline(deadline: float, step: float = 0.25) -> None:
    """Sleep in small slices so a signal is honoured promptly."""
    while not daemon_stop_requested():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(step, remaining))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read Hermes Companion location and health from iCloud")
    parser.add_argument("--json", action="store_true", help="Print the raw facts as JSON")
    parser.add_argument("--timeline", action="store_true", help="Print chronological timeline of visits, stays, and movements")
    parser.add_argument("--history", action="store_true", help="Alias for --timeline")
    parser.add_argument("--since", help="Filter history events: 'today', '24h', '7d', '60m', or ISO-8601 UTC")
    parser.add_argument("--sql", help="Run an in-memory SQL query against history records")
    parser.add_argument("--health", action="store_true", help="Print the Apple Health snapshot")
    parser.add_argument("--context", action="store_true", help="Print place and health together")
    parser.add_argument("--list", action="store_true", help="List known places")
    parser.add_argument("--add", action="store_true", help="Save a known place")
    parser.add_argument("--remove", metavar="ID", help="Remove a place by id or name")
    parser.add_argument("--threads", "--inbox", action="store_true", dest="threads", help="List Hermes Post inbox letters and threads")
    parser.add_argument("--pending", action="store_true", help="Filter threads awaiting agent reply")
    parser.add_argument("--thread", metavar="ID", help="Read full messages in a thread")
    parser.add_argument("--reply", metavar="ID", help="Post an agent reply to a thread")
    parser.add_argument("--archive", metavar="ID", help="Archive a dispatch thread and move to Documents/archive/")
    parser.add_argument("--unarchive", metavar="ID", help="Unarchive a dispatch thread and move to Documents/threads/")
    parser.add_argument("--archived", action="store_true", help="List archived dispatch threads from Documents/archive/")
    parser.add_argument("--delete-thread", "--delete", metavar="ID", dest="delete_thread", help="Permanently delete a dispatch thread from Documents/threads/ and Documents/archive/")
    parser.add_argument("--new-thread", "--new-post", action="store_true", dest="new_thread", help="Create a new post/thread")
    parser.add_argument("--subject", help="Subject for new dispatch thread")
    parser.add_argument("--message", help="Message body for reply or new thread")
    parser.add_argument("--profile", help="Active Hermes agent profile name (e.g. 'default', 'work', 'personal')")
    parser.add_argument("--to-profile", help="Target agent profile for new dispatch thread (defaults to 'default')")
    parser.add_argument("--register-profile", metavar="NAME", help="Register a specific agent profile in profiles.json")
    parser.add_argument("--discover-profiles", action="store_true", help="Discover and print all installed Hermes profiles")
    parser.add_argument("--dispatch-pending", action="store_true", help="Scan pending letters and generate autonomous replies using Hermes (one sweep)")
    parser.add_argument("--daemon", action="store_true", help="Run the long-lived dispatch daemon (launchd KeepAlive service)")
    parser.add_argument("--interval", type=float, default=DAEMON_DEFAULT_INTERVAL, help="Seconds between daemon sweeps (default 5)")
    parser.add_argument("--max-cycles", type=int, help="Stop the daemon after N sweeps (used by tests)")
    parser.add_argument("--doctor", action="store_true", help="Run system and environment pre-flight diagnostics")
    parser.add_argument("--install-service", action="store_true", help="Install macOS background LaunchAgent auto-responder")
    parser.add_argument("--uninstall-service", action="store_true", help="Uninstall macOS background LaunchAgent auto-responder")
    parser.add_argument("--service-status", action="store_true", help="Check status of background LaunchAgent auto-responder")
    parser.add_argument("--places", type=Path, help="Places JSON file")
    parser.add_argument("--icloud-dir", help="Directory that contains the latest_*.json files")
    parser.add_argument("--name")
    parser.add_argument("--category", default="general")
    parser.add_argument("--activity")
    parser.add_argument("--lat", type=float)
    parser.add_argument("--lon", type=float)
    parser.add_argument("--radius", type=float, default=150.0)
    parser.add_argument("--notes", default="")
    parser.add_argument("--serve", action="store_true", help="Start the Direct Connection HTTP daemon (Linux, Windows, macOS)")
    parser.add_argument("--pair", action="store_true", help="Display pairing URL, auth token, and QR code for iOS Direct Connection")
    parser.add_argument("--host", default="0.0.0.0", help="Host address to bind HTTP daemon (default 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8787, help="Port to listen on for HTTP daemon (default 8787)")
    parser.add_argument("--token", help="Bearer token for HTTP daemon (generated if missing)")
    parser.add_argument("--storage-dir", type=Path, help="Storage directory for HTTP daemon telemetry")
    return parser


def main(argv: Optional[Sequence[str]] = None, now: Optional[datetime] = None) -> int:
    if argv is None:
        argv_list = list(sys.argv[1:])
    else:
        argv_list = list(argv)

    if argv_list and argv_list[0] in ("serve", "pair"):
        argv_list[0] = f"--{argv_list[0]}"

    args = build_parser().parse_args(argv_list)

    if getattr(args, "serve", False):
        try:
            from companion_server import run_server
        except ImportError:
            from server.companion_server import run_server
        run_server(host=args.host, port=args.port, token=args.token, storage_dir=args.storage_dir)
        return 0

    if getattr(args, "pair", False):
        try:
            from companion_server import run_pairing_assistant
        except ImportError:
            from server.companion_server import run_pairing_assistant
        run_pairing_assistant(port=args.port, token=args.token)
        return 0
    if args.places is None:
        args.places = default_places_file()
    else:
        args.places = expand(str(args.places))
    moment = now or utcnow()

    if getattr(args, "doctor", False):
        return command_doctor(args)

    if args.install_service:
        if sys.platform != "darwin":
            print("Error: Background LaunchAgent requires macOS. Non-macOS systems (Linux, Docker, Windows) cannot run launchd.", file=sys.stderr)
            return 1
        hermes_bin = find_hermes_executable()
        if not hermes_bin:
            print("Warning: 'hermes' CLI executable was not found in PATH or standard application paths.", file=sys.stderr)
            print("Auto-dispatch of pending letters will fail until hermes is installed and discoverable.", file=sys.stderr)
            print("Run 'companion.py --doctor' to inspect your environment.", file=sys.stderr)
        ok = install_launch_agent(args.icloud_dir)
        print("LaunchAgent service installed and loaded" if ok else "Failed to install service")
        return 0 if ok else 1
    if args.uninstall_service:
        ok = uninstall_launch_agent()
        print("LaunchAgent service uninstalled" if ok else "Failed to uninstall service")
        return 0 if ok else 1
    if args.service_status:
        path = get_launch_agent_path()
        exists = path.is_file()
        print(f"Service plist: {path} (installed: {exists})")
        if exists:
            subprocess.run(["launchctl", "list", LAUNCH_AGENT_LABEL])
        return 0
    if args.dispatch_pending:
        return command_dispatch_pending(args, moment)
    if args.daemon:
        return command_daemon(args)

    # Automatically register active profile and discover installed profiles in iCloud profiles.json
    active_profile = active_profile_name(args.profile)
    if args.register_profile:
        register_active_profile(args.register_profile.strip(), args.icloud_dir, now=moment, auto_discover=True)
        print(f"Registered profile '{args.register_profile.strip()}' in profiles.json")
        return 0

    register_active_profile(active_profile, args.icloud_dir, now=moment, auto_discover=True)

    # Ensure background service is installed on macOS automatically (1-minute latency, 0 idle tokens)
    ensure_launch_agent_installed(args.icloud_dir)

    if args.discover_profiles:
        profs = discover_installed_hermes_profiles()
        print(f"Discovered Hermes profiles with companion skill: {', '.join(profs) if profs else 'none'}")
        return 0

    if args.threads:
        return command_threads(args)
    if args.archive:
        return command_archive_thread(args, moment)
    if args.unarchive:
        return command_unarchive_thread(args, moment)
    if getattr(args, "delete_thread", None):
        return command_delete_thread(args, moment)
    if args.thread:
        return command_thread_detail(args)
    if args.reply:
        return command_reply(args, moment)
    if args.new_thread:
        return command_new_thread(args, moment)
    if args.add:
        return command_add(args)
    if args.remove:
        return command_remove(args)
    if args.list:
        return command_list(args)
    if args.timeline or args.history or args.sql:
        return command_timeline(args, moment)
    if args.context:
        return command_context(args, moment)
    if args.health:
        return command_health(args, moment)
    return command_where(args, moment)


if __name__ == "__main__":
    sys.exit(main())
