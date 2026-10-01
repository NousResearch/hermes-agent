#!/usr/bin/env python3
"""Stdlib checks for the Hermes Companion skill reader. No network, no iCloud."""

from __future__ import annotations

import json
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import companion  # noqa: E402


NOW = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)
HOME = {
    "id": "home",
    "name": "Home",
    "category": "home",
    "activity": "at home",
    "latitude": 52.53,
    "longitude": 13.41,
    "radius_meters": 120,
}


def iso(delta_seconds: int) -> str:
    return (NOW - timedelta(seconds=delta_seconds)).strftime("%Y-%m-%dT%H:%M:%SZ")


def location(**overrides):
    raw = {
        "latitude": 52.53005,
        "longitude": 13.41005,
        "timestamp": iso(3 * 3600),
        "horizontal_accuracy": 8,
        "speed_mps": -1,
        "movement_reason": "moved",
        "motion_activity": "stationary",
        "motion_confidence": "high",
        "motion_timestamp": iso(20),
        "battery_level": 0.8,
        "battery_state": "unplugged",
    }
    raw.update(overrides)
    return companion.resolve_location(raw, [HOME], NOW, "fixture")


def check(name: str, cond: bool) -> None:
    if not cond:
        raise SystemExit(f"FAIL {name}")
    print(f"ok {name}")


def test_still_home_after_hours() -> None:
    loc = location()
    check("long stay is still home", loc["place_name"] == "Home" and loc["still_there"] is True)
    check("age is three hours", loc["minutes_since_last_move"] == 180)
    check("stationary is not moving", loc["is_moving_now"] is False and loc["motion_fresh"] is True)
    check("reason preserved", loc["movement_reason"] == "moved")


def test_walking_at_home() -> None:
    loc = location(motion_activity="walking", motion_timestamp=iso(12))
    check("walking stays at home", loc["place_name"] == "Home" and loc["is_moving_now"] is True)
    check("walking code kept", loc["motion_activity"] == "walking" and loc["activity"] == "walking")


def test_driving_is_transit() -> None:
    loc = location(motion_activity="automotive", motion_timestamp=iso(8), speed_mps=12)
    check("automotive leaves the place label", loc["place_name"] == "In Transit" and loc["place_category"] == "transit")


def test_unlisted() -> None:
    loc = location(latitude=48.1, longitude=11.5, motion_activity="stationary")
    check("unknown coordinate stays unlisted", loc["place_name"] == "Unlisted place" and loc["is_at_known_place"] is False)


def test_files_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "latest_location.json").write_text(
            json.dumps(
                {
                    "latitude": 52.53005,
                    "longitude": 13.41005,
                    "timestamp": iso(600),
                    "motion_activity": "stationary",
                    "motion_timestamp": iso(5),
                    "movement_reason": "no_motion_reading",
                    "battery_level": 0.5,
                }
            ),
            encoding="utf-8",
        )
        (root / "latest_health.json").write_text(
            json.dumps(
                {
                    "timestamp": iso(120),
                    "recovery_status": "recovered",
                    "step_count_today": 4200,
                    "sleep": {"formatted_duration": "7h 40m", "quality_rating": "good"},
                    "workout": None,
                }
            ),
            encoding="utf-8",
        )
        places = root / "places.json"
        code = companion.main(
            ["--icloud-dir", str(root), "--places", str(places), "--add", "--name", "Home", "--category", "home", "--lat", "52.53", "--lon", "13.41", "--radius", "120"],
            now=NOW,
        )
        check("add place exits 0", code == 0)
        code = companion.main(["--icloud-dir", str(root), "--places", str(places)], now=NOW)
        check("where exits 0", code == 0)
        raw, path = companion.first_reading([root], "latest_location.json")
        resolved = companion.resolve_location(raw, companion.load_places(places), NOW, str(path))
        check("file resolves to Home", resolved["place_name"] == "Home")
        check("no_motion_reading kept", resolved["movement_reason"] == "no_motion_reading")
        health_raw, health_path = companion.first_reading([root], "latest_health.json")
        health = companion.resolve_health(health_raw, NOW, str(health_path))
        check("sleep duration kept", health["sleep"]["formatted_duration"] == "7h 40m")


def test_battery_is_not_reported() -> None:
    loc = location()
    text = companion.format_location(loc)
    blob = json.dumps(loc)
    check("script omits battery", "battery" not in text and "battery" not in blob)


def test_text_has_no_english_script() -> None:
    loc = location(motion_activity="walking", motion_timestamp=iso(12))
    text = companion.format_location(loc)
    check("location text is facts", text.startswith("facts_only:") and "place_name: Home" in text)
    check("location text is not a sentence", "At Home" not in text and "context_summary" not in text)
    raw = {
        "timestamp": iso(60),
        "recovery_status": "fatigued",
        "sleep": {"formatted_duration": "7h", "quality_rating": "good", "summary": "Slept wonderfully"},
        "conversational_context": {
            "sleep_insight": "wonderful night",
            "nutrition_reminder": "30-40g of protein",
        },
        "suggested_openers": ["Time for your post-workout protein"],
    }
    health = companion.resolve_health(raw, NOW, "fixture")
    blob = json.dumps(health)
    health_text = companion.format_health(health)
    check("health payload drops prose", "wonderful" not in blob and "protein" not in blob)
    check("health text keeps the code", "recovery_status: fatigued" in health_text and "protein" not in health_text)


def main() -> int:
    test_still_home_after_hours()
    test_walking_at_home()
    test_driving_is_transit()
    test_unlisted()
    test_files_roundtrip()
    test_text_has_no_english_script()
    test_battery_is_not_reported()
    print("ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
