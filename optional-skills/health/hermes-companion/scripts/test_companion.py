#!/usr/bin/env python3
"""Stdlib checks for the Hermes Companion skill reader. No network, no iCloud."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Final, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parent))
import companion  # noqa: E402
from companion import PlaceRecord, ResolvedLocation

NOW: Final[datetime] = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)
HOME: Final[PlaceRecord] = {
    "id": "home",
    "name": "Home",
    "category": "home",
    "activity": "at home",
    "latitude": 52.53,
    "longitude": 13.41,
    "radius_meters": 120.0,
}


def iso(delta_seconds: int) -> str:
    return (NOW - timedelta(seconds=delta_seconds)).strftime("%Y-%m-%dT%H:%M:%SZ")


def location(**overrides: object) -> ResolvedLocation:
    raw: dict[str, object] = {
        "latitude": 52.53005,
        "longitude": 13.41005,
        "timestamp": iso(3 * 3600),
        "horizontal_accuracy": 8.0,
        "movement_reason": "moved",
        "motion_activity": "stationary",
        "motion_confidence": "high",
        "motion_timestamp": iso(20),
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
    check("moving walking is not still there", loc["still_there"] is False and loc["still_there_status"] == "no")


def test_driving_is_transit() -> None:
    loc = location(motion_activity="automotive", motion_timestamp=iso(8))
    check("automotive leaves the place label", loc["place_name"] == "In Transit" and loc["place_category"] == "transit")
    check("driving in transit is not still there", loc["still_there"] is False and loc["still_there_status"] == "no")


def test_unlisted() -> None:
    loc = location(latitude=48.1, longitude=11.5, motion_activity="stationary")
    check("unknown coordinate stays unlisted", loc["place_name"] == "Unlisted place" and loc["is_at_known_place"] is False)
    check("unlisted stationary with fresh motion is still there", loc["still_there"] is True and loc["still_there_status"] == "yes")


def test_unlisted_stale_unconfirmed() -> None:
    # Unlisted place with stale fix and no fresh motion
    loc = location(latitude=48.1, longitude=11.5, motion_activity="stationary", motion_timestamp=iso(3 * 3600))
    check("unlisted stale fix without fresh motion is unconfirmed", loc["still_there"] is False and "unconfirmed" in loc["still_there_status"])


def test_profile_places_isolation() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        fresh_home = Path(tmp) / "profiles" / "fresh_user"
        old_env = os.environ.get("HERMES_HOME")
        os.environ["HERMES_HOME"] = str(fresh_home)
        try:
            target = companion.default_places_file()
            expected = fresh_home / "state" / "places.json"
            check("default_places_file isolates fresh profile", target == expected)
        finally:
            if old_env is None:
                os.environ.pop("HERMES_HOME", None)
            else:
                os.environ["HERMES_HOME"] = old_env


def test_invalid_timestamp_graceful() -> None:
    check("corrupt timestamp returns None", companion.age_seconds("corrupt-date", NOW) is None)
    check("missing timestamp returns None", companion.age_seconds(None, NOW) is None)
    loc = location(timestamp="not-a-date")
    check("corrupt timestamp does not fake 0 age", loc["age_seconds"] is None)
    check("corrupt timestamp does not fake 0 minutes", loc["minutes_since_last_move"] is None)
    check("corrupt timestamp marks unconfirmed", loc["still_there"] is False and "unconfirmed" in loc["still_there_status"])


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
        assert raw is not None and path is not None
        resolved = companion.resolve_location(raw, companion.load_places(places), NOW, str(path))
        check("file resolves to Home", resolved["place_name"] == "Home")
        check("no_motion_reading kept", resolved["movement_reason"] == "no_motion_reading")
        health_raw, health_path = companion.first_reading([root], "latest_health.json")
        assert health_raw is not None and health_path is not None
        health = companion.resolve_health(health_raw, NOW, str(health_path))
        assert health["sleep"] is not None
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


def test_timeline_and_history() -> None:
    gym: PlaceRecord = {
        "id": "gym",
        "name": "Gym",
        "category": "gym",
        "activity": "working out",
        "latitude": 52.52,
        "longitude": 13.40,
        "radius_meters": 150.0,
    }
    places = [HOME, gym]

    records: list[dict[str, object]] = [
        {"timestamp": iso(4 * 3600), "latitude": 52.53, "longitude": 13.41, "motion_activity": "stationary"},
        {"timestamp": iso(3 * 3600), "latitude": 52.53, "longitude": 13.41, "motion_activity": "stationary"},
        {"timestamp": iso(2 * 3600), "latitude": 52.525, "longitude": 13.405, "motion_activity": "walking"},
        {"timestamp": iso(3600), "latitude": 52.52, "longitude": 13.40, "motion_activity": "stationary"},
        {"timestamp": iso(1800), "latitude": 52.52, "longitude": 13.40, "motion_activity": "stationary"},
        {"timestamp": iso(60), "latitude": 52.53, "longitude": 13.41, "motion_activity": "stationary"},
    ]

    timeline = companion.build_timeline(records, places, NOW)
    check("timeline has events", timeline["events_count"] >= 3)
    check("current place is Home", timeline["current_place"] == "Home")

    rows = companion.query_history_sqlite(records, places, "SELECT COUNT(*) as c FROM locations WHERE place_name = 'Gym'")
    check("sqlite found gym records", rows[0]["c"] == 2)

    text = companion.format_timeline(timeline)
    check("timeline formatting has facts", "facts_only:" in text and "stay: Gym" in text)


def test_bug_md_scenario_gap_detection() -> None:
    places: list[PlaceRecord] = [
        {"id": "home", "name": "Home", "category": "home", "latitude": 48.815047, "longitude": 9.232482, "radius_meters": 120.0},
        {"id": "cannstatt", "name": "Cannstatt", "category": "general", "latitude": 48.812884, "longitude": 9.221555, "radius_meters": 200.0},
    ]

    records: list[dict[str, object]] = [
        {"timestamp": "2026-10-02T16:48:25Z", "latitude": 48.815159, "longitude": 9.231775, "motion_activity": "stationary"},
        {"timestamp": "2026-10-02T18:29:06Z", "latitude": 48.812884, "longitude": 9.221555, "motion_activity": "walking"},
    ]
    now_moment = companion.parse_utc("2026-10-02T18:40:00Z")
    assert now_moment is not None
    timeline = companion.build_timeline(records, places, now_moment)

    cannstatt_event = timeline["events"][-1]
    check("cannstatt reached", cannstatt_event.get("place_name") == "Cannstatt")
    check("gap detected before cannstatt", "silent for 100m" in str(timeline) or "silent for 101m" in str(timeline) or "silent" in str(cannstatt_event.get("telemetry_gap_before")))

    fix_at_1801 = {
        "latitude": 48.815159,
        "longitude": 9.231775,
        "timestamp": "2026-10-02T16:48:25Z",
        "motion_activity": "stationary",
        "motion_timestamp": "2026-10-02T16:48:25Z",
    }
    resolved = companion.resolve_location(fix_at_1801, places, companion.parse_utc("2026-10-02T18:01:00Z") or NOW, "fixture")
    check("still_there is unconfirmed when old without fresh motion", resolved["still_there"] is False and "unconfirmed" in resolved["still_there_status"])


def test_apple_maps_placemark_resolution() -> None:
    places = [companion.PlaceRecord(id="home", name="Home", category="home", latitude=52.53, longitude=13.41, radius_meters=150.0)]
    
    # 1. Unlisted coordinate with Apple Maps placemark fallback
    raw_with_pm = {
        "latitude": 48.80196,
        "longitude": 9.22026,
        "timestamp": "2026-10-05T09:30:00Z",
        "motion_activity": "stationary",
        "motion_timestamp": "2026-10-05T09:30:00Z",
        "placemark_name": "Fitness First Cannstatt",
        "placemark_locality": "Stuttgart",
        "movement_reason": "moved",
    }
    resolved = companion.resolve_location(raw_with_pm, places, companion.parse_utc("2026-10-05T09:40:00Z") or NOW, "fixture")
    check("placemark name used when unlisted", resolved["place_name"] == "Fitness First Cannstatt, Stuttgart")
    check("placemark category is apple_maps", resolved["place_category"] == "apple_maps")
    formatted = companion.format_location(resolved)
    check("formatted text includes apple_maps_placemark", "apple_maps_placemark: Fitness First Cannstatt" in formatted)

    # 2. Known place overrides Apple Maps placemark
    raw_at_home = {
        "latitude": 52.53001,
        "longitude": 13.41001,
        "timestamp": "2026-10-05T12:00:00Z",
        "motion_activity": "stationary",
        "placemark_name": "Chausseestraße 42",
        "placemark_locality": "Berlin",
    }
    resolved_home = companion.resolve_location(raw_at_home, places, companion.parse_utc("2026-10-05T12:05:00Z") or NOW, "fixture")
    check("known place overrides placemark", resolved_home["place_name"] == "Home")
    check("known place category kept", resolved_home["place_category"] == "home")

    # 3. Timeline event uses placemark
    records = [
        {
            "latitude": 48.80196,
            "longitude": 9.22026,
            "timestamp": "2026-10-05T09:30:00Z",
            "motion_activity": "stationary",
            "placemark_name": "Fitness First Cannstatt",
            "placemark_locality": "Stuttgart",
        }
    ]
    tl = companion.build_timeline(records, places, companion.parse_utc("2026-10-05T10:00:00Z") or NOW)
    events = tl.get("events", [])
    check("timeline uses placemark", len(events) > 0 and events[0].get("place_name") == "Fitness First Cannstatt, Stuttgart")


def test_dispatch_threads_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        threads_dir = Path(tmp_dir) / "threads"
        threads_dir.mkdir(parents=True, exist_ok=True)

        # 1. Create a thread
        created = companion.create_dispatch_thread(
            threads_dir=threads_dir,
            subject="Weekly Mobility Briefing",
            initial_body="Can you summarize my running workouts?",
            sender="user",
            now=NOW,
        )
        check("thread created with pending_agent status", created["status"] == "pending_agent")
        check("thread created with 1 message", created["message_count"] == 1)

        # 2. List threads
        threads = companion.list_dispatch_threads(threads_dir)
        check("listed 1 thread", len(threads) == 1)
        check("subject matches", threads[0]["subject"] == "Weekly Mobility Briefing")

        # 3. Read messages
        msgs = companion.list_thread_messages(threads_dir, created["thread_id"])
        check("1 message in thread", len(msgs) == 1)
        check("user sender preserved", msgs[0]["sender"] == "user")
        check("message body matches", msgs[0]["body"] == "Can you summarize my running workouts?")

        # 4. Agent replies
        reply = companion.reply_to_thread(
            threads_dir=threads_dir,
            thread_id=created["thread_id"],
            body="You completed 3 running sessions totaling 18 km.",
            sender="agent",
            now=NOW + timedelta(minutes=5),
        )
        check("reply generated", reply is not None)
        check("reply sender is agent", reply is not None and reply["sender"] == "agent")

        # 5. Verify updated thread metadata
        threads_after = companion.list_dispatch_threads(threads_dir)
        check("thread status transitioned to replied", threads_after[0]["status"] == "replied")
        check("thread count incremented to 2", threads_after[0]["message_count"] == 2)
        check("snippet updated to reply", threads_after[0]["last_snippet"] == "You completed 3 running sessions totaling 18 km.")

        # 6. Verify chronological messages
        all_msgs = companion.list_thread_messages(threads_dir, created["thread_id"])
        check("chronology has 2 messages", len(all_msgs) == 2)
        check("first message is user", all_msgs[0]["sender"] == "user")
        check("second message is agent", all_msgs[1]["sender"] == "agent")


def test_profiles_registration_and_filtering() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        # 1. Register profiles
        companion.register_active_profile("work", icloud_dir=tmp_dir, now=NOW)
        companion.register_active_profile("fitness", icloud_dir=tmp_dir, now=NOW + timedelta(minutes=10))

        prof_file = Path(tmp_dir) / "profiles.json"
        check("profiles.json created", prof_file.is_file())
        config = json.loads(prof_file.read_text(encoding="utf-8"))
        profs = config.get("profiles", [])
        check("two profiles registered", len(profs) == 2)
        check("fitness is newest active", profs[0]["id"] == "fitness")
        check("work is second active", profs[1]["id"] == "work")

        # 2. Create threads targeted to different profiles
        threads_dir = Path(tmp_dir) / "threads"
        threads_dir.mkdir(parents=True, exist_ok=True)
        t_work = companion.create_dispatch_thread(
            threads_dir=threads_dir,
            subject="Quarterly Planning Review",
            initial_body="Review Q4 deliverables",
            target_profile="work",
            sender="user",
            now=NOW,
        )
        t_fit = companion.create_dispatch_thread(
            threads_dir=threads_dir,
            subject="Marathon Training Plan",
            initial_body="Adjust zone 2 miles",
            target_profile="fitness",
            sender="user",
            now=NOW,
        )
        check("work thread has work profile target", t_work.get("target_profile") == "work")
        check("fitness thread has fitness profile target", t_fit.get("target_profile") == "fitness")

        # 3. Test list filtering by profile
        all_threads = companion.list_dispatch_threads(threads_dir)
        check("total 2 threads created", len(all_threads) == 2)
        fitness_filtered = [t for t in all_threads if t.get("target_profile") == "fitness"]
        check("fitness filter yields 1 thread", len(fitness_filtered) == 1 and fitness_filtered[0]["subject"] == "Marathon Training Plan")


def test_discover_installed_hermes_profiles() -> None:
    with tempfile.TemporaryDirectory() as hermes_dir, tempfile.TemporaryDirectory() as icloud_dir:
        hermes_base = Path(hermes_dir)
        # Setup mock profiles
        (hermes_base / "profiles" / "love" / "skills" / "hermes-companion").mkdir(parents=True, exist_ok=True)
        (hermes_base / "profiles" / "fitness" / "skills" / "hermes-companion").mkdir(parents=True, exist_ok=True)
        (hermes_base / "profiles" / "other" / "skills").mkdir(parents=True, exist_ok=True)

        discovered = companion.discover_installed_hermes_profiles(hermes_base)
        check("discovered love and fitness", sorted(discovered) == ["fitness", "love"])

        companion.register_active_profile("love", icloud_dir=icloud_dir, now=NOW, auto_discover=True, hermes_base=hermes_base)
        prof_file = Path(icloud_dir) / "profiles.json"
        check("profiles.json created with discovery", prof_file.is_file())
        config = json.loads(prof_file.read_text(encoding="utf-8"))
        profs = config.get("profiles", [])
        prof_ids = [p["id"] for p in profs]
        check("love registered as active", "love" in prof_ids and profs[0]["id"] == "love")
        check("fitness discovered and added", "fitness" in prof_ids)


def test_clean_hermes_chat_output() -> None:
    raw = (
        "Warning: model foo is deprecated\n"
        "⚠️ Normalized model to gpt-4o\n"
        "↻ Resumed session post_th_123 (6 messages)\n"
        "session_id: 9a8b-123\n"
        "<think>pondering the memo...</think>\n"
        "Hola mi amor, te extraño mucho.\n"
        "Espero que tengas un buen dia."
    )
    cleaned = companion.clean_hermes_chat_output(raw)
    expected = "Hola mi amor, te extraño mucho.\nEspero que tengas un buen dia."
    check("cleaned chat output matches", cleaned == expected)


def test_error_handling_and_sync_marker() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        threads_dir = tmp_path / "threads"
        threads_dir.mkdir(parents=True, exist_ok=True)

        moment = datetime(2026, 10, 9, 14, 0, 0, tzinfo=timezone.utc)
        thread = companion.create_dispatch_thread(
            threads_dir,
            "Test Error",
            "Initial prompt",
            target_profile="default",
            sender="user",
            now=moment,
        )
        tid = thread["thread_id"]

        marker_file = threads_dir / ".sync_marker.json"
        check("sync marker created on thread create", marker_file.is_file())

        # Test retry attempts tracking without marking status failed
        companion.update_thread_retry(threads_dir, tid, 1, "Gateway busy", now=moment)
        meta_retry1 = companion.safe_read_json(threads_dir / tid / "meta.json")
        check("thread retry 1 recorded", meta_retry1 is not None and meta_retry1.get("dispatch_retries") == 1)
        check("thread retry 1 retains status pending_agent", meta_retry1 is not None and meta_retry1.get("status") == "pending_agent")

        companion.mark_thread_failed(threads_dir, tid, "Hermes timeout 120s", now=moment)
        meta_after = companion.safe_read_json(threads_dir / tid / "meta.json")
        check("thread marked failed", meta_after is not None and meta_after.get("status") == "failed")
        check("thread error recorded", meta_after is not None and meta_after.get("last_error") == "Hermes timeout 120s")

        # Now reply to it and ensure error is cleared and status is replied
        companion.reply_to_thread(threads_dir, tid, "Agent finally replied", sender="agent", now=moment)
        meta_replied = companion.safe_read_json(threads_dir / tid / "meta.json")
        check("thread status transitioned to replied", meta_replied is not None and meta_replied.get("status") == "replied")
        check("last_error cleared", meta_replied is not None and "last_error" not in meta_replied)
        check("dispatch_retries cleared after reply", meta_replied is not None and "dispatch_retries" not in meta_replied)

        # Test safe_is_file resilience
        check("safe_is_file on non-existent is False", not companion.safe_is_file(tmp_path / "does_not_exist.json"))
        check("safe_is_file on directory is False", not companion.safe_is_file(threads_dir))

        # Test local exclusion lock acquisition and release
        lock_file = tmp_path / "test.lock"
        check("acquire lock succeeds", companion.acquire_dispatch_lock(lock_file))
        check("acquire lock again fails", not companion.acquire_dispatch_lock(lock_file))
        companion.release_dispatch_lock(lock_file)
        companion.release_dispatch_lock(lock_file)


def test_physical_context_summary() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        loc_payload = {
            "latitude": 48.815,
            "longitude": 9.232,
            "timestamp": "2026-10-11T07:00:00Z",
            "placemark_name": "Zuckerleweg 50",
            "placemark_locality": "Stuttgart",
            "motion_activity": "stationary",
        }
        (tmp_path / "latest_location.json").write_text(json.dumps(loc_payload), encoding="utf-8")
        places_file = tmp_path / "places.json"
        places_file.write_text(json.dumps([HOME]), encoding="utf-8")

        summary = companion.get_physical_context_summary(icloud_dir=str(tmp_path), places_file=places_file, now=NOW)
        check("summary contains coordinates", "48.815" in summary and "9.232" in summary)
        check("summary contains placemark", "Zuckerleweg 50" in summary and "Stuttgart" in summary)


def test_atomic_write_creates_no_debris() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        target = Path(tmpdir) / "nested" / "meta.json"
        check("atomic write succeeds in a missing directory", companion.safe_write_text(target, '{"a": 1}'))
        check("atomic write wrote the payload", json.loads(target.read_text(encoding="utf-8"))["a"] == 1)
        leftovers = sorted(p.name for p in target.parent.iterdir() if p.name != "meta.json")
        check("atomic write leaves no temp file behind", leftovers == [])


def test_corrupt_payload_is_skipped_once() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        payload = Path(tmpdir) / "meta.json"
        payload.write_text("{not json", encoding="utf-8")
        check("corrupt payload reads as None", companion.safe_read_json(payload) is None)
        check("corrupt payload names the reason", "invalid JSON" in companion.read_json_checked(payload).reason)
        check("missing payload names the reason", companion.read_json_checked(Path(tmpdir) / "nope.json").reason == "missing")


def test_city_fallback_without_placemark() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        (tmp_path / "latest_location.json").write_text(json.dumps({
            "latitude": 48.814892907035208,
            "longitude": 9.2325277521644118,
            "timestamp": "2026-09-30T07:30:09Z",
            "motion_activity": "stationary",
        }), encoding="utf-8")
        (tmp_path / "location_history.json").write_text(json.dumps({
            "records": [
                {
                    "latitude": 48.80413631967493,
                    "longitude": 9.208474413971375,
                    "timestamp": "2026-09-29T10:51:12Z",
                    "placemark_name": "Neckartalstraße",
                    "placemark_locality": "Stuttgart",
                },
            ],
        }), encoding="utf-8")
        places_path = tmp_path / "places.json"
        places_path.write_text(json.dumps([]), encoding="utf-8")

        summary = companion.get_physical_context_summary(icloud_dir=str(tmp_path), places_file=places_path, now=NOW)
        check("city falls back to the last known locality", "City: Stuttgart" in summary)
        check("area falls back to the last known street", "Area/Street: Neckartalstraße" in summary)


def test_city_from_saved_place() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        (tmp_path / "latest_location.json").write_text(json.dumps({
            "latitude": 48.815047,
            "longitude": 9.232482,
            "timestamp": "2026-09-30T07:30:09Z",
            "motion_activity": "stationary",
        }), encoding="utf-8")
        places_path = tmp_path / "places.json"
        places_path.write_text(json.dumps([{**HOME, "latitude": 48.815047, "longitude": 9.232482, "city": "Stuttgart"}]), encoding="utf-8")

        summary = companion.get_physical_context_summary(icloud_dir=str(tmp_path), places_file=places_path, now=NOW)
        check("city comes from the matched saved place", "City: Stuttgart" in summary)
        check("place name still reported", "Place: Home" in summary)


def test_failed_at_recorded_and_cleared() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        threads_dir = Path(tmpdir) / "threads"
        threads_dir.mkdir(parents=True, exist_ok=True)
        moment = datetime(2026, 10, 9, 14, 0, 0, tzinfo=timezone.utc)
        thread = companion.create_dispatch_thread(
            threads_dir, "Failed letter", "body", target_profile="default", sender="user", now=moment
        )
        tid = thread["thread_id"]
        companion.mark_thread_failed(threads_dir, tid, "Timeout (120s) waiting for agent", now=moment)
        meta = companion.safe_read_json(threads_dir / tid / "meta.json")
        check("failed_at recorded", meta is not None and meta.get("failed_at") == "2026-10-09T14:00:00Z")
        check("failure status recorded", meta is not None and meta.get("status") == "failed")

        companion.reply_to_thread(threads_dir, tid, "late reply", sender="agent", now=moment)
        healed = companion.safe_read_json(threads_dir / tid / "meta.json")
        check("failed_at cleared after a reply", healed is not None and "failed_at" not in healed)


def test_render_launch_agent_plist() -> None:
    plist = companion.render_launch_agent_plist("/usr/bin/python3", "/tmp/companion.py", Path("/tmp/threads"), "/Users/tester")
    check("plist runs the daemon", "--daemon" in plist and "--interval" in plist)
    check("plist keeps the job alive", "<key>KeepAlive</key>" in plist)
    check("plist sets a throttle interval", "<key>ThrottleInterval</key>" in plist)
    check("plist forces unbuffered logging", "PYTHONUNBUFFERED" in plist)
    check("plist watches the threads container", "WatchPaths" in plist and "/tmp/threads" in plist)
    check("plist drops the transient StartInterval", "StartInterval" not in plist)


def test_daemon_loop_is_bounded() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        args = companion.build_parser().parse_args(
            ["--daemon", "--interval", "0.1", "--max-cycles", "2", "--icloud-dir", tmpdir]
        )
        started = time.monotonic()
        code = companion.command_daemon(args)
        check("daemon exits cleanly after max cycles", code == 0)
        check("daemon honours the sweep interval", time.monotonic() - started < 5.0)
        check("daemon left no dispatch lock behind", not companion.get_dispatch_lock_path().exists())



def test_unreadable_meta_is_reported() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        threads_dir = tmp_path / "threads"
        threads_dir.mkdir(parents=True, exist_ok=True)
        moment = datetime(2026, 10, 9, 14, 0, 0, tzinfo=timezone.utc)

        good = companion.create_dispatch_thread(
            threads_dir, "Good", "body", target_profile="default", sender="user", now=moment
        )
        good_id = good["thread_id"]
        companion.reply_to_thread(threads_dir, good_id, "answered", sender="agent", now=moment)

        blocked_dir = threads_dir / "th_blocked"
        (blocked_dir / "messages").mkdir(parents=True, exist_ok=True)
        blocked_meta = blocked_dir / "meta.json"
        blocked_meta.write_text(json.dumps({"thread_id": "th_blocked", "status": "pending_agent"}), encoding="utf-8")
        blocked_meta.chmod(0o000)
        try:
            scan = companion.scan_dispatch_threads(threads_dir)
            check("readable thread still returned by the scan", any(t.get("thread_id") == good_id for t in scan["threads"]))
            check("unreadable meta is named by the scan", scan["unreadable"] == ["th_blocked"])
            check("unreadable reason names the cause", "unreadable" in companion.read_json_checked(blocked_meta).reason)
            check("list_dispatch_threads still returns the readable ones", len(companion.list_dispatch_threads(threads_dir)) == 1)

            args = companion.build_parser().parse_args(["--dispatch-pending", "--icloud-dir", str(tmp_path)])
            report = companion.dispatch_pending_once(args, moment, threads_dir)
            check("sweep reports no replies for an unreadable thread", report["replies"] == 0)
            check("sweep counts the unreadable thread", report["unreadable"] == 1)
            check("sweep reports no pending letter it could act on", report["pending"] == 0)
        finally:
            blocked_meta.chmod(0o600)


def test_skip_log_is_throttled() -> None:
    key = "unit-test:throttle"
    check("first skip is logged", companion.log_skip_throttled(key, "INFO", "unit test skip"))
    check("repeat skip is throttled", not companion.log_skip_throttled(key, "INFO", "unit test skip"))


def test_sync_marker_gated_sweeps_and_cache() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        threads_dir = tmp_path / "threads"
        threads_dir.mkdir(parents=True, exist_ok=True)
        companion.reset_dispatch_cache()

        moment = datetime(2026, 10, 9, 14, 0, 0, tzinfo=timezone.utc)
        thread = companion.create_dispatch_thread(
            threads_dir,
            "Gating Test",
            "Initial body",
            target_profile="default",
            sender="user",
            now=moment,
        )
        tid = thread["thread_id"]

        # 1. Test in-memory cache avoids re-reading meta.json from disk
        original_read_json = companion.read_json_checked
        read_calls = 0

        def counting_read_json(p: Path):
            nonlocal read_calls
            read_calls += 1
            return original_read_json(p)

        companion.read_json_checked = counting_read_json
        try:
            # First scan loads and caches
            scan1 = companion.scan_dispatch_threads(threads_dir)
            check("first scan reads from disk", read_calls == 1)
            check("first scan found thread", len(scan1["threads"]) == 1)

            # Second scan with unchanged file should hit in-memory cache
            scan2 = companion.scan_dispatch_threads(threads_dir)
            check("second scan uses cache without disk read", read_calls == 1)
            check("second scan returns identical thread", scan2["threads"][0]["thread_id"] == tid)

            # Touch meta file with new timestamp/mtime: should trigger re-read
            meta_file = threads_dir / tid / "meta.json"
            time.sleep(0.01)
            meta_file.touch()
            scan3 = companion.scan_dispatch_threads(threads_dir)
            check("scan after mtime touch re-reads disk", read_calls == 2)
        finally:
            companion.read_json_checked = original_read_json

        # 2. Test sync marker gating in dispatch_pending_once
        # First, reply to the thread so there are 0 pending letters
        companion.reply_to_thread(threads_dir, tid, "Agent reply", sender="agent", now=moment)

        args = companion.build_parser().parse_args(["--daemon", "--icloud-dir", str(tmp_path)])
        # Baseline sweep: establishes sync marker baseline
        report1 = companion.dispatch_pending_once(args, moment, threads_dir)
        check("baseline sweep completes with 0 pending", report1["pending"] == 0 and report1["replies"] == 0)

        # Gated sweep: sync marker has NOT changed, should skip scan completely
        original_scan = companion.scan_dispatch_threads
        scan_called = False

        def tracking_scan(p: Path):
            nonlocal scan_called
            scan_called = True
            return original_scan(p)

        companion.scan_dispatch_threads = tracking_scan
        try:
            report2 = companion.dispatch_pending_once(args, moment, threads_dir)
            check("gated sweep skips scan entirely", not scan_called)
            check("gated sweep returns zero report", report2["pending"] == 0 and report2["replies"] == 0)

            # Touch sync marker: next sweep should notice change and scan
            companion.touch_sync_marker(threads_dir, tid, now=moment)
            time.sleep(0.01)
            (threads_dir / ".sync_marker.json").touch()
            report3 = companion.dispatch_pending_once(args, moment, threads_dir)
            check("sweep after marker touch invokes scan", scan_called)
        finally:
            companion.scan_dispatch_threads = original_scan


def test_archived_threads_segregation_and_migration() -> None:
    companion.reset_dispatch_cache()
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        threads_dir = tmp_path / "threads"
        archive_dir = tmp_path / "archive"
        threads_dir.mkdir(parents=True, exist_ok=True)
        moment = datetime(2026, 10, 11, 12, 0, 0, tzinfo=timezone.utc)

        # 1. Create two active threads
        t1 = companion.create_dispatch_thread(threads_dir, "Active letter 1", "Body 1", now=moment)
        t2 = companion.create_dispatch_thread(threads_dir, "Active letter 2", "Body 2", now=moment)
        tid1 = t1["thread_id"]
        tid2 = t2["thread_id"]

        check("both folders exist in threads_dir", (threads_dir / tid1).is_dir() and (threads_dir / tid2).is_dir())

        # 2. Archive t1
        archived_meta = companion.archive_dispatch_thread(threads_dir, tid1, now=moment)
        check("archive returns archived meta", archived_meta is not None and archived_meta["status"] == "archived")
        check("t1 moved out of threads_dir", not (threads_dir / tid1).exists())
        check("t1 exists in archive_dir", (archive_dir / tid1).is_dir())
        check("t1 messages still readable in archive", len(companion.list_thread_messages(threads_dir, tid1)) == 1)

        # 3. Active scanning loops only enumerate threads_dir -> only t2 returned
        scan = companion.scan_dispatch_threads(threads_dir)
        active_ids = [t["thread_id"] for t in scan["threads"]]
        check("active scan only returns t2", active_ids == [tid2])
        check("archive_dir is segregated and ignored by active scan", tid1 not in active_ids)

        # 4. Listing archived threads via list_dispatch_threads(archive_dir) returns t1
        archived_scan = companion.list_dispatch_threads(archive_dir)
        check("archived list returns t1", len(archived_scan) == 1 and archived_scan[0]["thread_id"] == tid1)

        # 5. Legacy migration: simulate an old thread that has status 'archived' inside threads_dir
        legacy_id = "th_legacy_archived"
        legacy_folder = threads_dir / legacy_id
        legacy_folder.mkdir()
        (legacy_folder / "messages").mkdir()
        legacy_meta = {
            "thread_id": legacy_id,
            "subject": "Legacy thread",
            "target_profile": "default",
            "status": "archived",
            "created_at": "2026-10-10T10:00:00Z",
            "updated_at": "2026-10-10T10:00:00Z",
            "message_count": 1,
        }
        (legacy_folder / "meta.json").write_text(json.dumps(legacy_meta), encoding="utf-8")

        # Scan should detect archived status, move to archive_dir, and exclude from active results
        scan_migration = companion.scan_dispatch_threads(threads_dir)
        active_ids_after = [t["thread_id"] for t in scan_migration["threads"]]
        check("scan auto-migrates legacy thread", not legacy_folder.exists())
        check("legacy thread now in archive_dir", (archive_dir / legacy_id).is_dir())
        check("active scan excludes migrated legacy thread", legacy_id not in active_ids_after)

        # 6. Unarchive t1
        unarchived_meta = companion.unarchive_dispatch_thread(threads_dir, tid1, now=moment)
        check("unarchive returns replied status", unarchived_meta is not None and unarchived_meta["status"] == "replied")
        check("t1 moved back to threads_dir", (threads_dir / tid1).is_dir())
        check("t1 removed from archive_dir", not (archive_dir / tid1).exists())
        scan_after_unarchive = companion.scan_dispatch_threads(threads_dir)
        res_ids = [t["thread_id"] for t in scan_after_unarchive["threads"]]
        check("t1 now back in active scan", tid1 in res_ids and tid2 in res_ids)


def test_delete_dispatch_thread() -> None:
    companion.reset_dispatch_cache()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        threads_dir = tmp_path / "threads"
        threads_dir.mkdir(parents=True)
        archive_dir = tmp_path / "archive"
        archive_dir.mkdir(parents=True)

        moment = datetime(2026, 10, 11, 14, 0, 0, tzinfo=timezone.utc)
        t1 = companion.create_dispatch_thread(threads_dir, "Thread to archive then delete", "Letter 1", now=moment)
        t2 = companion.create_dispatch_thread(threads_dir, "Thread to delete directly", "Letter 2", now=moment)
        tid1 = t1["thread_id"]
        tid2 = t2["thread_id"]

        # Archive tid1
        companion.archive_dispatch_thread(threads_dir, tid1, now=moment)
        check("tid1 in archive", (archive_dir / tid1).is_dir())
        check("tid2 in threads", (threads_dir / tid2).is_dir())

        # 1. Delete active thread tid2
        del_res2 = companion.delete_dispatch_thread(threads_dir, tid2, now=moment)
        check("delete active thread returns True", del_res2 is True)
        check("tid2 removed from threads_dir", not (threads_dir / tid2).exists())
        scan_after_del = companion.scan_dispatch_threads(threads_dir)
        active_ids = [t["thread_id"] for t in scan_after_del["threads"]]
        check("active scan does not include deleted tid2", tid2 not in active_ids)

        # 2. Delete archived thread tid1
        del_res1 = companion.delete_dispatch_thread(threads_dir, tid1, now=moment)
        check("delete archived thread returns True", del_res1 is True)
        check("tid1 removed from archive_dir", not (archive_dir / tid1).exists())
        archived_list = companion.list_dispatch_threads(archive_dir)
        archived_ids = [t["thread_id"] for t in archived_list]
        check("archived list does not include deleted tid1", tid1 not in archived_ids)

        # 3. Non-existent thread returns False
        del_nonexistent = companion.delete_dispatch_thread(threads_dir, "th_does_not_exist", now=moment)
        check("delete nonexistent thread returns False", del_nonexistent is False)


def test_icloud_placeholder_cache_fallback() -> None:
    companion.reset_dispatch_cache()
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        threads_dir = tmp_path / "threads"
        threads_dir.mkdir(parents=True, exist_ok=True)

        moment = datetime(2026, 10, 11, 10, 0, 0, tzinfo=timezone.utc)
        thread = companion.create_dispatch_thread(
            threads_dir,
            "Placeholder Fallback Test",
            "Initial letter body",
            target_profile="default",
            sender="user",
            now=moment,
        )
        tid = thread["thread_id"]

        # Initial scan caches thread
        scan1 = companion.scan_dispatch_threads(threads_dir)
        check("initial scan finds thread", len(scan1["threads"]) == 1)

        # Initial message read caches messages
        msgs1 = companion.list_thread_messages(threads_dir, tid)
        check("initial read finds message", len(msgs1) == 1)

        # Simulate iCloud eviction: rename meta.json -> .meta.json.icloud
        meta_file = threads_dir / tid / "meta.json"
        icloud_meta = threads_dir / tid / ".meta.json.icloud"
        meta_file.rename(icloud_meta)

        # Scan should gracefully retain cached meta and not mark unreadable
        scan2 = companion.scan_dispatch_threads(threads_dir)
        check("placeholder fallback retains cached thread", len(scan2["threads"]) == 1)
        check("placeholder fallback thread id matches", scan2["threads"][0]["thread_id"] == tid)
        check("placeholder fallback does not report unreadable", len(scan2["unreadable"]) == 0)

        # Simulate message eviction: rename message file -> .<filename>.icloud
        msg_files = list((threads_dir / tid / "messages").glob("*.json"))
        check("found message file to evict", len(msg_files) == 1)
        orig_msg_file = msg_files[0]
        evicted_msg_file = orig_msg_file.parent / f".{orig_msg_file.name}.icloud"
        orig_msg_file.rename(evicted_msg_file)

        # Message read should gracefully retain cached message
        msgs2 = companion.list_thread_messages(threads_dir, tid)
        check("message placeholder fallback retains cached message", len(msgs2) == 1)
        check("message placeholder fallback id matches", msgs2[0]["id"] == msgs1[0]["id"])


def test_turn_based_concurrency_guards_and_sequence_indexing() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        threads_dir = tmp_path / "threads"
        threads_dir.mkdir(parents=True, exist_ok=True)
        moment = datetime(2026, 10, 11, 12, 0, 0, tzinfo=timezone.utc)

        # 1. User creates a thread (status: pending_agent, seq: 1)
        created = companion.create_dispatch_thread(
            threads_dir, "Project Status", "How is the project going?", target_profile="default", sender="user", now=moment
        )
        tid = created["thread_id"]
        check("thread created with pending_agent", created["status"] == "pending_agent")
        check("initial message seq is 1", created["message_count"] == 1)

        # 2. Concurrency guard: User cannot reply while thread is pending_agent
        user_conflict_reply = companion.reply_to_thread(
            threads_dir, tid, "Follow-up question before agent answers", sender="user", now=moment
        )
        check("user reply refused during pending_agent turn", user_conflict_reply is None)

        # 3. Agent replies (status transitions to replied, seq: 2)
        agent_reply = companion.reply_to_thread(
            threads_dir, tid, "The project is on track.", sender="agent", now=moment + timedelta(minutes=1)
        )
        check("agent reply succeeds", agent_reply is not None and agent_reply["sender"] == "agent")
        meta = companion.safe_read_json(threads_dir / tid / "meta.json")
        check("thread status transitioned to replied", meta is not None and meta.get("status") == "replied")
        check("message count incremented to 2", meta is not None and meta.get("message_count") == 2)

        # 4. Concurrency guard: Agent cannot reply again while thread is replied
        agent_conflict_reply = companion.reply_to_thread(
            threads_dir, tid, "Duplicate unprompted reply", sender="agent", now=moment + timedelta(minutes=2)
        )
        check("agent reply refused during replied status", agent_conflict_reply is None)

        # 5. User replies now that thread is in replied status (status transitions to pending_agent, seq: 3)
        user_valid_reply = companion.reply_to_thread(
            threads_dir, tid, "Great, thanks for the update!", sender="user", now=moment + timedelta(minutes=3)
        )
        check("user reply succeeds in replied status", user_valid_reply is not None and user_valid_reply["sender"] == "user")
        meta = companion.safe_read_json(threads_dir / tid / "meta.json")
        check("thread status transitioned back to pending_agent", meta is not None and meta.get("status") == "pending_agent")
        check("message count incremented to 3", meta is not None and meta.get("message_count") == 3)

        # 6. Monotonic disk sequence defense: simulate a higher sequence file already present on disk (e.g. 005)
        msgs_dir = threads_dir / tid / "messages"
        rogue_filename = "005_agent_20261011120500_msg_rogue01.json"
        rogue_msg = {
            "id": "msg_rogue01",
            "thread_id": tid,
            "sender": "agent",
            "timestamp": "2026-10-11T12:05:00Z",
            "body": "Simulated rogue message",
        }
        (msgs_dir / rogue_filename).write_text(json.dumps(rogue_msg), encoding="utf-8")

        # Agent replies to user turn: next sequence must be max(3, 5) + 1 = 6, avoiding collision with 005
        next_agent_reply = companion.reply_to_thread(
            threads_dir, tid, "Acknowledged.", sender="agent", now=moment + timedelta(minutes=6)
        )
        check("agent reply succeeds after rogue file", next_agent_reply is not None)
        meta = companion.safe_read_json(threads_dir / tid / "meta.json")
        check("monotonic seq advanced to 6", meta is not None and meta.get("message_count") == 6)
        files_006 = list(msgs_dir.glob("006_*.json"))
        check("letter file with 006 prefix created", len(files_006) == 1)


def test_environment_diagnostics_and_doctor() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        (tmp_path / "latest_location.json").write_text("{}", encoding="utf-8")
        results = companion.check_environment_diagnostics(str(tmp_path))
        categories = {cat for _, cat, _, _ in results}
        check("diagnostics checks operating system", "Operating System" in categories)
        check("diagnostics checks hermes cli", "Hermes CLI" in categories)
        check("diagnostics checks app container", "App Container" in categories)

        args = companion.build_parser().parse_args(["--doctor", "--icloud-dir", str(tmp_path)])
        rc = companion.command_doctor(args)
        check("doctor command returns int status code", isinstance(rc, int))

        explanation = companion.missing_icloud_explanation([Path("/nonexistent/icloud")])
        check("missing explanation contains guidance", "iPhone" in explanation or "macOS" in explanation)


def main() -> int:
    test_still_home_after_hours()
    test_walking_at_home()
    test_driving_is_transit()
    test_unlisted()
    test_unlisted_stale_unconfirmed()
    test_profile_places_isolation()
    test_invalid_timestamp_graceful()
    test_files_roundtrip()
    test_text_has_no_english_script()
    test_battery_is_not_reported()
    test_timeline_and_history()
    test_bug_md_scenario_gap_detection()
    test_apple_maps_placemark_resolution()
    test_dispatch_threads_roundtrip()
    test_profiles_registration_and_filtering()
    test_discover_installed_hermes_profiles()
    test_clean_hermes_chat_output()
    test_error_handling_and_sync_marker()
    test_physical_context_summary()
    test_atomic_write_creates_no_debris()
    test_corrupt_payload_is_skipped_once()
    test_city_fallback_without_placemark()
    test_city_from_saved_place()
    test_failed_at_recorded_and_cleared()
    test_render_launch_agent_plist()
    test_daemon_loop_is_bounded()
    test_unreadable_meta_is_reported()
    test_skip_log_is_throttled()
    test_sync_marker_gated_sweeps_and_cache()
    test_archived_threads_segregation_and_migration()
    test_delete_dispatch_thread()
    test_icloud_placeholder_cache_fallback()
    test_turn_based_concurrency_guards_and_sequence_indexing()
    test_environment_diagnostics_and_doctor()
    print("ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())


