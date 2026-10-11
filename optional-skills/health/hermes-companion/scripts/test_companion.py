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
        check("corrupt payload warned once", str(payload) in companion._WARNED_PATHS)


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
    print("ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())


