"""Tests for the inline reminder queue (cron/reminder_queue.py).

Covers the acceptance criteria from Argos issue #3:
- Parsing (local tz): absolute + relative time expressions
- Fire-once: entry fires and is marked, never fires twice
- List/cancel: list pending sorted, cancel by query and by ID
- Catch-up: overdue reminders fire on the next poll (fire-late default)

E2E against a temp HERMES_HOME (per AGENTS.md: real path, not mocks).
"""
from __future__ import annotations

import importlib
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    """Isolate HERMES_HOME so the queue file doesn't leak into the real home."""
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "reminders").mkdir()

    monkeypatch.setenv("HERMES_HOME", str(home))

    # Reload modules that cache get_hermes_home() at import time.
    import hermes_constants
    importlib.reload(hermes_constants)
    import hermes_time
    importlib.reload(hermes_time)
    import cron.reminder_queue
    importlib.reload(cron.reminder_queue)

    return home


@pytest.fixture
def queue(hermes_home):
    """Return the reminder_queue module with an isolated HERMES_HOME."""
    from cron import reminder_queue
    return reminder_queue


@pytest.fixture
def tz():
    """The test timezone — use a fixed offset to avoid DST issues."""
    return timezone(timedelta(hours=2))  # SAST (UTC+2)


@pytest.fixture
def now(tz):
    """A fixed 'now' for deterministic tests: 2026-08-27 10:00 SAST."""
    return datetime(2026, 8, 27, 10, 0, tzinfo=tz)


# ---------------------------------------------------------------------------
# Parsing tests
# ---------------------------------------------------------------------------


class TestParseWhen:
    """parse_when handles absolute and relative time expressions."""

    def test_relative_minutes(self, queue, now):
        result = queue.parse_when("in 20 minutes", now)
        assert result == now + timedelta(minutes=20)

    def test_relative_seconds(self, queue, now):
        result = queue.parse_when("in 30 seconds", now)
        assert result == now + timedelta(seconds=30)

    def test_relative_hours(self, queue, now):
        result = queue.parse_when("in 2 hours", now)
        assert result == now + timedelta(hours=2)

    def test_relative_days(self, queue, now):
        result = queue.parse_when("in 3 days", now)
        assert result == now + timedelta(days=3)

    def test_relative_weeks(self, queue, now):
        result = queue.parse_when("in 1 week", now)
        assert result == now + timedelta(weeks=1)

    def test_absolute_tuesday_8am(self, queue, now):
        # 2026-08-27 is a Thursday.  Next Tuesday = 2026-09-01.
        result = queue.parse_when("tuesday 8am", now)
        assert result.weekday() == 1  # Tuesday
        assert result.hour == 8
        assert result.minute == 0
        assert result.date() == datetime(2026, 9, 1).date()

    def test_absolute_tue_8am_short(self, queue, now):
        result = queue.parse_when("tue 8am", now)
        assert result.weekday() == 1
        assert result.hour == 8

    def test_absolute_tuesday_830am(self, queue, now):
        result = queue.parse_when("tuesday 8:30am", now)
        assert result.hour == 8
        assert result.minute == 30

    def test_absolute_tomorrow_8am(self, queue, now):
        result = queue.parse_when("tomorrow 8am", now)
        assert result.date() == (now + timedelta(days=1)).date()
        assert result.hour == 8

    def test_absolute_today_6pm(self, queue, now):
        result = queue.parse_when("today 6pm", now)
        assert result.date() == now.date()
        assert result.hour == 18

    def test_absolute_pm_conversion(self, queue, now):
        result = queue.parse_when("today 6pm", now)
        assert result.hour == 18

    def test_absolute_am_midnight(self, queue, now):
        result = queue.parse_when("today 12am", now)
        assert result.hour == 0

    def test_absolute_pm_noon(self, queue, now):
        result = queue.parse_when("today 12pm", now)
        assert result.hour == 12

    def test_iso_format(self, queue, now):
        result = queue.parse_when("2026-08-28 08:00", now)
        assert result.hour == 8
        assert result.day == 28
        # Should be tz-aware (attached from 'now')
        assert result.tzinfo is not None

    def test_tuesday_today_time_passed_goes_next_week(self, queue, tz):
        # Thursday 8am — "tuesday 8am" with time already passed today
        # (but Tuesday is not today, so it's next Tuesday regardless)
        now_thu = datetime(2026, 8, 27, 10, 0, tzinfo=tz)  # Thursday
        result = queue.parse_when("tuesday 8am", now_thu)
        assert result.date() == datetime(2026, 9, 1).date()

    def test_today_time_passed_raises_or_future(self, queue, tz):
        # "today 6am" when it's 10am — should still resolve (to today 6am,
        # which is in the past, but parse_when doesn't check that — the
        # caller decides what to do with past times)
        now_10am = datetime(2026, 8, 27, 10, 0, tzinfo=tz)
        result = queue.parse_when("today 6am", now_10am)
        assert result.hour == 6
        assert result.date() == now_10am.date()

    def test_unparseable_raises(self, queue, now):
        with pytest.raises(ValueError, match="could not parse"):
            queue.parse_when("sometime next week maybe", now)

    def test_result_is_tz_aware(self, queue, now):
        result = queue.parse_when("in 5 minutes", now)
        assert result.tzinfo is not None


# ---------------------------------------------------------------------------
# Queue operations: add / list / cancel / fire-mark
# ---------------------------------------------------------------------------


class TestQueueOperations:
    """add/list/cancel/mark_fired behave correctly."""

    def test_add_and_list(self, queue, tz):
        due = datetime(2026, 8, 28, 8, 0, tzinfo=tz)
        entry = queue.add_reminder(due, "call the plumber")
        assert entry["status"] == "pending"
        assert entry["id"]

        pending = queue.list_pending()
        assert len(pending) == 1
        assert pending[0]["id"] == entry["id"]
        assert pending[0]["message"] == "call the plumber"

    def test_list_sorted_by_time(self, queue, tz):
        early = datetime(2026, 8, 28, 8, 0, tzinfo=tz)
        late = datetime(2026, 8, 28, 18, 0, tzinfo=tz)
        queue.add_reminder(late, "evening task")
        queue.add_reminder(early, "morning task")

        pending = queue.list_pending()
        assert pending[0]["message"] == "morning task"
        assert pending[1]["message"] == "evening task"

    def test_cancel_by_id(self, queue, tz):
        due = datetime(2026, 8, 28, 8, 0, tzinfo=tz)
        entry = queue.add_reminder(due, "call the plumber")
        assert queue.cancel_reminder(entry["id"]) is True
        assert len(queue.list_pending()) == 0

    def test_cancel_nonexistent_returns_false(self, queue):
        assert queue.cancel_reminder("nonexistent") is False

    def test_cancel_by_query(self, queue, tz):
        queue.add_reminder(datetime(2026, 8, 28, 8, 0, tzinfo=tz), "call the plumber")
        queue.add_reminder(datetime(2026, 8, 29, 9, 0, tzinfo=tz), "buy groceries")

        cancelled = queue.cancel_by_query("plumber")
        assert len(cancelled) == 1
        assert cancelled[0]["message"] == "call the plumber"
        remaining = queue.list_pending()
        assert len(remaining) == 1
        assert remaining[0]["message"] == "buy groceries"

    def test_cancel_by_query_case_insensitive(self, queue, tz):
        queue.add_reminder(datetime(2026, 8, 28, 8, 0, tzinfo=tz), "Call The Plumber")
        cancelled = queue.cancel_by_query("PLUMBER")
        assert len(cancelled) == 1

    def test_cancel_by_query_empty_string_noop(self, queue, tz):
        queue.add_reminder(datetime(2026, 8, 28, 8, 0, tzinfo=tz), "call the plumber")
        cancelled = queue.cancel_by_query("")
        assert len(cancelled) == 0
        assert len(queue.list_pending()) == 1


# ---------------------------------------------------------------------------
# Fire-once: entry fires and is marked, never fires twice
# ---------------------------------------------------------------------------


class TestFireOnce:
    """Reminders fire exactly once — mark_fired removes from pending."""

    def test_mark_fired_removes_from_pending(self, queue, tz):
        due = datetime(2026, 8, 28, 8, 0, tzinfo=tz)
        entry = queue.add_reminder(due, "call the plumber")
        assert queue.mark_fired(entry["id"]) is True
        assert len(queue.list_pending()) == 0

    def test_mark_fired_writes_to_fired_log(self, queue, tz):
        due = datetime(2026, 8, 28, 8, 0, tzinfo=tz)
        entry = queue.add_reminder(due, "call the plumber")
        queue.mark_fired(entry["id"])

        fired = queue.list_fired()
        assert len(fired) == 1
        assert fired[0]["id"] == entry["id"]
        assert fired[0]["status"] == "fired"
        assert "fired_at" in fired[0]

    def test_mark_fired_twice_returns_false(self, queue, tz):
        due = datetime(2026, 8, 28, 8, 0, tzinfo=tz)
        entry = queue.add_reminder(due, "call the plumber")
        assert queue.mark_fired(entry["id"]) is True
        # Already fired — second call should fail
        assert queue.mark_fired(entry["id"]) is False

    def test_due_now_does_not_mark_fired(self, queue, tz):
        """due_now returns due entries but does NOT mark them — poller marks after delivery."""
        past = datetime(2020, 1, 1, 0, 0, tzinfo=tz)
        queue.add_reminder(past, "overdue reminder")

        due = queue.due_now()
        assert len(due) == 1
        # Should still be pending (not marked)
        assert len(queue.list_pending()) == 1


# ---------------------------------------------------------------------------
# Catch-up: overdue reminders fire on next poll (fire-late default)
# ---------------------------------------------------------------------------


class TestCatchUp:
    """Machine asleep at due_at → catch-up on next poll (fire-late)."""

    def test_overdue_reminder_is_due_now(self, queue, tz):
        """A reminder from the past should be returned by due_now."""
        past = datetime(2020, 1, 1, 0, 0, tzinfo=tz)
        queue.add_reminder(past, "very overdue reminder")

        due = queue.due_now()
        assert len(due) == 1
        assert due[0]["message"] == "very overdue reminder"

    def test_future_reminder_not_due(self, queue, tz):
        future = datetime(2099, 1, 1, 0, 0, tzinfo=tz)
        queue.add_reminder(future, "far future reminder")

        due = queue.due_now()
        assert len(due) == 0

    def test_catch_up_fires_and_marks(self, queue, tz):
        """Full catch-up cycle: overdue → due_now → mark_fired → gone."""
        past = datetime(2020, 1, 1, 0, 0, tzinfo=tz)
        entry = queue.add_reminder(past, "overdue reminder")

        due = queue.due_now()
        assert len(due) == 1

        # Simulate the poller: mark fired after delivery
        assert queue.mark_fired(entry["id"]) is True

        # Next poll tick — should be empty
        due = queue.due_now()
        assert len(due) == 0

    def test_multiple_overdue_all_fire(self, queue, tz):
        """Multiple overdue reminders all fire on the same poll tick."""
        queue.add_reminder(datetime(2020, 1, 1, 0, 0, tzinfo=tz), "first")
        queue.add_reminder(datetime(2020, 6, 1, 0, 0, tzinfo=tz), "second")
        queue.add_reminder(datetime(2021, 1, 1, 0, 0, tzinfo=tz), "third")

        due = queue.due_now()
        assert len(due) == 3
        # Should be sorted by due_at (oldest first)
        assert due[0]["message"] == "first"
        assert due[2]["message"] == "third"


# ---------------------------------------------------------------------------
# Origin recording
# ---------------------------------------------------------------------------


class TestOriginRecording:
    """Origin (platform/chat/thread) is recorded on the entry."""

    def test_origin_stored(self, queue, tz):
        due = datetime(2026, 8, 28, 8, 0, tzinfo=tz)
        origin = {"platform": "telegram", "chat_id": "12345", "thread_id": "67890"}
        entry = queue.add_reminder(due, "test", origin=origin)
        assert entry["origin"] == origin

    def test_origin_none_defaults_empty(self, queue, tz):
        due = datetime(2026, 8, 28, 8, 0, tzinfo=tz)
        entry = queue.add_reminder(due, "test", origin=None)
        assert entry["origin"] == {}

    def test_origin_partial(self, queue, tz):
        """Origin can be partial (e.g. just platform+chat, no thread)."""
        due = datetime(2026, 8, 28, 8, 0, tzinfo=tz)
        origin = {"platform": "telegram", "chat_id": "12345"}
        entry = queue.add_reminder(due, "test", origin=origin)
        assert entry["origin"] == origin
        assert "thread_id" not in entry["origin"]


# ---------------------------------------------------------------------------
# Recurring flag
# ---------------------------------------------------------------------------


class TestRecurringFlag:
    """Recurring entries store a structured rule; the poller re-arms them."""

    def test_recurring_stored(self, queue, tz):
        due = datetime(2026, 8, 28, 8, 0, tzinfo=tz)
        recurring = {"kind": "weekly", "weekday": 1, "time": "08:00"}
        entry = queue.add_reminder(due, "take out trash", recurring=recurring)
        assert entry["recurring"] == recurring

    def test_non_recurring_is_none(self, queue, tz):
        due = datetime(2026, 8, 28, 8, 0, tzinfo=tz)
        entry = queue.add_reminder(due, "one-shot task")
        assert entry["recurring"] is None


# ---------------------------------------------------------------------------
# Rephrase helper
# ---------------------------------------------------------------------------


class TestRephrase:
    """rephrase_for_fire_time adjusts time words relative to fire time."""

    def test_tomorrow_to_today(self, queue, tz):
        now = datetime(2026, 8, 27, 10, 0, tzinfo=tz)
        due = datetime(2026, 8, 28, 8, 0, tzinfo=tz)  # tomorrow
        msg = "remind me tomorrow to call the plumber"
        result = queue.rephrase_for_fire_time(msg, due, now)
        # At fire time (tomorrow), "tomorrow" → "today"
        assert "today" in result
        assert "tomorrow" not in result

    def test_same_day_no_change(self, queue, tz):
        now = datetime(2026, 8, 27, 10, 0, tzinfo=tz)
        due = datetime(2026, 8, 27, 18, 0, tzinfo=tz)  # same day
        msg = "call the plumber"
        result = queue.rephrase_for_fire_time(msg, due, now)
        assert result == "call the plumber"


# ---------------------------------------------------------------------------
# Poller integration (E2E with the actual poller script)
# ---------------------------------------------------------------------------


class TestPollerIntegration:
    """The poller script reads the queue, prints due reminders, marks fired."""

    def test_poller_silent_when_empty(self, queue, hermes_home, monkeypatch):
        """Empty queue → poller prints nothing → silent run."""
        import subprocess, sys

        repo_root = Path(__file__).resolve().parent.parent.parent
        env = dict(os.environ)
        env["HERMES_HOME"] = str(hermes_home)
        env["PYTHONPATH"] = str(repo_root)
        env["PYTHONIOENCODING"] = "utf-8"

        result = subprocess.run(
            [sys.executable, str(repo_root / "cron" / "scripts" / "reminder_poller.py")],
            env=env, capture_output=True, text=True, encoding="utf-8", timeout=30,
        )
        assert result.returncode == 0
        assert (result.stdout or "").strip() == ""

    def test_poller_fires_due_reminder(self, queue, tz, hermes_home):
        """Due reminder → poller prints it + marks fired."""
        import subprocess, sys, os

        past = datetime(2020, 1, 1, 0, 0, tzinfo=tz)
        entry = queue.add_reminder(past, "test reminder from the past")

        repo_root = Path(__file__).resolve().parent.parent.parent
        env = dict(os.environ)
        env["HERMES_HOME"] = str(hermes_home)
        env["PYTHONPATH"] = str(repo_root)
        env["PYTHONIOENCODING"] = "utf-8"

        result = subprocess.run(
            [sys.executable, str(repo_root / "cron" / "scripts" / "reminder_poller.py")],
            env=env, capture_output=True, text=True, encoding="utf-8", timeout=30,
        )
        assert result.returncode == 0
        assert "test reminder from the past" in (result.stdout or "")
        assert entry["id"] in (result.stdout or "")

        # Should be marked fired now
        assert len(queue.list_pending()) == 0
        fired = queue.list_fired()
        assert len(fired) == 1
        assert fired[0]["id"] == entry["id"]

    def test_poller_silent_after_fire(self, queue, tz, hermes_home):
        """After firing, next poll is silent (no double-fire)."""
        import subprocess, sys, os

        past = datetime(2020, 1, 1, 0, 0, tzinfo=tz)
        queue.add_reminder(past, "one-shot reminder")

        repo_root = Path(__file__).resolve().parent.parent.parent
        env = dict(os.environ)
        env["HERMES_HOME"] = str(hermes_home)
        env["PYTHONPATH"] = str(repo_root)
        env["PYTHONIOENCODING"] = "utf-8"

        # First poll — fires
        r1 = subprocess.run(
            [sys.executable, str(repo_root / "cron" / "scripts" / "reminder_poller.py")],
            env=env, capture_output=True, text=True, encoding="utf-8", timeout=30,
        )
        assert "one-shot reminder" in (r1.stdout or "")

        # Second poll — silent
        r2 = subprocess.run(
            [sys.executable, str(repo_root / "cron" / "scripts" / "reminder_poller.py")],
            env=env, capture_output=True, text=True, encoding="utf-8", timeout=30,
        )
        assert (r2.stdout or "").strip() == ""
