"""Tests for recurring reminders (repeat-flag) — cron/reminder_queue.py.

Covers the repeat-flag decision on Argos issue #3 (design update 27/8):
recurring reminders stay on the queue and the poller re-arms them after each
fire, instead of graduating to a registered cron job (keeps the whole feature
on the drift-immune no-agent path).

- next_occurrence: daily/weekly rule math, strictly-after semantics
- poller E2E: a recurring entry fires, is logged, and a re-armed entry with
  the same message/origin/rule appears with a future due_at
- robustness: an invalid rule must not crash the poller

E2E against a temp HERMES_HOME (per AGENTS.md: real path, not mocks).
"""
from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Fixtures (mirror test_reminder_queue.py)
# ---------------------------------------------------------------------------


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    """Isolate HERMES_HOME so the queue file doesn't leak into the real home."""
    import importlib

    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "reminders").mkdir()

    monkeypatch.setenv("HERMES_HOME", str(home))

    import hermes_constants
    importlib.reload(hermes_constants)
    import hermes_time
    importlib.reload(hermes_time)
    import cron.reminder_queue
    importlib.reload(cron.reminder_queue)

    return home


@pytest.fixture
def queue(hermes_home):
    from cron import reminder_queue
    return reminder_queue


@pytest.fixture
def tz():
    """SAST (UTC+2) — fixed offset, no DST."""
    return timezone(timedelta(hours=2))


@pytest.fixture
def now(tz):
    """Fixed 'now': Thursday 2026-08-27 10:00 SAST."""
    return datetime(2026, 8, 27, 10, 0, tzinfo=tz)


# ---------------------------------------------------------------------------
# next_occurrence
# ---------------------------------------------------------------------------


class TestNextOccurrence:
    """Daily/weekly rule math with strictly-after semantics."""

    def test_weekly_later_this_week(self, queue, now):
        # Thu 2026-08-27 10:00 → next Tuesday 08:00 = 2026-09-01
        rule = {"kind": "weekly", "weekday": 1, "time": "08:00"}
        result = queue.next_occurrence(rule, now)
        assert result.date() == datetime(2026, 9, 1).date()
        assert result.hour == 8 and result.minute == 0
        assert result.tzinfo is now.tzinfo

    def test_weekly_same_day_time_in_future(self, queue, now, tz):
        # Monday 07:00 → today 08:00
        mon_morning = datetime(2026, 8, 31, 7, 0, tzinfo=tz)
        rule = {"kind": "weekly", "weekday": 0, "time": "08:00"}
        result = queue.next_occurrence(rule, mon_morning)
        assert result.date() == mon_morning.date()
        assert result.hour == 8

    def test_weekly_same_day_time_passed_rolls_next_week(self, queue, now, tz):
        # Monday 09:00 (08:00 passed) → next Monday 08:00
        mon_late = datetime(2026, 8, 31, 9, 0, tzinfo=tz)
        rule = {"kind": "weekly", "weekday": 0, "time": "08:00"}
        result = queue.next_occurrence(rule, mon_late)
        assert result.weekday() == 0
        assert result.hour == 8
        assert result.date() == datetime(2026, 9, 7).date()

    def test_daily_time_in_future_is_today(self, queue, now):
        rule = {"kind": "daily", "time": "18:00"}
        result = queue.next_occurrence(rule, now)
        assert result.date() == now.date()
        assert (result.hour, result.minute) == (18, 0)

    def test_daily_time_passed_rolls_tomorrow(self, queue, now, tz):
        evening = datetime(2026, 8, 27, 19, 0, tzinfo=tz)
        rule = {"kind": "daily", "time": "18:00"}
        result = queue.next_occurrence(rule, evening)
        assert result.date() == datetime(2026, 8, 28).date()
        assert result.hour == 18

    def test_daily_exact_boundary_is_tomorrow(self, queue, now):
        # Strictly-after: exactly at the scheduled time → next day.
        boundary = datetime(2026, 8, 27, 18, 0, 0, tzinfo=now.tzinfo)
        rule = {"kind": "daily", "time": "18:00"}
        result = queue.next_occurrence(rule, boundary)
        assert result.date() == datetime(2026, 8, 28).date()

    def test_unknown_kind_raises(self, queue, now):
        with pytest.raises(ValueError):
            queue.next_occurrence({"kind": "fortnightly", "time": "08:00"}, now)

    def test_invalid_time_raises(self, queue, now):
        with pytest.raises(ValueError):
            queue.next_occurrence({"kind": "daily", "time": "25:99"}, now)

    def test_weekly_weekday_out_of_range_raises(self, queue, now):
        with pytest.raises(ValueError):
            queue.next_occurrence({"kind": "weekly", "weekday": 9, "time": "08:00"}, now)

    def test_naive_after_raises(self, queue):
        naive = datetime(2026, 8, 27, 10, 0)
        with pytest.raises(ValueError):
            queue.next_occurrence({"kind": "daily", "time": "18:00"}, naive)

    def test_exact_equality_is_strictly_after(self, queue, now):
        # Weekly: today at exactly 08:00, now == 08:00:00 → next week, not today.
        tue_boundary = datetime(2026, 9, 1, 8, 0, 0, tzinfo=now.tzinfo)
        rule = {"kind": "weekly", "weekday": 1, "time": "08:00"}
        result = queue.next_occurrence(rule, tue_boundary)
        assert result.date() == datetime(2026, 9, 8).date()


# ---------------------------------------------------------------------------
# Poller E2E: recurring re-arm
# ---------------------------------------------------------------------------


class TestPollerRearmsRecurring:
    """The poller fires a recurring entry and re-arms the next occurrence."""

    def _run_poller(self, hermes_home):
        repo_root = Path(__file__).resolve().parent.parent.parent
        # Inherit the full environment (PATH/SystemRoot etc.) — a minimal env
        # breaks the child's Winsock init on Windows (WinError 10106).
        env = dict(os.environ)
        env["HERMES_HOME"] = str(hermes_home)
        env["PYTHONPATH"] = str(repo_root)
        env["PYTHONIOENCODING"] = "utf-8"
        return subprocess.run(
            [sys.executable, str(repo_root / "cron" / "scripts" / "reminder_poller.py")],
            env=env, capture_output=True, text=True, encoding="utf-8", timeout=30,
        )

    def test_recurring_daily_rearmed_after_fire(self, queue, tz, hermes_home):
        from hermes_time import now as hnow

        rule = {"kind": "daily", "time": "23:59"}
        past = datetime(2020, 1, 1, 0, 0, tzinfo=tz)
        entry = queue.add_reminder(
            past, "daily check-in", origin={"platform": "telegram", "chat_id": "123"},
            recurring=rule,
        )

        result = self._run_poller(hermes_home)
        assert result.returncode == 0
        assert "daily check-in" in (result.stdout or "")

        # Fired: original entry in the fired log with fired_at.
        fired = queue.list_fired()
        assert len(fired) == 1
        assert fired[0]["id"] == entry["id"]
        assert fired[0]["status"] == "fired"
        assert "fired_at" in fired[0]

        # Re-armed: exactly one pending entry, same message/origin/rule,
        # due strictly in the future.
        pending = queue.list_pending()
        assert len(pending) == 1
        rearmed = pending[0]
        assert rearmed["id"] != entry["id"]
        assert rearmed["message"] == "daily check-in"
        assert rearmed["origin"] == {"platform": "telegram", "chat_id": "123"}
        assert rearmed["recurring"] == rule
        assert datetime.fromisoformat(rearmed["due_at"]) > hnow()

    def test_recurring_weekly_rearmed(self, queue, tz, hermes_home):
        from hermes_time import now as hnow

        rule = {"kind": "weekly", "weekday": hnow().weekday(), "time": "23:59"}
        past = datetime(2020, 1, 1, 0, 0, tzinfo=tz)
        queue.add_reminder(past, "weekly sync", recurring=rule)

        result = self._run_poller(hermes_home)
        assert result.returncode == 0
        assert "weekly sync" in (result.stdout or "")

        pending = queue.list_pending()
        assert len(pending) == 1
        assert pending[0]["message"] == "weekly sync"
        assert datetime.fromisoformat(pending[0]["due_at"]) > hnow()
        assert len(queue.list_fired()) == 1

    def test_invalid_rule_fires_but_does_not_crash(self, queue, tz, hermes_home):
        past = datetime(2020, 1, 1, 0, 0, tzinfo=tz)
        queue.add_reminder(past, "bad rule reminder", recurring={"kind": "fortnightly"})

        result = self._run_poller(hermes_home)
        assert result.returncode == 0
        # The reminder still delivered…
        assert "bad rule reminder" in (result.stdout or "")
        # …but was NOT re-armed, with a warning on stderr.
        assert "WARNING" in (result.stderr or "")
        assert "invalid" in (result.stderr or "").lower()
        assert len(queue.list_pending()) == 0
        assert len(queue.list_fired()) == 1