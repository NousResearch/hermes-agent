"""Test that compute_next_run uses last_run_at for cron jobs.

Regression test for: cron jobs computing next_run_at from _hermes_now()
instead of from last_run_at, making them inconsistent with interval jobs.
"""
import pytest
from datetime import datetime
from zoneinfo import ZoneInfo

pytest.importorskip("croniter")

from cron.jobs import compute_next_run, parse_schedule


class TestCronComputeNextRunUsesLastRunAt:
    """compute_next_run MUST use last_run_at as the croniter base for cron jobs,
    consistent with how interval jobs work."""

    def test_cron_uses_last_run_at_for_every_6h_schedule(self, monkeypatch):
        """For a schedule like 'every 6 hours', the base time matters.
        If last_run_at is Apr 6 14:10, next should be Apr 6 18:00.
        If now is Apr 10 22:00, next should be Apr 11 00:00.
        compute_next_run must use last_run_at, not now."""
        morocco = ZoneInfo("Africa/Casablanca")

        # Job last ran April 6 at 14:10
        last_run = datetime(2026, 4, 6, 14, 10, 0, tzinfo=morocco)

        # But now it's April 10 at 22:00 (e.g., gateway restarted)
        now = datetime(2026, 4, 10, 22, 0, 0, tzinfo=morocco)
        monkeypatch.setattr("cron.jobs._hermes_now", lambda: now)

        schedule = {"kind": "cron", "expr": "0 */6 * * *"}  # every 6 hours

        result = compute_next_run(schedule, last_run_at=last_run.isoformat())
        assert result is not None
        next_dt = datetime.fromisoformat(result)

        # With last_run_at as base (Apr 6 14:10), next is Apr 6 18:00.
        # With now as base (Apr 10 22:00), next is Apr 11 00:00.
        # The fix should use last_run_at, returning Apr 6 18:00
        # (stale detection in get_due_jobs() fast-forwards from there).
        assert next_dt.date().isoformat() == "2026-04-06", (
            f"Expected next run on Apr 6 (from last_run_at), got {next_dt}"
        )
        assert next_dt.hour == 18

    def test_cron_without_last_run_at_uses_now(self, monkeypatch):
        """When last_run_at is NOT provided, compute_next_run falls back to
        _hermes_now() as the croniter base (existing behavior)."""
        morocco = ZoneInfo("Africa/Casablanca")

        now = datetime(2026, 4, 10, 22, 0, 0, tzinfo=morocco)
        monkeypatch.setattr("cron.jobs._hermes_now", lambda: now)

        schedule = {"kind": "cron", "expr": "0 */6 * * *"}

        result = compute_next_run(schedule)
        assert result is not None
        next_dt = datetime.fromisoformat(result)

        # Without last_run_at, should compute from now -> Apr 11 00:00
        assert next_dt.date().isoformat() == "2026-04-11", (
            f"Expected next run on Apr 11 (from now), got {next_dt}"
        )
        assert next_dt.hour == 0

    def test_cron_with_schedule_timezone_uses_that_zone_not_hermes_zone(self, monkeypatch):
        """A New York cron should advance in New York wall time even when
        Hermes itself runs in Bangkok.

        Regression: Alpaca jobs stored timezone=America/New_York but computed
        next_run_at as 12:45 Bangkok instead of 12:45 New York.
        """
        bangkok = ZoneInfo("Asia/Bangkok")

        now = datetime(2026, 5, 7, 12, 53, 0, tzinfo=bangkok)
        last_run = datetime(2026, 5, 6, 23, 51, 54, tzinfo=bangkok)
        monkeypatch.setattr("cron.jobs._hermes_now", lambda: now)

        schedule = {
            "kind": "cron",
            "expr": "45 12 * * 1-5",
            "timezone": "America/New_York",
        }

        result = compute_next_run(schedule, last_run_at=last_run.isoformat())
        assert result is not None
        next_dt = datetime.fromisoformat(result)

        assert next_dt.tzinfo is not None
        assert next_dt.astimezone(bangkok).isoformat() == "2026-05-07T23:45:00+07:00"

    def test_parse_schedule_accepts_cron_tz_prefix(self):
        schedule = parse_schedule("CRON_TZ=America/New_York 45 12 * * 1-5")

        assert schedule["kind"] == "cron"
        assert schedule["expr"] == "45 12 * * 1-5"
        assert schedule["timezone"] == "America/New_York"
        assert schedule["display"] == "CRON_TZ=America/New_York 45 12 * * 1-5"

    def test_cron_weekly_consistent_with_interval(self, monkeypatch):
        """Both cron and interval jobs should anchor to last_run_at when
        provided, producing consistent behavior after a crash/restart."""
        morocco = ZoneInfo("Africa/Casablanca")

        last_run = datetime(2026, 4, 6, 14, 10, 0, tzinfo=morocco)
        now = datetime(2026, 4, 10, 22, 0, 0, tzinfo=morocco)
        monkeypatch.setattr("cron.jobs._hermes_now", lambda: now)

        cron_schedule = {"kind": "cron", "expr": "0 14 * * 1"}
        interval_schedule = {"kind": "interval", "minutes": 7 * 24 * 60}

        cron_result = compute_next_run(cron_schedule, last_run_at=last_run.isoformat())
        interval_result = compute_next_run(interval_schedule, last_run_at=last_run.isoformat())

        # Both should be after last_run_at
        cron_dt = datetime.fromisoformat(cron_result)
        interval_dt = datetime.fromisoformat(interval_result)
        assert cron_dt > last_run, f"Cron next {cron_dt} should be after last_run {last_run}"
        assert interval_dt > last_run, f"Interval next {interval_dt} should be after last_run {last_run}"
