"""Test that compute_next_run uses last_run_at for cron jobs.

Regression test for: cron jobs computing next_run_at from _hermes_now()
instead of from last_run_at, making them inconsistent with interval jobs.
"""
import pytest
from datetime import datetime
from zoneinfo import ZoneInfo

pytest.importorskip("croniter")

from cron.jobs import compute_next_run


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


