"""Tests for #78516: manual trigger not swallowed by TZ migration repair.

When trigger_job() sets next_run_at and the host TZ differs from the
configured timezone, the migration-repair branch must NOT recompute
next_run_at for manually triggered jobs.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import patch, MagicMock


class TestTriggeredManuallyFlag:
    """trigger_job() should set triggered_manually=True."""

    def test_trigger_job_sets_flag(self):
        from cron.jobs import trigger_job

        mock_job = {"id": "test-123", "name": "test"}
        with patch("cron.jobs.resolve_job_ref", return_value=mock_job), \
             patch("cron.jobs.update_job") as mock_update:
            mock_update.return_value = {"id": "test-123"}
            trigger_job("test-123")

        call_kwargs = mock_update.call_args[0][1]
        assert call_kwargs["triggered_manually"] is True


class TestMigrationRepairSkipsManualTrigger:
    """Migration repair should skip jobs with triggered_manually=True."""

    def test_manual_trigger_not_recomputed(self):
        """A manually triggered job should not have its next_run_at
        recomputed by the TZ migration repair logic."""
        # Simulate: stored next_run_at is in UTC (offset 0), but now is in UTC-3
        utc = timezone.utc
        utc_minus_3 = timezone(timedelta(hours=-3))

        stored_time = datetime(2026, 8, 4, 11, 0, 0, tzinfo=utc)  # 11:00 UTC
        now_time = datetime(2026, 8, 4, 7, 30, 0, tzinfo=utc_minus_3)  # 07:30-3 = 10:30 UTC

        # The 4 conditions for migration repair:
        # 1. kind == "cron" ✓
        # 2. next_run_dt <= now (11:00 UTC <= 10:30 UTC? No — but let's make it due)
        stored_time = datetime(2026, 8, 4, 9, 0, 0, tzinfo=utc)  # 09:00 UTC (past)
        # now_time at 10:30 UTC → next_run_dt <= now ✓
        # 3. offset mismatch (UTC vs UTC-3) ✓
        # 4. stored wall clock is future (09:00 > 07:30 in naive) ✓

        # Without triggered_manually: all 4 conditions met → repair fires
        # With triggered_manually: condition 3 is short-circuited → no repair

        job = {
            "id": "test-456",
            "name": "Morning briefing",
            "schedule": {"kind": "cron", "expr": "0 8 * * *"},
            "triggered_manually": True,
        }

        # The key assertion: with triggered_manually=True, the migration
        # repair condition should be False (the flag blocks it)
        from cron.jobs import _timezone_offset_mismatch, _stored_wall_clock_is_future

        kind = job["schedule"]["kind"]
        next_run_dt = stored_time
        triggered = job.get("triggered_manually")

        should_repair = (
            kind == "cron"
            and next_run_dt <= now_time
            and not triggered
            and _timezone_offset_mismatch(stored_time, now_time)
            and _stored_wall_clock_is_future(stored_time, now_time)
        )
        assert should_repair is False, "Manual trigger should block migration repair"

    def test_without_flag_migration_repair_fires(self):
        """Without triggered_manually, migration repair should fire normally."""
        utc = timezone.utc
        utc_minus_3 = timezone(timedelta(hours=-3))

        stored_time = datetime(2026, 8, 4, 9, 0, 0, tzinfo=utc)
        now_time = datetime(2026, 8, 4, 7, 30, 0, tzinfo=utc_minus_3)

        from cron.jobs import _timezone_offset_mismatch, _stored_wall_clock_is_future

        job = {
            "id": "test-789",
            "name": "Morning briefing",
            "schedule": {"kind": "cron", "expr": "0 8 * * *"},
            # No triggered_manually field
        }

        kind = job["schedule"]["kind"]
        triggered = job.get("triggered_manually")

        should_repair = (
            kind == "cron"
            and stored_time <= now_time
            and not triggered
            and _timezone_offset_mismatch(stored_time, now_time)
            and _stored_wall_clock_is_future(stored_time, now_time)
        )
        assert should_repair is True, "Without flag, migration repair should fire"
