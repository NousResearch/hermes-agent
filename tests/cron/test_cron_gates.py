"""Tests for cron/cron_gates.py — pre-flight gate chain for cron jobs."""

from datetime import datetime, timezone, timedelta

import pytest

from cron.cron_gates import (
    check_job_gates,
    gate_cooldown,
    gate_max_daily,
    gate_active_hours,
)


def _now() -> datetime:
    """Return a fixed reference time for deterministic tests."""
    return datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


class TestGateCooldown:
    """cooldown_minutes: minimum gap since last run."""

    def test_no_gate_config(self):
        """No cooldown configured → always pass."""
        assert gate_cooldown({"gates": {}, "last_run_at": None}, _now()) == (True, "")

    def test_no_last_run(self):
        """First-time run → pass."""
        assert gate_cooldown(
            {"gates": {"cooldown_minutes": 30}, "last_run_at": None}, _now()
        ) == (True, "")

    def test_within_cooldown(self):
        """Job ran 5 minutes ago → block."""
        now = _now()
        last_run = now - timedelta(minutes=5)
        ok, reason = gate_cooldown(
            {"gates": {"cooldown_minutes": 30}, "last_run_at": last_run.isoformat()}, now
        )
        assert not ok
        assert "cooldown" in reason
        assert "5m ago" in reason

    def test_after_cooldown(self):
        """Job ran 45 minutes ago with 30min cooldown → pass."""
        now = _now()
        last_run = now - timedelta(minutes=45)
        assert gate_cooldown(
            {"gates": {"cooldown_minutes": 30}, "last_run_at": last_run.isoformat()}, now
        ) == (True, "")

    def test_exactly_at_boundary(self):
        """Job ran exactly 30 minutes ago with 30min cooldown → pass (check is <, not <=)."""
        now = _now()
        last_run = now - timedelta(minutes=30)
        assert gate_cooldown(
            {"gates": {"cooldown_minutes": 30}, "last_run_at": last_run.isoformat()}, now
        ) == (True, "")

    def test_missing_timezone_last_run(self):
        """last_run_at without tzinfo is treated as UTC (same zone as now)."""
        now = _now()
        last_run_naive = datetime(2026, 6, 1, 11, 55, 0)  # naive → default UTC
        ok, reason = gate_cooldown(
            {"gates": {"cooldown_minutes": 30}, "last_run_at": last_run_naive.isoformat()}, now
        )
        assert not ok
        assert "5m ago" in reason


class TestGateMaxDaily:
    """max_daily: max runs in rolling 24h window."""

    def test_no_gate_config(self):
        assert gate_max_daily({"gates": {}, "run_history": []}, _now()) == (True, "")

    def test_no_history(self):
        assert gate_max_daily(
            {"gates": {"max_daily": 5}, "run_history": []}, _now()
        ) == (True, "")

    def test_under_limit(self):
        now = _now()
        history = [{"timestamp": (now - timedelta(hours=i)).isoformat()} for i in range(3)]
        assert gate_max_daily(
            {"gates": {"max_daily": 5}, "run_history": history}, now
        ) == (True, "")

    def test_over_limit(self):
        now = _now()
        history = [{"timestamp": (now - timedelta(hours=i)).isoformat()} for i in range(10)]
        ok, reason = gate_max_daily(
            {"gates": {"max_daily": 5}, "run_history": history}, now
        )
        assert not ok
        assert "max_daily" in reason
        assert "10 runs" in reason

    def test_old_entries_dont_count(self):
        """Entries older than 24h are excluded."""
        now = _now()
        history = [
            {"timestamp": (now - timedelta(hours=25)).isoformat()},  # too old
            {"timestamp": (now - timedelta(hours=2)).isoformat()},   # recent
        ]
        assert gate_max_daily(
            {"gates": {"max_daily": 5}, "run_history": history}, now
        ) == (True, "")

    def test_plain_string_history(self):
        """run_history entries can be plain ISO strings (not dicts)."""
        now = _now()
        history = [now.isoformat()] * 5
        ok, reason = gate_max_daily(
            {"gates": {"max_daily": 3}, "run_history": history}, now
        )
        assert not ok
        assert "5 runs" in reason


class TestGateActiveHours:
    """active_hours: time window restriction."""

    def test_no_gate_config(self):
        assert gate_active_hours({"gates": {}}, _now()) == (True, "")

    def test_inside_window(self):
        now = datetime(2026, 6, 1, 14, 30, 0, tzinfo=timezone.utc)
        assert gate_active_hours(
            {"gates": {"active_hours": "09:00-18:00"}}, now
        ) == (True, "")

    def test_outside_window(self):
        now = datetime(2026, 6, 1, 20, 0, 0, tzinfo=timezone.utc)
        ok, reason = gate_active_hours(
            {"gates": {"active_hours": "09:00-18:00"}}, now
        )
        assert not ok
        assert "outside window" in reason

    def test_exactly_at_boundary_open(self):
        """At 09:00 exactly → inside window (start is inclusive)."""
        now = datetime(2026, 6, 1, 9, 0, 0, tzinfo=timezone.utc)
        assert gate_active_hours(
            {"gates": {"active_hours": "09:00-18:00"}}, now
        ) == (True, "")

    def test_exactly_at_boundary_close(self):
        """At 18:00 exactly → outside window (end is exclusive)."""
        now = datetime(2026, 6, 1, 18, 0, 0, tzinfo=timezone.utc)
        ok, reason = gate_active_hours(
            {"gates": {"active_hours": "09:00-18:00"}}, now
        )
        assert not ok

    def test_crosses_midnight_inside(self):
        """Window 22:00-06:00, current time 02:00 → pass."""
        now = datetime(2026, 6, 1, 2, 0, 0, tzinfo=timezone.utc)
        assert gate_active_hours(
            {"gates": {"active_hours": "22:00-06:00"}}, now
        ) == (True, "")

    def test_crosses_midnight_outside(self):
        """Window 22:00-06:00, current time 12:00 → block."""
        now = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        ok, reason = gate_active_hours(
            {"gates": {"active_hours": "22:00-06:00"}}, now
        )
        assert not ok

    def test_invalid_format(self):
        """Malformed window string → pass (logged as warning)."""
        assert gate_active_hours(
            {"gates": {"active_hours": "not-a-valid-format"}}, _now()
        ) == (True, "")


class TestCheckJobGates:
    """Integration: check_job_gates runs the full gate chain."""

    def test_no_gates_configured(self):
        """No gates dict → always pass."""
        assert check_job_gates({"id": "test"}) == (True, "")

    def test_empty_gates(self):
        assert check_job_gates({"id": "test", "gates": {}}) == (True, "")

    def test_all_gates_pass(self):
        """Job with cooldown met + daily under limit + inside window → pass."""
        assert check_job_gates({
            "id": "test",
            "gates": {"cooldown_minutes": 30, "max_daily": 10, "active_hours": "00:00-23:59"},
        }) == (True, "")

    def test_cooldown_blocks(self):
        """cooldown gate fires first."""
        from hermes_time import now as _hn
        now = _hn()
        ok, reason = check_job_gates({
            "id": "test",
            "gates": {"cooldown_minutes": 30},
            "last_run_at": now.isoformat(),
        })
        assert not ok
        assert "cooldown" in reason

    def test_max_daily_blocks(self):
        from hermes_time import now as _hn
        now = _hn()
        ok, reason = check_job_gates({
            "id": "test",
            "gates": {"max_daily": 1},
            "run_history": [{"timestamp": now.isoformat()}, {"timestamp": now.isoformat()}],
        })
        assert not ok
        assert "max_daily" in reason

    def test_active_hours_blocks(self):
        """Active hours fires when cooldown and max_daily pass."""
        ok, reason = check_job_gates({
            "id": "test",
            "gates": {"active_hours": "01:00-01:01"},
        })
        assert not ok
        assert "outside window" in reason
