"""Test that compute_next_run uses last_run_at for cron jobs.

Regression test for: cron jobs computing next_run_at from _hermes_now()
instead of from last_run_at, making them inconsistent with interval jobs.
"""
import pytest
from datetime import datetime, timezone
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


class TestIntervalNextRunIsAbsoluteAcrossDST:
    """Interval schedules mean N minutes of *real* time. When last_run_at is
    anchored on the far side of a DST transition, the next run must land at the
    correct absolute instant (last_run + interval in UTC), not at a wall-clock
    offset that the DST jump shifts by ~1h."""

    def test_spring_forward_does_not_fire_early(self, monkeypatch):
        """last_run 2026-03-08 01:30 EST, every 120 min. Spring-forward is at
        02:00. Wall-clock addition yields 03:30 EDT = 07:30 UTC (only 60 real
        minutes -> fires ~1h early). Absolute math yields 08:30 UTC."""
        ny = ZoneInfo("America/New_York")
        now = datetime(2026, 3, 8, 12, 0, 0, tzinfo=ny)
        monkeypatch.setattr("cron.jobs._hermes_now", lambda: now)

        schedule = {"kind": "interval", "minutes": 120}
        result = compute_next_run(
            schedule, last_run_at="2026-03-08T01:30:00-05:00"
        )
        assert result is not None
        got_utc = datetime.fromisoformat(result).astimezone(timezone.utc)
        assert got_utc == datetime(2026, 3, 8, 8, 30, tzinfo=timezone.utc), (
            f"Expected 08:30 UTC (last_run + 120 real minutes), got {got_utc}"
        )

    def test_fall_back_does_not_fire_late(self, monkeypatch):
        """last_run 2026-11-01 01:30 EDT, every 120 min. Fall-back is at 02:00.
        Wall-clock addition yields 03:30 EST = 08:30 UTC (150 real minutes ->
        fires ~1h late). Absolute math yields 07:30 UTC."""
        ny = ZoneInfo("America/New_York")
        now = datetime(2026, 11, 1, 12, 0, 0, tzinfo=ny)
        monkeypatch.setattr("cron.jobs._hermes_now", lambda: now)

        schedule = {"kind": "interval", "minutes": 120}
        result = compute_next_run(
            schedule, last_run_at="2026-11-01T01:30:00-04:00"
        )
        assert result is not None
        got_utc = datetime.fromisoformat(result).astimezone(timezone.utc)
        assert got_utc == datetime(2026, 11, 1, 7, 30, tzinfo=timezone.utc), (
            f"Expected 07:30 UTC (last_run + 120 real minutes), got {got_utc}"
        )

    def test_first_run_spring_forward_does_not_fire_early(self, monkeypatch):
        """First-run branch (no last_run_at): the interval is measured from
        `now` and must still be absolute.

        now = 2026-03-08 01:30 EST (06:30 UTC), every 120 min. Wall-clock
        addition yields 03:30, which on that date resolves to EDT = 07:30 UTC
        — only 60 real minutes, so the first run fires ~1h early. Absolute
        math yields 08:30 UTC.
        """
        ny = ZoneInfo("America/New_York")
        now = datetime(2026, 3, 8, 1, 30, 0, tzinfo=ny)
        monkeypatch.setattr("cron.jobs._hermes_now", lambda: now)

        schedule = {"kind": "interval", "minutes": 120}
        result = compute_next_run(schedule)
        assert result is not None
        got_utc = datetime.fromisoformat(result).astimezone(timezone.utc)
        assert got_utc == datetime(2026, 3, 8, 8, 30, tzinfo=timezone.utc), (
            f"Expected 08:30 UTC (now + 120 real minutes), got {got_utc}"
        )

    def test_first_run_fall_back_does_not_fire_late(self, monkeypatch):
        """First-run branch across the fall-back boundary.

        now = 2026-11-01 01:30 EDT (05:30 UTC), every 120 min. Wall-clock
        addition yields 03:30 EST = 08:30 UTC — 150 real minutes, so the first
        run fires ~1h late. Absolute math yields 07:30 UTC.
        """
        ny = ZoneInfo("America/New_York")
        now = datetime(2026, 11, 1, 1, 30, 0, tzinfo=ny)
        monkeypatch.setattr("cron.jobs._hermes_now", lambda: now)

        schedule = {"kind": "interval", "minutes": 120}
        result = compute_next_run(schedule)
        assert result is not None
        got_utc = datetime.fromisoformat(result).astimezone(timezone.utc)
        assert got_utc == datetime(2026, 11, 1, 7, 30, tzinfo=timezone.utc), (
            f"Expected 07:30 UTC (now + 120 real minutes), got {got_utc}"
        )

    @pytest.mark.parametrize(
        "bad_last_run_at",
        ["not-a-timestamp", "2026-13-45T99:99:99", "1772000000"],
        ids=["garbage", "out-of-range", "epoch-seconds"],
    )
    def test_exception_fallback_spring_forward_does_not_fire_early(
        self, monkeypatch, bad_last_run_at
    ):
        """Exception-fallback branch: an unparseable `last_run_at` falls back to
        `now`, and that fallback must be absolute too.

        `datetime.fromisoformat` raises on these, so the handler recomputes from
        now = 2026-03-08 01:30 EST (06:30 UTC). Wall-clock addition would give
        03:30 EDT = 07:30 UTC (~1h early); absolute math gives 08:30 UTC.
        """
        ny = ZoneInfo("America/New_York")
        now = datetime(2026, 3, 8, 1, 30, 0, tzinfo=ny)
        monkeypatch.setattr("cron.jobs._hermes_now", lambda: now)

        schedule = {"kind": "interval", "minutes": 120}
        result = compute_next_run(schedule, last_run_at=bad_last_run_at)
        assert result is not None
        got_utc = datetime.fromisoformat(result).astimezone(timezone.utc)
        assert got_utc == datetime(2026, 3, 8, 8, 30, tzinfo=timezone.utc), (
            f"Expected 08:30 UTC (now + 120 real minutes), got {got_utc}"
        )

    def test_exception_fallback_fall_back_does_not_fire_late(self, monkeypatch):
        """Exception-fallback branch across the fall-back boundary.

        now = 2026-11-01 01:30 EDT (05:30 UTC). Wall-clock addition would give
        03:30 EST = 08:30 UTC (~1h late); absolute math gives 07:30 UTC.
        """
        ny = ZoneInfo("America/New_York")
        now = datetime(2026, 11, 1, 1, 30, 0, tzinfo=ny)
        monkeypatch.setattr("cron.jobs._hermes_now", lambda: now)

        schedule = {"kind": "interval", "minutes": 120}
        result = compute_next_run(schedule, last_run_at="not-a-timestamp")
        assert result is not None
        got_utc = datetime.fromisoformat(result).astimezone(timezone.utc)
        assert got_utc == datetime(2026, 11, 1, 7, 30, tzinfo=timezone.utc), (
            f"Expected 07:30 UTC (now + 120 real minutes), got {got_utc}"
        )

    def test_non_dst_interval_is_exact(self, monkeypatch):
        """An interval that does not cross a DST boundary is unchanged: the
        absolute-time round-trip is a no-op when the offset is stable."""
        ny = ZoneInfo("America/New_York")
        now = datetime(2026, 6, 1, 18, 0, 0, tzinfo=ny)
        monkeypatch.setattr("cron.jobs._hermes_now", lambda: now)

        schedule = {"kind": "interval", "minutes": 120}
        result = compute_next_run(
            schedule, last_run_at="2026-06-01T12:00:00-04:00"
        )
        assert result is not None
        got_utc = datetime.fromisoformat(result).astimezone(timezone.utc)
        assert got_utc == datetime(2026, 6, 1, 18, 0, tzinfo=timezone.utc)
