"""Regression: ``compute_next_run`` must hold a cron job's wall-clock hour in the configured
IANA timezone across DST.

croniter works in the UTC *offset* of its start time, never the zone, so a ``0 9 * * *`` job
in America/Toronto fired at 08:00 local on spring-forward day and 10:00 on fall-back day, and
stayed an hour off for the rest of each season. The zone is configured the way production
configures it (``HERMES_TIMEZONE``) instead of by patching ``get_timezone``, so
``get_timezone()`` and the stored-timestamp normalization agree and these tests fail on the
unfixed code with the real symptom. Filed from the 09:00 America/Toronto morning routine.
"""

import pytest
from datetime import datetime
from zoneinfo import ZoneInfo

pytest.importorskip("croniter")

import hermes_time
from cron.jobs import compute_next_run

TORONTO = ZoneInfo("America/Toronto")
MORNING = {"kind": "cron", "expr": "0 9 * * *"}


@pytest.fixture
def toronto(monkeypatch):
    """Configure the active profile's zone through the real resolution path."""
    monkeypatch.setenv("HERMES_TIMEZONE", "America/Toronto")
    hermes_time.reset_cache()
    yield
    hermes_time.reset_cache()


def _next_local(last_run_at: str) -> datetime:
    return datetime.fromisoformat(
        compute_next_run(MORNING, last_run_at=last_run_at)).astimezone(TORONTO)


class TestCronNextRunHoldsConfiguredWallClock:
    def test_dst_transition_days_fire_at_9am_local(self, toronto):
        """Spring-forward (Mar 8 2026) and fall-back (Nov 1 2026) days: 08:00 / 10:00 before."""
        spring = _next_local("2026-03-07T09:00:00-05:00")
        fall = _next_local("2026-10-31T09:00:00-04:00")
        assert (spring.hour, spring.minute) == (9, 0)
        assert (fall.hour, fall.minute) == (9, 0)

    def test_full_year_walk_never_drifts(self, toronto):
        """Walking 2026 crosses both transitions; the hour must stay 09:00 throughout."""
        last = datetime(2026, 1, 1, 9, 0, tzinfo=TORONTO)
        for _ in range(365):
            nxt = datetime.fromisoformat(
                compute_next_run(MORNING, last_run_at=last.isoformat()))
            wall = nxt.astimezone(TORONTO)
            assert (wall.hour, wall.minute) == (9, 0), \
                f"drift at {last.isoformat()}: {wall.isoformat()}"
            last = nxt


class TestFallBackStrictlyAfterContract:
    """qwen-code#11723 class: a naive wall clock re-attached to the zone resolves the
    repeated autumn hour to its earlier occurrence, so a base inside the second
    occurrence used to get a next_run in the PAST (immediate re-fire loop)."""

    def test_base_in_second_occurrence_gets_future_instant(self, toronto):
        base = datetime(2026, 11, 1, 1, 15, tzinfo=TORONTO, fold=1)  # 01:15 EST
        nxt = datetime.fromisoformat(
            compute_next_run({"kind": "cron", "expr": "30 1 * * *"},
                             last_run_at=base.isoformat()))
        assert nxt.timestamp() > base.timestamp()
        wall = nxt.astimezone(TORONTO)
        assert (wall.hour, wall.minute) == (1, 30)

    def test_every_minute_never_returns_past(self, toronto):
        base = datetime(2026, 11, 1, 1, 59, tzinfo=TORONTO, fold=0)  # last EDT minute
        nxt = datetime.fromisoformat(
            compute_next_run({"kind": "cron", "expr": "* * * * *"},
                             last_run_at=base.isoformat()))
        assert nxt.timestamp() > base.timestamp()


class TestScheduleSpecificTimezone:
    def test_schedule_tz_overrides_profile_timezone(self):
        """When schedule specifies a timezone (e.g. America/New_York), compute_next_run must honor it."""
        ny = ZoneInfo("America/New_York")
        schedule = {"kind": "cron", "expr": "0 9 * * 1-5", "tz": "America/New_York"}
        # Base is Friday 09:00 NY time
        base = datetime(2026, 10, 9, 9, 0, tzinfo=ny)
        nxt_str = compute_next_run(schedule, last_run_at=base.isoformat())
        assert nxt_str is not None
        nxt = datetime.fromisoformat(nxt_str)
        assert nxt.tzinfo is not None
        assert nxt.astimezone(ny).hour == 9
        assert nxt.astimezone(ny).minute == 0
        assert nxt.astimezone(ny).day == 12  # Monday

