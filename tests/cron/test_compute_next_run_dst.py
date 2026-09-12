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
