"""Regression test: compute_next_run must honor the configured IANA timezone.

Background (task t_392f55e9 -- "morning routine at 9:00 America/Toronto"):

croniter 6.0.0 ignores the tzinfo on its start time and uses the start's
UTC *offset* as its working offset. The original code passed a tz-aware
``last_run_at`` straight into ``croniter(expr, base_time)``, which caused two
distinct bugs:

  1. When ``last_run_at`` is stored in UTC (e.g. ``+00:00`` -- what the gateway
     wrote while it was running without ``timezone`` configured), the next
     occurrence was computed 24h later in UTC terms, so a ``0 9 * * *`` job
     fired at 09:00 UTC = 05:00 Toronto (four hours early).

  2. On DST transition days the wall-clock hour drifted: 08:00 on the
     spring-forward day and 10:00 on the fall-back day, instead of 09:00.

The fix renders the base as the *configured* IANA zone's naive wall clock for
croniter, then re-attaches the zone, so the wall-clock hour is correct on
every calendar day including DST boundaries.
"""

import pytest
from datetime import datetime
from zoneinfo import ZoneInfo

pytest.importorskip("croniter")

from cron.jobs import compute_next_run

# The configured IANA zone the fix must anchor to.
TORONTO = ZoneInfo("America/Toronto")


class TestCronComputeNextRunHonorsConfiguredTz:
    """``compute_next_run`` must keep the wall-clock hour at the cron hour in
    the configured IANA timezone, regardless of the offset carried by
    ``last_run_at`` and across DST transitions."""

    def test_toronto_utc_stored_last_run_stays_at_9am_local(self, monkeypatch):
        """A last_run_at stored in UTC must NOT push the next fire to 09:00 UTC."""
        monkeypatch.setattr("cron.jobs.get_timezone", lambda: TORONTO)
        # Pretend the last fire was recorded at 09:00 UTC (the buggy history).
        last_run = "2026-08-03T09:00:39+00:00"
        result = compute_next_run(
            {"kind": "cron", "expr": "0 9 * * *"}, last_run_at=last_run
        )
        nxt = datetime.fromisoformat(result)
        wall = nxt.astimezone(TORONTO)
        assert (wall.hour, wall.minute) == (9, 0), f"expected 09:00 Toronto, got {wall}"
        assert nxt.utcoffset() is not None, "result must carry a concrete zone offset"

    def test_spring_forward_keeps_9am(self, monkeypatch):
        monkeypatch.setattr("cron.jobs.get_timezone", lambda: TORONTO)
        # Base just before the spring-forward (Mar 8 2026, 02:00 EST -> 03:00 EDT).
        last_run = "2026-03-07T09:00:00-05:00"
        result = compute_next_run(
            {"kind": "cron", "expr": "0 9 * * *"}, last_run_at=last_run
        )
        wall = datetime.fromisoformat(result).astimezone(TORONTO)
        assert (wall.hour, wall.minute) == (9, 0), f"got {wall}"

    def test_fall_back_keeps_9am(self, monkeypatch):
        monkeypatch.setattr("cron.jobs.get_timezone", lambda: TORONTO)
        # Base just before the fall-back (Nov 1 2026, 02:00 EDT -> 01:00 EST).
        last_run = "2026-10-31T09:00:00-04:00"
        result = compute_next_run(
            {"kind": "cron", "expr": "0 9 * * *"}, last_run_at=last_run
        )
        wall = datetime.fromisoformat(result).astimezone(TORONTO)
        assert (wall.hour, wall.minute) == (9, 0), f"got {wall}"

    def test_full_year_stable_at_9am(self, monkeypatch):
        """Walking a full year (both DST transitions) must never drift off 09:00."""
        monkeypatch.setattr("cron.jobs.get_timezone", lambda: TORONTO)
        last = datetime(2026, 1, 1, 9, 0, 0, tzinfo=TORONTO)
        for _ in range(365):
            nxt = datetime.fromisoformat(
                compute_next_run(
                    {"kind": "cron", "expr": "0 9 * * *"},
                    last_run_at=last.isoformat(),
                )
            )
            wall = nxt.astimezone(TORONTO)
            assert (wall.hour, wall.minute) == (9, 0), f"drift at {last}: {wall}"
            last = nxt
