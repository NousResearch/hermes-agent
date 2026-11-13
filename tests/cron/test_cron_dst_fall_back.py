"""DST fall-back: a cron expression must not skip the repeated hour (#147553).

At the autumn fall-back the wall clock repeats an hour (America/Los_Angeles: 02:00 PDT -> 01:00
PST). croniter matches against the configured zone's *naive* wall clock, so from a base late in
the first occurrence (01:55 PDT) the next wall-clock match is 02:00 — an instant 65 minutes later.
The entire second pass through 01:00–01:59 is behind the base on the wall clock and was never a
candidate, so a `*/5` health job went quiet for just over an hour, once a year.
"""

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from cron import jobs

LA = ZoneInfo("America/Los_Angeles")


@pytest.fixture(autouse=True)
def _la_zone(monkeypatch):
    """Configure the profile zone as America/Los_Angeles (the reported case)."""
    monkeypatch.setattr(jobs, "get_timezone", lambda: LA)


def _next(expr: str, last_run: str) -> str:
    return jobs.compute_next_run({"kind": "cron", "expr": expr}, last_run)


def test_step_expression_runs_in_the_repeated_hour():
    # 01:55 PDT is the last 5-minute slot of the first pass; the clock then repeats 01:00-01:59 PST.
    assert _next("*/5 * * * *", "2026-11-01T01:55:16-07:00") == "2026-11-01T01:00:00-08:00"


def test_hourly_expression_runs_in_the_repeated_hour():
    # 0 * * * * from 01:50 PDT: the 01:00 PST fire is 10 minutes away, not the 02:00 PST one.
    assert _next("0 * * * *", "2026-11-01T01:50:00-07:00") == "2026-11-01T01:00:00-08:00"


def test_quarter_hour_expression_runs_in_the_repeated_hour():
    assert _next("*/15 * * * *", "2026-11-01T01:55:00-07:00") == "2026-11-01T01:00:00-08:00"


def test_a_match_earlier_in_the_first_pass_still_wins():
    # 01:45 PDT comes before the repeat starts, so it stays the next run.
    assert _next("*/15 * * * *", "2026-11-01T01:40:00-07:00") == "2026-11-01T01:45:00-07:00"


def test_a_daily_expression_is_unaffected():
    assert _next("0 9 * * *", "2026-11-01T01:55:00-07:00") == "2026-11-01T09:00:00-08:00"


def test_a_base_inside_the_second_pass_still_moves_forward():
    # Base already carries the new offset (fold=1): no repeat to rescan, just the next slot.
    assert _next("*/15 * * * *", "2026-11-01T01:10:00-08:00") == "2026-11-01T01:15:00-08:00"


def test_an_ordinary_day_is_untouched():
    assert _next("*/5 * * * *", "2026-11-02T01:55:16-08:00") == "2026-11-02T02:00:00-08:00"


def test_a_far_off_match_is_not_pulled_back_a_year():
    # Feb 29 is months away and crosses a fall-back in between; the scan must stay bounded.
    assert _next("0 0 29 2 *", "2026-03-01T09:00:00-08:00").startswith("2028-02-29T00:00:00")


def test_the_returned_instant_still_parses_back_to_itself():
    iso = _next("*/5 * * * *", "2026-11-01T01:55:16-07:00")
    parsed = datetime.fromisoformat(iso)
    assert parsed.timestamp() == datetime.fromisoformat("2026-11-01T01:00:00-08:00").timestamp()
