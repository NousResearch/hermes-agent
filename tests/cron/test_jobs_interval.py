"""Recurring intervals below one minute are rejected (#55727).

``every 0m`` (or a bare ``0m``) parsed to a 0-minute interval whose next run is ``last_run + 0``,
always in the past, so the job was perpetually due and re-fired on every scheduler tick.
"""

from datetime import datetime, timedelta, UTC

import pytest

from cron.jobs import compute_next_run, parse_schedule


class TestSubMinuteIntervals:
    def test_zero_interval_rejected(self):
        # "every 0m" parsed to a 0-minute interval, whose next run computes to
        # last_run + 0 == last_run (always in the past), so the job re-fired on
        # every scheduler tick. It must be rejected.
        with pytest.raises(ValueError):
            parse_schedule("every 0m")

    def test_zero_hour_interval_rejected(self):
        with pytest.raises(ValueError):
            parse_schedule("every 0h")

    def test_zero_bare_duration_interval_rejected(self):
        # A bare duration is a RECURRING interval per the documented tool
        # contract, so "0m" reaches the same runaway as "every 0m". Both forms
        # now build through _interval_schedule, so one guard covers each.
        with pytest.raises(ValueError) as exc:
            parse_schedule("0m")
        assert "at least 1 minute" in str(exc.value)

    def test_rejection_names_the_schedule_the_user_typed(self):
        with pytest.raises(ValueError) as exc:
            parse_schedule("every 0h")
        assert "every 0h" in str(exc.value)

    def test_one_minute_interval_still_allowed(self):
        result = parse_schedule("every 1m")
        assert result["kind"] == "interval"
        assert result["minutes"] == 1

    def test_bare_duration_interval_still_allowed(self):
        result = parse_schedule("30m")
        assert result["kind"] == "interval"
        assert result["minutes"] == 30

    def test_compute_next_run_clamps_zero_interval(self):
        # A legacy/persisted job carrying a 0-minute interval must still
        # advance past last_run, not return last_run (which would be eternally
        # due and re-fire every tick).
        last = datetime(2020, 1, 1, tzinfo=UTC)
        nxt = compute_next_run(
            {"kind": "interval", "minutes": 0}, last_run_at=last.isoformat()
        )
        assert nxt is not None
        assert datetime.fromisoformat(nxt) >= last + timedelta(minutes=1)
