"""Regression tests for #119219: PeriodicScheduler must reject intervals that
cannot represent a valid periodic delay (zero, negative, or non-finite)."""

import math

import pytest

from agent.periodic_scheduler import PeriodicScheduler, schedule


@pytest.mark.parametrize("interval", [0, 0.0, -1, -0.5, math.nan, math.inf, -math.inf])
def test_scheduler_rejects_invalid_interval(interval):
    scheduler = PeriodicScheduler()
    with pytest.raises(ValueError):
        scheduler.schedule(lambda: True, interval)


def test_module_level_schedule_rejects_invalid_interval():
    with pytest.raises(ValueError):
        schedule(lambda: True, 0)


def test_scheduler_accepts_positive_interval():
    scheduler = PeriodicScheduler()
    handle = scheduler.schedule(lambda: False, 3600.0)
    try:
        assert not handle.cancelled
    finally:
        handle.cancel(wait=1)
