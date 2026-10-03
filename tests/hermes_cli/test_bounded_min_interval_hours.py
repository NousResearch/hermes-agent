"""``sessions.min_interval_hours`` / ``checkpoints.min_interval_hours`` must floor at 1.

``maybe_auto_archive`` / ``maybe_auto_prune_and_vacuum`` / ``maybe_auto_prune_checkpoints`` gate
their sweep on ``now - last_run < min_interval_hours * 3600``. A value <= 0 makes that comparison
false on every call — the sweep then runs on every housekeeping tick instead of at most once per
interval, the same class of bug ``agent.curator._bounded_count`` already guards against for
``curator.interval_hours`` / ``curator.stale_after_days`` / ``curator.archive_after_days``.
"""

import logging

import pytest

import hermes_cli.config as config
from hermes_cli.config import bounded_min_interval_hours


@pytest.fixture(autouse=True)
def _reset_warned_values():
    """The warn-once set is module-level state shared across tests in this file."""
    config._warned_bad_min_interval_hours.clear()


@pytest.mark.parametrize("bad_value", [0, -1, -24])
def test_non_positive_value_falls_back_to_default(bad_value):
    assert bounded_min_interval_hours(bad_value) == 24


@pytest.mark.parametrize("bad_value", [None, "not-a-number", object()])
def test_non_numeric_value_falls_back_to_default(bad_value):
    assert bounded_min_interval_hours(bad_value) == 24


def test_valid_value_passes_through():
    assert bounded_min_interval_hours(1) == 1
    assert bounded_min_interval_hours(6) == 6
    assert bounded_min_interval_hours("12") == 12


def test_custom_default_is_honoured():
    assert bounded_min_interval_hours(0, default=7) == 7
    assert bounded_min_interval_hours(None, default=7) == 7


def test_bad_value_warns_once_per_distinct_value(caplog):
    with caplog.at_level(logging.WARNING, logger="hermes_cli.config"):
        for _ in range(3):
            bounded_min_interval_hours(0)
        bounded_min_interval_hours(-5)
        bounded_min_interval_hours(-5)
    msgs = [r.getMessage() for r in caplog.records if "min_interval_hours" in r.getMessage()]
    assert len(msgs) == 2, msgs
    assert "got 0" in msgs[0] and "got -5" in msgs[1]
