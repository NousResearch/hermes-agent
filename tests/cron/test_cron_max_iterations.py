"""Regression tests for cron agent max-iteration config resolution."""

from cron.scheduler import _resolve_agent_max_iterations


def test_cron_nested_zero_max_turns_is_unbounded_not_defaulted():
    assert _resolve_agent_max_iterations({"agent": {"max_turns": 0}}) == 0


def test_cron_root_zero_max_turns_is_unbounded_not_defaulted():
    assert _resolve_agent_max_iterations({"agent": {}, "max_turns": 0}) == 0


def test_cron_missing_max_turns_uses_default():
    assert _resolve_agent_max_iterations({"agent": {}}) == 90
