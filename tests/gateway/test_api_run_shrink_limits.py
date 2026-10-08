"""Request-local /v1/runs limits only shrink the profile that was already resolved."""

import pytest

from gateway.platforms.api_server_run_limits import (
    parse_shrink_only_limits, shrink_iterations, shrink_run_budget, shrink_toolsets)


def test_absent_policy_is_empty():
    assert parse_shrink_only_limits({}) == {}
    assert parse_shrink_only_limits({"owner": {"type": "a2a"}}) == {}


def test_unknown_field_is_rejected():
    with pytest.raises(ValueError, match="Unsupported"):
        parse_shrink_only_limits({"execution_policy": {"max_turns": 2, "model": "x"}})


def test_bool_and_widen_bounds_are_rejected():
    with pytest.raises(ValueError):
        parse_shrink_only_limits({"execution_policy": {"max_turns": True}})
    with pytest.raises(ValueError):
        parse_shrink_only_limits({"execution_policy": {"run_budget_seconds": 10801}})


def test_shrink_does_not_widen_turns_toolsets_or_budget():
    limits = parse_shrink_only_limits({
        "execution_policy": {
            "max_turns": 4,
            "run_budget_seconds": 30,
            "toolsets": ["web", "terminal", "not-enabled"],
        }
    })
    assert shrink_iterations(20, limits) == 4
    assert shrink_iterations(3, limits) == 3
    assert shrink_toolsets(["web", "file", "memory"], limits) == ["web"]
    assert shrink_run_budget(100.0, limits) == 30.0
    assert shrink_run_budget(10.0, limits) == 10.0
    assert shrink_run_budget(None, limits) == 30.0


def test_missing_keys_leave_profile_values():
    assert shrink_iterations(20, {}) == 20
    assert shrink_toolsets(["web"], {}) == ["web"]
    assert shrink_run_budget(100.0, {}) is None
