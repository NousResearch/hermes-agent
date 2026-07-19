from __future__ import annotations

import math

import pytest

from agent.runtime_display import (
    effective_reasoning_effort,
    format_reasoning_label,
    format_session_cost,
)


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (None, "provider-default"),
        ({}, "provider-default"),
        ({"enabled": False}, "none"),
        ({"enabled": True}, "provider-default"),
        ({"enabled": True, "effort": "minimal"}, "minimal"),
        ({"enabled": True, "effort": "low"}, "low"),
        ({"enabled": True, "effort": "medium"}, "medium"),
        ({"enabled": True, "effort": "high"}, "high"),
        ({"enabled": True, "effort": "xhigh"}, "xhigh"),
        ({"enabled": True, "effort": "max"}, "max"),
        ({"enabled": True, "effort": " HIGH "}, "high"),
        ({"enabled": True, "effort": "turbo"}, "provider-default"),
    ],
)
def test_effective_reasoning_effort_normalizes_runtime_config(config, expected):
    assert effective_reasoning_effort(config) == expected


@pytest.mark.parametrize(
    ("config", "compact", "expected"),
    [
        ({"enabled": False}, False, "reasoning none"),
        ({"enabled": True, "effort": "high"}, False, "reasoning high"),
        (None, False, "reasoning provider-default"),
        ({"enabled": False}, True, "r:none"),
        ({"enabled": True, "effort": "high"}, True, "r:high"),
        (None, True, "r:default"),
    ],
)
def test_format_reasoning_label_has_full_and_compact_forms(config, compact, expected):
    assert format_reasoning_label(config, compact=compact) == expected


@pytest.mark.parametrize(
    ("amount", "status", "compact", "expected"),
    [
        (0.001234, "actual", False, "cost $0.0012"),
        (0.1234, "actual", False, "cost $0.12"),
        (123.456, "actual", False, "cost $123.46"),
        (0.001234, "estimated", False, "cost ~$0.0012"),
        (0.1234, "estimated", True, "~$0.12"),
        (0, "included", False, "cost included"),
        (0, "included", True, "included"),
        (None, "unknown", False, "cost unavailable"),
        (None, "unknown", True, "cost n/a"),
    ],
)
def test_format_session_cost_preserves_billing_semantics(amount, status, compact, expected):
    assert format_session_cost(amount, status, compact=compact) == expected


@pytest.mark.parametrize(
    "amount",
    [None, "not-a-number", -1, math.inf, -math.inf, math.nan],
)
def test_format_session_cost_rejects_invalid_amounts(amount):
    assert format_session_cost(amount, "estimated") == "cost unavailable"


def test_format_session_cost_rejects_unknown_status_even_with_zero_amount():
    assert format_session_cost(0, "unknown") == "cost unavailable"
