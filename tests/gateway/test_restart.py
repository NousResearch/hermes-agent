"""Tests for shared gateway restart parsing helpers."""

import math

import pytest

from gateway.restart import (
    DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT,
    parse_restart_drain_timeout,
)


def test_parse_restart_drain_timeout_uses_default_for_empty_or_invalid_values():
    assert parse_restart_drain_timeout("") == DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT
    assert parse_restart_drain_timeout(None) == DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT
    assert parse_restart_drain_timeout("not-a-number") == DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT


def test_parse_restart_drain_timeout_clamps_negative_values_to_zero():
    assert parse_restart_drain_timeout("-1") == 0.0
    assert parse_restart_drain_timeout(-5) == 0.0


@pytest.mark.parametrize("raw", ["inf", "Infinity", float("inf"), float("-inf"), float("nan")])
def test_parse_restart_drain_timeout_rejects_non_finite_values(raw):
    value = parse_restart_drain_timeout(raw)

    assert value == DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT
    assert math.isfinite(value)
