"""Tests for gateway.restart — parse_restart_drain_timeout and constants."""

from __future__ import annotations

import pytest

from gateway.restart import (
    DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT,
    GATEWAY_SERVICE_RESTART_EXIT_CODE,
    parse_restart_drain_timeout,
)


# ============================================================================
# parse_restart_drain_timeout
# ============================================================================
class TestParseRestartDrainTimeout:
    def test_positive_float(self):
        assert parse_restart_drain_timeout(30.0) == 30.0

    def test_positive_int(self):
        assert parse_restart_drain_timeout(30) == 30.0

    def test_string_number(self):
        assert parse_restart_drain_timeout("45.5") == 45.5

    def test_negative_clamped_to_zero(self):
        assert parse_restart_drain_timeout(-10.0) == 0.0

    def test_none_returns_default(self):
        assert parse_restart_drain_timeout(None) == DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT

    def test_empty_string_returns_default(self):
        assert parse_restart_drain_timeout("") == DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT

    def test_whitespace_string_returns_default(self):
        assert parse_restart_drain_timeout("   ") == DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT

    def test_invalid_string_returns_default(self):
        assert parse_restart_drain_timeout("not-a-number") == DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT

    def test_list_returns_default(self):
        """Non-numeric type → TypeError → default."""
        assert parse_restart_drain_timeout([1, 2, 3]) == DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT

    def test_dict_returns_default(self):
        assert parse_restart_drain_timeout({"a": 1}) == DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT

    def test_zero_int_returns_default(self):
        """0 is falsy in Python, so `0 or ''` → '' → default."""
        assert parse_restart_drain_timeout(0) == DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT

    def test_zero_float_returns_default(self):
        """0.0 is falsy → same as 0."""
        assert parse_restart_drain_timeout(0.0) == DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT

    def test_very_large_value(self):
        assert parse_restart_drain_timeout(1_000_000.0) == 1_000_000.0

    def test_small_positive(self):
        assert parse_restart_drain_timeout(0.001) == 0.001

    def test_boolean_true(self):
        """bool is a subclass of int — True=1 → float(True)=1.0."""
        assert parse_restart_drain_timeout(True) == 1.0

    def test_boolean_false(self):
        """False=0 → falsy → default."""
        assert parse_restart_drain_timeout(False) == DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT

    def test_string_zero(self):
        """'0' is a truthy string → float('0')=0.0 → max(0,0)=0.0."""
        assert parse_restart_drain_timeout("0") == 0.0

    def test_string_negative(self):
        """'-5'→ float('-5')=-5.0 → max(0,-5)=0.0."""
        assert parse_restart_drain_timeout("-5") == 0.0


# ============================================================================
# Constants
# ============================================================================
class TestConstants:
    def test_exit_code_is_75(self):
        assert GATEWAY_SERVICE_RESTART_EXIT_CODE == 75

    def test_exit_code_is_int(self):
        assert isinstance(GATEWAY_SERVICE_RESTART_EXIT_CODE, int)

    def test_default_timeout_is_positive(self):
        assert DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT > 0

    def test_default_timeout_is_float(self):
        assert isinstance(DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT, float)
