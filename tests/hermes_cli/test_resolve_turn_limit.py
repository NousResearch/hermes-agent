"""Tests for :func:`hermes_cli.config.resolve_turn_limit` and the
``TURN_LIMIT_UNLIMITED`` sentinel.

Covers the full spelling table (int, float, numeric string, ``"none"``,
``"unlimited"``, ``"infinite"``, ``"-1"``, ``"0"``, YAML ``None``, bool,
garbage) and the str→int env-var round-trip that the gateway bridge relies on.
"""
import sys
import pytest

from hermes_cli.config import resolve_turn_limit, TURN_LIMIT_UNLIMITED


class TestNumericValues:
    def test_int_passthrough(self):
        assert resolve_turn_limit(90) == 90
        assert resolve_turn_limit(120) == 120
        assert resolve_turn_limit(1) == 1

    def test_float_truncated(self):
        assert resolve_turn_limit(3.7) == 3
        assert resolve_turn_limit(3.0) == 3
        assert resolve_turn_limit(100.9) == 100

    def test_numeric_string(self):
        assert resolve_turn_limit("120") == 120
        assert resolve_turn_limit("3") == 3
        assert resolve_turn_limit("3.7") == 3  # float string → int

    def test_negative_int_is_unlimited(self):
        assert resolve_turn_limit(-5) == TURN_LIMIT_UNLIMITED

    def test_negative_string_is_unlimited(self):
        assert resolve_turn_limit("-1") == TURN_LIMIT_UNLIMITED
        assert resolve_turn_limit("-42") == TURN_LIMIT_UNLIMITED


class TestUnlimitedSpellings:
    @pytest.mark.parametrize("spelling", [
        "none", "None", "NONE", "nOnE",
        "unlimited", "UNLIMITED", "Unlimited",
        "infinite", "INFINITE",
        "∞",
        "-1", "0",
    ])
    def test_string_spellings_resolve_to_sentinel(self, spelling):
        assert resolve_turn_limit(spelling) == TURN_LIMIT_UNLIMITED

    @pytest.mark.parametrize("spelling", [" none ", "  unlimited  ", "\tinfinite\t"])
    def test_whitespace_tolerant(self, spelling):
        assert resolve_turn_limit(spelling) == TURN_LIMIT_UNLIMITED

    def test_zero_int_is_unlimited(self):
        assert resolve_turn_limit(0) == TURN_LIMIT_UNLIMITED

    def test_zero_float_is_unlimited(self):
        assert resolve_turn_limit(0.0) == TURN_LIMIT_UNLIMITED


class TestAbsentAndDefault:
    def test_none_returns_default(self):
        assert resolve_turn_limit(None) == 90

    def test_none_custom_default(self):
        assert resolve_turn_limit(None, default=500) == 500

    def test_empty_string_returns_default(self):
        assert resolve_turn_limit("") == 90

    def test_whitespace_only_returns_default(self):
        assert resolve_turn_limit("   ") == 90

    def test_absent_env_var_returns_default(self):
        """Simulates os.getenv() returning None when HERMES_MAX_ITERATIONS unset."""
        assert resolve_turn_limit(None) == 90


class TestInvalidInputs:
    def test_bool_rejected(self):
        # bool is an int subclass — must not silently become 1/0
        assert resolve_turn_limit(True) == 90
        assert resolve_turn_limit(False) == 90

    def test_garbage_string_returns_default(self):
        assert resolve_turn_limit("garbage") == 90
        assert resolve_turn_limit("not_a_number") == 90

    def test_list_returns_default(self):
        assert resolve_turn_limit([]) == 90
        assert resolve_turn_limit([90]) == 90

    def test_dict_returns_default(self):
        assert resolve_turn_limit({}) == 90
        assert resolve_turn_limit({"max_turns": 90}) == 90


class TestSentinelProperties:
    def test_sentinel_is_sys_maxsize(self):
        assert TURN_LIMIT_UNLIMITED == sys.maxsize

    def test_sentinel_str_int_round_trip(self):
        """The gateway bridge writes str(value) to HERMES_MAX_ITERATIONS,
        then _current_max_iterations reads it back.  The sentinel must survive."""
        s = str(TURN_LIMIT_UNLIMITED)
        assert int(s) == TURN_LIMIT_UNLIMITED

    def test_sentinel_greater_than_any_realistic_count(self):
        assert TURN_LIMIT_UNLIMITED > 10_000_000
        assert TURN_LIMIT_UNLIMITED > 1_000_000_000


class TestEnvVarBridgeSimulation:
    """Simulates the full gateway chain: config value → str() → env var →
    resolve_turn_limit()."""

    def test_none_string_round_trip(self):
        # config has: agent.max_turns: "none"
        env_val = str("none")
        assert resolve_turn_limit(env_val) == TURN_LIMIT_UNLIMITED

    def test_yaml_null_round_trip(self):
        # YAML bare 'none' parses to Python None, str(None) = "None"
        env_val = str(None)  # "None"
        assert resolve_turn_limit(env_val) == TURN_LIMIT_UNLIMITED

    def test_int_round_trip(self):
        env_val = str(120)
        assert resolve_turn_limit(env_val) == 120

    def test_unlimited_string_round_trip(self):
        env_val = str("unlimited")
        assert resolve_turn_limit(env_val) == TURN_LIMIT_UNLIMITED
