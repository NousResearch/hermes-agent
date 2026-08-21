"""Tests for :func:`hermes_cli.config.resolve_turn_limit` and the
``TURN_LIMIT_UNLIMITED`` sentinel.

Covers the full spelling table (int, float, numeric string, ``"none"``,
``"unlimited"``, ``"infinite"``, ``"-1"``, ``"0"``, YAML ``None``, bool,
garbage) and the str→int env-var round-trip that the gateway bridge relies on.
"""
import os
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
        # Default is now unlimited (max_turns caused more problems than it solved).
        assert resolve_turn_limit(None) == TURN_LIMIT_UNLIMITED

    def test_none_custom_default(self):
        assert resolve_turn_limit(None, default=500) == 500

    def test_empty_string_returns_default(self):
        assert resolve_turn_limit("") == TURN_LIMIT_UNLIMITED

    def test_whitespace_only_returns_default(self):
        assert resolve_turn_limit("   ") == TURN_LIMIT_UNLIMITED

    def test_absent_env_var_returns_default(self):
        """Simulates os.getenv() returning None when HERMES_MAX_ITERATIONS unset."""
        assert resolve_turn_limit(None) == TURN_LIMIT_UNLIMITED


class TestInvalidInputs:
    def test_bool_rejected(self):
        # bool is an int subclass — must not silently become 1/0
        assert resolve_turn_limit(True) == TURN_LIMIT_UNLIMITED
        assert resolve_turn_limit(False) == TURN_LIMIT_UNLIMITED

    def test_garbage_string_returns_default(self):
        assert resolve_turn_limit("garbage") == TURN_LIMIT_UNLIMITED
        assert resolve_turn_limit("not_a_number") == TURN_LIMIT_UNLIMITED

    def test_list_returns_default(self):
        assert resolve_turn_limit([]) == TURN_LIMIT_UNLIMITED
        assert resolve_turn_limit([90]) == TURN_LIMIT_UNLIMITED

    def test_dict_returns_default(self):
        assert resolve_turn_limit({}) == TURN_LIMIT_UNLIMITED
        assert resolve_turn_limit({"max_turns": 90}) == TURN_LIMIT_UNLIMITED


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


class TestGatewayBridgeNullHandling:
    """Verify that _bridge_max_turns_from_config does NOT serialize Python None
    (from YAML ``null`` or bare ``key:``) into the env var as the string
    "None", which would resolve to unlimited instead of the default."""

    def test_none_value_not_bridged(self, monkeypatch, tmp_path):
        """YAML ``max_turns: null`` should not set HERMES_MAX_ITERATIONS."""
        import yaml
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("agent:\n  max_turns: null\n", encoding="utf-8")
        monkeypatch.setenv("HERMES_MAX_ITERATIONS", "stale-120")
        # Import here to avoid module-level gateway dependency
        from gateway.run import _bridge_max_turns_from_config
        _bridge_max_turns_from_config(tmp_path)
        # The stale env var should be cleared so downstream resolver
        # applies the default rather than using the stale value.
        assert "HERMES_MAX_ITERATIONS" not in os.environ

    def test_string_none_bridged_correctly(self, monkeypatch, tmp_path):
        """YAML ``max_turns: "none"`` should bridge as the string "none"."""
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text('agent:\n  max_turns: "none"\n', encoding="utf-8")
        monkeypatch.delenv("HERMES_MAX_ITERATIONS", raising=False)
        from gateway.run import _bridge_max_turns_from_config
        _bridge_max_turns_from_config(tmp_path)
        assert os.environ.get("HERMES_MAX_ITERATIONS") == "none"

    def test_int_bridged_correctly(self, monkeypatch, tmp_path):
        """YAML ``max_turns: 120`` should bridge as "120"."""
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("agent:\n  max_turns: 120\n", encoding="utf-8")
        monkeypatch.delenv("HERMES_MAX_ITERATIONS", raising=False)
        from gateway.run import _bridge_max_turns_from_config
        _bridge_max_turns_from_config(tmp_path)
        assert os.environ.get("HERMES_MAX_ITERATIONS") == "120"

    def test_bare_key_treated_as_null(self, monkeypatch, tmp_path):
        """YAML ``max_turns:`` (bare key, no value) parses as Python None."""
        import yaml
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("agent:\n  max_turns:\n", encoding="utf-8")
        monkeypatch.setenv("HERMES_MAX_ITERATIONS", "stale-90")
        from gateway.run import _bridge_max_turns_from_config
        _bridge_max_turns_from_config(tmp_path)
        assert "HERMES_MAX_ITERATIONS" not in os.environ


class TestTUIResolver:
    """Verify that _cfg_max_turns in tui_gateway/server.py routes through
    resolve_turn_limit instead of bare int()."""

    def test_string_none_resolves_to_unlimited(self):
        from tui_gateway.server import _cfg_max_turns
        cfg = {"agent": {"max_turns": "none"}}
        assert _cfg_max_turns(cfg, default=25) == TURN_LIMIT_UNLIMITED

    def test_int_zero_resolves_to_unlimited(self):
        from tui_gateway.server import _cfg_max_turns
        cfg = {"agent": {"max_turns": 0}}
        assert _cfg_max_turns(cfg, default=25) == TURN_LIMIT_UNLIMITED

    def test_string_unlimited_resolves_to_unlimited(self):
        from tui_gateway.server import _cfg_max_turns
        cfg = {"agent": {"max_turns": "unlimited"}}
        assert _cfg_max_turns(cfg, default=25) == TURN_LIMIT_UNLIMITED

    def test_int_passthrough(self):
        from tui_gateway.server import _cfg_max_turns
        cfg = {"agent": {"max_turns": 50}}
        assert _cfg_max_turns(cfg, default=25) == 50

    def test_none_falls_through_to_default(self):
        from tui_gateway.server import _cfg_max_turns
        cfg = {"agent": {}}
        assert _cfg_max_turns(cfg, default=25) == 25

    def test_env_var_override(self, monkeypatch):
        from tui_gateway.server import _cfg_max_turns
        monkeypatch.setenv("HERMES_TUI_MAX_TURNS", "none")
        cfg = {"agent": {"max_turns": 50}}
        assert _cfg_max_turns(cfg, default=25) == TURN_LIMIT_UNLIMITED

    def test_env_var_zero_is_unlimited(self, monkeypatch):
        """Old code swallowed 0 via `int(0) > 0` → False. Now it routes
        through resolve_turn_limit which treats 0 as unlimited."""
        from tui_gateway.server import _cfg_max_turns
        monkeypatch.setenv("HERMES_TUI_MAX_TURNS", "0")
        cfg = {"agent": {"max_turns": 50}}
        assert _cfg_max_turns(cfg, default=25) == TURN_LIMIT_UNLIMITED

    def test_root_level_max_turns(self):
        """Legacy root-level max_turns still works."""
        from tui_gateway.server import _cfg_max_turns
        cfg = {"max_turns": "none"}
        assert _cfg_max_turns(cfg, default=25) == TURN_LIMIT_UNLIMITED
