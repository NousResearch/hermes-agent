"""Regression test: config loaded in on_session_start must reach the live LcmEngine.

Fix: after rebuilding self._config in on_session_start, propagate it to
self._engine.config so compaction reads the user-supplied thresholds, not the
constructor-time defaults.

Fix 2: lcm.enabled=false must be honoured — no tools exposed, compress() is a
pass-through, should_compress() always returns False.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from plugins.context_engine.lcm.__init__ import LcmContextEngine
from plugins.context_engine.lcm.config import LcmConfig


class TestConfigPropagationOnSessionStart:
    """on_session_start must write the reloaded config back to the live engine."""

    def test_tau_soft_propagated_to_engine(self):
        """engine._engine.config.tau_soft reflects the value from config.yaml."""
        default_config = LcmConfig()
        assert default_config.tau_soft != 0.42, "pick a non-default value for this test"

        engine = LcmContextEngine()

        fake_cfg = {"lcm": {"tau_soft": "0.42"}}
        with patch(
            "hermes_cli.config.load_config",
            return_value=fake_cfg,
        ):
            engine.on_session_start("test-session-cfg1")

        assert engine._engine.config.tau_soft == pytest.approx(0.42), (
            f"engine._engine.config.tau_soft={engine._engine.config.tau_soft!r} "
            "but expected 0.42 — config was not propagated to the live engine"
        )

    def test_tau_hard_propagated_to_engine(self):
        """engine._engine.config.tau_hard reflects the value from config.yaml."""
        engine = LcmContextEngine()
        fake_cfg = {"lcm": {"tau_hard": "0.95"}}
        with patch(
            "hermes_cli.config.load_config",
            return_value=fake_cfg,
        ):
            engine.on_session_start("test-session-cfg2")

        assert engine._engine.config.tau_hard == pytest.approx(0.95), (
            f"engine._engine.config.tau_hard={engine._engine.config.tau_hard!r} "
            "but expected 0.95"
        )

    def test_protect_last_n_propagated_to_engine(self):
        """engine._engine.config.protect_last_n reflects config.yaml."""
        engine = LcmContextEngine()
        fake_cfg = {"lcm": {"protect_last_n": "10"}}
        with patch(
            "hermes_cli.config.load_config",
            return_value=fake_cfg,
        ):
            engine.on_session_start("test-session-cfg3")

        assert engine._engine.config.protect_last_n == 10, (
            f"engine._engine.config.protect_last_n={engine._engine.config.protect_last_n!r} "
            "but expected 10"
        )

    def test_wrapper_attributes_also_updated(self):
        """threshold_percent and protect_last_n on the wrapper also sync."""
        engine = LcmContextEngine()
        fake_cfg = {"lcm": {"tau_soft": "0.60", "protect_last_n": "8"}}
        with patch(
            "hermes_cli.config.load_config",
            return_value=fake_cfg,
        ):
            engine.on_session_start("test-session-cfg4")

        assert engine.threshold_percent == pytest.approx(0.60)
        assert engine.protect_last_n == 8

    def test_no_config_does_not_crash(self):
        """If load_config raises, on_session_start must not crash."""
        engine = LcmContextEngine()
        with patch(
            "hermes_cli.config.load_config",
            side_effect=RuntimeError("no config"),
        ):
            engine.on_session_start("test-session-no-cfg")
        # Engine should still be usable
        assert engine._engine is not None


class TestEnabledFalseDisablesEngine:
    """lcm.enabled=false must suppress tool surface, pass-through compress,
    and prevent should_compress() from ever returning True."""

    def test_get_tool_schemas_returns_empty_when_disabled(self):
        """No LCM or memory tools exposed when enabled=False."""
        engine = LcmContextEngine(lcm_config={"enabled": False})
        schemas = engine.get_tool_schemas()
        assert schemas == [], (
            f"get_tool_schemas() returned {len(schemas)} schemas; expected [] when disabled"
        )

    def test_compress_is_passthrough_when_disabled(self):
        """compress() must return the input messages unchanged when enabled=False."""
        engine = LcmContextEngine(lcm_config={"enabled": False})
        msgs = [
            {"role": "user", "content": f"msg {i}"}
            for i in range(5)
        ]
        result = engine.compress(msgs)
        assert result == msgs, (
            f"compress() changed messages when disabled: got {result!r}"
        )

    def test_compress_passthrough_for_large_context(self):
        """compress() still passes through even at very high token counts."""
        engine = LcmContextEngine(lcm_config={"enabled": False})
        msgs = [{"role": "user", "content": "x" * 100} for _ in range(200)]
        result = engine.compress(msgs)
        assert result == msgs

    def test_should_compress_false_when_disabled(self):
        """should_compress() always returns False when enabled=False."""
        engine = LcmContextEngine(lcm_config={"enabled": False})
        # Even at 10,000,000 tokens — far above any threshold
        assert engine.should_compress(prompt_tokens=10_000_000) is False, (
            "should_compress() returned True despite enabled=False"
        )

    def test_handle_tool_call_returns_error_when_disabled(self):
        """handle_tool_call() must return an error JSON when enabled=False."""
        import json as _json
        engine = LcmContextEngine(lcm_config={"enabled": False})
        result = engine.handle_tool_call("lcm_search", {"query": "test"})
        parsed = _json.loads(result)
        assert "error" in parsed, (
            f"handle_tool_call() did not return an error dict when disabled: {parsed!r}"
        )

    def test_enabled_true_still_works_normally(self):
        """Sanity: enabled=True (default) must still expose tools."""
        engine = LcmContextEngine(lcm_config={"enabled": True})
        schemas = engine.get_tool_schemas()
        assert len(schemas) > 0, "get_tool_schemas() returned [] for enabled=True engine"
