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


# ---------------------------------------------------------------------------
# Fix 5 — _disabled re-evaluated when config reloads in on_session_start
# ---------------------------------------------------------------------------

class TestDisabledFlagOnConfigReload:
    """on_session_start must re-evaluate _disabled after reloading config.

    Previously only tau_soft/tau_hard/protect_last_n were propagated;
    _disabled was only set in __init__, so changing lcm.enabled in config.yaml
    had no effect on a running session.
    """

    def test_enabled_to_disabled_via_on_session_start(self):
        """Engine inited with enabled=True, config reloads enabled=False -> _disabled=True."""
        engine = LcmContextEngine(lcm_config={"enabled": True})
        assert engine._disabled is False, "precondition: engine starts enabled"

        fake_cfg = {"lcm": {"enabled": "false"}}
        with patch("hermes_cli.config.load_config", return_value=fake_cfg):
            engine.on_session_start("s-disable-test")

        assert engine._disabled is True, (
            f"_disabled={engine._disabled!r} after reloading enabled=False; "
            "expected True — _disabled was not re-evaluated from the reloaded config"
        )

    def test_compress_passthrough_when_disabled_via_reload(self):
        """After reload sets _disabled=True, compress() must be a no-op pass-through."""
        engine = LcmContextEngine(lcm_config={"enabled": True})

        fake_cfg = {"lcm": {"enabled": "false"}}
        with patch("hermes_cli.config.load_config", return_value=fake_cfg):
            engine.on_session_start("s-compress-passthrough")

        msg = {"role": "user", "content": "hello"}
        result = engine.compress([msg])
        assert result == [msg], (
            f"compress() did not pass through unchanged after _disabled=True; got: {result!r}"
        )

    def test_disabled_to_enabled_via_on_session_start(self):
        """Engine inited with enabled=False, config reloads enabled=True -> _disabled=False."""
        engine = LcmContextEngine(lcm_config={"enabled": False})
        assert engine._disabled is True, "precondition: engine starts disabled"

        fake_cfg = {"lcm": {"enabled": "true"}}
        with patch("hermes_cli.config.load_config", return_value=fake_cfg):
            engine.on_session_start("s-enable-test")

        assert engine._disabled is False, (
            f"_disabled={engine._disabled!r} after reloading enabled=True; "
            "expected False"
        )


# ---------------------------------------------------------------------------
# Fix 6 — lcm_config passed to constructor so disabled mode suppresses tools
#          at registration time (before on_session_start is ever called)
# ---------------------------------------------------------------------------

class TestDisabledAtConstructionTimeViaLcmConfig:
    """lcm_config passed to LcmContextEngine at construction time must suppress
    tool registration immediately — no on_session_start() call required.

    Regression: previously the engine was constructed with default kwargs (enabled=True),
    then on_session_start() would re-evaluate _disabled.  This meant get_tool_schemas()
    returned non-empty tools between construction and the first on_session_start call.
    """

    def test_get_tool_schemas_empty_before_session_start(self):
        """Tools must be suppressed purely from constructor kwargs, before any session."""
        engine = LcmContextEngine(lcm_config={"enabled": False})
        # Do NOT call on_session_start — that was the previous dependency
        schemas = engine.get_tool_schemas()
        assert schemas == [], (
            f"get_tool_schemas() returned {len(schemas)} schemas before any "
            "on_session_start() call; disabled mode must suppress tools at construction"
        )

    def test_disabled_flag_set_at_construction(self):
        """_disabled=True must be set immediately in __init__ from lcm_config."""
        engine = LcmContextEngine(lcm_config={"enabled": False})
        assert engine._disabled is True, (
            "_disabled should be True immediately after construction with enabled=False"
        )

    def test_load_context_engine_with_lcm_config_disabled(self):
        """load_context_engine('lcm', lcm_config={'enabled': False}) must return
        an engine whose get_tool_schemas() is empty without any session lifecycle call."""
        from plugins.context_engine import load_context_engine
        engine = load_context_engine("lcm", lcm_config={"enabled": False})
        assert engine is not None, "load_context_engine should return an engine instance"
        schemas = engine.get_tool_schemas()
        assert schemas == [], (
            f"load_context_engine with enabled=False returned {len(schemas)} schemas; "
            "expected [] — lcm_config must be forwarded through the loader to the constructor"
        )

    def test_load_context_engine_with_lcm_config_enabled(self):
        """Sanity: load_context_engine('lcm', lcm_config={'enabled': True}) must expose tools."""
        from plugins.context_engine import load_context_engine
        engine = load_context_engine("lcm", lcm_config={"enabled": True})
        assert engine is not None
        schemas = engine.get_tool_schemas()
        assert len(schemas) > 0, (
            "load_context_engine with enabled=True returned no schemas; "
            "enabled engines must expose their tools"
        )


# ---------------------------------------------------------------------------
# Fix: summarizer model must be re-synced when config reloads in on_session_start
# ---------------------------------------------------------------------------

class TestSummarizerModelPropagation:
    """After on_session_start reloads config, the live Summarizer must pick up
    the new lcm.summary_model value instead of keeping its constructor-time model.
    """

    def test_summary_model_propagated_on_session_start(self):
        """engine._engine.summarizer.config.model reflects the reloaded summary_model."""
        engine = LcmContextEngine()
        # Confirm the summarizer starts with no forced model (empty string default)
        assert engine._engine.summarizer.config.model != "custom-model-x"

        fake_cfg = {"lcm": {"summary_model": "custom-model-x"}}
        with patch("hermes_cli.config.load_config", return_value=fake_cfg):
            engine.on_session_start("s1")

        assert engine._engine.summarizer.config.model == "custom-model-x", (
            f"summarizer.config.model={engine._engine.summarizer.config.model!r} "
            "but expected 'custom-model-x' — summary_model not re-synced after config reload"
        )

    def test_summary_model_not_overwritten_when_not_set(self):
        """When config reloads without summary_model, the summarizer keeps its current model."""
        engine = LcmContextEngine()
        # Manually set a model to verify it isn't wiped by a config without summary_model
        engine._engine.summarizer.config.model = "pre-set-model"

        fake_cfg = {"lcm": {"tau_soft": "0.55"}}  # no summary_model key
        with patch("hermes_cli.config.load_config", return_value=fake_cfg):
            engine.on_session_start("s2")

        # summary_model is falsy (empty string) so the branch must NOT overwrite
        assert engine._engine.summarizer.config.model == "pre-set-model", (
            "summarizer.config.model was overwritten despite no summary_model in config"
        )


# ---------------------------------------------------------------------------
# Fix: runtime enable→disable transition logs a warning about tool retraction
# ---------------------------------------------------------------------------

class TestRuntimeDisableTransitionWarning:
    """When on_session_start detects enabled→disabled, a warning must be logged.

    The agent tool list cannot be retracted at runtime (it was populated once
    at startup).  The warning tells the user to restart for the change to take
    effect.  All LCM tool calls already return {"error": "LCM is disabled"} so
    the behaviour is safe; this is purely a visibility fix.
    """

    def test_warning_logged_on_enabled_to_disabled_transition(self, caplog):
        """A warning containing 'restart' must be emitted when transitioning
        from enabled to disabled via config reload."""
        import logging
        engine = LcmContextEngine(lcm_config={"enabled": True})
        assert engine._disabled is False, "precondition: engine starts enabled"

        fake_cfg = {"lcm": {"enabled": "false"}}
        with patch("hermes_cli.config.load_config", return_value=fake_cfg):
            with caplog.at_level(logging.WARNING, logger="plugins.context_engine.lcm"):
                engine.on_session_start("s-warn-test")

        assert engine._disabled is True, "precondition: engine is now disabled"
        assert any(
            "restart" in record.message.lower()
            for record in caplog.records
        ), (
            "Expected a warning containing 'restart' when LCM transitions "
            f"enabled→disabled at runtime; got: {[r.message for r in caplog.records]!r}"
        )

    def test_no_warning_when_already_disabled(self, caplog):
        """No spurious warning when the engine was already disabled before reload."""
        import logging
        engine = LcmContextEngine(lcm_config={"enabled": False})
        assert engine._disabled is True

        fake_cfg = {"lcm": {"enabled": "false"}}
        with patch("hermes_cli.config.load_config", return_value=fake_cfg):
            with caplog.at_level(logging.WARNING, logger="plugins.context_engine.lcm"):
                engine.on_session_start("s-no-warn-test")

        restart_warnings = [
            r for r in caplog.records if "restart" in r.message.lower()
        ]
        assert restart_warnings == [], (
            "Unexpected 'restart' warning when engine was already disabled: "
            f"{[r.message for r in restart_warnings]!r}"
        )

    def test_no_warning_on_disable_to_enable_transition(self, caplog):
        """Enabling an already-disabled engine must not emit a 'restart' warning."""
        import logging
        engine = LcmContextEngine(lcm_config={"enabled": False})

        fake_cfg = {"lcm": {"enabled": "true"}}
        with patch("hermes_cli.config.load_config", return_value=fake_cfg):
            with caplog.at_level(logging.WARNING, logger="plugins.context_engine.lcm"):
                engine.on_session_start("s-enable-no-warn")

        restart_warnings = [
            r for r in caplog.records if "restart" in r.message.lower()
        ]
        assert restart_warnings == [], (
            "Unexpected 'restart' warning on disabled→enabled transition: "
            f"{[r.message for r in restart_warnings]!r}"
        )
