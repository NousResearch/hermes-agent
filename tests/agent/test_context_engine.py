"""Tests for the ContextEngine ABC and plugin slot."""

import json
import pytest
from typing import Any, Dict, List

from agent.context_engine import ContextEngine
from agent.context_compressor import ContextCompressor


# ---------------------------------------------------------------------------
# A minimal concrete engine for testing the ABC
# ---------------------------------------------------------------------------

class StubEngine(ContextEngine):
    """Minimal engine that satisfies the ABC without doing real work."""

    def __init__(self, context_length=200000, threshold_pct=0.50):
        self.context_length = context_length
        self.threshold_tokens = int(context_length * threshold_pct)
        self._compress_called = False
        self._tools_called = []

    @property
    def name(self) -> str:
        return "stub"

    def update_from_response(self, usage: Dict[str, Any]) -> None:
        self.last_prompt_tokens = usage.get("prompt_tokens", 0)
        self.last_completion_tokens = usage.get("completion_tokens", 0)
        self.last_total_tokens = usage.get("total_tokens", 0)

    def should_compress(self, prompt_tokens: int = None) -> bool:
        tokens = prompt_tokens if prompt_tokens is not None else self.last_prompt_tokens
        return tokens >= self.threshold_tokens

    def compress(self, messages: List[Dict[str, Any]], current_tokens: int = None) -> List[Dict[str, Any]]:
        self._compress_called = True
        self.compression_count += 1
        # Trivial: just return as-is
        return messages

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "stub_search",
                "description": "Search the stub engine",
                "parameters": {"type": "object", "properties": {}},
            }
        ]

    def handle_tool_call(self, name: str, args: Dict[str, Any]) -> str:
        self._tools_called.append(name)
        return json.dumps({"ok": True, "tool": name})


# ---------------------------------------------------------------------------
# ABC contract tests
# ---------------------------------------------------------------------------

class TestContextEngineABC:
    """Verify the ABC enforces the required interface."""

    def test_cannot_instantiate_abc_directly(self):
        with pytest.raises(TypeError):
            ContextEngine()

    def test_missing_methods_raises(self):
        """A subclass missing required methods cannot be instantiated."""
        class Incomplete(ContextEngine):
            @property
            def name(self):
                return "incomplete"
        with pytest.raises(TypeError):
            Incomplete()

    def test_stub_engine_satisfies_abc(self):
        engine = StubEngine()
        assert isinstance(engine, ContextEngine)
        assert engine.name == "stub"

    def test_compressor_is_context_engine(self):
        c = ContextCompressor(model="test", quiet_mode=True, config_context_length=200000)
        assert isinstance(c, ContextEngine)
        assert c.name == "compressor"


# ---------------------------------------------------------------------------
# Default method behavior
# ---------------------------------------------------------------------------

class TestDefaults:
    """Verify ABC default implementations work correctly."""

    def test_default_tool_schemas_empty(self):
        engine = StubEngine()
        # StubEngine overrides this, so test the base via super
        assert ContextEngine.get_tool_schemas(engine) == []

    def test_default_handle_tool_call_returns_error(self):
        engine = StubEngine()
        result = ContextEngine.handle_tool_call(engine, "unknown", {})
        data = json.loads(result)
        assert "error" in data

    def test_default_get_status(self):
        engine = StubEngine()
        engine.last_prompt_tokens = 50000
        status = engine.get_status()
        assert status["last_prompt_tokens"] == 50000
        assert status["context_length"] == 200000
        assert status["threshold_tokens"] == 100000
        assert 0 < status["usage_percent"] <= 100

    def test_on_session_reset(self):
        engine = StubEngine()
        engine.last_prompt_tokens = 999
        engine.compression_count = 3
        engine.on_session_reset()
        assert engine.last_prompt_tokens == 0
        assert engine.compression_count == 0

    def test_should_compress_preflight_default_false(self):
        engine = StubEngine()
        assert engine.should_compress_preflight([]) is False


# ---------------------------------------------------------------------------
# StubEngine behavior
# ---------------------------------------------------------------------------

class TestStubEngine:

    def test_should_compress(self):
        engine = StubEngine(context_length=100000, threshold_pct=0.50)
        assert not engine.should_compress(40000)
        assert engine.should_compress(50000)
        assert engine.should_compress(60000)

    def test_compress_tracks_count(self):
        engine = StubEngine()
        msgs = [{"role": "user", "content": "hello"}]
        result = engine.compress(msgs)
        assert result == msgs
        assert engine._compress_called
        assert engine.compression_count == 1

    def test_tool_schemas(self):
        engine = StubEngine()
        schemas = engine.get_tool_schemas()
        assert len(schemas) == 1
        assert schemas[0]["name"] == "stub_search"

    def test_handle_tool_call(self):
        engine = StubEngine()
        result = engine.handle_tool_call("stub_search", {})
        assert json.loads(result)["ok"] is True
        assert "stub_search" in engine._tools_called

    def test_update_from_response(self):
        engine = StubEngine()
        engine.update_from_response({"prompt_tokens": 1000, "completion_tokens": 200, "total_tokens": 1200})
        assert engine.last_prompt_tokens == 1000
        assert engine.last_completion_tokens == 200


# ---------------------------------------------------------------------------
# ContextCompressor session reset via ABC
# ---------------------------------------------------------------------------

class TestCompressorSessionReset:
    """Verify ContextCompressor.on_session_reset() clears all state."""

    def test_reset_clears_state(self):
        c = ContextCompressor(model="test", quiet_mode=True, config_context_length=200000)
        c.last_prompt_tokens = 50000
        c.compression_count = 3
        c._previous_summary = "some old summary"
        c._context_probed = True
        c._context_probe_persistable = True

        c.on_session_reset()

        assert c.last_prompt_tokens == 0
        assert c.last_completion_tokens == 0
        assert c.last_total_tokens == 0
        assert c.compression_count == 0
        assert c._context_probed is False
        assert c._context_probe_persistable is False
        assert c._previous_summary is None


# ---------------------------------------------------------------------------
# Plugin slot (PluginManager integration)
# ---------------------------------------------------------------------------

class TestPluginContextEngineSlot:
    """Test register_context_engine on PluginContext."""

    def test_register_engine(self):
        from hermes_cli.plugins import PluginManager, PluginContext, PluginManifest
        mgr = PluginManager()
        manifest = PluginManifest(name="test-lcm")
        ctx = PluginContext(manifest, mgr)

        engine = StubEngine()
        ctx.register_context_engine(engine)

        assert mgr._context_engine is engine
        assert mgr._context_engine.name == "stub"

    def test_reject_second_engine(self):
        from hermes_cli.plugins import PluginManager, PluginContext, PluginManifest
        mgr = PluginManager()
        manifest = PluginManifest(name="test-lcm")
        ctx = PluginContext(manifest, mgr)

        engine1 = StubEngine()
        engine2 = StubEngine()
        ctx.register_context_engine(engine1)
        ctx.register_context_engine(engine2)  # should be rejected

        assert mgr._context_engine is engine1

    def test_reject_non_engine(self):
        from hermes_cli.plugins import PluginManager, PluginContext, PluginManifest
        mgr = PluginManager()
        manifest = PluginManifest(name="test-bad")
        ctx = PluginContext(manifest, mgr)

        ctx.register_context_engine("not an engine")
        assert mgr._context_engine is None

    def test_get_plugin_context_engine(self):
        from hermes_cli.plugins import PluginManager, PluginContext, PluginManifest, get_plugin_context_engine, _plugin_manager
        import hermes_cli.plugins as plugins_mod

        # Inject a test manager
        old_mgr = plugins_mod._plugin_manager
        try:
            mgr = PluginManager()
            plugins_mod._plugin_manager = mgr

            assert get_plugin_context_engine() is None

            engine = StubEngine()
            mgr._context_engine = engine
            assert get_plugin_context_engine() is engine
        finally:
            plugins_mod._plugin_manager = old_mgr
