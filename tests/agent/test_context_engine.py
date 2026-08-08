"""Tests for the ContextEngine ABC and plugin slot."""

import json
import pytest
from pathlib import Path
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

    def update_model(self, model="", context_length=0, base_url="", api_key="",
                     provider="", api_mode="", **kwargs) -> None:
        """Mirror ContextCompressor.update_model — recompute threshold from the
        new context_length. This is the mutation that corrupted the shared
        singleton in #42449."""
        self.context_length = context_length
        self.threshold_tokens = int(context_length * 0.20)

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



# ---------------------------------------------------------------------------
# Default method behavior
# ---------------------------------------------------------------------------

class TestDefaults:
    """Verify ABC default implementations work correctly."""



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



# ---------------------------------------------------------------------------
# StubEngine behavior
# ---------------------------------------------------------------------------

class TestStubEngine:



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



    def test_get_plugin_context_engine(self):
        from hermes_cli.plugins import PluginManager, get_plugin_context_engine
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


class _RuntimeEngine(StubEngine):
    """Engine whose factory gives each agent independent mutable state."""

    def __init__(self, state=None):
        super().__init__()
        self.state = [] if state is None else state
        self.closed = 0

    def create_runtime(self):
        return type(self)()

    def close(self):
        if self.closed == 0:
            self.closed = 1


def _write_engine_plugin(tmp_path: Path, body: str, name: str = "test_runtime") -> Path:
    engine_dir = tmp_path / name
    engine_dir.mkdir()
    (engine_dir / "__init__.py").write_text(body)
    return engine_dir


def test_context_engine_factory_isolates_mutable_runtime_state(tmp_path, monkeypatch):
    """The loader caches registration but creates an isolated runtime per load."""
    import plugins.context_engine as loader

    engine_dir = _write_engine_plugin(
        tmp_path,
        """
from agent.context_engine import ContextEngine

class Engine(ContextEngine):
    @property
    def name(self): return 'test_runtime'
    def update_from_response(self, usage): pass
    def should_compress(self, prompt_tokens=None): return False
    def compress(self, messages, current_tokens=None): return messages
    def create_runtime(self):
        runtime = type(self)()
        runtime.state = []
        return runtime
    def __init__(self): self.state = []

def register(ctx): ctx.register_context_engine(Engine())
""",
    )
    monkeypatch.setattr(loader, "_CONTEXT_ENGINE_PLUGINS_DIR", tmp_path)
    loader._ENGINE_PROTOTYPES.clear()

    first = loader.load_context_engine(engine_dir.name)
    second = loader.load_context_engine(engine_dir.name)
    assert first is not None
    assert second is not None
    assert first is not second
    first.state.append("first")
    assert second.state == []


def test_discovery_checks_prototype_without_creating_or_leaking_runtime(tmp_path, monkeypatch):
    """Repeated lightweight discovery never creates a resource-owning runtime."""
    import plugins.context_engine as loader

    engine_dir = _write_engine_plugin(
        tmp_path,
        """
from agent.context_engine import ContextEngine

factory_calls = 0
prototype_calls = 0
close_calls = 0

class Engine(ContextEngine):
    def __init__(self):
        global prototype_calls
        prototype_calls += 1

    @property
    def name(self): return 'test_discovery'
    def update_from_response(self, usage): pass
    def should_compress(self, prompt_tokens=None): return False
    def compress(self, messages, current_tokens=None): return messages
    def is_available(self): return True
    def create_runtime(self):
        global factory_calls
        factory_calls += 1
        return type(self)()
    def close(self):
        global close_calls
        close_calls += 1

def register(ctx): ctx.register_context_engine(Engine())
""",
        name="test_discovery",
    )
    monkeypatch.setattr(loader, "_CONTEXT_ENGINE_PLUGINS_DIR", tmp_path)
    loader._ENGINE_PROTOTYPES.clear()

    assert loader.discover_context_engines() == [("test_discovery", "", True)]
    assert loader.discover_context_engines() == [("test_discovery", "", True)]
    module = __import__("plugins.context_engine.test_discovery", fromlist=["*"])
    assert module.prototype_calls == 1
    assert module.factory_calls == 0
    assert module.close_calls == 0


def test_create_runtime_normalizes_factory_exception():
    """A malformed plugin factory has one actionable lifecycle error shape."""
    from agent.context_engine import (
        ContextEngineLifecycleError,
        create_context_engine_runtime,
    )

    class MalformedEngine(StubEngine):
        def create_runtime(self):
            raise ValueError("bad plugin state")

    with pytest.raises(ContextEngineLifecycleError, match=r"create_runtime\(\) failed: bad plugin state"):
        create_context_engine_runtime(MalformedEngine(), name="Context engine 'bad'")


def test_repo_loader_preserves_normalized_factory_error(tmp_path, monkeypatch):
    """The repo loader exposes the same lifecycle error for malformed factories."""
    import plugins.context_engine as loader

    engine_dir = _write_engine_plugin(
        tmp_path,
        """
from agent.context_engine import ContextEngine

class Engine(ContextEngine):
    @property
    def name(self): return 'test_bad_factory'
    def update_from_response(self, usage): pass
    def should_compress(self, prompt_tokens=None): return False
    def compress(self, messages, current_tokens=None): return messages
    def create_runtime(self): raise ValueError('factory exploded')

def register(ctx): ctx.register_context_engine(Engine())
""",
        name="test_bad_factory",
    )
    monkeypatch.setattr(loader, "_CONTEXT_ENGINE_PLUGINS_DIR", tmp_path)
    loader._ENGINE_PROTOTYPES.clear()

    with pytest.raises(loader.ContextEngineLifecycleError, match="factory exploded"):
        loader.load_context_engine(engine_dir.name)


def test_context_engine_close_boundary_is_idempotent():
    engine = _RuntimeEngine()
    engine.close()
    engine.close()
    assert engine.closed == 1


def test_loader_rejects_engine_using_shared_base_factory(tmp_path, monkeypatch):
    """A plugin must opt into the explicit factory instead of sharing state."""
    import plugins.context_engine as loader

    engine_dir = _write_engine_plugin(
        tmp_path,
        """
from agent.context_engine import ContextEngine
class Engine(ContextEngine):
    @property
    def name(self): return 'test_shared'
    def update_from_response(self, usage): pass
    def should_compress(self, prompt_tokens=None): return False
    def compress(self, messages, current_tokens=None): return messages
def register(ctx): ctx.register_context_engine(Engine())
""",
        name="test_shared",
    )
    monkeypatch.setattr(loader, "_CONTEXT_ENGINE_PLUGINS_DIR", tmp_path)
    loader._ENGINE_PROTOTYPES.clear()

    with pytest.raises(loader.ContextEngineLifecycleError, match="implement create_runtime"):
        loader.load_context_engine(engine_dir.name)


def test_agent_shutdown_closes_context_engine_after_session_end():
    """Agent teardown closes the runtime, while repeated teardown is harmless."""
    from run_agent import AIAgent

    events = []
    engine = _RuntimeEngine()
    engine.on_session_end = lambda session_id, messages: events.append("session_end")
    engine.close = lambda: events.append("close")
    agent = AIAgent.__new__(AIAgent)
    agent._memory_manager = None
    agent.context_compressor = engine
    agent.session_id = "session-1"

    agent.commit_memory_session([])
    assert events == ["session_end"]

    agent.close()
    agent.close()

    assert events == ["session_end", "session_end", "close"]


def test_agent_shutdown_retries_context_engine_close_after_failure():
    """A failed close is retried, while successful cleanup is not repeated."""
    from run_agent import AIAgent

    events = []
    close_attempts = 0
    engine = _RuntimeEngine()

    def close():
        nonlocal close_attempts
        close_attempts += 1
        if close_attempts == 1:
            raise RuntimeError("transient close failure")
        events.append("close")

    engine.on_session_end = lambda session_id, messages: events.append("session_end")
    engine.close = close
    agent = AIAgent.__new__(AIAgent)
    agent._memory_manager = None
    agent.context_compressor = engine
    agent.session_id = "session-1"

    agent.shutdown_memory_provider([])
    assert events == ["session_end"]
    assert not getattr(agent, "_shutdown_memory_provider_done", False)

    agent.shutdown_memory_provider([])
    agent.shutdown_memory_provider([])
    assert events == ["session_end", "close"]
    assert close_attempts == 2
    assert agent._shutdown_memory_provider_done is True


def test_default_context_engine_behavior_remains_unchanged():
    engine = ContextCompressor(model="test", quiet_mode=True, config_context_length=200000)
    assert engine.name == "compressor"
    assert engine.close() is None
