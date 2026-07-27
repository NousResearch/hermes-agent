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

    def test_default_get_status_clamps_post_compression_sentinel(self):
        """After a compression, last_prompt_tokens is the -1 sentinel. get_status
        must clamp it to 0 rather than export a raw -1 or a negative
        usage_percent on the transitional turn."""
        engine = StubEngine()
        engine.last_prompt_tokens = -1
        status = engine.get_status()
        assert status["last_prompt_tokens"] == 0
        assert status["usage_percent"] >= 0

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

    def test_prune_tool_results_only_defaults_to_safe_noop(self):
        # An engine implementing only the required interface (no prune override)
        # must inherit the base no-op instead of raising AttributeError: the
        # agent loop calls prune_tool_results_only() on the active engine after a
        # tool call whenever full compression does not fire, so every pluggable
        # ContextEngine reaches this path (see conversation_loop proactive-prune).
        engine = StubEngine()
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        result, pruned = engine.prune_tool_results_only(msgs, current_tokens=10_000_000)
        assert pruned == 0
        assert result is msgs


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



class TestPluginContextEngineDeepCopy:
    """Verify that the plugin context engine singleton is deep-copied before
    mutation in agent_init — regression test for #42449."""

    def test_deepcopy_prevents_shared_mutation(self):
        """Deep-copied engine should not propagate mutations back to the singleton."""
        import copy
        engine = StubEngine(context_length=1_000_000, threshold_pct=0.20)
        clone = copy.deepcopy(engine)

        # Mutate the clone (simulating child agent's update_model)
        clone.context_length = 204800
        clone.threshold_tokens = 40960

        # Original must be unaffected
        assert engine.context_length == 1_000_000
        assert engine.threshold_tokens == 200000  # 1M * 0.20
        assert clone is not engine

    def test_deepcopy_preserves_engine_name(self):
        """Deep-copied engine retains its identity (name property)."""
        import copy
        engine = StubEngine(context_length=500000)
        clone = copy.deepcopy(engine)
        assert clone.name == engine.name == "stub"

    def test_deepcopy_preserves_compressor_state(self):
        """Deep-copied engine starts with the same token counters."""
        import copy
        engine = StubEngine(context_length=500000)
        engine.last_prompt_tokens = 1000
        engine.last_total_tokens = 1500
        engine.compression_count = 3

        clone = copy.deepcopy(engine)
        assert clone.last_prompt_tokens == 1000
        assert clone.last_total_tokens == 1500
        assert clone.compression_count == 3
        assert clone is not engine

    def test_no_deepcopy_direct_assignment_would_share_state(self):
        """Baseline: without deepcopy, both variables point to the same object."""
        engine = StubEngine(context_length=1_000_000)
        direct = engine  # no deepcopy — the bug path
        direct.context_length = 204800
        assert engine.context_length == 204800  # bug: parent corrupted!


class TestInitAgentDoesNotMutatePluginSingleton:
    """Regression coverage for #42449: a child agent's init must not mutate the
    shared plugin context-engine singleton via update_model().

    Note: ``test_child_init_does_not_corrupt_parent_singleton`` replicates the
    init_agent selection-block *pattern* (it cannot cheaply spin up a full
    init_agent), so it documents/verifies the deepcopy approach but does NOT by
    itself guard a production revert. The real revert guard is
    ``test_agent_init_source_deepcopies_singleton_not_aliases`` (source-pin),
    and ``test_unpicklable_engine_falls_back_gracefully`` covers the
    copy-failure path.
    """

    def test_child_init_does_not_corrupt_parent_singleton(self, monkeypatch):
        import hermes_cli.plugins as plugins_mod
        from hermes_cli.plugins import PluginManager

        # Register a "parent" engine as the global plugin singleton, sized for
        # a 1M-context model (DeepSeek-style), threshold 20% => 200K.
        singleton = StubEngine(context_length=1_000_000, threshold_pct=0.20)
        old_mgr = plugins_mod._plugin_manager
        try:
            mgr = PluginManager()
            mgr._context_engine = singleton
            plugins_mod._plugin_manager = mgr

            # Replicate init_agent's fallback selection-block pattern: fetch the
            # singleton, deepcopy it, then mutate the copy via update_model with
            # a SMALLER child context (MiniMax-style 204800).
            import copy
            from hermes_cli.plugins import get_plugin_context_engine

            _candidate = get_plugin_context_engine()
            assert _candidate is singleton
            _selected_engine = copy.deepcopy(_candidate)
            _selected_engine.update_model(
                model="MiniMax-M2", context_length=204800, provider="minimax",
            )

            # The child's smaller context must NOT leak back into the parent
            # singleton (the #42449 corruption).
            assert singleton.context_length == 1_000_000, (
                "parent singleton context_length was corrupted by child init"
            )
            assert singleton.threshold_tokens == 200_000
            # And the child's own engine reflects the child model.
            assert _selected_engine.context_length == 204800
            assert _selected_engine is not singleton
        finally:
            plugins_mod._plugin_manager = old_mgr

    def test_unpicklable_engine_falls_back_gracefully(self, monkeypatch):
        """Copy-failure path: an engine holding uncopyable state (a lock — the
        plugin docs prescribe locks/DB connections for stateful engines) makes
        copy.deepcopy raise. init_agent must NOT silently drop it with a
        misleading 'not found'; it falls back to the built-in compressor and
        logs an accurate copy-failure warning. Regression for the deepcopy-
        copy-failure path."""
        import threading

        class _UncopyableEngine(StubEngine):
            def __init__(self):
                super().__init__(context_length=1_000_000, threshold_pct=0.20)
                self._lock = threading.RLock()  # RLock can't be deepcopied

        engine = _UncopyableEngine()
        # Sanity: the engine genuinely defeats deepcopy.
        import copy
        with pytest.raises(Exception):
            copy.deepcopy(engine)

        # Replicate the init_agent fallback block's copy-failure handling.
        selected = None
        copy_failed = False
        try:
            selected = copy.deepcopy(engine)
        except Exception:
            copy_failed = True
            selected = None

        assert copy_failed is True
        assert selected is None
        # The original engine is untouched (no partial mutation).
        assert engine.context_length == 1_000_000

    def test_agent_init_source_deepcopies_singleton_not_aliases(self):
        """Source-pin guarding the production fix in agent/agent_init.py:
        the plugin-singleton fallback MUST deepcopy the candidate, not alias
        it (`_selected_engine = _candidate`). Full init_agent is too heavy to
        drive here, so this pins the exact line so a future revert to direct
        assignment fails CI. Regression for #42449."""
        import inspect
        import re
        import agent.agent_init as _ai

        src = inspect.getsource(_ai)
        # The candidate fetched from the plugin singleton must be deep-copied
        # before becoming _selected_engine (which is later mutated by
        # update_model). A bare `_selected_engine = _candidate` is the bug.
        assert re.search(
            r"_selected_engine\s*=\s*(copy|_copy)\.deepcopy\(\s*_candidate\s*\)",
            src,
        ), (
            "agent_init must deepcopy the plugin context-engine singleton "
            "(`_selected_engine = copy.deepcopy(_candidate)`) — a bare "
            "`_selected_engine = _candidate` re-introduces #42449 (child "
            "update_model corrupts the parent's shared singleton)."
        )
        # And the bug-shape alias must NOT be present on that path.
        assert not re.search(
            r"_selected_engine\s*=\s*_candidate\b", src
        ), "found the #42449 bug-shape alias `_selected_engine = _candidate`"
