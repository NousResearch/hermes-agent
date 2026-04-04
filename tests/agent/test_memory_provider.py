"""Tests for the memory provider interface, manager, and builtin provider."""

import json
import pytest
from unittest.mock import MagicMock, patch

from agent.memory_provider import MemoryProvider
from agent.memory_manager import MemoryManager
from agent.builtin_memory_provider import BuiltinMemoryProvider


# ---------------------------------------------------------------------------
# Concrete test provider
# ---------------------------------------------------------------------------


class FakeMemoryProvider(MemoryProvider):
    """Minimal concrete provider for testing."""

    def __init__(self, name="fake", available=True, tools=None):
        self._name = name
        self._available = available
        self._tools = tools or []
        self.initialized = False
        self.synced_turns = []
        self.prefetch_queries = []
        self.queued_prefetches = []
        self.turn_starts = []
        self.session_end_called = False
        self.pre_compress_called = False
        self.memory_writes = []
        self.shutdown_called = False
        self._prefetch_result = ""
        self._prompt_block = ""

    @property
    def name(self) -> str:
        return self._name

    def is_available(self) -> bool:
        return self._available

    def initialize(self, session_id, **kwargs):
        self.initialized = True
        self._init_kwargs = {"session_id": session_id, **kwargs}

    def system_prompt_block(self) -> str:
        return self._prompt_block

    def prefetch(self, query, *, session_id=""):
        self.prefetch_queries.append(query)
        return self._prefetch_result

    def queue_prefetch(self, query, *, session_id=""):
        self.queued_prefetches.append(query)

    def sync_turn(self, user_content, assistant_content, *, session_id=""):
        self.synced_turns.append((user_content, assistant_content))

    def get_tool_schemas(self):
        return self._tools

    def handle_tool_call(self, tool_name, args, **kwargs):
        return json.dumps({"handled": tool_name, "args": args})

    def shutdown(self):
        self.shutdown_called = True

    def on_turn_start(self, turn_number, message):
        self.turn_starts.append((turn_number, message))

    def on_session_end(self, messages):
        self.session_end_called = True

    def on_pre_compress(self, messages):
        self.pre_compress_called = True

    def on_memory_write(self, action, target, content):
        self.memory_writes.append((action, target, content))


# ---------------------------------------------------------------------------
# MemoryProvider ABC tests
# ---------------------------------------------------------------------------


class TestMemoryProviderABC:
    def test_cannot_instantiate_abstract(self):
        """ABC cannot be instantiated directly."""
        with pytest.raises(TypeError):
            MemoryProvider()

    def test_concrete_provider_works(self):
        """Concrete implementation can be instantiated."""
        p = FakeMemoryProvider()
        assert p.name == "fake"
        assert p.is_available()

    def test_default_optional_hooks_are_noop(self):
        """Optional hooks have default no-op implementations."""
        p = FakeMemoryProvider()
        # These should not raise
        p.on_turn_start(1, "hello")
        p.on_session_end([])
        p.on_pre_compress([])
        p.on_memory_write("add", "memory", "test")
        p.queue_prefetch("query")
        p.sync_turn("user", "assistant")
        p.shutdown()


# ---------------------------------------------------------------------------
# MemoryManager tests
# ---------------------------------------------------------------------------


class TestMemoryManager:
    def test_empty_manager(self):
        mgr = MemoryManager()
        assert mgr.providers == []
        assert mgr.provider_names == []
        assert mgr.get_all_tool_schemas() == []
        assert mgr.build_system_prompt() == ""
        assert mgr.prefetch_all("test") == ""

    def test_add_provider(self):
        mgr = MemoryManager()
        p = FakeMemoryProvider("test1")
        mgr.add_provider(p)
        assert len(mgr.providers) == 1
        assert mgr.provider_names == ["test1"]

    def test_get_provider_by_name(self):
        mgr = MemoryManager()
        p = FakeMemoryProvider("test1")
        mgr.add_provider(p)
        assert mgr.get_provider("test1") is p
        assert mgr.get_provider("nonexistent") is None

    def test_builtin_plus_external(self):
        mgr = MemoryManager()
        p1 = FakeMemoryProvider("builtin")
        p2 = FakeMemoryProvider("external")
        mgr.add_provider(p1)
        mgr.add_provider(p2)
        assert mgr.provider_names == ["builtin", "external"]

    def test_second_external_rejected(self):
        """Only one non-builtin provider is allowed."""
        mgr = MemoryManager()
        builtin = FakeMemoryProvider("builtin")
        ext1 = FakeMemoryProvider("mem0")
        ext2 = FakeMemoryProvider("hindsight")
        mgr.add_provider(builtin)
        mgr.add_provider(ext1)
        mgr.add_provider(ext2)  # should be rejected
        assert mgr.provider_names == ["builtin", "mem0"]
        assert len(mgr.providers) == 2

    def test_system_prompt_merges_blocks(self):
        mgr = MemoryManager()
        p1 = FakeMemoryProvider("builtin")
        p1._prompt_block = "Block from builtin"
        p2 = FakeMemoryProvider("external")
        p2._prompt_block = "Block from external"
        mgr.add_provider(p1)
        mgr.add_provider(p2)

        result = mgr.build_system_prompt()
        assert "Block from builtin" in result
        assert "Block from external" in result

    def test_system_prompt_skips_empty(self):
        mgr = MemoryManager()
        p1 = FakeMemoryProvider("builtin")
        p1._prompt_block = "Has content"
        p2 = FakeMemoryProvider("external")
        p2._prompt_block = ""
        mgr.add_provider(p1)
        mgr.add_provider(p2)

        result = mgr.build_system_prompt()
        assert result == "Has content"

    def test_prefetch_merges_results(self):
        mgr = MemoryManager()
        p1 = FakeMemoryProvider("builtin")
        p1._prefetch_result = "Memory from builtin"
        p2 = FakeMemoryProvider("external")
        p2._prefetch_result = "Memory from external"
        mgr.add_provider(p1)
        mgr.add_provider(p2)

        result = mgr.prefetch_all("what do you know?")
        assert "Memory from builtin" in result
        assert "Memory from external" in result
        assert p1.prefetch_queries == ["what do you know?"]
        assert p2.prefetch_queries == ["what do you know?"]

    def test_prefetch_skips_empty(self):
        mgr = MemoryManager()
        p1 = FakeMemoryProvider("builtin")
        p1._prefetch_result = "Has memories"
        p2 = FakeMemoryProvider("external")
        p2._prefetch_result = ""
        mgr.add_provider(p1)
        mgr.add_provider(p2)

        result = mgr.prefetch_all("query")
        assert result == "Has memories"

    def test_queue_prefetch_all(self):
        mgr = MemoryManager()
        p1 = FakeMemoryProvider("builtin")
        p2 = FakeMemoryProvider("external")
        mgr.add_provider(p1)
        mgr.add_provider(p2)

        mgr.queue_prefetch_all("next turn")
        assert p1.queued_prefetches == ["next turn"]
        assert p2.queued_prefetches == ["next turn"]

    def test_sync_all(self):
        mgr = MemoryManager()
        p1 = FakeMemoryProvider("builtin")
        p2 = FakeMemoryProvider("external")
        mgr.add_provider(p1)
        mgr.add_provider(p2)

        mgr.sync_all("user msg", "assistant msg")
        assert p1.synced_turns == [("user msg", "assistant msg")]
        assert p2.synced_turns == [("user msg", "assistant msg")]

    def test_sync_failure_doesnt_block_others(self):
        """If one provider's sync fails, others still run."""
        mgr = MemoryManager()
        p1 = FakeMemoryProvider("builtin")
        p1.sync_turn = MagicMock(side_effect=RuntimeError("boom"))
        p2 = FakeMemoryProvider("external")
        mgr.add_provider(p1)
        mgr.add_provider(p2)

        mgr.sync_all("user", "assistant")
        # p1 failed but p2 still synced
        assert p2.synced_turns == [("user", "assistant")]

    # -- Tool routing -------------------------------------------------------

    def test_tool_schemas_collected(self):
        mgr = MemoryManager()
        p1 = FakeMemoryProvider("builtin", tools=[
            {"name": "recall_builtin", "description": "Builtin recall", "parameters": {}}
        ])
        p2 = FakeMemoryProvider("external", tools=[
            {"name": "recall_ext", "description": "External recall", "parameters": {}}
        ])
        mgr.add_provider(p1)
        mgr.add_provider(p2)

        schemas = mgr.get_all_tool_schemas()
        names = {s["name"] for s in schemas}
        assert names == {"recall_builtin", "recall_ext"}

    def test_tool_name_conflict_first_wins(self):
        mgr = MemoryManager()
        p1 = FakeMemoryProvider("builtin", tools=[
            {"name": "shared_tool", "description": "From builtin", "parameters": {}}
        ])
        p2 = FakeMemoryProvider("external", tools=[
            {"name": "shared_tool", "description": "From external", "parameters": {}}
        ])
        mgr.add_provider(p1)
        mgr.add_provider(p2)

        assert mgr.has_tool("shared_tool")
        result = json.loads(mgr.handle_tool_call("shared_tool", {"q": "test"}))
        assert result["handled"] == "shared_tool"
        # Should be handled by p1 (first registered)

    def test_handle_unknown_tool(self):
        mgr = MemoryManager()
        result = json.loads(mgr.handle_tool_call("nonexistent", {}))
        assert "error" in result

    def test_tool_routing(self):
        mgr = MemoryManager()
        p1 = FakeMemoryProvider("builtin", tools=[
            {"name": "builtin_tool", "description": "Builtin", "parameters": {}}
        ])
        p2 = FakeMemoryProvider("external", tools=[
            {"name": "ext_tool", "description": "External", "parameters": {}}
        ])
        mgr.add_provider(p1)
        mgr.add_provider(p2)

        r1 = json.loads(mgr.handle_tool_call("builtin_tool", {"a": 1}))
        assert r1["handled"] == "builtin_tool"
        r2 = json.loads(mgr.handle_tool_call("ext_tool", {"b": 2}))
        assert r2["handled"] == "ext_tool"

    # -- Lifecycle hooks -----------------------------------------------------

    def test_on_turn_start(self):
        mgr = MemoryManager()
        p = FakeMemoryProvider("p")
        mgr.add_provider(p)
        mgr.on_turn_start(3, "hello")
        assert p.turn_starts == [(3, "hello")]

    def test_on_session_end(self):
        mgr = MemoryManager()
        p = FakeMemoryProvider("p")
        mgr.add_provider(p)
        mgr.on_session_end([{"role": "user", "content": "hi"}])
        assert p.session_end_called

    def test_on_pre_compress(self):
        mgr = MemoryManager()
        p = FakeMemoryProvider("p")
        mgr.add_provider(p)
        mgr.on_pre_compress([{"role": "user", "content": "old"}])
        assert p.pre_compress_called

    def test_on_memory_write_skips_builtin(self):
        """on_memory_write should skip the builtin provider."""
        mgr = MemoryManager()
        builtin = BuiltinMemoryProvider()
        external = FakeMemoryProvider("external")
        mgr.add_provider(builtin)
        mgr.add_provider(external)

        mgr.on_memory_write("add", "memory", "test fact")
        assert external.memory_writes == [("add", "memory", "test fact")]

    def test_shutdown_all_reverse_order(self):
        mgr = MemoryManager()
        order = []
        p1 = FakeMemoryProvider("builtin")
        p1.shutdown = lambda: order.append("builtin")
        p2 = FakeMemoryProvider("external")
        p2.shutdown = lambda: order.append("external")
        mgr.add_provider(p1)
        mgr.add_provider(p2)

        mgr.shutdown_all()
        assert order == ["external", "builtin"]  # reverse order

    def test_initialize_all(self):
        mgr = MemoryManager()
        p1 = FakeMemoryProvider("builtin")
        p2 = FakeMemoryProvider("external")
        mgr.add_provider(p1)
        mgr.add_provider(p2)

        mgr.initialize_all(session_id="test-123", platform="cli")
        assert p1.initialized
        assert p2.initialized
        assert p1._init_kwargs["session_id"] == "test-123"
        assert p1._init_kwargs["platform"] == "cli"

    # -- Error resilience ---------------------------------------------------

    def test_prefetch_failure_doesnt_block(self):
        mgr = MemoryManager()
        p1 = FakeMemoryProvider("builtin")
        p1.prefetch = MagicMock(side_effect=RuntimeError("network error"))
        p2 = FakeMemoryProvider("external")
        p2._prefetch_result = "external memory"
        mgr.add_provider(p1)
        mgr.add_provider(p2)

        result = mgr.prefetch_all("query")
        assert "external memory" in result

    def test_system_prompt_failure_doesnt_block(self):
        mgr = MemoryManager()
        p1 = FakeMemoryProvider("builtin")
        p1.system_prompt_block = MagicMock(side_effect=RuntimeError("broken"))
        p2 = FakeMemoryProvider("external")
        p2._prompt_block = "works fine"
        mgr.add_provider(p1)
        mgr.add_provider(p2)

        result = mgr.build_system_prompt()
        assert result == "works fine"


# ---------------------------------------------------------------------------
# BuiltinMemoryProvider tests
# ---------------------------------------------------------------------------


class TestBuiltinMemoryProvider:
    def test_name(self):
        p = BuiltinMemoryProvider()
        assert p.name == "builtin"

    def test_always_available(self):
        p = BuiltinMemoryProvider()
        assert p.is_available()

    def test_no_tools(self):
        """Builtin provider exposes no tools (memory tool is agent-level)."""
        p = BuiltinMemoryProvider()
        assert p.get_tool_schemas() == []

    def test_system_prompt_with_store(self):
        store = MagicMock()
        store.format_for_system_prompt.side_effect = lambda t: f"BLOCK_{t}" if t == "memory" else f"BLOCK_{t}"

        p = BuiltinMemoryProvider(
            memory_store=store,
            memory_enabled=True,
            user_profile_enabled=True,
        )
        block = p.system_prompt_block()
        assert "BLOCK_memory" in block
        assert "BLOCK_user" in block

    def test_system_prompt_memory_disabled(self):
        store = MagicMock()
        store.format_for_system_prompt.return_value = "content"

        p = BuiltinMemoryProvider(
            memory_store=store,
            memory_enabled=False,
            user_profile_enabled=False,
        )
        assert p.system_prompt_block() == ""

    def test_system_prompt_no_store(self):
        p = BuiltinMemoryProvider(memory_store=None, memory_enabled=True)
        assert p.system_prompt_block() == ""

    def test_prefetch_returns_empty(self):
        p = BuiltinMemoryProvider()
        assert p.prefetch("anything") == ""

    def test_store_property(self):
        store = MagicMock()
        p = BuiltinMemoryProvider(memory_store=store)
        assert p.store is store

    def test_initialize_loads_from_disk(self):
        store = MagicMock()
        p = BuiltinMemoryProvider(memory_store=store)
        p.initialize(session_id="test")
        store.load_from_disk.assert_called_once()


# ---------------------------------------------------------------------------
# Plugin registration tests
# ---------------------------------------------------------------------------


class TestSingleProviderGating:
    """Only the configured provider should activate."""

    def test_no_provider_configured_means_builtin_only(self):
        """When memory.provider is empty, no plugin providers activate."""
        mgr = MemoryManager()
        builtin = BuiltinMemoryProvider()
        mgr.add_provider(builtin)

        # Simulate what run_agent.py does when provider="" 
        configured = ""
        available_plugins = [
            FakeMemoryProvider("holographic"),
            FakeMemoryProvider("mem0"),
        ]
        # With empty config, no plugins should be added
        if configured:
            for p in available_plugins:
                if p.name == configured and p.is_available():
                    mgr.add_provider(p)

        assert mgr.provider_names == ["builtin"]

    def test_configured_provider_activates(self):
        """Only the named provider should be added."""
        mgr = MemoryManager()
        builtin = BuiltinMemoryProvider()
        mgr.add_provider(builtin)

        configured = "holographic"
        p1 = FakeMemoryProvider("holographic")
        p2 = FakeMemoryProvider("mem0")
        p3 = FakeMemoryProvider("hindsight")

        for p in [p1, p2, p3]:
            if p.name == configured and p.is_available():
                mgr.add_provider(p)

        assert mgr.provider_names == ["builtin", "holographic"]
        assert p1.initialized is False  # not initialized by the gating logic itself

    def test_unavailable_provider_skipped(self):
        """If the configured provider is unavailable, it should be skipped."""
        mgr = MemoryManager()
        builtin = BuiltinMemoryProvider()
        mgr.add_provider(builtin)

        configured = "holographic"
        p1 = FakeMemoryProvider("holographic", available=False)

        for p in [p1]:
            if p.name == configured and p.is_available():
                mgr.add_provider(p)

        assert mgr.provider_names == ["builtin"]

    def test_nonexistent_provider_results_in_builtin_only(self):
        """If the configured name doesn't match any plugin, only builtin remains."""
        mgr = MemoryManager()
        builtin = BuiltinMemoryProvider()
        mgr.add_provider(builtin)

        configured = "nonexistent"
        plugins = [FakeMemoryProvider("holographic"), FakeMemoryProvider("mem0")]

        for p in plugins:
            if p.name == configured and p.is_available():
                mgr.add_provider(p)

        assert mgr.provider_names == ["builtin"]


class TestPluginMemoryDiscovery:
    """Memory providers are discovered from plugins/memory/ directory."""

    def test_discover_finds_providers(self):
        """discover_memory_providers returns available providers."""
        from plugins.memory import discover_memory_providers
        providers = discover_memory_providers()
        names = [name for name, _, _ in providers]
        assert "holographic" in names  # always available (no external deps)

    def test_load_provider_by_name(self):
        """load_memory_provider returns a working provider instance."""
        from plugins.memory import load_memory_provider
        p = load_memory_provider("holographic")
        assert p is not None
        assert p.name == "holographic"
        assert p.is_available()

    def test_load_nonexistent_returns_none(self):
        """load_memory_provider returns None for unknown names."""
        from plugins.memory import load_memory_provider
        assert load_memory_provider("nonexistent_provider") is None


# ---------------------------------------------------------------------------
# Sequential dispatch routing tests
# ---------------------------------------------------------------------------


class TestSequentialDispatchRouting:
    """Verify that memory provider tools are correctly routed through
    memory_manager.has_tool() and handle_tool_call().

    This is a regression test for a bug where _execute_tool_calls_sequential
    in run_agent.py had its own inline dispatch chain that skipped
    memory_manager.has_tool(), causing all memory provider tools to fall
    through to the registry and return "Unknown tool". The fix added
    has_tool() + handle_tool_call() to the sequential path.

    These tests verify the memory_manager contract that both dispatch
    paths rely on: has_tool() returns True for registered provider tools,
    and handle_tool_call() routes to the correct provider.
    """

    def test_has_tool_returns_true_for_provider_tools(self):
        """has_tool returns True for tools registered by memory providers."""
        mgr = MemoryManager()
        provider = FakeMemoryProvider("ext", tools=[
            {"name": "ext_recall", "description": "Ext recall", "parameters": {}},
            {"name": "ext_retain", "description": "Ext retain", "parameters": {}},
        ])
        mgr.add_provider(provider)

        assert mgr.has_tool("ext_recall")
        assert mgr.has_tool("ext_retain")

    def test_has_tool_returns_false_for_builtin_tools(self):
        """has_tool returns False for agent-level tools (terminal, memory, etc.)."""
        mgr = MemoryManager()
        provider = FakeMemoryProvider("ext", tools=[
            {"name": "ext_recall", "description": "Ext", "parameters": {}},
        ])
        mgr.add_provider(provider)

        assert not mgr.has_tool("terminal")
        assert not mgr.has_tool("memory")
        assert not mgr.has_tool("todo")
        assert not mgr.has_tool("session_search")
        assert not mgr.has_tool("nonexistent")

    def test_handle_tool_call_routes_to_provider(self):
        """handle_tool_call dispatches to the correct provider's handler."""
        mgr = MemoryManager()
        provider = FakeMemoryProvider("hindsight", tools=[
            {"name": "hindsight_recall", "description": "Recall", "parameters": {}},
            {"name": "hindsight_retain", "description": "Retain", "parameters": {}},
        ])
        mgr.add_provider(provider)

        result = json.loads(mgr.handle_tool_call("hindsight_recall", {"query": "alice"}))
        assert result["handled"] == "hindsight_recall"
        assert result["args"] == {"query": "alice"}

    def test_handle_tool_call_unknown_returns_error(self):
        """handle_tool_call returns error for tools not in any provider."""
        mgr = MemoryManager()
        provider = FakeMemoryProvider("ext", tools=[
            {"name": "ext_recall", "description": "Ext", "parameters": {}},
        ])
        mgr.add_provider(provider)

        result = json.loads(mgr.handle_tool_call("terminal", {"command": "ls"}))
        assert "error" in result

    def test_multiple_providers_route_to_correct_one(self):
        """Tools from different providers route to the right handler."""
        mgr = MemoryManager()
        builtin = FakeMemoryProvider("builtin", tools=[
            {"name": "builtin_tool", "description": "Builtin", "parameters": {}},
        ])
        external = FakeMemoryProvider("hindsight", tools=[
            {"name": "hindsight_recall", "description": "Recall", "parameters": {}},
        ])
        mgr.add_provider(builtin)
        mgr.add_provider(external)

        r1 = json.loads(mgr.handle_tool_call("builtin_tool", {}))
        assert r1["handled"] == "builtin_tool"

        r2 = json.loads(mgr.handle_tool_call("hindsight_recall", {"query": "test"}))
        assert r2["handled"] == "hindsight_recall"

    def test_tool_names_include_all_providers(self):
        """get_all_tool_names returns tools from all registered providers."""
        mgr = MemoryManager()
        builtin = FakeMemoryProvider("builtin", tools=[
            {"name": "builtin_tool", "description": "B", "parameters": {}},
        ])
        external = FakeMemoryProvider("ext", tools=[
            {"name": "ext_recall", "description": "E1", "parameters": {}},
            {"name": "ext_retain", "description": "E2", "parameters": {}},
        ])
        mgr.add_provider(builtin)
        mgr.add_provider(external)

        names = mgr.get_all_tool_names()
        assert names == {"builtin_tool", "ext_recall", "ext_retain"}


# ---------------------------------------------------------------------------
# Setup wizard field filtering tests (when clause and default_from)
# ---------------------------------------------------------------------------


class TestSetupFieldFiltering:
    """Test the 'when' clause and 'default_from' logic used by the
    memory setup wizard in hermes_cli/memory_setup.py.

    These features are generic — any memory plugin can use them in
    get_config_schema(). Currently used by the hindsight plugin.
    """

    def _filter_fields(self, schema, provider_config):
        """Simulate the setup wizard's field filtering logic.

        Returns list of (key, effective_default) for fields that pass
        the 'when' filter.
        """
        results = []
        for field in schema:
            key = field["key"]
            default = field.get("default")

            # Dynamic default
            default_from = field.get("default_from")
            if default_from and isinstance(default_from, dict):
                ref_field = default_from.get("field", "")
                ref_map = default_from.get("map", {})
                ref_value = provider_config.get(ref_field, "")
                if ref_value and ref_value in ref_map:
                    default = ref_map[ref_value]

            # When clause
            when = field.get("when")
            if when and isinstance(when, dict):
                if not all(provider_config.get(k) == v for k, v in when.items()):
                    continue

            results.append((key, default))
        return results

    def test_when_clause_filters_fields(self):
        """Fields with 'when' are skipped if the condition doesn't match."""
        schema = [
            {"key": "mode", "default": "cloud"},
            {"key": "api_url", "default": "https://api.example.com", "when": {"mode": "cloud"}},
            {"key": "api_key", "default": None, "when": {"mode": "cloud"}},
            {"key": "llm_provider", "default": "openai", "when": {"mode": "local"}},
            {"key": "llm_model", "default": "gpt-4o-mini", "when": {"mode": "local"}},
            {"key": "budget", "default": "mid"},
        ]

        # Cloud mode: should see mode, api_url, api_key, budget
        cloud_fields = self._filter_fields(schema, {"mode": "cloud"})
        cloud_keys = [k for k, _ in cloud_fields]
        assert cloud_keys == ["mode", "api_url", "api_key", "budget"]

        # Local mode: should see mode, llm_provider, llm_model, budget
        local_fields = self._filter_fields(schema, {"mode": "local"})
        local_keys = [k for k, _ in local_fields]
        assert local_keys == ["mode", "llm_provider", "llm_model", "budget"]

    def test_when_clause_no_condition_always_shown(self):
        """Fields without 'when' are always included."""
        schema = [
            {"key": "bank_id", "default": "hermes"},
            {"key": "budget", "default": "mid"},
        ]
        fields = self._filter_fields(schema, {"mode": "cloud"})
        assert [k for k, _ in fields] == ["bank_id", "budget"]

    def test_default_from_resolves_dynamic_default(self):
        """default_from looks up the default from another field's value."""
        provider_models = {
            "openai": "gpt-4o-mini",
            "groq": "openai/gpt-oss-120b",
            "anthropic": "claude-haiku-4-5",
        }
        schema = [
            {"key": "llm_provider", "default": "openai"},
            {"key": "llm_model", "default": "gpt-4o-mini",
             "default_from": {"field": "llm_provider", "map": provider_models}},
        ]

        # Groq selected: model should default to groq's default
        fields = self._filter_fields(schema, {"llm_provider": "groq"})
        model_default = dict(fields)["llm_model"]
        assert model_default == "openai/gpt-oss-120b"

        # Anthropic selected
        fields = self._filter_fields(schema, {"llm_provider": "anthropic"})
        model_default = dict(fields)["llm_model"]
        assert model_default == "claude-haiku-4-5"

    def test_default_from_falls_back_to_static_default(self):
        """default_from falls back to static default if provider not in map."""
        schema = [
            {"key": "llm_model", "default": "gpt-4o-mini",
             "default_from": {"field": "llm_provider", "map": {"groq": "openai/gpt-oss-120b"}}},
        ]

        # Unknown provider: should fall back to static default
        fields = self._filter_fields(schema, {"llm_provider": "unknown_provider"})
        model_default = dict(fields)["llm_model"]
        assert model_default == "gpt-4o-mini"

    def test_default_from_with_no_ref_value(self):
        """default_from keeps static default if referenced field is not set."""
        schema = [
            {"key": "llm_model", "default": "gpt-4o-mini",
             "default_from": {"field": "llm_provider", "map": {"groq": "openai/gpt-oss-120b"}}},
        ]

        # No provider set at all
        fields = self._filter_fields(schema, {})
        model_default = dict(fields)["llm_model"]
        assert model_default == "gpt-4o-mini"

    def test_when_and_default_from_combined(self):
        """when clause and default_from work together correctly."""
        provider_models = {"groq": "openai/gpt-oss-120b", "openai": "gpt-4o-mini"}
        schema = [
            {"key": "mode", "default": "local"},
            {"key": "llm_provider", "default": "openai", "when": {"mode": "local"}},
            {"key": "llm_model", "default": "gpt-4o-mini",
             "default_from": {"field": "llm_provider", "map": provider_models},
             "when": {"mode": "local"}},
            {"key": "api_url", "default": "https://api.example.com", "when": {"mode": "cloud"}},
        ]

        # Local + groq: should see llm_model with groq default, no api_url
        fields = self._filter_fields(schema, {"mode": "local", "llm_provider": "groq"})
        keys = [k for k, _ in fields]
        assert "llm_model" in keys
        assert "api_url" not in keys
        assert dict(fields)["llm_model"] == "openai/gpt-oss-120b"

        # Cloud: should see api_url, no llm_model
        fields = self._filter_fields(schema, {"mode": "cloud"})
        keys = [k for k, _ in fields]
        assert "api_url" in keys
        assert "llm_model" not in keys
