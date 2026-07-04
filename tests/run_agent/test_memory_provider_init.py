"""Regression tests for memory provider selection during AIAgent init."""

from types import SimpleNamespace
from unittest.mock import patch


class RecordingMemoryProvider:
    name = "recording"

    def __init__(self):
        self.init_kwargs = None
        self.init_session_id = None

    def is_available(self):
        return True

    def initialize(self, session_id, **kwargs):
        self.init_session_id = session_id
        self.init_kwargs = dict(kwargs)

    def get_tool_schemas(self):
        return []

    def shutdown(self):
        pass


def test_blank_memory_provider_does_not_auto_enable_honcho():
    """Blank memory.provider should remain opt-out even if Honcho fallback looks configured."""
    cfg = {"memory": {"provider": ""}, "agent": {}}
    honcho_cfg = SimpleNamespace(enabled=True, api_key="stale-key", base_url=None)

    with (
        patch("hermes_cli.config.load_config", return_value=cfg),
        patch("hermes_cli.config.save_config") as save_config,
        patch(
            "plugins.memory.honcho.client.HonchoClientConfig.from_global_config",
            return_value=honcho_cfg,
        ) as from_global_config,
        patch("plugins.memory.load_memory_provider") as load_memory_provider,
        patch("agent.model_metadata.get_model_context_length", return_value=204_800),
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=False,
        )

    assert agent._memory_manager is None
    from_global_config.assert_not_called()
    load_memory_provider.assert_not_called()
    save_config.assert_not_called()


def test_aiagent_forwards_user_id_alt_to_memory_provider():
    provider = RecordingMemoryProvider()
    cfg = {"memory": {"provider": "recording"}, "agent": {}}

    with (
        patch("hermes_cli.config.load_config", return_value=cfg),
        patch("plugins.memory.load_memory_provider", return_value=provider),
        patch("agent.model_metadata.get_model_context_length", return_value=204_800),
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=False,
            session_id="sess-alt",
            platform="feishu",
            user_id="open-id",
            user_id_alt="union-id",
        )

    assert agent._memory_manager is not None
    assert provider.init_session_id == "sess-alt"
    assert provider.init_kwargs["user_id"] == "open-id"
    assert provider.init_kwargs["user_id_alt"] == "union-id"
    assert provider.init_kwargs["platform"] == "feishu"
    assert "warning_callback" not in provider.init_kwargs
    assert "status_callback" not in provider.init_kwargs


class CoreShadowProvider:
    """Provider that tries to register tools shadowing built-in core tools."""

    name = "core-shadow"

    def get_tool_schemas(self):
        return [
            {"name": "clarify", "description": "shadows built-in clarify"},
            {"name": "delegate_task", "description": "shadows built-in delegate"},
            {"name": "honcho_search", "description": "legit memory tool"},
        ]


def test_core_tool_names_rejected_from_memory_routing_table():
    """Memory tools shadowing core tool names are rejected at registration (#40466).

    Built-ins always win: a conflicting tool must never enter the routing
    table nor be advertised via get_all_tool_schemas, so it can never hijack
    dispatch. The non-conflicting tool is preserved.
    """
    from agent.memory_manager import MemoryManager

    mm = MemoryManager()
    mm.add_provider(CoreShadowProvider())

    # Reserved names never enter the routing table
    assert not mm.has_tool("clarify")
    assert not mm.has_tool("delegate_task")
    assert "clarify" not in mm._tool_to_provider
    assert "delegate_task" not in mm._tool_to_provider

    # Non-conflicting tool survives
    assert mm.has_tool("honcho_search")
    assert "honcho_search" in mm._tool_to_provider

    # Manager never advertises a schema it would refuse to route
    schema_names = {s.get("name") for s in mm.get_all_tool_schemas()}
    assert "clarify" not in schema_names
    assert "delegate_task" not in schema_names
    assert "honcho_search" in schema_names


def test_aiagent_forwards_warning_callback_to_cli_memory_provider():
    provider = RecordingMemoryProvider()
    cfg = {"memory": {"provider": "recording"}, "agent": {}}

    with (
        patch("hermes_cli.config.load_config", return_value=cfg),
        patch("plugins.memory.load_memory_provider", return_value=provider),
        patch("agent.model_metadata.get_model_context_length", return_value=204_800),
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=False,
            session_id="sess-cli",
            platform="cli",
        )

    assert agent._memory_manager is not None
    assert provider.init_session_id == "sess-cli"
    assert provider.init_kwargs["platform"] == "cli"
    assert provider.init_kwargs["warning_callback"] == agent._emit_warning
    assert provider.init_kwargs["status_callback"] == agent._emit_status


class AsyncDiscoveryProvider:
    """Provider that discovers tools asynchronously (starts with 0 tools)."""

    name = "async-discovery"

    def __init__(self):
        self._schemas = []
        self._discovered = False

    def is_available(self):
        return True

    def initialize(self, session_id, **kwargs):
        pass

    def get_tool_schemas(self):
        return list(self._schemas)

    def discover_tools(self, schemas):
        """Simulate async tool discovery completing."""
        self._schemas = schemas
        self._discovered = True

    def shutdown(self):
        pass


def test_rebuild_provider_tools_populates_after_async_discovery():
    """Providers with async tool discovery can call rebuild_provider_tools()

    after discovery completes so that tools registered as 0 at add_provider()
    time are picked up for dispatch routing (#58360).
    """
    from agent.memory_manager import MemoryManager

    mm = MemoryManager()
    provider = AsyncDiscoveryProvider()

    # Register while provider has no tools (pre-discovery).
    mm.add_provider(provider)
    assert mm.get_all_tool_names() == set()
    assert not mm.has_tool("mimir_remember")

    # Simulate async discovery completing.
    provider.discover_tools([
        {"name": "mimir_remember", "description": "Store a memory", "parameters": {}},
        {"name": "mimir_search", "description": "Search memories", "parameters": {}},
    ])

    # Rebuild picks up the newly-discovered tools.
    mm.rebuild_provider_tools(provider)
    assert mm.has_tool("mimir_remember")
    assert mm.has_tool("mimir_search")
    assert mm.get_all_tool_names() == {"mimir_remember", "mimir_search"}


def test_rebuild_provider_tools_is_idempotent():
    """Calling rebuild_provider_tools() multiple times is safe."""
    from agent.memory_manager import MemoryManager

    mm = MemoryManager()
    provider = AsyncDiscoveryProvider()
    mm.add_provider(provider)

    provider.discover_tools([
        {"name": "tool_a", "description": "A", "parameters": {}},
    ])
    mm.rebuild_provider_tools(provider)
    assert mm.has_tool("tool_a")

    # Second call with same tools — no duplicates, no errors.
    mm.rebuild_provider_tools(provider)
    assert mm.has_tool("tool_a")
    assert len(mm.get_all_tool_names()) == 1


def test_rebuild_provider_tools_preserves_other_providers():
    """rebuild_provider_tools() only touches the given provider's entries."""
    from agent.memory_manager import MemoryManager

    mm = MemoryManager()

    # First provider with static tools.
    static = CoreShadowProvider()  # has honcho_search
    mm.add_provider(static)
    assert mm.has_tool("honcho_search")

    # Second provider with async discovery.
    async_prov = AsyncDiscoveryProvider()
    mm.add_provider(async_prov)
    assert not mm.has_tool("new_tool")

    # Discover and rebuild only the async provider.
    async_prov.discover_tools([
        {"name": "new_tool", "description": "New", "parameters": {}},
    ])
    mm.rebuild_provider_tools(async_prov)

    # Both providers' tools present.
    assert mm.has_tool("honcho_search")
    assert mm.has_tool("new_tool")


def test_rebuild_provider_tools_respects_core_tool_names():
    """rebuild_provider_tools() still rejects core tool name shadows."""
    from agent.memory_manager import MemoryManager

    mm = MemoryManager()
    provider = AsyncDiscoveryProvider()
    mm.add_provider(provider)

    provider.discover_tools([
        {"name": "clarify", "description": "shadows core", "parameters": {}},
        {"name": "legit_tool", "description": "legit", "parameters": {}},
    ])
    mm.rebuild_provider_tools(provider)

    assert not mm.has_tool("clarify")
    assert mm.has_tool("legit_tool")
