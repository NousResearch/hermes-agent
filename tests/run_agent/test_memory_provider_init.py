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


class HonchoToolProvider:
    """Provider exposing the Honcho tool names without importing Honcho SDK."""

    name = "honcho"

    def get_tool_schemas(self):
        return [
            {"name": "honcho_search", "description": "Search Honcho memory"},
            {"name": "honcho_conclude", "description": "Save a Honcho conclusion"},
        ]


class OtherMemoryToolProvider:
    """Non-Honcho provider used to prove the honcho toolset is provider-specific."""

    name = "other-memory"

    def get_tool_schemas(self):
        return [
            {"name": "other_memory_search", "description": "Search another memory backend"},
        ]


class OverbroadHonchoToolProvider:
    """Honcho-like provider that reports a non-Honcho tool too."""

    name = "honcho"

    def get_tool_schemas(self):
        return [
            {"name": "honcho_search", "description": "Search Honcho memory"},
            {"name": "other_memory_search", "description": "Search another memory backend"},
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


def test_honcho_toolset_enables_memory_provider_tool_injection():
    """Restricted sessions can opt into Honcho tools by naming the honcho toolset."""
    from agent.memory_manager import MemoryManager, inject_memory_provider_tools

    mm = MemoryManager()
    mm.add_provider(HonchoToolProvider())
    agent = SimpleNamespace(
        _memory_manager=mm,
        tools=[],
        enabled_toolsets=["honcho"],
        valid_tool_names=set(),
    )

    added = inject_memory_provider_tools(agent)
    tool_names = {tool["function"]["name"] for tool in agent.tools}

    assert added == 2
    assert tool_names == {"honcho_search", "honcho_conclude"}
    assert agent.valid_tool_names == tool_names


def test_honcho_toolset_only_injects_matching_provider_tool_names():
    from agent.memory_manager import MemoryManager, inject_memory_provider_tools

    mm = MemoryManager()
    mm.add_provider(OverbroadHonchoToolProvider())
    agent = SimpleNamespace(
        _memory_manager=mm,
        tools=[],
        enabled_toolsets=["honcho"],
        valid_tool_names=set(),
    )

    added = inject_memory_provider_tools(agent)
    tool_names = {tool["function"]["name"] for tool in agent.tools}

    assert added == 1
    assert tool_names == {"honcho_search"}
    assert agent.valid_tool_names == tool_names


def test_existing_core_memory_tool_keeps_provider_injection_compatible():
    from agent.memory_manager import MemoryManager, inject_memory_provider_tools

    mm = MemoryManager()
    mm.add_provider(OverbroadHonchoToolProvider())
    agent = SimpleNamespace(
        _memory_manager=mm,
        tools=[{"type": "function", "function": {"name": "memory"}}],
        enabled_toolsets=["terminal"],
        valid_tool_names=set(),
    )

    added = inject_memory_provider_tools(agent)
    tool_names = {tool["function"]["name"] for tool in agent.tools}

    assert added == 2
    assert tool_names == {"memory", "honcho_search", "other_memory_search"}
    assert agent.valid_tool_names == {"honcho_search", "other_memory_search"}


def test_honcho_toolset_does_not_enable_other_memory_provider_tools():
    from agent.memory_manager import MemoryManager, inject_memory_provider_tools

    mm = MemoryManager()
    mm.add_provider(OtherMemoryToolProvider())
    agent = SimpleNamespace(
        _memory_manager=mm,
        tools=[],
        enabled_toolsets=["honcho"],
        valid_tool_names=set(),
    )

    added = inject_memory_provider_tools(agent)

    assert added == 0
    assert agent.tools == []
    assert agent.valid_tool_names == set()


def test_unrelated_restricted_toolset_does_not_inject_memory_provider_tools():
    from agent.memory_manager import MemoryManager, inject_memory_provider_tools

    mm = MemoryManager()
    mm.add_provider(HonchoToolProvider())
    agent = SimpleNamespace(
        _memory_manager=mm,
        tools=[],
        enabled_toolsets=["terminal"],
        valid_tool_names=set(),
    )

    added = inject_memory_provider_tools(agent)

    assert added == 0
    assert agent.tools == []
    assert agent.valid_tool_names == set()


def test_disabled_honcho_toolset_suppresses_provider_tool_injection():
    from agent.memory_manager import MemoryManager, inject_memory_provider_tools

    mm = MemoryManager()
    mm.add_provider(HonchoToolProvider())
    agent = SimpleNamespace(
        _memory_manager=mm,
        tools=[],
        enabled_toolsets=["honcho"],
        disabled_toolsets=["honcho"],
        valid_tool_names=set(),
    )

    added = inject_memory_provider_tools(agent)

    assert added == 0
    assert agent.tools == []
    assert agent.valid_tool_names == set()


def test_disabled_memory_toolset_suppresses_provider_tool_injection():
    from agent.memory_manager import MemoryManager, inject_memory_provider_tools

    mm = MemoryManager()
    mm.add_provider(HonchoToolProvider())
    agent = SimpleNamespace(
        _memory_manager=mm,
        tools=[],
        enabled_toolsets=None,
        disabled_toolsets=["memory"],
        valid_tool_names=set(),
    )

    added = inject_memory_provider_tools(agent)

    assert added == 0
    assert agent.tools == []
    assert agent.valid_tool_names == set()


def test_disabled_all_toolsets_suppresses_provider_tool_injection():
    from agent.memory_manager import MemoryManager, inject_memory_provider_tools

    mm = MemoryManager()
    mm.add_provider(HonchoToolProvider())
    agent = SimpleNamespace(
        _memory_manager=mm,
        tools=[],
        enabled_toolsets=None,
        disabled_toolsets=["*"],
        valid_tool_names=set(),
    )

    added = inject_memory_provider_tools(agent)

    assert added == 0
    assert agent.tools == []
    assert agent.valid_tool_names == set()


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
