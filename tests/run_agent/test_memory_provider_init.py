"""Regression tests for memory provider selection during AIAgent init."""

import json
from types import SimpleNamespace
from unittest.mock import patch


class RecordingMemoryProvider:
    name = "recording"

    def __init__(self):
        self.init_kwargs = None
        self.init_session_id = None
        self.lifecycle_calls = []
        self.shutdown_called = False

    def is_available(self):
        return True

    def initialize(self, session_id, **kwargs):
        self.init_session_id = session_id
        self.init_kwargs = dict(kwargs)

    def get_tool_schemas(self):
        return [{
            "name": "recording_recall",
            "description": "Test-only explicit memory lookup",
            "parameters": {"type": "object", "properties": {}},
        }]

    def system_prompt_block(self):
        self.lifecycle_calls.append("prompt")
        return "recording prompt"

    def prefetch(self, query, **kwargs):
        self.lifecycle_calls.append("prefetch")
        return "recording context"

    def queue_prefetch(self, query, **kwargs):
        self.lifecycle_calls.append("queue_prefetch")

    def sync_turn(self, user_content, assistant_content, **kwargs):
        self.lifecycle_calls.append("sync")

    def on_turn_start(self, turn_number, message, **kwargs):
        self.lifecycle_calls.append("turn_start")

    def on_session_end(self, messages):
        self.lifecycle_calls.append("session_end")

    def on_session_switch(self, new_session_id, **kwargs):
        self.lifecycle_calls.append("session_switch")

    def on_pre_compress(self, messages):
        self.lifecycle_calls.append("pre_compress")
        return "recording compression context"

    def on_memory_write(self, action, target, content, **kwargs):
        self.lifecycle_calls.append("memory_write")

    def on_delegation(self, task, result, **kwargs):
        self.lifecycle_calls.append("delegation")

    def handle_tool_call(self, tool_name, args, **kwargs):
        return json.dumps({"success": True, "tool": tool_name})

    def shutdown(self):
        self.shutdown_called = True


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


def test_aiagent_can_enable_provider_tools_mode_even_when_skip_memory_is_true():
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
            skip_memory=True,
            memory_provider_mode="tools",
            session_id="sess-cron",
            platform="cron",
        )

    assert agent._memory_manager is not None
    assert agent._memory_manager.mode == "tools"
    assert provider.init_session_id == "sess-cron"
    assert provider.init_kwargs["platform"] == "cron"
    assert provider.init_kwargs["agent_context"] == "primary"

    manager = agent._memory_manager
    assert manager.has_tool("recording_recall")
    assert json.loads(manager.handle_tool_call("recording_recall", {}))["success"] is True

    assert manager.build_system_prompt() == ""
    assert manager.prefetch_all("cron prompt") == ""
    manager.queue_prefetch_all("cron prompt")
    manager.sync_all("cron prompt", "cron response")
    manager.on_turn_start(1, "cron prompt")
    manager.on_session_end([])
    manager.commit_session_boundary_async([], new_session_id="sess-next")
    manager.on_session_switch("sess-next")
    assert manager.on_pre_compress([]) == ""
    manager.on_memory_write("add", "memory", "cron content")
    manager.notify_memory_tool_write(
        json.dumps({"success": True}),
        {"action": "add", "target": "memory", "content": "cron content"},
    )
    manager.on_delegation("cron task", "cron result")
    assert provider.lifecycle_calls == []

    manager.shutdown_all()
    assert provider.shutdown_called is True


def test_skip_memory_defaults_provider_mode_to_off():
    provider = RecordingMemoryProvider()
    cfg = {"memory": {"provider": "recording"}, "agent": {}}

    with (
        patch("hermes_cli.config.load_config", return_value=cfg),
        patch("plugins.memory.load_memory_provider", return_value=provider) as loader,
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
            skip_memory=True,
            session_id="sess-cron-off",
            platform="cron",
        )

    assert agent._memory_provider_mode == "off"
    assert agent._memory_manager is None
    loader.assert_not_called()


def test_memory_manager_initialization_uses_its_authoritative_mode():
    from agent.memory_manager import MemoryManager

    provider = RecordingMemoryProvider()
    manager = MemoryManager(mode="tools")
    manager.add_provider(provider)

    manager.initialize_all(
        session_id="direct-manager-session",
        memory_provider_mode="full",
    )

    assert provider.init_kwargs["memory_provider_mode"] == "tools"


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
