"""E2E contracts for route-aware tool admission plugins."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import yaml

import hermes_cli.plugins as plugins_mod
from agent.agent_runtime_helpers import tool_hook_route_metadata
from hermes_cli.plugins import PluginManager
from run_agent import AIAgent


def _install_admission_plugin(
    hermes_home: Path, active_work: list[dict] | None = None
) -> None:
    plugin_dir = hermes_home / "plugins" / "route-admission"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text(
        yaml.safe_dump({"name": "route-admission", "version": "0.1.0"}),
        encoding="utf-8",
    )
    (plugin_dir / "__init__.py").write_text(
        "import json\n"
        "from pathlib import Path\n\n"
        "def register(ctx):\n"
        "    def admit(**kwargs):\n"
        "        route = kwargs.get('route_metadata', {})\n"
        "        if kwargs['tool_name'] != 'delegate_task':\n"
        "            return None\n"
        "        work = json.loads((Path(__file__).parent / 'active-work.json').read_text())\n"
        "        admitted = any(\n"
        "            item.get('active') and item.get('subscribed') and\n"
        "            item.get('session_id') == kwargs.get('session_id') and\n"
        "            item.get('route') == route\n"
        "            for item in work\n"
        "        )\n"
        "        if not admitted:\n"
        "            return {'action': 'block', 'message': 'mission admission denied'}\n"
        "    ctx.register_hook('pre_tool_call', admit)\n",
        encoding="utf-8",
    )
    (plugin_dir / "active-work.json").write_text(
        json.dumps(active_work or []), encoding="utf-8"
    )
    (hermes_home / "config.yaml").write_text(
        yaml.safe_dump({"plugins": {"enabled": ["route-admission"]}}),
        encoding="utf-8",
    )


def _tool_defs() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "delegate_task",
                "description": "delegate",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]


def _admitted_work() -> list[dict]:
    return [
        {
            "active": True,
            "subscribed": True,
            "session_id": "session-123",
            "route": {
                "platform": "telegram",
                "chat_id": "chat-42",
                "thread_id": "topic-7",
                "session_key": "agent:main:telegram:group:chat-42:topic-7",
            },
        }
    ]


def test_route_aware_plugin_vetoes_delegate_task_before_dispatch(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    _install_admission_plugin(hermes_home)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    manager = PluginManager()
    monkeypatch.setattr(plugins_mod, "_plugin_manager", manager)
    monkeypatch.setattr(plugins_mod, "get_plugin_manager", lambda: manager)
    manager.discover_and_load()

    with (
        patch("run_agent.get_tool_definitions", return_value=_tool_defs()),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            session_id="session-123",
            platform="telegram",
            chat_id="chat-42",
            thread_id="topic-7",
            gateway_session_key="agent:main:telegram:group:chat-42:topic-7",
        )
    agent.client = MagicMock()
    dispatched = []
    agent._dispatch_delegate_task = lambda function_args: (
        dispatched.append(function_args) or "dispatched"
    )

    cached_prompt = object()
    setattr(agent, "_cached_system_prompt", cached_prompt)
    with patch.object(agent, "_invalidate_system_prompt") as invalidate_prompt:
        result = json.loads(
            agent._invoke_tool(
                "delegate_task",
                {"goal": "start downstream work"},
                "task-parent",
                tool_call_id="call-1",
            )
        )

    assert result == {"error": "mission admission denied"}
    assert dispatched == []
    assert getattr(agent, "_cached_system_prompt") is cached_prompt
    invalidate_prompt.assert_not_called()


def test_plugin_correlates_active_subscribed_work_and_permits_dispatch(
    tmp_path, monkeypatch
):
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    _install_admission_plugin(hermes_home, _admitted_work())
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    manager = PluginManager()
    monkeypatch.setattr(plugins_mod, "_plugin_manager", manager)
    monkeypatch.setattr(plugins_mod, "get_plugin_manager", lambda: manager)
    manager.discover_and_load()

    with (
        patch("run_agent.get_tool_definitions", return_value=_tool_defs()),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            session_id="session-123",
            platform="telegram",
            chat_id="chat-42",
            thread_id="topic-7",
            gateway_session_key="agent:main:telegram:group:chat-42:topic-7",
        )
    agent.client = MagicMock()
    dispatched = []
    agent._dispatch_delegate_task = lambda function_args: (
        dispatched.append(function_args) or "dispatched"
    )

    result = agent._invoke_tool(
        "delegate_task", {"goal": "admitted work"}, "task-parent"
    )

    assert result == "dispatched"
    assert dispatched == [{"goal": "admitted work"}]


def test_delegate_dispatch_is_unchanged_when_admission_plugin_is_absent(
    tmp_path, monkeypatch
):
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    manager = PluginManager()
    monkeypatch.setattr(plugins_mod, "_plugin_manager", manager)
    monkeypatch.setattr(plugins_mod, "get_plugin_manager", lambda: manager)
    manager.discover_and_load()

    with (
        patch("run_agent.get_tool_definitions", return_value=_tool_defs()),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            session_id="session-123",
            platform="telegram",
        )
    agent.client = MagicMock()
    dispatched = []
    agent._dispatch_delegate_task = lambda function_args: (
        dispatched.append(function_args) or "dispatched"
    )

    result = agent._invoke_tool(
        "delegate_task", {"goal": "ordinary work"}, "task-parent"
    )

    assert result == "dispatched"
    assert dispatched == [{"goal": "ordinary work"}]


def test_sequential_agent_loop_uses_route_metadata_before_delegate_dispatch(
    tmp_path, monkeypatch
):
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    _install_admission_plugin(hermes_home)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    manager = PluginManager()
    monkeypatch.setattr(plugins_mod, "_plugin_manager", manager)
    monkeypatch.setattr(plugins_mod, "get_plugin_manager", lambda: manager)
    manager.discover_and_load()

    with (
        patch("run_agent.get_tool_definitions", return_value=_tool_defs()),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            session_id="session-123",
            platform="telegram",
            chat_id="chat-42",
            thread_id="topic-7",
            gateway_session_key="agent:main:telegram:group:chat-42:topic-7",
        )
    dispatched = []
    agent._dispatch_delegate_task = lambda function_args: (
        dispatched.append(function_args) or "dispatched"
    )
    tool_call = SimpleNamespace(
        id="call-1",
        function=SimpleNamespace(
            name="delegate_task",
            arguments=json.dumps({"goal": "start downstream work"}),
        ),
    )
    assistant_message = SimpleNamespace(tool_calls=[tool_call])
    messages = []

    agent._execute_tool_calls_sequential(assistant_message, messages, "task-parent")

    transcript = [
        {"role": "user", "content": "start downstream work"},
        {"role": "assistant", "tool_calls": [tool_call]},
        *messages,
    ]
    assert [message["role"] for message in transcript] == [
        "user",
        "assistant",
        "tool",
    ]
    assert json.loads(messages[0]["content"]) == {"error": "mission admission denied"}
    assert "route_metadata" not in messages[0]["content"]
    assert dispatched == []


def test_concurrent_agent_loop_uses_route_metadata_before_delegate_dispatch(
    tmp_path, monkeypatch
):
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    _install_admission_plugin(hermes_home)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    manager = PluginManager()
    monkeypatch.setattr(plugins_mod, "_plugin_manager", manager)
    monkeypatch.setattr(plugins_mod, "get_plugin_manager", lambda: manager)
    manager.discover_and_load()

    with (
        patch("run_agent.get_tool_definitions", return_value=_tool_defs()),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            session_id="session-123",
            platform="telegram",
            chat_id="chat-42",
            thread_id="topic-7",
            gateway_session_key="agent:main:telegram:group:chat-42:topic-7",
        )
    dispatched = []
    agent._dispatch_delegate_task = lambda function_args: (
        dispatched.append(function_args) or "dispatched"
    )
    tool_call = SimpleNamespace(
        id="call-1",
        function=SimpleNamespace(
            name="delegate_task",
            arguments=json.dumps({"goal": "start downstream work"}),
        ),
    )
    assistant_message = SimpleNamespace(tool_calls=[tool_call])
    messages = []

    agent._execute_tool_calls_concurrent(assistant_message, messages, "task-parent")

    assert json.loads(messages[0]["content"]) == {"error": "mission admission denied"}
    assert dispatched == []


def test_route_identity_is_stable_and_copied_across_tool_calls(monkeypatch):
    manager = PluginManager()
    monkeypatch.setattr(plugins_mod, "_plugin_manager", manager)
    monkeypatch.setattr(plugins_mod, "get_plugin_manager", lambda: manager)
    observed = []

    def capture(**kwargs):
        observed.append((kwargs["session_id"], kwargs["route_metadata"]))

    manager._hooks["pre_tool_call"] = [capture]

    with (
        patch("run_agent.get_tool_definitions", return_value=_tool_defs()),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            session_id="session-stable",
            platform="telegram",
            user_id="user-9",
            chat_id="chat-42",
            thread_id="topic-7",
            gateway_session_key="agent:main:telegram:group:chat-42:topic-7",
        )
    agent.client = MagicMock()
    agent._dispatch_delegate_task = lambda function_args: "dispatched"

    agent._invoke_tool("delegate_task", {"goal": "first"}, "task-parent")
    observed[0][1]["chat_id"] = "plugin-mutated-copy"
    agent._invoke_tool("delegate_task", {"goal": "second"}, "task-parent")

    expected_route = {
        "platform": "telegram",
        "user_id": "user-9",
        "chat_id": "chat-42",
        "thread_id": "topic-7",
        "session_key": "agent:main:telegram:group:chat-42:topic-7",
    }
    assert observed[0][0] == observed[1][0] == "session-stable"
    assert observed[1][1] == expected_route
    assert tool_hook_route_metadata(agent) == expected_route


def test_route_identity_is_isolated_between_sessions(monkeypatch):
    manager = PluginManager()
    monkeypatch.setattr(plugins_mod, "_plugin_manager", manager)
    monkeypatch.setattr(plugins_mod, "get_plugin_manager", lambda: manager)
    observed = []
    manager._hooks["pre_tool_call"] = [
        lambda **kwargs: observed.append(
            (
                kwargs["session_id"],
                kwargs["route_metadata"],
            )
        )
    ]

    agents = []
    with (
        patch("run_agent.get_tool_definitions", return_value=_tool_defs()),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        for suffix in ("one", "two"):
            agent = AIAgent(
                api_key="test-key-1234567890",
                base_url="https://openrouter.ai/api/v1",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
                session_id=f"session-{suffix}",
                platform="discord",
                chat_id=f"chat-{suffix}",
                gateway_session_key=f"agent:main:discord:channel:chat-{suffix}",
            )
            agent.client = MagicMock()
            agent._dispatch_delegate_task = lambda function_args: "dispatched"
            agents.append(agent)

    for agent in agents:
        agent._invoke_tool("delegate_task", {"goal": "work"}, "task-parent")

    assert observed == [
        (
            "session-one",
            {
                "platform": "discord",
                "chat_id": "chat-one",
                "session_key": "agent:main:discord:channel:chat-one",
            },
        ),
        (
            "session-two",
            {
                "platform": "discord",
                "chat_id": "chat-two",
                "session_key": "agent:main:discord:channel:chat-two",
            },
        ),
    ]


def test_route_identity_defaults_to_empty_for_non_gateway_agents(monkeypatch):
    manager = PluginManager()
    monkeypatch.setattr(plugins_mod, "_plugin_manager", manager)
    monkeypatch.setattr(plugins_mod, "get_plugin_manager", lambda: manager)
    observed = []
    manager._hooks["pre_tool_call"] = [
        lambda **kwargs: observed.append(kwargs["route_metadata"])
    ]

    assert tool_hook_route_metadata(SimpleNamespace()) == {}
    assert plugins_mod.resolve_pre_tool_block("read_file", {}) is None
    assert observed == [{}]
