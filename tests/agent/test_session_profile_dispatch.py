"""Profile identity crosses real agent construction, dispatch and observer boundaries."""
import json
import sys
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import model_tools
from agent.tool_executor import execute_tool_calls_sequential, execute_tool_calls_concurrent
from gateway.run_turn_runner import TurnRunner
from gateway.run_agent_cache import GatewayAgentCacheMixin
from gateway.session_identity import RoutingIdentity
from hermes_constants import set_hermes_home_override, reset_hermes_home_override
from run_agent import AIAgent

NAME = "mcp_profile_probe"
SCHEMA = {"name": NAME, "description": "Profile probe",
          "parameters": {"type": "object", "properties": {}}}
DEFS = [{"type": "function", "function": SCHEMA}]


@pytest.fixture
def harness(monkeypatch):
    from hermes_cli import plugins
    registry = model_tools.registry
    monkeypatch.setattr(registry, "_tools", dict(registry._tools))
    seen, hooks = [], []
    def handler(args, **kwargs):
        seen.append(kwargs)
        return json.dumps({"ok": True})
    registry.register(NAME, "mcp-profile-probe", SCHEMA, handler)
    manager = plugins.PluginManager()
    manager._discovered = True
    for name in ("pre_tool_call", "post_tool_call", "transform_tool_result"):
        def observe(_name=name, **kwargs):
            hooks.append((_name, kwargs))
        manager._hooks[name] = [observe]
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)
    monkeypatch.setattr(model_tools, "get_tool_definitions", lambda **kwargs: DEFS)
    monkeypatch.setattr(model_tools, "check_toolset_requirements", lambda **kwargs: {})
    monkeypatch.setattr("agent.process_bootstrap.OpenAI", MagicMock())
    monkeypatch.setattr("agent.model_metadata.fetch_model_metadata", lambda *a, **kw: {})
    return seen, hooks


def make_agent(**kwargs):
    return AIAgent(api_key="test", base_url="https://openrouter.ai/api/v1", model="test",
                   quiet_mode=True, skip_context_files=True, skip_memory=True,
                   skip_background_review=True, **kwargs)


def gateway_agent(profile, tmp_path):
    source = SimpleNamespace(profile="wrong-source", user_id="u", user_id_alt=None,
                             user_name="U", chat_id="c", chat_name=None,
                             chat_type="dm", thread_id=None)
    source._identity = RoutingIdentity("transport", profile, tmp_path, tmp_path)
    ctx = SimpleNamespace(source=source, AIAgent=AIAgent, user_config={},
                          session_id="session-" + profile, session_key="agent:" + profile + ":api:dm:c",
                          enabled_toolsets=None, disabled_toolsets=None)
    runner = SimpleNamespace(_prefill_messages=[], _service_tier=None, _session_db=None,
                             _refresh_fallback_model=lambda: None)
    turn = TurnRunner(runner, ctx)
    return turn._build_fresh_agent(
        {"model": "test", "runtime": {"api_key": "test", "base_url": "https://openrouter.ai/api/v1"}},
        "api", None, 5, None, {}, True)


@pytest.mark.parametrize("path", ["invoke", "sequential", "concurrent"])
def test_gateway_real_agent_dispatch_a_b_a(harness, monkeypatch, tmp_path, path):
    seen, hooks = harness
    monkeypatch.setenv("HERMES_PROFILE_NAME", "launcher")
    agents = [gateway_agent(p, tmp_path) for p in ("research", "work")]
    try:
        for agent in (agents[0], agents[1], agents[0]):
            hooks.clear()
            if path == "invoke":
                result = agent._invoke_tool(NAME, {}, "task", tool_call_id="call")
                assert json.loads(result)["ok"]
            else:
                agent._flush_messages_to_session_db = MagicMock(return_value=True)
                call = SimpleNamespace(id="call", type="function", function=SimpleNamespace(name=NAME, arguments="{}"))
                messages = []
                execute = execute_tool_calls_sequential if path == "sequential" else execute_tool_calls_concurrent
                execute(agent, SimpleNamespace(tool_calls=[call]), messages, "task")
                assert json.loads(messages[-1]["content"])["ok"]
            for hook_name in ("pre_tool_call", "post_tool_call", "transform_tool_result"):
                events = [kw for name, kw in hooks if name == hook_name]
                assert len(events) == 1
                assert events[0]["profile"] == agent._profile_name
                assert events[0]["session_id"] == agent.session_id
        assert [kw["profile"] for kw in seen] == ["research", "work", "research"]
    finally:
        for agent in agents:
            agent.close()


def test_agent_snapshots_request_scope_not_launcher(harness, monkeypatch, tmp_path):
    from hermes_cli import profiles
    monkeypatch.setattr(profiles, "_get_default_hermes_home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_PROFILE_NAME", "launcher")
    token = set_hermes_home_override(str(tmp_path / "profiles" / "research"))
    try:
        agent = make_agent()
    finally:
        reset_hermes_home_override(token)
    try:
        assert agent._profile_name == "research"
        assert json.loads(agent._invoke_tool(NAME, {}, "task"))["ok"]
        assert harness[0][-1]["profile"] == "research"
    finally:
        agent.close()


def test_profile_participates_in_cache_signature():
    sig = GatewayAgentCacheMixin._agent_config_signature
    assert sig("m", {}, [], "", profile_name="research") != sig("m", {}, [], "", profile_name="work")
    assert sig("m", {}, [], "", profile_name="research") == sig("m", {}, [], "", profile_name="research")


def test_real_bridge_keeps_profile_for_handler_and_hooks(harness):
    seen, hooks = harness
    result = model_tools.handle_function_call("tool_call", {"calls": [{"name": NAME, "arguments": {}}]},
                                              profile="research", session_id="session")
    assert json.loads(result)["ok"]
    assert seen[-1]["profile"] == "research"
    assert {name for name, kw in hooks} == {"pre_tool_call", "post_tool_call", "transform_tool_result"}
    assert all(kw["profile"] == "research" and kw["tool_name"] == NAME for _, kw in hooks)


def test_profileless_dispatch_preserves_kwargs(harness):
    seen, hooks = harness
    assert json.loads(model_tools.handle_function_call(NAME, {}))["ok"]
    assert "profile" not in seen[-1]
    assert all("profile" not in kw for _, kw in hooks)


def test_connector_batch_keeps_profile_per_entry(harness, monkeypatch):
    from tools.connectors.gateway import bridge, config
    from tools.registry import invalidate_check_fn_cache
    seen, hooks = harness
    class Client:
        def execute(self, plans):
            return [{"data": "ok", "error": None} for _ in plans]
    monkeypatch.setattr(config, "connectors_available", lambda: True)
    monkeypatch.setattr(bridge, "connectors_available", lambda: True)
    monkeypatch.setattr(bridge, "_default_client_factory", Client)
    # Keep the actual connector catalog, bridge, batch and per-entry dispatcher.
    monkeypatch.setattr(model_tools, "get_tool_definitions", lambda **kw: [
        {"type": "function", "function": {"name": "manage_connections", "parameters": {"type": "object"}}}])
    invalidate_check_fn_cache()
    names = ["connectors__gmail__SEND_EMAIL", "connectors__slack__POST_MESSAGE"]
    result = json.loads(model_tools.handle_function_call("tool_call", {"calls": [
        {"name": name, "arguments": {}} for name in names]}, profile="work", session_id="batch",
        enabled_toolsets=["connections"]))
    assert result["success_count"] == 2
    for name in names:
        events = [kw for _, kw in hooks if kw.get("tool_name") == name]
        assert len(events) == 3
        assert all(kw["profile"] == "work" and kw["session_id"] == "batch" for kw in events)


def test_mcp_registered_handler_resolves_call_scope_a_b_a(harness, monkeypatch, tmp_path):
    from hermes_cli import profiles
    from agent.transports import hermes_tools_mcp_server as server
    monkeypatch.setattr(profiles, "_get_default_hermes_home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_PROFILE_NAME", "launcher")
    class SDK:
        def __init__(self, *a, **kw):
            self.handlers = {}
        def add_tool(self, fn, *, name, description):
            self.handlers[name] = fn
    monkeypatch.setitem(sys.modules, "mcp.server", SimpleNamespace(MCPServer=SDK))
    monkeypatch.setattr(server, "EXPOSED_TOOLS", (NAME,))
    mcp = server._build_server()
    for profile in ("research", "work", "research"):
        token = set_hermes_home_override(str(tmp_path / "profiles" / profile))
        try:
            assert json.loads(mcp.handlers[NAME]())["ok"]
        finally:
            reset_hermes_home_override(token)
    assert [kw["profile"] for kw in harness[0]] == ["research", "work", "research"]
    assert all(kw["profile"] != "launcher" for _, kw in harness[1])


def test_gateway_cache_rebuilds_on_runtime_owner_change(harness, monkeypatch, tmp_path):
    source = SimpleNamespace(profile="research", user_id="u", user_id_alt=None,
                             user_name="U", chat_id="c", chat_name=None, chat_type="dm", thread_id=None)
    ctx = SimpleNamespace(source=source, AIAgent=AIAgent, user_config={}, session_id="sid",
                          session_key="same-key", enabled_toolsets=None, disabled_toolsets=None,
                          _interrupt_depth=0)
    runner = SimpleNamespace(_prefill_messages=[], _service_tier=None, _session_db=None,
                             _refresh_fallback_model=lambda: None, _agent_cache={},
                             _agent_cache_lock=threading.RLock(), _enforce_agent_cache_cap=lambda: None,
                             _agent_config_signature=GatewayAgentCacheMixin._agent_config_signature,
                             _extract_cache_busting_config=lambda cfg: {},
                             _init_cached_agent_for_turn=lambda agent, depth: None,
                             _apply_fallback_chain_to_agent=lambda agent, fallback: None)
    turn = TurnRunner(runner, ctx)
    route = {"model": "test", "runtime": {"api_key": "test", "base_url": "https://openrouter.ai/api/v1"}}
    agents = []
    try:
        first, reused = turn._resolve_turn_agent(route, "api", None, 5, None, {})
        agents.append(first)
        assert not reused
        same, reused = turn._resolve_turn_agent(route, "api", None, 5, None, {})
        assert reused and same is first
        source.profile = "work"
        second, reused = turn._resolve_turn_agent(route, "api", None, 5, None, {})
        agents.append(second)
        assert not reused and second is not first
        assert second._profile_name == "work"
    finally:
        for agent in agents:
            agent.close()


@pytest.mark.parametrize("path", ["direct", "invoke", "sequential"])
def test_profile_hooks_can_still_veto(harness, tmp_path, path):
    from hermes_cli import plugins
    seen, hooks = harness
    plugins.get_plugin_manager()._hooks["pre_tool_call"].append(
        lambda **kw: {"action": "block", "message": "profile policy"})
    agent = gateway_agent("research", tmp_path)
    try:
        if path == "direct":
            result = model_tools.handle_function_call(NAME, {}, profile="research")
        elif path == "invoke":
            result = agent._invoke_tool(NAME, {}, "task")
        else:
            agent._flush_messages_to_session_db = MagicMock(return_value=True)
            call = SimpleNamespace(id="call", type="function", function=SimpleNamespace(name=NAME, arguments="{}"))
            messages = []
            execute_tool_calls_sequential(agent, SimpleNamespace(tool_calls=[call]), messages, "task")
            result = messages[-1]["content"]
        assert "profile policy" in json.loads(result)["error"]
        assert not seen
        events = [kw for name, kw in hooks if name == "post_tool_call"]
        assert len(events) == 1 and events[0]["status"] == "blocked"
        assert events[0]["profile"] == "research"
    finally:
        agent.close()


def test_real_codex_spawn_carries_agent_profile_to_managed_mcp(harness, monkeypatch, tmp_path):
    from agent import codex_runtime
    from agent.transports import codex_app_server as cas, hermes_tools_mcp_server as server
    commands = []
    class Process:
        def __init__(self, command, **kw):
            commands.append(command)
            self.stdin = self.stdout = self.stderr = None
            self.pid = 2147483647
        def poll(self):
            return None
        def terminate(self):
            pass
        def wait(self, timeout=None):
            return 0
    class SDK:
        def __init__(self, *a, **kw):
            self.handlers = {}
        def add_tool(self, fn, *, name, description):
            self.handlers[name] = fn
    monkeypatch.setattr(cas.subprocess, "Popen", Process)
    # No host process tree belongs to this simulated transport, including on a
    # failing assertion. Never pass a real PID to the production teardown.
    monkeypatch.setattr(cas, "_snapshot_descendants", lambda pid: [])
    monkeypatch.setattr(cas.CodexAppServerClient, "initialize", lambda *a, **kw: {})
    monkeypatch.setattr(cas.CodexAppServerClient, "request", lambda *a, **kw: {"thread": {"id": "thread"}})
    monkeypatch.setattr(codex_runtime, "_codex_developer_instructions", lambda agent: "test prompt")
    monkeypatch.setenv("HERMES_PROFILE_NAME", "launcher")
    monkeypatch.delenv("HERMES_TOOL_PROFILE_NAME", raising=False)
    monkeypatch.setitem(sys.modules, "mcp.server", SimpleNamespace(MCPServer=SDK))
    monkeypatch.setattr(server, "EXPOSED_TOOLS", (NAME,))
    agent = gateway_agent("research", tmp_path)
    try:
        codex_runtime._ensure_codex_session(agent, [])
        assert agent._codex_session.ensure_started() == "thread"
        prefix = f"mcp_servers.{server.HERMES_TOOLS_MCP_SERVER_NAME}.env.HERMES_TOOL_PROFILE_NAME="
        overrides = [arg for arg in commands[-1] if arg.startswith(prefix)]
        assert len(overrides) == 1
        # Simulate only Codex applying its per-server environment; server registration
        # and dispatch remain real, with a conflicting launcher profile.
        monkeypatch.setenv("HERMES_TOOL_PROFILE_NAME", json.loads(overrides[0][len(prefix):]))
        mcp = server._build_server()
        assert json.loads(mcp.handlers[NAME]())["ok"]
        assert harness[0][-1]["profile"] == "research"
        assert all(kw["profile"] == "research" for _, kw in harness[1])
        agent._codex_session._client._closed = True
    finally:
        agent.close()
