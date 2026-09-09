"""Hosted API construction and refresh keep the room-file surface direct."""

import json
from types import SimpleNamespace

import pytest

from agent.agent_init import _load_tools
from agent.conversation_compression import _rebuild_system_prompt_at_boundary
from gateway.hosted_room_execution_policy import execution_policy_mapping
from gateway.platforms.api_server import APIServerAdapter
from gateway.platforms.base import PlatformConfig
from hermes_constants import get_hermes_home
from model_tools import _dispatch_bridge_tool, get_tool_definitions
from tools.hosted_room_artifact import ensure_share_group_file_tool
from tools.mcp_tool_agent import refresh_agent_mcp_tools
from tools.registry import registry


@pytest.fixture
def build(monkeypatch, tmp_path):
    import run_agent
    from gateway import run

    monkeypatch.chdir(tmp_path)
    config = {"platform_toolsets": {"api_server": ["file"]}, "tools": {"tool_search": {"enabled": "on"}}}

    def tool_agent(**kwargs):
        # Only provider/client initialization is replaced; the real tool phase runs.
        agent = SimpleNamespace(disabled_toolsets=None, **kwargs)
        _load_tools(agent, agent.enabled_toolsets, agent.disabled_toolsets)
        return agent

    monkeypatch.setattr(run_agent, "AIAgent", tool_agent)
    monkeypatch.setattr(run, "_resolve_runtime_agent_kwargs", lambda: {})
    monkeypatch.setattr(run, "_resolve_gateway_model", lambda: "offline")
    monkeypatch.setattr(run, "_load_gateway_config", lambda: config)
    monkeypatch.setattr(run.GatewayRunner, "_load_reasoning_config", staticmethod(lambda model="": None))
    monkeypatch.setattr(run.GatewayRunner, "_load_fallback_model", staticmethod(lambda: None))
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    monkeypatch.setattr(adapter, "_ensure_session_db", lambda: None)

    def construct(*, hosted=True, defer=False):
        config["tools"]["tool_search"]["defer"] = ["share_group_file"] if defer else []
        (get_hermes_home() / "config.yaml").write_text(json.dumps(config))
        kwargs = {}
        if hosted:
            kwargs = {"room_dispatch": {"room_id": "room"}, "room_execution_policy": execution_policy_mapping(
                target_profile="writer", config=config)}
        return adapter._create_agent(session_id="room-session", **kwargs)

    return construct


def assert_share(agent, count):
    names = [tool["function"]["name"] for tool in agent.tools]
    assert names.count("share_group_file") == count
    assert ("share_group_file" in agent.valid_tool_names) is bool(count)
    assert set(names) == agent.valid_tool_names


def compact(agent):
    agent._cached_system_prompt = None
    agent._invalidate_system_prompt = lambda: None
    agent._build_system_prompt = lambda _: "offline prompt"
    assert _rebuild_system_prompt_at_boundary(agent, "") == "offline prompt"


@pytest.mark.parametrize("refresh", [refresh_agent_mcp_tools, compact])
def test_hosted_construction_and_refresh_retain_one_direct_share(build, refresh):
    agent = build()
    assert "bot_room" in agent.enabled_toolsets
    assert_share(agent, 1)
    assert ensure_share_group_file_tool(agent, force=True)
    for _ in range(2):
        refresh(agent)
        assert_share(agent, 1)
    assert json.loads(registry.dispatch("share_group_file", {"path": "/not/read"}))["ok"] is False


def test_explicit_disable_removes_direct_and_deferred_share(build):
    agent = build()
    refresh_agent_mcp_tools(agent, disabled_override=["bot_room"])
    compact(agent)
    assert_share(agent, 0)
    raw = get_tool_definitions(enabled_toolsets=agent.enabled_toolsets, disabled_toolsets=agent.disabled_toolsets,
                               quiet_mode=True, skip_tool_search_assembly=True)
    assert all(tool["function"]["name"] != "share_group_file" for tool in raw)


def test_explicit_deferral_remains_deferred_without_granting_dispatch(build):
    agent = build(defer=True)
    refresh_agent_mcp_tools(agent)
    compact(agent)
    assert_share(agent, 0)
    assert _dispatch_bridge_tool(
        "tool_call", {"name": "share_group_file", "arguments": {"path": "/not/read"}},
        agent.enabled_toolsets, agent.disabled_toolsets,
    ) == (None, ("share_group_file", {"path": "/not/read"}))
    result = json.loads(registry.dispatch("share_group_file", {"path": "/not/read"}))
    assert result["ok"] is False and "only during a Group Chat turn" in result["error"]


def test_ordinary_api_agent_is_not_enabled(build):
    agent = build(hosted=False)
    assert "bot_room" not in agent.enabled_toolsets
    refresh_agent_mcp_tools(agent)
    compact(agent)
    assert_share(agent, 0)
