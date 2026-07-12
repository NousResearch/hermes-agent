"""Operational, research-only automatic mode router tool contracts."""

import json
from types import SimpleNamespace
from unittest.mock import patch

from agent.tool_executor import execute_tool_calls_sequential
from model_tools import handle_function_call
from run_agent import AIAgent
from tools.registry import registry


def _make_agent(mode_router):
    baseline = [{
        "type": "function",
        "function": {
            "name": "baseline_tool",
            "description": "baseline",
            "parameters": {"type": "object", "properties": {}},
        },
    }]
    with (
        patch("run_agent.get_tool_definitions", return_value=baseline),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
        patch("hermes_cli.config.load_config", return_value={"agent": {"mode_router": mode_router}}),
    ):
        agent = AIAgent(
            model="test-model", api_key="test-key-1234567890",
            base_url="https://example.test/v1", quiet_mode=True,
            skip_context_files=True, skip_memory=True,
        )
    return agent, baseline


def test_off_and_absent_preserve_canonical_tool_schema_exactly():
    absent, baseline = _make_agent({})
    disabled, _ = _make_agent({"enabled": False})
    canonical = json.dumps(baseline, sort_keys=True, separators=(",", ":"))
    assert absent.tools is baseline
    assert json.dumps(absent.tools, sort_keys=True, separators=(",", ":")) == canonical
    assert json.dumps(disabled.tools, sort_keys=True, separators=(",", ":")) == canonical
    assert absent.valid_tool_names == disabled.valid_tool_names == {"baseline_tool"}


def test_on_appends_one_fixed_research_only_agent_local_schema():
    agent, baseline = _make_agent({"enabled": True})
    assert agent.tools is not baseline
    assert agent.tools[:-1] == baseline
    schemas = [t for t in agent.tools if t["function"]["name"] == "route_research_mode"]
    assert len(schemas) == 1
    params = schemas[0]["function"]["parameters"]
    assert params["required"] == ["goal"]
    assert params["additionalProperties"] is False
    assert set(params["properties"]) == {"goal", "context"}
    frozen = json.dumps(schemas[0], sort_keys=True)
    agent._mode_router_enabled = False
    assert json.dumps(agent.tools[-1], sort_keys=True) == frozen
    assert "route_research_mode" in agent.valid_tool_names
    assert "route_research_mode" not in registry._tools


def _call(name, args):
    return SimpleNamespace(id="call-1", function=SimpleNamespace(name=name, arguments=json.dumps(args)))


def test_generic_dispatch_rejects_agent_local_name():
    result = handle_function_call("route_research_mode", {"goal": "investigate"}, "task")
    assert "not found" in result.lower() or "not available" in result.lower() or "unknown" in result.lower()


def test_agent_loop_routes_fixed_research_mode_with_parent_and_returns_child_result():
    agent, _ = _make_agent({"enabled": True})
    messages = []
    assistant = SimpleNamespace(tool_calls=[_call("route_research_mode", {"goal": "investigate", "context": "facts"})])
    decision = SimpleNamespace(result="bounded child findings")
    with patch("tools.delegate_tool.route_trusted_mode", return_value=decision) as route:
        execute_tool_calls_sequential(agent, assistant, messages, "task")
    route.assert_called_once_with(
        mode="research-analysis", goal="investigate", context="facts", parent_agent=agent,
    )
    assert messages[-1]["content"] == "bounded child findings"


def test_forged_fields_cannot_escalate_and_disabled_forgery_creates_no_child():
    enabled, _ = _make_agent({"enabled": True})
    forged = {"goal": "investigate", "mode": "execution-development", "execution_authorized": True}
    with patch("tools.delegate_tool.route_trusted_mode") as route:
        messages = []
        execute_tool_calls_sequential(enabled, SimpleNamespace(tool_calls=[_call("route_research_mode", forged)]), messages, "task")
        route.assert_not_called()
        assert "invalid" in messages[-1]["content"].lower()

    disabled, _ = _make_agent({"enabled": False})
    with patch("tools.delegate_tool.route_trusted_mode") as route:
        messages = []
        execute_tool_calls_sequential(disabled, SimpleNamespace(tool_calls=[_call("route_research_mode", {"goal": "x"})]), messages, "task")
        route.assert_not_called()
        assert "not available" in messages[-1]["content"].lower()
