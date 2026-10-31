"""Bridge rejections remain visible and count toward loop guards (#139728)."""

import json
import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.tool_guardrails import LoopCapConfig, ToolCallGuardrailConfig, ToolCallGuardrailController
from tests.agent.test_tool_call_guardrail_runtime import _make_agent, _mock_tool_call
from tools.registry import registry


TOOL = "mcp_bridge_guardrail_probe"


@pytest.fixture
def deferred_tool(monkeypatch, request):
    calls = []
    tool_name = getattr(request, "param", TOOL)
    argument = "query" if tool_name == "web_search" else "value"
    schema = {
        "name": tool_name,
        "description": "Deferred loop-guard probe",
        "parameters": {
            "type": "object",
            "properties": {argument: {"type": "string"}},
            "required": [argument],
        },
    }

    def handler(args, **kwargs):
        calls.append(dict(args))
        return json.dumps({"ok": args[argument]})

    monkeypatch.setattr(
        "model_tools.get_tool_definitions",
        lambda **kwargs: [{"type": "function", "function": schema}],
    )
    with patch.dict(registry._tools):
        registry.register(tool_name, "mcp-bridge-guardrail-probe", schema, handler, override=True)
        yield calls
        registry.deregister(tool_name)


def _execute(agent, messages, call_ids, *, args=None, concurrent=False, tool_name=TOOL):
    calls = [
        _mock_tool_call(
            "tool_call", json.dumps({"name": tool_name, "arguments": args or {}}), call_id,
        )
        for call_id in call_ids
    ]
    execute = agent._execute_tool_calls_concurrent if concurrent else agent._execute_tool_calls_sequential
    execute(SimpleNamespace(content="", tool_calls=calls), messages, "bridge-guardrail-task")


@pytest.mark.parametrize("rejection", ["schema", "scope"])
def test_unattended_rejections_halt_without_dispatch(deferred_tool, monkeypatch, rejection):
    agent = _make_agent("tool_call", TOOL, platform="telegram")
    if rejection == "scope":
        monkeypatch.setattr("model_tools.get_tool_definitions", lambda **kwargs: [])
    messages = []

    for index in range(5):
        _execute(agent, messages, [f"rejected-{index}"])

    assert deferred_tool == []
    assert len(messages) == 5
    decision = agent._tool_guardrail_halt_decision
    assert decision is not None
    assert decision.code in {"identical_call_streak_halt", "same_tool_failure_halt"}
    assert decision.tool_name == TOOL
    assert decision.code in messages[-1]["content"]
    if rejection == "scope":
        assert "not available in this session" in messages[-1]["content"]


def test_attended_rejection_keeps_schema_and_appends_warning(deferred_tool):
    agent = _make_agent("tool_call", TOOL, platform="cli")
    messages = []

    for index in range(2):
        _execute(agent, messages, [f"missing-{index}"])

    content = messages[-1]["content"]
    payload, end = json.JSONDecoder().raw_decode(content)
    assert "missing required argument(s): value" in payload["error"]
    assert 'Parameters schema: {"type": "object"' in payload["error"]
    assert '"required": ["value"]' in payload["error"]
    assert "repeated_exact_failure_warning" in content[end:]
    assert content.index("missing required") < content.index("repeated_exact_failure_warning")
    assert agent._tool_guardrail_halt_decision is None
    assert deferred_tool == []


@pytest.mark.parametrize("deferred_tool", ["process_manage"], indirect=True)
@pytest.mark.parametrize("platform", ["telegram", "cli"])
def test_rejected_poller_replay_obeys_surface_hard_stop(deferred_tool, platform):
    agent = _make_agent("tool_call", "process_manage", platform=platform)
    messages = []
    with patch.object(agent, "_append_guardrail_observation", wraps=agent._append_guardrail_observation) as observe:
        for index in range(6):
            _execute(agent, messages, [f"poll-{index}"], tool_name="process_manage")

    assert deferred_tool == []
    assert len(messages) == 6
    if platform == "telegram":
        assert "repeated_exact_failure_block" in messages[-1]["content"]
        assert agent._tool_guardrail_halt_decision.code == "repeated_exact_failure_block"
        assert agent._tool_guardrail_halt_decision.tool_name == "process_manage"
        assert observe.call_count == 5
    else:
        assert "repeated_exact_failure_warning" in messages[-1]["content"]
        assert "repeated_exact_failure_block" not in messages[-1]["content"]
        assert agent._tool_guardrail_halt_decision is None
        assert observe.call_count == 6


def test_concurrent_rejections_observed_on_main_thread_in_call_order(deferred_tool):
    agent = _make_agent("tool_call", TOOL, platform="cron")
    messages, observations = [], []
    main_thread = threading.get_ident()
    append_observation = agent._append_guardrail_observation

    def observe(name, args, result, *, failed, tool_call_id):
        observations.append((threading.get_ident(), tool_call_id, name, args, failed))
        return append_observation(name, args, result, failed=failed, tool_call_id=tool_call_id)

    call_ids = [f"parallel-{index}" for index in range(6)]
    with patch.object(agent, "_append_guardrail_observation", side_effect=observe):
        for index in range(0, 6, 2):
            _execute(agent, messages, call_ids[index:index + 2], concurrent=True)

    assert observations == [(main_thread, call_id, TOOL, {}, True) for call_id in call_ids]
    assert [message["tool_call_id"] for message in messages] == call_ids
    assert agent._tool_guardrail_halt_decision is not None
    assert agent._tool_guardrail_halt_decision.code == "identical_call_streak_halt"
    assert deferred_tool == []
    for message in messages:
        content = message["content"]
        payload, _ = json.JSONDecoder().raw_decode(content)
        rejection = json.loads(payload["error"])
        assert "missing required argument(s): value" in rejection["error"]
        assert rejection["parameters"]["required"] == ["value"]
    assert "identical_call_streak_halt" in messages[-1]["content"]


def test_corrected_call_resets_underlying_tool_failure_count(deferred_tool):
    agent = _make_agent("tool_call", TOOL, platform="telegram")
    messages = []

    for index in range(8):
        # Distinct failures exercise per-tool reset without triggering exact replay blocking.
        _execute(agent, messages, [f"bad-{index}"], args={"attempt": index})
        assert agent._tool_guardrails._same_tool_failure_counts == {TOOL: 1}
        _execute(agent, messages, [f"fixed-{index}"], args={"value": f"fixed-{index}"})
        assert agent._tool_guardrails._same_tool_failure_counts == {}
        assert json.dumps({"ok": f"fixed-{index}"}) in messages[-1]["content"]
        assert agent._tool_guardrail_halt_decision is None

    assert deferred_tool == [{"value": f"fixed-{index}"} for index in range(8)]


@pytest.mark.parametrize("concurrent", [False, True])
def test_plugin_blocks_stay_unobserved(deferred_tool, concurrent):
    agent = _make_agent("tool_call", TOOL, platform="telegram")
    messages = []
    with (
        patch("hermes_cli.plugins._dispatch_pre_tool_call_hooks", return_value=("plugin policy", None)),
        patch.object(agent, "_append_guardrail_observation", wraps=agent._append_guardrail_observation) as observe,
    ):
        for index in range(5):
            _execute(agent, messages, [f"plugin-{index}"], args={"value": "ok"}, concurrent=concurrent)

    observe.assert_not_called()
    assert agent._tool_guardrail_halt_decision is None
    assert agent._tool_guardrails._same_tool_failure_counts == {}
    assert deferred_tool == []
    assert len(messages) == 5
    assert all("plugin policy" in message["content"] for message in messages)
    assert all("repeated_exact_failure_warning" not in message["content"] for message in messages)


def test_before_call_without_loop_caps_neither_spends_nor_meets_search_cap():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(loop_caps=LoopCapConfig(max_web_searches=1))
    )
    args = {"query": "x"}

    assert controller.before_call("web_search", args, loop_caps=False).allows_execution
    assert controller._turn_web_search_count == 0
    assert controller.before_call("web_search", args).allows_execution
    assert controller._turn_web_search_count == 1
    assert controller.before_call("web_search", args, loop_caps=False).allows_execution
    assert controller._turn_web_search_count == 1
    assert controller.halt_decision is None

    decision = controller.before_call("web_search", args)
    assert decision.code == "loop_web_search_cap"
    assert decision.should_halt
    assert controller._turn_web_search_count == 1


@pytest.mark.parametrize("deferred_tool", ["web_search"], indirect=True)
def test_attended_rejected_search_preserves_cap_for_corrected_call(deferred_tool, monkeypatch):
    config = {
        "tools": {"tool_search": {"defer": ["web_search"]}},
        "tool_loop_guardrails": {"loop_caps": {"max_web_searches": 1}},
    }
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: config)
    agent = _make_agent(
        "tool_call", "web_search", platform="cli", config=config,
    )
    messages = []

    _execute(agent, messages, ["missing-query"], tool_name="web_search")
    assert "missing required argument(s): query" in messages[-1]["content"]
    assert deferred_tool == []
    _execute(agent, messages, ["corrected-query"], tool_name="web_search", args={"query": "x"})

    assert deferred_tool == [{"query": "x"}]
    assert json.dumps({"ok": "x"}) in messages[-1]["content"]
    assert all("loop_web_search_cap" not in message["content"] for message in messages)
    assert agent._tool_guardrails._turn_web_search_count == 1
    assert agent._tool_guardrail_halt_decision is None
