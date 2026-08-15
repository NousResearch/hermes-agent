"""Lifecycle rejection tests for incomplete historical tool calls."""

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tests.run_agent.test_run_agent import agent, _mock_assistant_msg, _mock_tool_call

RESERVED = "__hermes_incomplete_tool_arguments__"


def _run(agent, concurrent, call, messages):
    msg = _mock_assistant_msg(content="", tool_calls=[call])
    method = agent._execute_tool_calls_concurrent if concurrent else agent._execute_tool_calls_sequential
    method(msg, messages, "task-1")


@pytest.mark.parametrize("concurrent", [False, True])
def test_initial_rejection_skips_hooks_and_callbacks(agent, monkeypatch, concurrent):
    events = []
    monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda name: True)
    monkeypatch.setattr(
        "hermes_cli.lifecycle.invoke_hook",
        lambda *args, **kwargs: events.append("hook") or [],
    )
    agent.tool_progress_callback = lambda *args, **kwargs: events.append("progress")
    agent.tool_start_callback = lambda *args, **kwargs: events.append("start")
    agent.tool_complete_callback = lambda *args, **kwargs: events.append("complete")
    agent._touch_activity = lambda *args, **kwargs: events.append("activity")
    payload = {RESERVED: {"version": 1, "replayable": False}}
    call = _mock_tool_call(name="terminal", arguments=json.dumps(payload), call_id="c1")
    messages = []
    with patch("run_agent.handle_function_call", side_effect=AssertionError("dispatch")):
        _run(agent, concurrent, call, messages)
    assert events == []
    assert len(messages) == 1
    assert json.loads(messages[0]["content"])["error_type"] == "incomplete_historical_tool_arguments"


@pytest.mark.parametrize("concurrent", [False, True])
def test_deferred_rejection_skips_middleware_hooks_and_callbacks(
    agent, monkeypatch, concurrent
):
    from tools.registry import registry
    from tools.tool_search import TOOL_CALL_NAME

    events = []
    tool_name = "mcp__integrity_lifecycle__run"
    registry.register(
        name=tool_name,
        toolset="integrity-lifecycle",
        schema={
            "name": tool_name,
            "description": "probe",
            "parameters": {"type": "object", "properties": {}},
        },
        handler=lambda args, **kwargs: events.append("handler") or "handled",
    )
    middleware = lambda **kwargs: events.append("middleware") or None
    manager = SimpleNamespace(
        _middleware={"tool_request": [middleware], "tool_execution": [middleware]}
    )
    monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: manager)
    monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda name: True)
    monkeypatch.setattr(
        "hermes_cli.lifecycle.invoke_hook",
        lambda *args, **kwargs: events.append("hook") or [],
    )
    agent.tool_progress_callback = lambda *args, **kwargs: events.append("progress")
    agent.tool_start_callback = lambda *args, **kwargs: events.append("start")
    agent.tool_complete_callback = lambda *args, **kwargs: events.append("complete")
    agent._touch_activity = lambda *args, **kwargs: events.append("activity")
    inner = json.dumps({RESERVED: {"version": 1, "replayable": False}})
    outer = json.dumps({"name": tool_name, "arguments": inner})
    call = _mock_tool_call(name=TOOL_CALL_NAME, arguments=outer, call_id="c1")
    messages = []
    with patch("run_agent.handle_function_call", side_effect=AssertionError("dispatch")):
        _run(agent, concurrent, call, messages)
    assert events == []
    assert len(messages) == 1
    result = json.loads(messages[0]["content"])
    assert result["error_type"] == "incomplete_historical_tool_arguments"


@pytest.mark.parametrize("concurrent", [False, True])
def test_schema_decoded_rejection_skips_lifecycle(agent, monkeypatch, concurrent):
    from tools.registry import registry

    events = []
    tool_name = "integrity_schema_lifecycle_probe"
    registry.register(
        name=tool_name,
        toolset="integrity-lifecycle",
        schema={
            "name": tool_name,
            "description": "probe",
            "parameters": {
                "type": "object",
                "properties": {"items": {"type": "array"}},
            },
        },
        handler=lambda args, **kwargs: events.append("handler") or "handled",
    )
    middleware = lambda **kwargs: events.append("middleware") or None
    manager = SimpleNamespace(
        _middleware={"tool_request": [middleware], "tool_execution": [middleware]}
    )
    monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: manager)
    monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda name: True)
    monkeypatch.setattr(
        "hermes_cli.lifecycle.invoke_hook",
        lambda *args, **kwargs: events.append("hook") or [],
    )
    agent.tool_progress_callback = lambda *args, **kwargs: events.append("progress")
    agent.tool_start_callback = lambda *args, **kwargs: events.append("start")
    agent.tool_complete_callback = lambda *args, **kwargs: events.append("complete")
    agent._touch_activity = lambda *args, **kwargs: events.append("activity")
    encoded = json.dumps([{RESERVED: {"version": 1, "replayable": False}}])
    call = _mock_tool_call(
        name=tool_name,
        arguments=json.dumps({"items": encoded}),
        call_id="c1",
    )
    messages = []
    with patch("run_agent.handle_function_call", side_effect=AssertionError("dispatch")):
        _run(agent, concurrent, call, messages)
    assert events == []
    assert json.loads(messages[0]["content"])["error_type"] == "incomplete_historical_tool_arguments"


def test_concurrent_spinner_counts_only_runnable_calls(agent, monkeypatch):
    spinner_texts = []
    completion_texts = []

    class SpinnerProbe:
        @staticmethod
        def get_waiting_faces():
            return ["face"]

        def __init__(self, text, **kwargs):
            del kwargs
            spinner_texts.append(text)

        def start(self):
            pass

        def stop(self, text=None):
            completion_texts.append(text)

    monkeypatch.setattr("agent.tool_executor.KawaiiSpinner", SpinnerProbe)
    monkeypatch.setattr(agent, "_should_emit_quiet_tool_messages", lambda: True)
    monkeypatch.setattr(agent, "_should_start_quiet_spinner", lambda: True)
    blocked = _mock_tool_call(
        name="terminal",
        arguments=json.dumps({RESERVED: {"version": 1, "replayable": False}}),
        call_id="blocked",
    )
    runnable = _mock_tool_call(
        name="web_search",
        arguments=json.dumps({"query": "Hermes"}),
        call_id="runnable",
    )
    message = _mock_assistant_msg(content="", tool_calls=[blocked, runnable])

    with patch("run_agent.handle_function_call", return_value="ok"):
        agent._execute_tool_calls_concurrent(message, [], "task-1")

    assert len(spinner_texts) == 1
    assert "running 1 tools concurrently" in spinner_texts[0]
    assert len(completion_texts) == 1
    assert completion_texts[0].startswith("⚡ 1/1 tools completed")


@pytest.mark.parametrize("concurrent", [False, True])
def test_clean_arguments_skip_schema_integrity_preview(
    agent, monkeypatch, concurrent
):
    call = _mock_tool_call(
        name="web_search",
        arguments=json.dumps({"query": "Hermes"}),
        call_id="clean",
    )
    messages = []
    monkeypatch.setattr(
        "agent.tool_executor._schema_decoded_integrity_result",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("schema preview should be skipped")
        ),
    )

    with patch("run_agent.handle_function_call", return_value="ok"):
        _run(agent, concurrent, call, messages)

    assert len(messages) == 1
    assert messages[0]["content"] == "ok"
