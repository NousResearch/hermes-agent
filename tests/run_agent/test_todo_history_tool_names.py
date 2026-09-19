"""Current and legacy todo calls must survive paired history hydration."""

import json
from unittest.mock import patch

import pytest

from hermes_constants import get_hermes_home
from run_agent import AIAgent
from tools.todo_tool import TODO_SCHEMA, TodoStore


@pytest.fixture
def agent():
    (get_hermes_home() / "config.yaml").write_text(
        "model:\n  default: test-model\n  provider: custom\n"
        "  base_url: http://localhost:9/v1\n  context_length: 128000\n",
        encoding="utf-8",
    )
    # Keep client/discovery offline; dispatch, TodoStore and hydration are real.
    with (
        patch("model_tools.get_tool_definitions", return_value=[
            {"type": "function", "function": TODO_SCHEMA}
        ]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        instance = AIAgent(
            api_key="test", provider="custom", model="test-model",
            base_url="http://localhost:9/v1", quiet_mode=True,
            skip_context_files=True, skip_memory=True,
        )
        try:
            yield instance
        finally:
            instance.close()


@pytest.mark.parametrize("tool_name", [TODO_SCHEMA["name"], "todo"])
def test_dispatched_todo_snapshot_survives_history_hydration(agent, tool_name):
    todos = [{"id": "verify", "content": "Verify the change", "status": "in_progress"}]
    result = agent._invoke_tool(TODO_SCHEMA["name"], {"todos": todos}, "test-task")
    snapshot = agent._todo_store.snapshot()
    assert snapshot["todos"] == todos
    history = [
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": "todo-call", "type": "function",
            "function": {"name": tool_name, "arguments": json.dumps({"todos": todos})},
        }]},
        {"role": "tool", "tool_call_id": "todo-call", "content": result},
    ]
    # New agent per gateway message starts with an empty store.
    agent._todo_store = TodoStore()
    agent._hydrate_todo_store(history)
    assert agent._todo_store.snapshot() == snapshot
    injection = agent._todo_store.format_for_injection()
    assert injection is not None and todos[0]["content"] in injection


@pytest.mark.parametrize("break_pair", ["wrong_id", "wrong_tool", "missing_call", "user", "system"])
def test_todo_snapshot_requires_matching_uninterrupted_call(agent, break_pair):
    todos = [{"id": "verify", "content": "Verify the change", "status": "pending"}]
    result = agent._invoke_tool(TODO_SCHEMA["name"], {"todos": todos}, "test-task")
    history = [
        {"role": "assistant", "tool_calls": [{
            "id": "todo-call", "type": "function",
            "function": {"name": TODO_SCHEMA["name"], "arguments": "{}"},
        }]},
        {"role": "tool", "tool_call_id": "todo-call", "content": result},
    ]
    if break_pair == "wrong_id":
        history[-1]["tool_call_id"] = "other-call"
    elif break_pair == "wrong_tool":
        history[0]["tool_calls"][0]["function"]["name"] = "not_todo_list"
    elif break_pair == "missing_call":
        history.pop(0)
    else:
        history.insert(1, {"role": break_pair, "content": "new boundary"})
    agent._todo_store = TodoStore()
    agent._hydrate_todo_store(history)
    assert agent._todo_store.snapshot() == {"todos": [], "revision": 0}
    assert agent._todo_store.format_for_injection() is None
