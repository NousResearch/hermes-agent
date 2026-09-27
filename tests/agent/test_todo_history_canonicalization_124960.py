"""Regression for #124960: Todo history pairing follows dispatch canonicalization."""

import json
from unittest.mock import patch

from model_tools import _LEGACY_TOOL_ALIASES
from run_agent import AIAgent
from tools.todo_tool import TODO_SCHEMA, TodoStore


def _agent() -> AIAgent:
    agent = object.__new__(AIAgent)
    agent.quiet_mode = True
    agent._todo_store = TodoStore()
    return agent


def _assistant_call(name: str, call_id: str = "todo-call") -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": "{}"},
        }],
    }


def test_current_todo_list_name_hydrates_eleven_items_across_turns():
    todos = [
        {"id": str(index), "content": f"Task {index}", "status": "pending"}
        for index in range(11)
    ]
    history = [
        _assistant_call(TODO_SCHEMA["name"]),
        {
            "role": "tool",
            "tool_call_id": "todo-call",
            "content": json.dumps({"todos": todos, "revision": 7}),
        },
    ]
    agent = _agent()

    with patch("run_agent._set_interrupt"):
        agent._hydrate_todo_store(history)

    assert agent._todo_store.snapshot() == {"todos": todos, "revision": 7}


def test_history_pairing_uses_the_same_alias_table_as_dispatch(monkeypatch):
    synthetic_alias = "legacy_todo_probe"
    monkeypatch.setitem(_LEGACY_TOOL_ALIASES, synthetic_alias, TODO_SCHEMA["name"])

    assert AIAgent._assistant_has_todo_tool_call(
        _assistant_call(synthetic_alias), "todo-call"
    )


def test_unrelated_alias_and_wrong_call_id_remain_rejected(monkeypatch):
    monkeypatch.setitem(_LEGACY_TOOL_ALIASES, "legacy_reader_probe", "read_file")

    assert not AIAgent._assistant_has_todo_tool_call(
        _assistant_call("legacy_reader_probe"), "todo-call"
    )
    assert not AIAgent._assistant_has_todo_tool_call(
        _assistant_call(TODO_SCHEMA["name"]), "different-call"
    )
