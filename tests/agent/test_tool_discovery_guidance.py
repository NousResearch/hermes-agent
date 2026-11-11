"""Deferred tools stay reachable after a direct call, without changing the prompt or schemas."""

import copy
import json

import pytest
from openai.types.chat import ChatCompletion

from agent.coding_context import _NO_TODO_SENTENCE, _TODO_SENTENCE


def _agent(tmp_path, monkeypatch, config="", disabled=None):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "agent:\n  coding_context: on\n" + config, encoding="utf-8"
    )
    from run_agent import AIAgent

    return AIAgent(
        api_key="test-key", base_url="http://127.0.0.1:1/v1",
        provider="openai-compat", model="nemotron-test", max_iterations=10,
        enabled_toolsets=["terminal", "todo"], disabled_toolsets=disabled,
        quiet_mode=True, skip_context_files=True, skip_memory=True,
        save_trajectories=False, platform="cli",
    )


def _response(*calls):
    return ChatCompletion.model_validate({
        "id": "test-response", "object": "chat.completion", "created": 0,
        "model": "nemotron-test",
        "choices": [{"index": 0, "finish_reason": "tool_calls" if calls else "stop",
                     "message": {"role": "assistant", "content": "" if calls else "done",
                                 "tool_calls": [
                                     {"id": call_id, "type": "function", "function": {
                                         "name": name, "arguments": json.dumps(args)}}
                                     for call_id, name, args in calls
                                 ] or None}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
    })


def _run(agent, monkeypatch, responses):
    requests = []
    replies = iter(responses)

    def reply(api_kwargs, **kwargs):
        requests.append(copy.deepcopy(api_kwargs))
        return next(replies)

    monkeypatch.setattr(agent, "_interruptible_api_call", reply)
    monkeypatch.setattr(agent, "_interruptible_streaming_api_call", reply)
    result = agent.run_conversation("Track the work and check background commands.")
    return result, requests


@pytest.mark.parametrize("mixed", [False, True])
def test_deferred_call_recovers_through_bridge_without_changing_prompt(tmp_path, monkeypatch, mixed):
    agent = _agent(tmp_path, monkeypatch)
    todos = [{"id": "verify", "content": "Verify the change", "status": "in_progress"}]
    direct = [("direct", "todo_list", {"todos": todos})]
    if mixed:
        direct.append(("search", "tool_search", {"queries": ["track task progress"]}))
    result, requests = _run(agent, monkeypatch, [
        _response(*direct),
        _response(("describe", "tool_describe", {"names": ["todo_list"]})),
        _response(("invoke", "tool_call", {"calls": [{"name": "todo_list", "arguments": {"todos": todos}}]})),
        _response(),
    ])
    assert result["completed"]
    messages = result["messages"]
    error = next(m["content"] for m in messages if m.get("tool_call_id") == "direct")
    assert "does not exist" not in error
    assert "tool_describe" in error and "tool_call" in error
    assert agent._todo_store.read() == todos
    assert {"todo_list", "process_manage"}.isdisjoint(agent.valid_tool_names)
    assert "tool_search" in agent.valid_tool_names
    prompt = agent._cached_system_prompt
    # Recovery lives in the tool result only: no discovery block, no rewritten todo sentence.
    assert "# Tool discovery" not in prompt
    assert "tool_describe(names=" not in prompt
    assert _NO_TODO_SENTENCE in prompt and _TODO_SENTENCE not in prompt
    # Every request keeps the same system prompt and schema; recovery is a tool result.
    assert len(requests) > 1
    for request in requests[1:]:
        assert request["tools"] == requests[0]["tools"]
        assert [m for m in request["messages"] if m["role"] == "system"] == [
            m for m in requests[0]["messages"] if m["role"] == "system"
        ]
    call_ids = [tc["id"] for m in messages for tc in m.get("tool_calls", [])]
    assert sorted(call_ids) == sorted(m["tool_call_id"] for m in messages if m["role"] == "tool")


_FAKE_START = {"output": "Background process started", "session_id": "proc_fake", "pid": 1,
               "exit_code": 0, "error": None, "notify_on_complete": True}


@pytest.mark.parametrize("mode", ["deferred", "eager", "disabled"])
def test_workflow_hints_follow_session_tool_scope(tmp_path, monkeypatch, mode):
    from tools import terminal_tool

    # Never start a real process: the spawn returns main's background-start payload.
    monkeypatch.setattr(terminal_tool, "spawn_background_process",
                        lambda **_kw: json.dumps(_FAKE_START))
    config = "tools:\n  tool_search:\n    enabled: off\n" if mode == "eager" else ""
    agent = _agent(tmp_path, monkeypatch, config, ["todo"] if mode == "disabled" else None)
    result, _ = _run(agent, monkeypatch, [
        _response(("background", "terminal", {
            "command": "echo discovery-probe", "background": True, "notify_on_complete": True,
        })),
        _response(("missing", "todo_list", {})) if mode == "disabled" else _response(),
        _response(),
    ])
    assert result["completed"]
    prompt = agent._cached_system_prompt
    background = next(m["content"] for m in result["messages"] if m.get("tool_call_id") == "background")
    # Terminal results keep main's shape in every mode: no key added after the tool returns.
    assert json.loads(background) == _FAKE_START
    assert "# Tool discovery" not in prompt
    if mode == "disabled":
        assert "todo_list" not in prompt
        error = next(m["content"] for m in result["messages"] if m.get("tool_call_id") == "missing")
        assert "does not exist" in error
        assert 'tool_describe(names=["todo_list"])' not in error
    else:
        # Same todo sentence main picks from the direct tool list (deferred = not direct).
        assert (_TODO_SENTENCE in prompt) == (mode == "eager")


def test_process_manage_notes_carry_the_bridge_hint():
    from tools.terminal_tool_background import (
        _AUTO_BACKGROUND_NOTE, _PROCESS_MANAGE_BRIDGE_HINT, _SUBAGENT_NOTIFY_NOTE,
    )

    for note in (_AUTO_BACKGROUND_NOTE, _SUBAGENT_NOTIFY_NOTE):
        assert note.endswith(_PROCESS_MANAGE_BRIDGE_HINT)
        assert "tool_call" in note
