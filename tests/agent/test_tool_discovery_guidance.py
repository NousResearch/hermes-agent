"""Successful terminal background starts and the system prompt stay identical to main in every Tool Search mode."""

import copy
import json

import pytest
from openai.types.chat import ChatCompletion

from agent.coding_context import _TODO_SENTENCE


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


# main's note text, verbatim: successful background starts must stay byte-identical in every mode.
_AUTO_BACKGROUND_NOTE = (
    "Ran in the background (the trailing & was converted to background=true; Hermes tracks the "
    "process). Do NOT re-run it. Use process_manage (poll/log/wait/kill, session_id above) to check it."
)
_SUBAGENT_NOTIFY_NOTE = (
    "You are a subagent: this process's completion notice will NOT reach your parent, and the process is killed when "
    "you finish. Before you finish, either wait for it (process_manage wait), kill it, or hand it to your parent with "
    "process_manage(action='handoff', session_id=..., data='<purpose>') so the parent receives its completion. For CI "
    "watchers prefer returning the fact (PR number, SHA) and letting the parent watch."
)
_START = {"output": "Background process started", "session_id": "proc_fake", "pid": 1,
          "exit_code": 0, "error": None}
_EXPECTED = {
    "notify": {**_START, "notify_on_complete": True},
    "auto": {**_START, "auto_backgrounded": _AUTO_BACKGROUND_NOTE},
    "subagent": {**_START, "notify_on_complete": False, "subagent_note": _SUBAGENT_NOTIFY_NOTE},
}


def _stub_spawn(monkeypatch):
    """Stub only the process launch, so the real background-start code builds the result."""
    from types import SimpleNamespace

    from tools import terminal_tool_background

    monkeypatch.setattr(terminal_tool_background, "_spawn", lambda *_a, **_kw: SimpleNamespace(
        id="proc_fake", pid=1, watcher_platform="", parent_session_id=None))


@pytest.mark.parametrize("case", sorted(_EXPECTED))
@pytest.mark.parametrize("mode", ["deferred", "eager", "disabled"])
def test_workflow_hints_follow_session_tool_scope(tmp_path, monkeypatch, mode, case):
    _stub_spawn(monkeypatch)
    if case == "subagent":
        monkeypatch.setattr("agent.delegation_context.is_delegated_child_context", lambda: True)
    args = ({"command": "echo discovery-probe &"} if case == "auto" else
            {"command": "echo discovery-probe", "background": True, "notify_on_complete": True})
    config = "tools:\n  tool_search:\n    enabled: off\n" if mode == "eager" else ""
    agent = _agent(tmp_path, monkeypatch, config, ["todo"] if mode == "disabled" else None)
    result, _ = _run(agent, monkeypatch, [
        _response(("background", "terminal", args)),
        _response(("missing", "todo_list", {})) if mode == "disabled" else _response(),
        _response(),
    ])
    assert result["completed"]
    prompt = agent._cached_system_prompt
    background = next(m["content"] for m in result["messages"] if m.get("tool_call_id") == "background")
    # Successful background starts keep main's keys and note text in every mode.
    assert json.loads(background) == _EXPECTED[case]
    assert "# Tool discovery" not in prompt
    if mode == "disabled":
        assert "todo_list" not in prompt
        error = next(m["content"] for m in result["messages"] if m.get("tool_call_id") == "missing")
        assert "does not exist" in error
    else:
        # Same todo sentence main picks from the direct tool list (deferred = not direct).
        assert (_TODO_SENTENCE in prompt) == (mode == "eager")
