"""Inline transforms retain the same per-call identity as the SDK tool observer."""

from types import SimpleNamespace

from agent.agent_runtime_helpers import invoke_tool


def test_inline_transform_uses_call_snapshot_after_executor_changes_agent_identity(monkeypatch):
    agent = SimpleNamespace(session_id="session-original", _current_turn_id="turn-original",
                            _current_api_request_id="request-original")
    events = []

    def execute(_agent, _args, _ctx):
        agent.session_id = "session-next"
        agent._current_turn_id = "turn-next"
        agent._current_api_request_id = "request-next"
        return '{"ok":true}'

    def hook(name, **kwargs):
        if name == "transform_tool_result":
            events.append((name, kwargs))
            return ["transformed"]
        return []

    monkeypatch.setattr("agent.inline_tool_executors.resolve_invoke_tool_executor", lambda *_: execute)
    monkeypatch.setattr("model_tools._emit_post_tool_call_hook",
                        lambda **kwargs: events.append(("post_tool_call", kwargs)))
    monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda name: name == "transform_tool_result")
    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", hook)
    result = invoke_tool(agent, "todo_list", {}, "task-original", tool_call_id="tool-original",
                         pre_tool_block_checked=True, skip_tool_request_middleware=True,
                         skip_tool_execution_middleware=True)

    assert result == "transformed"
    assert [name for name, _ in events] == ["post_tool_call", "transform_tool_result"]
    for _, event in events:
        assert event["result"] == '{"ok":true}'
        assert {key: event[key] for key in (
            "task_id", "session_id", "tool_call_id", "turn_id", "api_request_id"
        )} == {"task_id": "task-original", "session_id": "session-original",
               "tool_call_id": "tool-original", "turn_id": "turn-original",
               "api_request_id": "request-original"}
