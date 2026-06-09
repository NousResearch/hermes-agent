"""Live-event bridge for codex app-server turns (agent/codex_runtime.py).

Regression: codex turns were fully silent in connected UIs — the session's
on_event hook was never wired, so streaming text, reasoning, and tool
activity all vanished and clients saw only message.start → message.complete.
"""

from types import SimpleNamespace

from agent.codex_runtime import (
    _codex_live_event,
    _codex_tool_descriptor,
    _codex_tool_result,
)


def _recording_agent():
    calls = {"stream": [], "reasoning": [], "tool_start": [], "tool_complete": []}
    agent = SimpleNamespace(
        _fire_stream_delta=lambda text: calls["stream"].append(text),
        _fire_reasoning_delta=lambda text: calls["reasoning"].append(text),
        tool_start_callback=lambda cid, name, args: calls["tool_start"].append((cid, name, args)),
        tool_complete_callback=lambda cid, name, args, result: calls["tool_complete"].append(
            (cid, name, args, result)
        ),
    )
    return agent, calls


def test_agent_message_delta_streams_text():
    agent, calls = _recording_agent()
    _codex_live_event(agent, {"method": "item/agentMessage/delta", "params": {"delta": "hel"}})
    _codex_live_event(agent, {"method": "item/agentMessage/delta", "params": {"delta": "lo"}})
    assert calls["stream"] == ["hel", "lo"]
    assert calls["reasoning"] == []


def test_reasoning_delta_fires_reasoning_callback():
    agent, calls = _recording_agent()
    _codex_live_event(agent, {"method": "item/reasoning/delta", "params": {"delta": "thinking…"}})
    assert calls["reasoning"] == ["thinking…"]
    assert calls["stream"] == []


def test_command_execution_start_and_complete_fire_tool_events():
    agent, calls = _recording_agent()
    item = {"type": "commandExecution", "id": "abc123", "command": "echo hi", "cwd": "/tmp"}
    _codex_live_event(agent, {"method": "item/started", "params": {"item": item}})
    done = dict(item, aggregatedOutput="hi\n", exitCode=0)
    _codex_live_event(agent, {"method": "item/completed", "params": {"item": done}})

    assert calls["tool_start"] == [
        ("codex_exec_abc123", "exec_command", {"command": "echo hi", "cwd": "/tmp"})
    ]
    (cid, name, args, result) = calls["tool_complete"][0]
    assert cid == "codex_exec_abc123"
    assert name == "exec_command"
    assert result == "hi\n"


def test_failed_command_result_carries_exit_code():
    item = {"type": "commandExecution", "id": "x", "aggregatedOutput": "boom", "exitCode": 2}
    assert _codex_tool_result(item) == "[exit 2]\nboom"


def test_non_tool_items_and_junk_are_ignored():
    agent, calls = _recording_agent()
    _codex_live_event(agent, {"method": "item/started", "params": {"item": {"type": "reasoning"}}})
    _codex_live_event(agent, {"method": "item/completed", "params": {"item": {"type": "agentMessage"}}})
    _codex_live_event(agent, {"method": "turn/completed", "params": {}})
    _codex_live_event(agent, {})
    assert calls == {"stream": [], "reasoning": [], "tool_start": [], "tool_complete": []}


def test_bridge_never_raises_even_when_callbacks_blow_up():
    def _boom(*_a, **_k):
        raise RuntimeError("display crashed")

    agent = SimpleNamespace(_fire_stream_delta=_boom)
    _codex_live_event(agent, {"method": "item/agentMessage/delta", "params": {"delta": "x"}})


def test_descriptor_call_ids_match_projector_history_ids():
    """Live tool cards must correlate with the tool_calls persisted by the
    projector — same deterministic id scheme."""
    from agent.transports.codex_event_projector import _deterministic_call_id

    mcp = {"type": "mcpToolCall", "id": "i1", "server": "srv", "tool": "t", "arguments": {"a": 1}}
    cid, name, args = _codex_tool_descriptor(mcp)
    assert cid == _deterministic_call_id("mcp_srv_t", "i1")
    assert name == "mcp.srv.t"
    assert args == {"a": 1}

    patch = {
        "type": "fileChange",
        "id": "i2",
        "changes": [{"kind": {"type": "add"}, "path": "a.py"}],
    }
    cid, name, args = _codex_tool_descriptor(patch)
    assert cid == _deterministic_call_id("apply_patch", "i2")
    assert name == "apply_patch"
    assert args == {"changes": [{"kind": "add", "path": "a.py"}]}
