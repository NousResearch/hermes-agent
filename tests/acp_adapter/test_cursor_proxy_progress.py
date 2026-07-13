import base64
import json
from typing import Any

from acp_adapter import events


def marker(payload):
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    return "\x1ehermes_cursor_event:" + encoded


def test_cursor_tool_markers_become_native_acp_tool_cards(monkeypatch):
    updates = []
    monkeypatch.setattr(events, "_send_update", lambda _conn, _sid, _loop, update: updates.append(update))
    placeholder: Any = object()
    callback = events.make_thinking_cb(placeholder, "session-1", placeholder)

    callback(marker({
        "kind": "tool",
        "phase": "started",
        "callId": "cursor-call-1",
        "name": "read",
        "args": {"path": "/workspace/package.json"},
    }))
    callback(marker({
        "kind": "tool",
        "phase": "completed",
        "callId": "cursor-call-1",
        "name": "read",
        "args": {"path": "/workspace/package.json"},
        "result": {"success": {"content": "{}"}},
    }))

    assert [update.session_update for update in updates] == ["tool_call", "tool_call_update"]
    assert updates[0].title == "read: /workspace/package.json"
    assert updates[1].tool_call_id == updates[0].tool_call_id
    assert updates[1].status == "completed"


def test_regular_cursor_reasoning_stays_in_acp_thought_stream(monkeypatch):
    updates = []
    monkeypatch.setattr(events, "_send_update", lambda _conn, _sid, _loop, update: updates.append(update))
    placeholder: Any = object()
    callback = events.make_thinking_cb(placeholder, "session-1", placeholder)

    callback("Inspecting the package manifest")

    assert len(updates) == 1
    assert updates[0].session_update == "agent_thought_chunk"
    assert updates[0].content.text == "Inspecting the package manifest"
