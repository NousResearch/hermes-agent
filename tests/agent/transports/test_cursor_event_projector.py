"""Tests for Cursor SDK event projection into Hermes messages."""

from agent.transports.cursor_event_projector import CursorEventProjector


def test_assistant_text_projects_to_message():
    projector = CursorEventProjector()
    result = projector.project(
        {
            "type": "assistant",
            "message": {
                "content": [{"type": "text", "text": "Hello from Cursor"}],
            },
        }
    )
    assert len(result.messages) == 1
    assert result.messages[0]["role"] == "assistant"
    assert result.messages[0]["content"] == "Hello from Cursor"
    assert result.final_text == "Hello from Cursor"


def test_tool_call_running_then_completed():
    projector = CursorEventProjector()
    running = projector.project(
        {
            "type": "tool_call",
            "call_id": "call-1",
            "name": "web_search",
            "status": "running",
            "args": {"query": "hermes agent"},
        }
    )
    assert running.messages == []

    completed = projector.project(
        {
            "type": "tool_call",
            "call_id": "call-1",
            "name": "web_search",
            "status": "completed",
            "result": {"hits": 1},
        }
    )
    assert completed.is_tool_iteration is True
    assert len(completed.messages) == 2
    assert completed.messages[0]["tool_calls"][0]["function"]["name"] == "web_search"
    assert completed.messages[1]["role"] == "tool"
