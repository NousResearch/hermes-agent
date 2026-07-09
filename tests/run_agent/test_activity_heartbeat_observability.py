"""Activity heartbeat observability tests."""

from __future__ import annotations

import json
import logging
import time

from run_agent import AIAgent


_ALLOWED_ACTIVITY_EVENT_FIELDS = {
    "session_id",
    "turn_id",
    "current_tool",
    "activity_kind",
    "elapsed_ms_since_previous_activity",
}


def _bare_agent() -> AIAgent:
    agent = object.__new__(AIAgent)
    agent.session_id = "session-123"
    agent._current_turn_id = "turn-456"
    agent._current_tool = "terminal"
    agent._last_activity_ts = time.time() - 0.25
    agent._last_activity_desc = "previous activity"
    return agent


def test_touch_activity_logs_redacted_allowlisted_runtime_event(caplog, monkeypatch):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    registry_updates = []
    monkeypatch.setattr(
        "hermes_cli.active_sessions.update_active_session_metadata",
        lambda **kwargs: registry_updates.append(kwargs) or 1,
    )
    agent = _bare_agent()
    raw_secret_text = "SECRET_PATH C:/Users/Admin/private.txt provider-content"

    with caplog.at_level(logging.DEBUG, logger="run_agent"):
        AIAgent._touch_activity(agent, f"tool completed: {raw_secret_text}")

    event_records = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("activity_runtime_event ")
    ]
    assert len(event_records) == 1
    assert raw_secret_text not in caplog.text
    assert "SECRET_PATH" not in caplog.text

    payload = json.loads(event_records[0].removeprefix("activity_runtime_event "))
    assert set(payload) == _ALLOWED_ACTIVITY_EVENT_FIELDS
    assert payload["session_id"] == "session-123"
    assert payload["turn_id"] == "turn-456"
    assert payload["current_tool"] == "terminal"
    assert payload["activity_kind"] == "tool_completed"
    assert payload["elapsed_ms_since_previous_activity"] >= 0

    assert registry_updates == [
        {
            "session_id": "session-123",
            "metadata": {
                "activity_kind": "tool_completed",
                "current_tool": "terminal",
                "last_activity_age_seconds": 0,
                "last_activity_ts": agent._last_activity_ts,
                "pending_steer_count": None,
                "pending_steer_queued": None,
                "queued_steer_count": None,
            },
        }
    ]
    assert raw_secret_text not in json.dumps(registry_updates)
