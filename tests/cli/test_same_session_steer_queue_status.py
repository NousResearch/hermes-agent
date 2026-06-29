import threading
import time
from types import SimpleNamespace

from run_agent import AIAgent


def _status_agent() -> AIAgent:
    agent = object.__new__(AIAgent)
    agent.session_id = "session-queued-steer"
    agent._current_turn_id = "turn-active-123"
    agent._last_activity_ts = time.time() - 2.0
    agent._last_activity_desc = "executing tool: terminal"
    agent._current_tool = "terminal"
    agent._api_call_count = 4
    agent.max_iterations = 90
    agent.iteration_budget = SimpleNamespace(used=7, max_total=90)
    agent._pending_steer = None
    agent._pending_steer_count = 0
    agent._pending_steer_lock = threading.Lock()
    return agent


def test_activity_summary_reports_queued_steer_without_leaking_or_draining_text(monkeypatch):
    registry_updates = []
    monkeypatch.setattr(
        "hermes_cli.active_sessions.update_active_session_metadata",
        lambda **kwargs: registry_updates.append(kwargs) or 1,
    )
    agent = _status_agent()

    assert agent.steer("private operator instruction") is True
    assert agent.steer("second private instruction") is True

    summary = agent.get_activity_summary()

    assert summary["session_id"] == "session-queued-steer"
    assert summary["active_turn_id"] == "turn-active-123"
    assert summary["last_activity_age_seconds"] >= 1
    assert summary["pending_steer_count"] == 2
    assert summary["pending_steer_queued"] is True
    assert "private operator instruction" not in str(summary)
    assert "second private instruction" not in str(summary)

    assert agent._drain_pending_steer() == (
        "private operator instruction\nsecond private instruction"
    )

    assert registry_updates == [
        {
            "session_id": "session-queued-steer",
            "metadata": {
                "pending_steer_count": 1,
                "pending_steer_queued": True,
                "queued_steer_count": 1,
            },
        },
        {
            "session_id": "session-queued-steer",
            "metadata": {
                "pending_steer_count": 2,
                "pending_steer_queued": True,
                "queued_steer_count": 2,
            },
        },
        {
            "session_id": "session-queued-steer",
            "metadata": {
                "pending_steer_count": None,
                "pending_steer_queued": None,
                "queued_steer_count": None,
            },
        },
    ]
    assert "private operator instruction" not in str(registry_updates)
    assert "second private instruction" not in str(registry_updates)
