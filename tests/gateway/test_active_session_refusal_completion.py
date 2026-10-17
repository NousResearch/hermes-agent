"""Refused turns must publish a completion instead of failing during payload copy."""
import pytest

from gateway.session_events import SessionEvents
from gateway.session_settlement_recovery import completion_payload
from hermes_cli.active_sessions import (
    MAX_CONCURRENT_SESSIONS,
    SESSION_COORDINATION_UNAVAILABLE,
    SESSION_NOT_OWNED,
    ActiveSessionRefusal,
)


@pytest.mark.parametrize("reason", [
    MAX_CONCURRENT_SESSIONS, SESSION_NOT_OWNED, SESSION_COORDINATION_UNAVAILABLE, None,
])
def test_completion_publishes_and_replays_refusal_or_ordinary_response(reason):
    response = ActiveSessionRefusal("Session unavailable.", reason) if reason else "Done."
    outcome = "failed" if reason else "completed"
    payload = completion_payload(
        {"admission_id": "a1"}, {"outcome": outcome}, response, None,
    )
    stream = SessionEvents()
    received = []
    stream.observers.add(received.append)

    assert stream.publish("session", payload) is True
    assert len(received) == 1
    params = received[0]["params"]
    assert params["type"] == "message.complete"
    assert params["session_id"] == "session"
    assert params["payload"] == payload
    assert params["payload"]["status"] == ("error" if reason else "complete")
    if reason:
        assert params["payload"]["text"].reason == reason
        assert params["payload"]["content"].reason == reason

    replay = stream.since(stream.epoch, 0)
    assert replay["truncated"] is False
    assert replay["count"] == 1
    assert replay["events"][0]["payload"] == payload
