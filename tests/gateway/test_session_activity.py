"""Behaviour tests for the replayable cross-surface activity journal."""

import pytest

from gateway import session_activity
from gateway.session_activity import ActivityCursorExpired, SessionActivityStore


def test_activity_events_are_replayable_and_streaming_deltas_coalesce(tmp_path):
    store = SessionActivityStore(tmp_path / "activity.db")
    first = store.append(
        session_id="session-a",
        turn_id="turn-a",
        event_type="message.start",
        payload={"user_text": "Ship this"},
        surface="tui_gateway",
    )
    first_delta = store.append(
        session_id="session-a",
        turn_id="turn-a",
        event_type="message.delta",
        payload={"delta": "Hello "},
        surface="tui_gateway",
    )
    second_delta = store.append(
        session_id="session-a",
        turn_id="turn-a",
        event_type="message.delta",
        payload={"delta": "world"},
        surface="tui_gateway",
    )

    assert first is not None
    assert first_delta is not None
    assert second_delta is not None
    assert second_delta["event_id"] == first_delta["event_id"]
    events = store.events("session-a")
    assert [event["type"] for event in events] == ["message.start", "message.delta"]
    assert events[-1]["payload"]["text"] == "Hello world"
    assert store.latest_cursor("session-a") == second_delta["event_id"]


def test_activity_cursor_expiry_and_turn_retention_are_per_session(tmp_path, monkeypatch):
    monkeypatch.setattr(session_activity, "MAX_TURNS_PER_SESSION", 1)
    store = SessionActivityStore(tmp_path / "activity.db")
    old = store.append(
        session_id="session-a",
        turn_id="turn-old",
        event_type="message.start",
        payload={},
        surface="api_server",
    )
    store.append(
        session_id="session-a",
        turn_id="turn-new",
        event_type="message.start",
        payload={},
        surface="api_server",
    )
    store.append(
        session_id="session-b",
        turn_id="turn-b",
        event_type="message.start",
        payload={},
        surface="api_server",
    )
    store.append(
        session_id="session-a",
        turn_id="turn-newest",
        event_type="message.start",
        payload={},
        surface="api_server",
    )

    assert old is not None
    assert [event["turn_id"] for event in store.events("session-a")] == ["turn-newest"]
    assert [event["turn_id"] for event in store.events("session-b")] == ["turn-b"]
    with pytest.raises(ActivityCursorExpired):
        store.assert_cursor_available("session-a", int(old["event_id"]))


def test_activity_rekeys_and_deletes_with_the_session(tmp_path):
    store = SessionActivityStore(tmp_path / "activity.db")
    store.append(
        session_id="parent",
        turn_id="turn-1",
        event_type="tool.complete",
        payload={"tool_id": "tool-1"},
        surface="tui_gateway",
    )

    store.rekey_session("parent", "continuation")
    assert store.events("parent") == []
    assert store.events("continuation")[0]["payload"]["tool_id"] == "tool-1"
    store.delete_session("continuation")
    assert store.events("continuation") == []
