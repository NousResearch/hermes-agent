"""The accepted input owns its row; transport echoes do not own new rows."""
import threading
from types import SimpleNamespace

import pytest

from agent.codex_runtime import _persist_projected_messages
from agent.message_metadata import append_message
from agent.session_persistence import SessionPersistenceMixin
from agent.transports.codex_event_projector import CodexEventProjector
from hermes_state import SessionDB


@pytest.mark.parametrize("platform_id", ["2146", None])
@pytest.mark.parametrize("projection", ["echo", "assistant_only", "different", "later_equal"])
def test_only_submitted_leading_echo_is_excluded(tmp_path, platform_id, projection):
    db = SessionDB(tmp_path / "state.db")
    try:
        db.create_session(session_id="echo", source="telegram", model="codex")
        agent = SessionPersistenceMixin()
        agent.session_id = "echo"
        agent._session_db = db
        agent._session_db_created = True
        agent._last_flushed_db_idx = 0
        agent._session_persist_lock = threading.RLock()
        messages = []
        # Two independently accepted identical inputs must both survive.
        for turn_index in range(2):
            append_message(messages, {"role": "user", "content": "accepted", "platform_message_id": platform_id})
            assert agent._flush_messages_to_session_db(messages)
            projector = CodexEventProjector()
            items = []
            if projection != "assistant_only":
                items.append({"type": "userMessage", "id": f"u{turn_index}", "content": [
                    {"type": "text", "text": "different" if projection == "different" else "wire caption"}]})
            items.append({"type": "agentMessage", "id": f"a{turn_index}", "text": "reply"})
            if projection == "later_equal":
                items.append({"type": "userMessage", "id": f"s{turn_index}", "content": [
                    {"type": "text", "text": "wire caption"}]})
            projected = []
            for item in items:
                projected.extend(projector.project({"method": "item/completed", "params": {"item": item}}).messages)
            _persist_projected_messages(agent, SimpleNamespace(
                projected_messages=projected, submitted_user_text="wire caption"), messages)
            assert agent._flush_messages_to_session_db(messages)
        users = [row["content"] for row in db.get_messages_as_conversation("echo") if row["role"] == "user"]
        extra = {"different": ["different"], "later_equal": ["wire caption"]}.get(projection, [])
        assert users == (["accepted"] + extra) * 2
    finally:
        db.close()


@pytest.mark.parametrize("user_input", [
    [{"type": "text", "text": "alpha"}, {"type": "text", "text": "beta"}],
    [{"type": "text", "text": "note"}, {"type": "image", "url": "data:image/png;base64,abc"},
     {"type": "text", "text": "caption"}],
    ["alpha", "beta"],
    [{"type": "image", "url": "data:image/png;base64,abc"}],
    [{"type": "text", "text": "  alpha  "}, {"type": "text", "text": ""},
     {"type": "text", "text": "beta\n"}],
])
def test_rich_wire_echo_does_not_create_another_sqlite_user_row(tmp_path, user_input):
    from tests.agent.transports.test_codex_app_server_session import FakeClient, make_session

    client = FakeClient()

    def respond(method, params):
        if method == "thread/start":
            return {"thread": {"id": "thread-fake-001"}}
        if method == "turn/start":
            client._notifications.extend([
                {"method": "item/completed", "params": {"item": {
                    "type": "userMessage", "content": params["input"]}}},
                {"method": "item/completed", "params": {"item": {
                    "type": "agentMessage", "text": "reply"}}},
                {"method": "turn/completed", "params": {"threadId": "thread-fake-001",
                    "turn": {"id": "turn-fake-001", "status": "completed"}}},
            ])
            return {"turn": {"id": "turn-fake-001"}}
        return {}

    client._request_handler = respond
    session = make_session(client)
    db = SessionDB(tmp_path / "rich-echo.db")
    try:
        db.create_session(session_id="rich", source="cli", model="codex")
        agent = SessionPersistenceMixin()
        agent.session_id = "rich"
        agent._session_db = db
        agent._session_db_created = True
        agent._last_flushed_db_idx = 0
        agent._session_persist_lock = threading.RLock()
        messages = []
        append_message(messages, {"role": "user", "content": "accepted original"})
        assert agent._flush_messages_to_session_db(messages)
        turn = session.run_turn(user_input, turn_timeout=1)
        assert turn.error is None and turn.terminal_acknowledged
        _persist_projected_messages(agent, turn, messages)
        assert agent._flush_messages_to_session_db(messages)
        rows = db.get_messages_as_conversation("rich")
        assert [row["role"] for row in rows] == ["user", "assistant"]
        assert rows[0]["content"] == "accepted original"
    finally:
        session.close()
        db.close()
