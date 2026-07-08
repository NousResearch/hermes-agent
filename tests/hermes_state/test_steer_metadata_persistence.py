import json

import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    session_db = SessionDB(db_path=tmp_path / "state.db")
    try:
        yield session_db
    finally:
        session_db.close()


def test_update_message_steer_round_trips_content_and_metadata(db):
    db.create_session(session_id="s1", source="cli")
    db.append_message(
        "s1",
        role="tool",
        content="tool output",
        tool_call_id="call-1",
        tool_name="terminal",
    )

    updated = db.update_message_steer(
        "s1",
        "call-1",
        "tool output\n\n<steer>look at stderr</steer>",
        ["look at stderr"],
    )

    assert updated is True
    messages = db.get_messages("s1")
    assert messages[0]["content"].endswith("<steer>look at stderr</steer>")
    assert messages[0]["_steer_applied"] == ["look at stderr"]

    conversation = db.get_messages_as_conversation("s1")
    assert conversation[0]["content"].endswith("<steer>look at stderr</steer>")
    assert conversation[0]["_steer_applied"] == ["look at stderr"]


def test_update_message_steer_only_updates_active_tool_row(db):
    db.create_session(session_id="s1", source="cli")
    db.append_message(
        "s1",
        role="tool",
        content="archived output",
        tool_call_id="call-1",
    )
    with db._lock:
        db._conn.execute(
            "UPDATE messages SET active = 0 WHERE session_id = ?",
            ("s1",),
        )
        db._conn.commit()
    db.append_message(
        "s1",
        role="tool",
        content="active output",
        tool_call_id="call-1",
    )

    updated = db.update_message_steer(
        "s1",
        "call-1",
        "active output\nsteered",
        ["steered"],
    )

    assert updated is True
    rows = db._conn.execute(
        "SELECT content, steer_applied, active FROM messages "
        "WHERE session_id = ? ORDER BY id",
        ("s1",),
    ).fetchall()
    assert rows[0]["content"] == "archived output"
    assert rows[0]["steer_applied"] is None
    assert rows[0]["active"] == 0
    assert rows[1]["content"] == "active output\nsteered"
    assert json.loads(rows[1]["steer_applied"]) == ["steered"]
    assert rows[1]["active"] == 1


def test_replace_messages_preserves_steer_metadata(db):
    db.create_session(session_id="s1", source="cli")
    db.replace_messages(
        "s1",
        [
            {
                "role": "tool",
                "content": "tool output",
                "tool_call_id": "call-1",
                "_steer_applied": ["focus here"],
            },
        ],
    )

    assert db.get_messages("s1")[0]["_steer_applied"] == ["focus here"]
    assert db.get_messages_as_conversation("s1")[0]["_steer_applied"] == ["focus here"]


def test_archive_and_compact_preserves_steer_metadata(db):
    db.create_session(session_id="s1", source="cli")
    db.append_message("s1", role="user", content="before compact")

    inserted = db.archive_and_compact(
        "s1",
        [
            {
                "role": "tool",
                "content": "compacted tool output",
                "tool_call_id": "call-1",
                "_steer_applied": ["keep this steer"],
            },
        ],
    )

    assert inserted == 1
    conversation = db.get_messages_as_conversation("s1")
    assert len(conversation) == 1
    assert conversation[0]["_steer_applied"] == ["keep this steer"]


def test_get_resume_conversations_preserves_steer_metadata(db):
    db.create_session("solo", "cli")
    db.append_message("solo", role="user", content="hi")
    db.append_message(
        "solo",
        role="assistant",
        content="",
        tool_calls=[
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "terminal", "arguments": "{}"},
            }
        ],
    )
    db.append_message(
        "solo",
        role="tool",
        content="tool output<steer>focus here</steer>",
        tool_call_id="call-1",
        steer_applied=["focus here"],
    )
    db.append_message("solo", role="assistant", content="hello")

    model_history, display_history = db.get_resume_conversations("solo")

    assert model_history[2]["_steer_applied"] == ["focus here"]
    assert display_history[2]["_steer_applied"] == ["focus here"]
