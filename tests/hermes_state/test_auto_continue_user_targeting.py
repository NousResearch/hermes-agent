"""Synthetic auto-continuation users persist but are not user-turn targets."""

import pytest

from hermes_state import SessionDB


MARKER = "[Continuing after max-iteration exhaustion]"


@pytest.fixture()
def db(tmp_path):
    session_db = SessionDB(db_path=tmp_path / "state.db")
    session_db.create_session("session", source="test")
    yield session_db
    session_db.close()


def test_recent_user_messages_ignore_persisted_auto_continuation_markers(db):
    original_id = db.append_message("session", "user", "original task")
    lowercase_literal_id = db.append_message(
        "session",
        "user",
        "[continuing after max-iteration exhaustion] is text I authored",
    )
    db.append_message(
        "session",
        "assistant",
        "The iteration budget was exhausted; starting a fresh continuation turn.",
    )
    marker_id = db.append_message("session", "user", f"{MARKER}\nkeep going")
    db.append_message("session", "assistant", "finished")

    rows = db.list_recent_user_messages("session", limit=2)
    persisted = db.get_messages("session")

    assert [row["id"] for row in rows] == [lowercase_literal_id, original_id]
    assert any(row["id"] == marker_id and row["role"] == "user" for row in persisted)
