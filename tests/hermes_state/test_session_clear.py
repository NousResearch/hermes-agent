"""Durable same-session conversation clearing."""

import sqlite3

import pytest

from hermes_state import SessionDB


@pytest.fixture()
def db(tmp_path):
    database = SessionDB(tmp_path / "state.db")
    database.create_session("same-session", source="api_server", session_key="api:room-1")
    conn = database._conn
    assert conn is not None
    conn.execute(
        "UPDATE sessions SET title = ?, cwd = ? WHERE id = ?",
        ("Keep this title", "/workspace", "same-session"),
    )
    conn.commit()
    database.append_message("same-session", "user", "before clear")
    database.append_message("same-session", "assistant", "before clear reply")
    yield database
    database.close()


def test_clear_conversation_keeps_session_metadata_and_hides_prior_messages(db):
    """A clear starts a fresh context in the same durable session, not /new."""
    epoch = db.clear_conversation("same-session")

    session = db.get_session("same-session")
    assert epoch == 1
    assert session["id"] == "same-session"
    assert session["title"] == "Keep this title"
    assert session["cwd"] == "/workspace"
    assert session["conversation_epoch"] == 1
    assert session["message_count"] == 0
    assert db.get_messages_as_conversation("same-session") == []
    assert [message["content"] for message in db.get_messages("same-session", include_inactive=True)] == [
        "before clear",
        "before clear reply",
    ]


def test_existing_state_db_gains_conversation_epoch_on_open(tmp_path):
    """Schema reconciliation upgrades stores created before same-session clear."""
    path = tmp_path / "legacy-state.db"
    original = SessionDB(path)
    original.close()
    conn = sqlite3.connect(path)
    try:
        conn.execute("ALTER TABLE sessions DROP COLUMN conversation_epoch")
        conn.commit()
    finally:
        conn.close()

    migrated = SessionDB(path)
    try:
        migrated_conn = migrated._conn
        assert migrated_conn is not None
        session_columns = {
            row[1] for row in migrated_conn.execute("PRAGMA table_info(sessions)")
        }
        assert "conversation_epoch" in session_columns
    finally:
        migrated.close()


def test_first_message_after_clear_is_active_conversation(db):
    """Fresh turns after a clear persist and reload without reviving the old epoch."""
    db.clear_conversation("same-session")
    db.append_message("same-session", "user", "after clear")
    db.append_message("same-session", "assistant", "after clear reply")

    assert [message["content"] for message in db.get_messages_as_conversation("same-session")] == [
        "after clear",
        "after clear reply",
    ]
    assert db.get_session("same-session")["message_count"] == 2


def test_clear_rejects_late_writer_from_prior_conversation_epoch(db):
    """An interrupted turn captured before /clear cannot repopulate the fresh context."""
    from hermes_state_errors import SessionConversationEpochStaleError

    old_epoch = db.get_session("same-session")["conversation_epoch"]
    db.clear_conversation("same-session")
    with pytest.raises(SessionConversationEpochStaleError):
        db.append_messages_batch(
            "same-session",
            [{"role": "assistant", "content": "late pre-clear worker output"}],
            expected_conversation_epoch=old_epoch,
        )
    assert db.get_messages_as_conversation("same-session") == []


def test_pending_recovery_rejects_payload_before_clear(db):
    """Shutdown input queued before /clear cannot be recovered into the new epoch."""
    from hermes_state_errors import SessionConversationEpochStaleError

    db.clear_conversation("same-session")
    with pytest.raises(SessionConversationEpochStaleError):
        db.append_message(
            "same-session", "user", "pre-clear queued prompt", not_after_conversation_clear=0.0,
        )


def test_atomic_turn_lease_epoch_fences_a_clear_after_admission(db):
    """Lease admission returns the pre-clear epoch from the same write transaction."""
    from hermes_state_errors import SessionConversationEpochStaleError

    epoch = db.acquire_session_turn_lease_with_epoch(
        "same-session", "test-turn", wait_seconds=0,
    )
    assert epoch == 0
    db.clear_conversation("same-session")

    with pytest.raises(SessionConversationEpochStaleError):
        db.append_message(
            "same-session", "assistant", "old worker output",
            expected_conversation_epoch=epoch,
        )
    db.release_session_turn_lease("same-session", "test-turn")


def test_atomic_lease_admission_rejects_stale_expected_epoch(db):
    """A queued completion cannot start a fresh internal turn after /clear."""
    old_epoch = 0
    db.clear_conversation("same-session")
    assert db.acquire_session_turn_lease_with_epoch("same-session", "old-completion", wait_seconds=0, expected_conversation_epoch=old_epoch) is None

def test_delegation_delivery_rejects_late_pre_clear_epoch(db):
    """Detached completion persistence has the same atomic boundary as a normal turn."""
    from hermes_state_errors import SessionConversationEpochStaleError

    old_epoch = db.get_session("same-session")["conversation_epoch"]
    db.clear_conversation("same-session")
    with pytest.raises(SessionConversationEpochStaleError):
        db.append_delegation_delivery(
            "same-session", "old completion", {"delegation_id": "deleg-old"},
            expected_conversation_epoch=old_epoch,
        )
    assert db.get_messages_as_conversation("same-session") == []
