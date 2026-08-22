"""Behavior contract for incremental session-history revisions."""

import sqlite3

import pytest

from hermes_state import SessionDB


def _inventory(db: SessionDB) -> dict[str, str]:
    rows = db._conn.execute(
        "SELECT session_id, revision FROM session_history_revisions"
    ).fetchall()
    return {row["session_id"]: row["revision"] for row in rows}


def _assert_revision(token: str) -> None:
    assert len(token) == 32
    int(token, 16)


@pytest.fixture()
def db(tmp_path):
    session_db = SessionDB(tmp_path / "state.db")
    yield session_db
    session_db.close()


def test_contract_tracks_session_and_message_changes(db):
    version = db._conn.execute(
        "SELECT version FROM session_history_contract WHERE singleton = 1"
    ).fetchone()[0]
    assert version >= 1
    assert _inventory(db) == {}

    db.create_session("s1", source="cli")
    created = _inventory(db)["s1"]
    _assert_revision(created)

    db.update_session_cwd("s1", "/tmp/revision-test")
    session_updated = _inventory(db)["s1"]
    assert session_updated != created

    message_id = db.append_message("s1", role="user", content="before", timestamp=123.0)
    message_inserted = _inventory(db)["s1"]
    assert message_inserted != session_updated

    # The row identity and timestamp stay fixed: payload-only rewrites still
    # have to invalidate an incremental consumer's cached session.
    db._execute_write(
        lambda conn: conn.execute(
            "UPDATE messages SET content = ? WHERE id = ?",
            ("after", message_id),
        )
    )
    message_rewritten = _inventory(db)["s1"]
    assert message_rewritten != message_inserted

    # Reads never advance the token.
    assert db.get_messages("s1")[0]["content"] == "after"
    assert _inventory(db)["s1"] == message_rewritten


def test_reparenting_message_invalidates_both_sessions(db):
    db.create_session("old", source="cli")
    db.create_session("new", source="cli")
    message_id = db.append_message("old", role="user", content="move me")
    before = _inventory(db)

    db._execute_write(
        lambda conn: conn.execute(
            "UPDATE messages SET session_id = ? WHERE id = ?",
            ("new", message_id),
        )
    )

    after = _inventory(db)
    assert after["old"] != before["old"]
    assert after["new"] != before["new"]


def test_message_change_is_scoped_to_its_owning_session(db):
    db.create_session("changed", source="cli")
    db.create_session("unchanged", source="cli")
    changed_id = db.append_message("changed", role="user", content="remove me")
    db.append_message("unchanged", role="user", content="keep me")
    before = _inventory(db)

    db._execute_write(
        lambda conn: conn.execute("DELETE FROM messages WHERE id = ?", (changed_id,))
    )

    after = _inventory(db)
    assert after["changed"] != before["changed"]
    assert after["unchanged"] == before["unchanged"]


def test_deletion_is_visible_as_a_missing_inventory_entry(db):
    db.create_session("delete-me", source="cli")
    db.append_message("delete-me", role="user", content="temporary")
    assert "delete-me" in _inventory(db)

    assert db.delete_session("delete-me") is True
    assert "delete-me" not in _inventory(db)


def test_revisions_follow_transaction_rollback(db):
    db.create_session("s1", source="cli")
    db.append_message("s1", role="user", content="stable")
    before = _inventory(db)["s1"]

    def _fail(conn):
        conn.execute(
            "UPDATE messages SET content = 'rolled back' WHERE session_id = 's1'"
        )
        raise RuntimeError("abort transaction")

    with pytest.raises(RuntimeError, match="abort transaction"):
        db._execute_write(_fail)

    assert db.get_messages("s1")[0]["content"] == "stable"
    assert _inventory(db)["s1"] == before


def test_reopening_database_is_a_revision_noop(tmp_path):
    path = tmp_path / "state.db"
    first = SessionDB(path)
    first.create_session("stable", source="cli")
    first.append_message("stable", role="user", content="unchanged")
    revision = _inventory(first)["stable"]
    first.close()

    reopened = SessionDB(path)
    try:
        assert _inventory(reopened)["stable"] == revision
    finally:
        reopened.close()


def test_writable_open_backfills_missing_revision_state(tmp_path):
    path = tmp_path / "state.db"
    original = SessionDB(path)
    original.create_session("legacy", source="cli")
    original.append_message("legacy", role="user", content="preserved")
    original.close()

    # Model a database created before the contract existed. Dropping the
    # contract and its triggers leaves canonical history untouched.
    conn = sqlite3.connect(path)
    for name in (
        "session_history_session_insert",
        "session_history_session_update",
        "session_history_message_insert",
        "session_history_message_update",
        "session_history_message_move",
        "session_history_message_delete",
    ):
        conn.execute(f"DROP TRIGGER {name}")
    conn.execute("DROP TABLE session_history_revisions")
    conn.execute("DROP TABLE session_history_contract")
    conn.commit()
    conn.close()

    reopened = SessionDB(path)
    try:
        revision = _inventory(reopened)["legacy"]
        _assert_revision(revision)
        assert reopened.get_messages("legacy")[0]["content"] == "preserved"
    finally:
        reopened.close()
