"""state.db keeps a write-side ``sessions_revision`` in ``state_meta`` that moves exactly when a
sessions row changes in a column the Desktop session projections read (the change watcher's
signature fields), so a watcher can compare one row instead of hashing the whole table."""
import sqlite3

import pytest

from hermes_state import SessionDB

_NUMERIC = {"started_at", "ended_at", "message_count", "tool_call_count", "archived", "pinned", "hidden",
            "last_read_at"}


def _meta(path, key):
    conn = sqlite3.connect(path)
    try:
        row = conn.execute("SELECT value FROM state_meta WHERE key = ?", (key,)).fetchone()
    finally:
        conn.close()
    return None if row is None else row[0]


def _revision(path):
    value = _meta(path, "sessions_revision")
    return None if value is None else int(value)


@pytest.fixture
def store(tmp_path):
    path = tmp_path / "state.db"
    db = SessionDB(path)
    db.create_session("parent", "cli")
    db.create_session("s1", "cli")
    yield db, path
    db.close()


def _raw(path):
    return sqlite3.connect(path, isolation_level=None)


def test_fresh_store_seeds_revision_and_store_identity(store):
    _db, path = store
    assert _revision(path) is not None
    assert _meta(path, "store_instance_id")


def test_heartbeat_noop_and_untracked_columns_do_not_move_revision(store):
    db, path = store
    before = _revision(path)
    assert before is not None

    db.touch_session_activity("s1", description="thinking")
    conn = _raw(path)
    conn.execute("UPDATE sessions SET title = title, message_count = message_count WHERE id = 's1'")
    conn.execute("UPDATE sessions SET input_tokens = input_tokens + 10, last_activity_at = 99 WHERE id = 's1'")
    conn.close()

    assert _revision(path) == before


def test_every_signature_field_moves_revision(store):
    from hermes_state_common import SESSION_SIGNATURE_FIELDS

    _db, path = store
    assert len(SESSION_SIGNATURE_FIELDS) == 22
    conn = _raw(path)
    sid, moved = "s1", {}
    for field in SESSION_SIGNATURE_FIELDS:
        before = _revision(path)
        if field == "id":
            conn.execute("UPDATE sessions SET id = 's1-renamed' WHERE id = ?", (sid,))
            sid = "s1-renamed"
        elif field == "parent_session_id":
            conn.execute("UPDATE sessions SET parent_session_id = 'parent' WHERE id = ?", (sid,))
        elif field in _NUMERIC:
            conn.execute(f"UPDATE sessions SET {field} = COALESCE({field}, 0) + 1 WHERE id = ?", (sid,))
        else:
            conn.execute(f"UPDATE sessions SET {field} = 'changed-{field}' WHERE id = ?", (sid,))
        moved[field] = _revision(path) > before
    conn.close()

    assert [f for f, ok in moved.items() if not ok] == []


def test_api_writes_move_revision_once_per_row_change(store):
    db, path = store
    steps = []
    for write in (
        lambda: db.set_session_title("s1", "Renamed"),
        lambda: db.set_session_archived("s1", True),
        lambda: db.end_session("s1", "user_exit"),
        lambda: db.create_session("s2", "cli"),
        lambda: db.delete_session("s2"),
    ):
        before = _revision(path)
        write()
        steps.append(_revision(path) > before)

    assert steps == [True, True, True, True, True]


def test_rolled_back_write_leaves_revision_unchanged(store):
    _db, path = store
    before = _revision(path)
    writer, reader = _raw(path), _raw(path)
    writer.execute("BEGIN IMMEDIATE")
    writer.execute("UPDATE sessions SET title = 'never' WHERE id = 's1'")
    writer.execute("INSERT INTO sessions (id, source, started_at) VALUES ('ghost', 'cli', 1)")
    assert int(reader.execute("SELECT value FROM state_meta WHERE key = 'sessions_revision'").fetchone()[0]) == before
    writer.execute("ROLLBACK")
    writer.close()
    reader.close()

    assert _revision(path) == before


def test_upgrade_installs_triggers_and_keeps_existing_identity(tmp_path):
    """A store written by a build without the revision (no triggers, no key) is caught up on the
    next writable open; an existing store_instance_id is kept, and reopening never resets."""
    path = tmp_path / "state.db"
    SessionDB(path).close()
    conn = _raw(path)
    for (name,) in conn.execute("SELECT name FROM sqlite_master WHERE type = 'trigger' AND tbl_name = 'sessions' "
                                "AND name LIKE 'sessions_revision_%'").fetchall():
        conn.execute(f"DROP TRIGGER {name}")
    conn.execute("DELETE FROM state_meta WHERE key = 'sessions_revision'")
    conn.execute("UPDATE state_meta SET value = 'kept-id' WHERE key = 'store_instance_id'")
    conn.close()

    db = SessionDB(path)
    db.create_session("s1", "cli")
    db.set_session_title("s1", "One")
    db.close()
    advanced = _revision(path)
    assert advanced is not None and advanced > 0

    SessionDB(path).close()
    assert _revision(path) == advanced
    assert _meta(path, "store_instance_id") == "kept-id"


def test_null_to_value_and_back_moves_revision_each_time(store):
    """``old.f IS NOT new.f`` is NULL-safe: ``!=`` would read NULL comparisons as unchanged."""
    _db, path = store
    conn = _raw(path)
    seen = [_revision(path)]
    for value in (5.0, None, None):
        conn.execute("UPDATE sessions SET ended_at = ? WHERE id = 's1'", (value,))
        seen.append(_revision(path))
    conn.close()

    assert seen[1] == seen[0] + 1 and seen[2] == seen[1] + 1 and seen[3] == seen[2]
