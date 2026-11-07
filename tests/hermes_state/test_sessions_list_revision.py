"""The store keeps one revision for what a session list renders, whoever writes the row.

The Desktop change watcher used to hash every session row on each state.db write (heartbeats and
cron runs included) to decide whether to broadcast ``sessions.changed``, so its cost grew with the
whole history (#146643). It now reads this trigger-kept counter instead.
"""
import sqlite3
from contextlib import closing

from hermes_state import SessionDB
from hermes_state_common import SESSIONS_LIST_REVISION_KEY


def _revision(path):
    with closing(sqlite3.connect(path)) as conn:
        row = conn.execute("SELECT value FROM state_meta WHERE key = ?", (SESSIONS_LIST_REVISION_KEY,)).fetchone()
    return row and row[0]


def _write(path, sql, *params):
    """A bare writer, as an older build or another process would issue it."""
    with closing(sqlite3.connect(path)) as conn:
        conn.execute(sql, params)
        conn.commit()


def test_revision_moves_for_list_changes_and_holds_for_activity_restamps(tmp_path):
    path = tmp_path / "state.db"
    db = SessionDB(db_path=path)
    try:
        db.create_session("s1", source="cli")
        created = _revision(path)
        assert created is not None

        _write(path, "UPDATE sessions SET last_activity_at = 123456 WHERE id = 's1'")
        _write(path, "UPDATE sessions SET title = title, pinned = pinned WHERE id = 's1'")
        assert _revision(path) == created

        _write(path, "UPDATE sessions SET title = 'Renamed' WHERE id = 's1'")
        renamed = _revision(path)
        assert renamed != created

        _write(path, "UPDATE sessions SET title = NULL WHERE id = 's1'")
        cleared = _revision(path)
        assert cleared != renamed

        _write(path, "DELETE FROM sessions WHERE id = 's1'")
        assert _revision(path) != cleared
    finally:
        db.close()
