"""Bulk in-place mutations (not just delete/prune) must stay below SQLite's bound-variable ceiling.

Sibling of ``test_corrupt_row_robustness.py::test_bulk_delete_and_prune_stay_below_sqlite_variable_limit``:
that test pins ``prune_sessions``/``delete_sessions`` against an oversized ``IN (...)`` list via
``_id_chunks``. ``sweep_orphaned_sessions``, ``rewind_to_message`` and ``purge_stale_tool_call_markers``
build the exact same shape of unbounded id list for an ``UPDATE ... WHERE id IN (...)`` and must chunk
the same way — each is reachable automatically (gateway startup) or by an operator on exactly the kind
of long-accumulated store where the id count is largest.
"""

import sqlite3
import time

from hermes_state import SessionDB


def _force_legacy_variable_ceiling(db):
    db._conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 999)  # the legacy ceiling, deterministic


def test_sweep_orphaned_sessions_stays_below_sqlite_variable_limit(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        old = time.time() - 100_000
        ids = [f"orphan_{i}" for i in range(1200)]

        def seed(conn):
            conn.executemany(
                "INSERT INTO sessions (id, source, started_at, ended_at) VALUES (?, 'tui', ?, NULL)",
                [(sid, old) for sid in ids])

        db._execute_write(seed)
        _force_legacy_variable_ceiling(db)
        reaped = db.sweep_orphaned_sessions(max_idle_seconds=60)
        assert len(reaped) == 1200
        assert db._read_one(
            "SELECT COUNT(*) FROM sessions WHERE end_reason = 'startup_orphan_reap'")[0] == 1200
    finally:
        db.close()


def test_rewind_to_message_stays_below_sqlite_variable_limit(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("s", "cli")
        target_id = db.append_message("s", "user", "rewind target")
        now = time.time()

        def seed(conn):
            conn.executemany(
                "INSERT INTO messages (session_id, role, content, timestamp, active) "
                "VALUES ('s', 'assistant', 'x', ?, 1)",
                [(now,) for _ in range(1200)])

        db._execute_write(seed)
        _force_legacy_variable_ceiling(db)
        result = db.rewind_to_message("s", target_id)
        assert result["rewound_count"] == 1201  # the target row plus the 1200 seeded rows after it
        assert db._read_one(
            "SELECT COUNT(*) FROM messages WHERE session_id = 's' AND active = 1")[0] == 0
    finally:
        db.close()


def test_purge_stale_tool_call_markers_stays_below_sqlite_variable_limit(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("s", "cli")
        now = time.time()

        def seed(conn):
            conn.executemany(
                "INSERT INTO messages (session_id, role, content, tool_calls, timestamp) "
                "VALUES ('s', 'assistant', '[memory]', 'x', ?)",
                [(now,) for _ in range(1200)])

        db._execute_write(seed)
        _force_legacy_variable_ceiling(db)
        result = db.purge_stale_tool_call_markers(dry_run=False, backup=False)
        assert result["rows_affected"] == 1200
        assert db._read_one("SELECT COUNT(*) FROM messages WHERE content = '[memory]'")[0] == 0
    finally:
        db.close()
