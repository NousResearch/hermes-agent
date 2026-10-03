"""Regression for the ``fts_align_empty`` savepoint on an unused pre-v3 home
(issue #121882).

The ``not has_messages`` branch of ``_migrate_misaligned_fts_source`` wraps the
realignment in a savepoint, but the callee chain reached
``_ensure_fts_schema`` → ``cursor.executescript(ddl)``, and ``executescript``
issues an implicit ``COMMIT`` before running the script. The outer savepoint
was therefore already gone when ``RELEASE SAVEPOINT fts_align_empty`` ran, and
the open of a store whose ``messages`` table is empty and whose ``messages_fts``
still reads the raw ``messages`` table raised ``no such savepoint:
fts_align_empty`` — surfacing as "session store unavailable" with that
conversation's transcript lost, even though the DDL itself had (implicitly)
committed. ``_execute_ddl_script_transactional`` exists precisely to run the
DDL statement-by-statement without that implicit commit.

These are behaviour contracts on the open path: the realignment of an empty,
misaligned store must succeed atomically or leave the store unopened — the
caller's transaction scope must stay valid across ``do_align()``.
"""

import sqlite3

import pytest

from hermes_state import SessionDB
from hermes_state_common import FTS_STORAGE_VERSION


def _build_pre_v3_empty_store(path) -> None:
    """Create a store at the current schema, then rewind it to the pre-v3
    shape with zero messages: external-content FTS over the raw ``messages``
    table (no ``messages_fts_src``), no rows, ``fts_storage_version = 1``."""
    first = SessionDB(db_path=path)
    if not first._fts_enabled:
        first.close()
        pytest.skip("SQLite FTS5 unavailable")
    first.close()

    conn = sqlite3.connect(path)
    v3_ddl = conn.execute(
        "SELECT sql FROM sqlite_master WHERE name = 'messages_fts'"
    ).fetchone()[0]
    legacy_ddl = v3_ddl.replace("'messages_fts_src'", "'messages'")
    for trigger in ("messages_fts_insert", "messages_fts_delete", "messages_fts_update"):
        conn.execute(f"DROP TRIGGER IF EXISTS {trigger}")
    conn.execute("DROP TABLE IF EXISTS messages_fts")
    conn.execute("DROP VIEW IF EXISTS messages_fts_src")
    conn.execute(legacy_ddl)
    conn.execute("DELETE FROM messages")
    conn.execute(
        "INSERT INTO state_meta(key, value) VALUES('fts_storage_version', '1') "
        "ON CONFLICT(key) DO UPDATE SET value = '1'"
    )
    conn.commit()
    conn.close()


def test_empty_misaligned_store_opens_and_realigns(tmp_path):
    """Opening an empty store whose index still reads raw ``messages`` realigns
    without raising and stamps the current storage version.

    Pre-fix, the open raises ``OperationalError: no such savepoint:
    fts_align_empty``: ``executescript``'s implicit commit destroyed the
    enclosing savepoint before ``RELEASE`` ran, and the handler's own
    ``ROLLBACK TO SAVEPOINT`` raised the same error, masking the original.
    """
    path = tmp_path / "state.db"
    _build_pre_v3_empty_store(path)

    reopened = SessionDB(db_path=path)
    try:
        assert reopened.get_meta("fts_storage_version") == str(FTS_STORAGE_VERSION)
        index_sql = reopened._conn.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'messages_fts'"
        ).fetchone()[0]
        assert "messages_fts_src" in index_sql
    finally:
        reopened.close()

    # And the realigned store keeps opening clean.
    again = SessionDB(db_path=path)
    try:
        assert again.get_meta("fts_storage_version") == str(FTS_STORAGE_VERSION)
    finally:
        again.close()
