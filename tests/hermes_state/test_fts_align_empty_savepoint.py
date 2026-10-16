"""Regression (est-qtya lane, discovered while replaying the ghost-tail fixture):
opening an EMPTY store whose ``messages_fts`` still reads raw ``messages``
(FTS_STORAGE_VERSION < 3) died with ``no such savepoint: fts_align_empty`` —
the empty-DB align path runs under SAVEPOINT fts_align_empty, but the DDL was
applied via ``executescript`` (inside ``_ensure_fts_schema``), and executescript
implicitly COMMITs the open savepoint. The RELEASE — and the ROLLBACK TO in the
error handler — then die, killing EVERY SessionDB open on that file.

Fix: ``_execute_ddl_script_transactional`` runs the FTS DDL statement-by-
statement so no implicit commit occurs and the savepoint stays alive.
"""

import sqlite3

import pytest

from hermes_state import SessionDB


def _make_empty_misaligned(path):
    """Rewind a fresh, message-less store to the pre-fix on-disk shape: base
    index external-content over raw ``messages`` + storage version 2."""
    db = SessionDB(db_path=path)
    enabled = db._fts_enabled
    db.close()
    if not enabled:
        return False
    conn = sqlite3.connect(path)
    conn.execute("DROP TABLE IF EXISTS messages_fts")
    for name in ("messages_fts_insert", "messages_fts_delete", "messages_fts_update"):
        conn.execute(f"DROP TRIGGER IF EXISTS {name}")
    conn.execute(
        "CREATE VIRTUAL TABLE messages_fts USING fts5("
        "content, tool_name, tool_calls, content='messages', content_rowid='id')"
    )
    conn.execute(
        "INSERT INTO state_meta(key, value) VALUES('fts_storage_version', '2') "
        "ON CONFLICT(key) DO UPDATE SET value = '2'"
    )
    conn.commit()
    conn.close()
    return True


def test_empty_misaligned_store_opens_under_the_align_savepoint(tmp_path):
    """The realign must stay INSIDE fts_align_empty: open succeeds, the index
    ends aligned, and the version marker records the migration."""
    from hermes_state_common import FTS_STORAGE_VERSION

    path = tmp_path / "state.db"
    if not _make_empty_misaligned(path):
        pytest.skip("SQLite FTS5 unavailable")

    migrated = SessionDB(db_path=path)  # pre-fix: OperationalError no such savepoint
    try:
        assert migrated.get_meta("fts_storage_version") == str(FTS_STORAGE_VERSION)
        index_sql = migrated._conn.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'messages_fts'"
        ).fetchone()[0]
        assert "messages_fts_src" in index_sql
        migrated._conn.execute(
            "INSERT INTO messages_fts(messages_fts, rank) VALUES('integrity-check', 1)"
        )
    finally:
        migrated.close()
    # And the store stays openable (no lingering savepoint damage).
    again = SessionDB(db_path=path)
    again.close()


def test_align_rolls_back_atomically_when_the_ddl_fails(tmp_path, monkeypatch):
    """A failing DDL mid-align must roll the savepoint back, not strand the
    store half-migrated with the savepoint already implicitly committed."""
    import hermes_state_schema as schema

    path = tmp_path / "state.db"
    if not _make_empty_misaligned(path):
        pytest.skip("SQLite FTS5 unavailable")

    calls = {"n": 0}

    def boom(cursor, ddl):
        calls["n"] += 1
        raise sqlite3.OperationalError("simulated mid-align failure")

    monkeypatch.setattr(
        schema.SessionSchemaMixin, "_execute_ddl_script_transactional",
        staticmethod(boom))
    with pytest.raises(sqlite3.OperationalError, match="simulated mid-align failure"):
        SessionDB(db_path=path)
    assert calls["n"] >= 1
    monkeypatch.undo()

    # The failed attempt rolled back: the file still holds the old shape and a
    # clean retry realigns it (the savepoint was never implicitly committed).
    conn = sqlite3.connect(path)
    stored = conn.execute(
        "SELECT sql FROM sqlite_master WHERE name = 'messages_fts'").fetchone()[0]
    conn.close()
    assert "messages_fts_src" not in stored, "failed align must not half-publish"
    migrated = SessionDB(db_path=path)
    migrated.close()
