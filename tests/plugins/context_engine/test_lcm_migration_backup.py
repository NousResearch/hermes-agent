"""Schema migration and backup.

Both are acceptance-listed ("migration, backup, ... tests pass with exact
evidence"), and both must be non-destructive: migrating a populated v1 database
keeps every row, and a backup of a live database is openable and current.
"""

from __future__ import annotations

import sqlite3

from plugins.context_engine.lcm import storage as st
from plugins.context_engine.lcm.storage import (
    SCHEMA_VERSION,
    LCMStore,
    backup_db,
    migrate,
    schema_version,
)


def _populate_v1(path):
    """Build a v1 database by hand (the state a pre-v2 install is in)."""
    conn = sqlite3.connect(str(path))
    st._migrate_to_1(conn)
    conn.execute(
        "INSERT INTO messages (session_id, seq, message_id, role, body, payload, "
        "compacted, created_at) VALUES (?, ?, ?, ?, ?, ?, 0, ?)",
        ("s1", 1, "m-legacy", "user", "legacy body", '{"role":"user"}', 1.0),
    )
    conn.execute(
        "INSERT INTO nodes (node_id, session_id, level, summary, source_count, "
        "created_at) VALUES (?, ?, ?, ?, ?, ?)",
        ("n-legacy", "s1", 1, "legacy node", 1, 1.0),
    )
    conn.execute(
        "INSERT INTO node_sources (node_id, source_kind, source_id, ordinal) "
        "VALUES (?, ?, ?, ?)",
        ("n-legacy", "message", "m-legacy", 0),
    )
    conn.execute(
        "INSERT INTO schema_meta (key, value) VALUES ('schema_version', '1')"
    )
    conn.commit()
    conn.close()


def test_fresh_database_is_created_at_current_version(tmp_path):
    store = LCMStore(tmp_path / "lcm.db").open()
    assert store.version() == SCHEMA_VERSION
    assert store.fts_available() is True
    store.close()


def test_migrate_from_v1_preserves_rows_and_adds_the_index(tmp_path):
    db = tmp_path / "lcm.db"
    _populate_v1(db)

    conn = sqlite3.connect(str(db))
    assert schema_version(conn) == 1
    assert migrate(conn) == SCHEMA_VERSION
    conn.commit()

    # The v1 rows survive the upgrade unchanged...
    assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM node_sources").fetchone()[0] == 1
    # ...and the new capability is present.
    conn.execute(
        "INSERT INTO messages_fts (session_id, message_id, body) VALUES (?, ?, ?)",
        ("s1", "m-legacy", "legacy body"),
    )
    conn.commit()
    conn.close()

    store = LCMStore(db).open()  # opening migrates again — must be a no-op
    assert store.version() == SCHEMA_VERSION
    stats = store.stats("s1")
    assert stats["messages"] == 1
    assert stats["nodes"] == 1
    assert stats["lineage_edges"] == 1
    assert store.message("s1", "m-legacy")["body"] == "legacy body"
    assert store.node("n-legacy")["summary"] == "legacy node"
    store.close()


def test_migrate_is_idempotent(tmp_path):
    conn = sqlite3.connect(str(tmp_path / "lcm.db"))
    assert migrate(conn) == SCHEMA_VERSION
    assert migrate(conn) == SCHEMA_VERSION
    assert schema_version(conn) == SCHEMA_VERSION
    conn.close()


def test_backup_snapshots_a_live_database(tmp_path):
    db = tmp_path / "lcm.db"
    store = LCMStore(db).open()
    store.sync_transcript("s1", [{"role": "user", "content": "before backup"}])

    dest = backup_db(db, tmp_path / "snapshot.db")
    assert dest.exists()

    # The snapshot is a real database at the current schema with the data.
    check = sqlite3.connect(str(dest))
    try:
        assert schema_version(check) == SCHEMA_VERSION
        assert check.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 1
    finally:
        check.close()

    # Writes after the snapshot must not leak into it (it is a copy, not a link).
    store.sync_transcript("s1", [{"role": "assistant", "content": "after backup"}])
    check = sqlite3.connect(str(dest))
    try:
        assert check.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 1
    finally:
        check.close()
    assert store.stats("s1")["messages"] == 2
    store.close()


def test_backup_of_a_missing_database_raises(tmp_path):
    try:
        backup_db(tmp_path / "nope.db")
    except FileNotFoundError:
        pass
    else:  # pragma: no cover
        raise AssertionError("backup of a missing database must fail loudly")
