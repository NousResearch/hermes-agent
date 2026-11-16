"""#103840: a ``sqlite3 .recover`` restore re-emits FTS5 shadow tables as ordinary tables
but cannot re-emit the ``CREATE VIRTUAL TABLE`` row. The next SessionDB open then failed
in ``_ensure_fts_schema`` with "fts5: error creating shadow table messages_fts_data: table
already exists". Only families whose vtable row is absent may be repaired; a healthy family's
shadows must survive untouched.
"""

import sqlite3

from hermes_state import SessionDB


def _orphan_family(db_path, family: str) -> None:
    """Emulate the ``.recover`` residue for one family: vtable row gone, shadows kept."""
    raw = sqlite3.connect(db_path)
    raw.isolation_level = None
    raw.execute("PRAGMA writable_schema=ON")
    raw.execute(
        "DELETE FROM sqlite_master WHERE name = ? AND sql LIKE 'CREATE VIRTUAL TABLE%'", (family,),
    )
    version = raw.execute("PRAGMA schema_version").fetchone()[0]
    raw.execute(f"PRAGMA schema_version={version + 1}")
    raw.execute("PRAGMA writable_schema=OFF")
    raw.close()


def _fts_master_rows(db_path, prefix: str) -> list:
    raw = sqlite3.connect(db_path)
    try:
        return raw.execute(
            "SELECT rowid, type, name FROM sqlite_master WHERE name LIKE ? ESCAPE '\\' ORDER BY rowid",
            (prefix.replace("_", "\\_") + "%",),
        ).fetchall()
    finally:
        raw.close()


def test_orphaned_base_family_is_repaired_and_healthy_trigram_untouched(tmp_path):
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    db.create_session("s1", source="cli", model="m")
    for i in range(3):
        db.append_message("s1", "user", f"recovered orphan {i}")
    db.close()

    _orphan_family(db_path, "messages_fts")
    trigram_before = _fts_master_rows(db_path, "messages_fts_trigram")
    assert trigram_before, "fixture needs a live trigram family"
    raw = sqlite3.connect(db_path)
    orphan_shadows = raw.execute(
        "SELECT count(*) FROM sqlite_master WHERE name IN ('messages_fts_data', 'messages_fts_config')"
    ).fetchone()[0]
    raw.close()
    assert orphan_shadows == 2, "fixture must leave the base shadows behind"

    reopened = SessionDB(db_path=db_path)
    try:
        assert reopened._fts_enabled is True
        with reopened._lock:
            hits = reopened._conn.execute(
                "SELECT count(*) FROM messages_fts WHERE messages_fts MATCH 'orphan'"
            ).fetchone()[0]
        assert hits == 3, "recreated index must be rebuilt from the canonical messages table"
    finally:
        reopened.close()
    # Same rowids: the healthy family was neither dropped nor recreated.
    assert _fts_master_rows(db_path, "messages_fts_trigram") == trigram_before
