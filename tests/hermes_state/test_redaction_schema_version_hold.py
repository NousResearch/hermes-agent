"""Physical redaction cleanup must not override unfinished data-migration gates."""

import json
import sqlite3

import pytest

from hermes_state import SessionDB
from hermes_state_common import SCHEMA_VERSION


@pytest.mark.parametrize("old_version", [30, 31])
def test_redaction_sanitation_preserves_incomplete_fts_version_hold(
    tmp_path, monkeypatch, old_version,
):
    db_path = tmp_path / "state.db"
    secret = "sk-" + "legacySchemaHoldSecret" * 3
    origin = {"tool": "foreign-tool", "foreign_session_id": secret, "path": "/legacy/history"}
    original_migration = SessionDB._migrate_trigram_cron_exclusion
    original_sanitation = SessionDB._sanitize_v31_legacy_redaction_storage
    calls = []

    db = SessionDB(db_path)
    try:
        if not db._trigram_available:
            pytest.skip("trigram tokenizer unavailable in this SQLite build")
        db.create_session("legacy", source="cli")
        db.create_session("imported", source="foreign-tool")
        row_id = db.append_message("legacy", "assistant", content="safe")
    finally:
        db.close()
    with sqlite3.connect(db_path) as conn:
        conn.execute("UPDATE messages SET content = ? WHERE id = ?", (secret, row_id))
        conn.execute(
            "UPDATE sessions SET origin_json = ? WHERE id = 'imported'",
            (json.dumps({"imported_from": origin}),),
        )
        conn.execute("UPDATE schema_version SET version = ?", (old_version,))

    def incomplete_migration(self, cursor):
        calls.append("incomplete")
        return False

    def sanitize(self):
        original_sanitation(self)
        calls.append("sanitized")

    # Model the documented soft-failure return (e.g. an unqueryable trigram
    # index). All other migration, FTS init, and physical sanitation work is real.
    monkeypatch.setattr(SessionDB, "_migrate_trigram_cron_exclusion", incomplete_migration)
    monkeypatch.setattr(SessionDB, "_sanitize_v31_legacy_redaction_storage", sanitize)
    migrated = SessionDB(db_path)
    try:
        assert calls == ["incomplete", "sanitized"]
        assert secret not in migrated._conn.execute(
            "SELECT content FROM messages WHERE id = ?", (row_id,),
        ).fetchone()[0]
        assert migrated._conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert migrated.find_foreign_import(origin) == "imported"
        assert migrated._conn.execute("SELECT version FROM schema_version").fetchone()[0] == old_version
    finally:
        migrated.close()

    for path in (db_path, db_path.with_name(db_path.name + "-wal")):
        assert not path.exists() or secret.encode("utf-8") not in path.read_bytes()
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT version FROM schema_version").fetchone()[0] == old_version

    def complete_migration(self, cursor):
        calls.append("retry")
        return original_migration(self, cursor)

    monkeypatch.setattr(SessionDB, "_migrate_trigram_cron_exclusion", complete_migration)
    retried = SessionDB(db_path)
    try:
        assert calls == ["incomplete", "sanitized", "retry", "sanitized"]
        assert retried._conn.execute("SELECT version FROM schema_version").fetchone()[0] == SCHEMA_VERSION
        assert retried._conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert retried.find_foreign_import(origin) == "imported"
        retried._conn.execute(
            "INSERT INTO messages_fts_trigram(messages_fts_trigram, rank) VALUES('integrity-check', 1)"
        )
    finally:
        retried.close()
