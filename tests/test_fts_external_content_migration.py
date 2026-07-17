import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from hermes_state import SessionDB, _rebuild_fts_table_for_repair


def _schema_sql(db: SessionDB, name: str) -> str:
    row = db._conn.execute(
        "SELECT sql FROM sqlite_master WHERE name = ?",
        (name,),
    ).fetchone()
    assert row is not None
    return str(row[0])


def test_new_database_uses_external_content_fts_and_preserves_search_documents(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        if not db._fts_enabled:
            pytest.skip("SQLite build has no FTS5")

        base_sql = _schema_sql(db, "messages_fts")
        assert "content='messages_fts_source'" in base_sql
        assert "content_rowid='id'" in base_sql
        assert db.get_meta("fts_storage_revision") == "external-content-v1"

        view = db._conn.execute(
            "SELECT type, sql FROM sqlite_master WHERE name = 'messages_fts_source'"
        ).fetchone()
        assert view is not None
        assert view[0] == "view"
        assert "COALESCE(content, '')" in view[1]
        assert "COALESCE(tool_name, '')" in view[1]
        assert "COALESCE(tool_calls, '')" in view[1]

        # External-content FTS keeps only its index/docsize/config data. The
        # canonical document text is supplied by the zero-storage view.
        shadow_tables = {
            row[0]
            for row in db._conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        assert "messages_fts_source" not in shadow_tables
        assert "messages_fts_content" not in shadow_tables
        if db._trigram_available:
            trigram_sql = _schema_sql(db, "messages_fts_trigram")
            assert "content='messages_fts_source'" in trigram_sql
            assert "content_rowid='id'" in trigram_sql
            assert "messages_fts_trigram_content" not in shadow_tables

        db.create_session(session_id="s1", source="cli")
        content_id = db.append_message(
            "s1", role="user", content="alpha content needle"
        )
        tool_id = db.append_message(
            "s1",
            role="assistant",
            content="tool result body",
            tool_name="beta_tool_needle",
            tool_calls=[{"name": "gamma_call_needle"}],
        )

        content_hit = db.search_messages("alpha")
        assert [row["id"] for row in content_hit] == [content_id]
        assert ">>>alpha<<<" in content_hit[0]["snippet"]

        tool_hit = db.search_messages("beta_tool_needle")
        assert [row["id"] for row in tool_hit] == [tool_id]
        assert ">>>beta_tool_needle<<<" in tool_hit[0]["snippet"]

        call_hit = db.search_messages("gamma_call_needle")
        assert [row["id"] for row in call_hit] == [tool_id]
        assert ">>>gamma_call_needle<<<" in call_hit[0]["snippet"]

        if db._trigram_available:
            cjk_id = db.append_message(
                "s1", role="user", content="中文迁移测试内容"
            )
            cjk_hit = db.search_messages("迁移测试")
            assert [row["id"] for row in cjk_hit] == [cjk_id]

        # rank=1 makes FTS5 compare the inverted index with the external view.
        db._conn.execute(
            "INSERT INTO messages_fts(messages_fts, rank) "
            "VALUES('integrity-check', 1)"
        )
        if db._trigram_available:
            db._conn.execute(
                "INSERT INTO messages_fts_trigram(messages_fts_trigram, rank) "
                "VALUES('integrity-check', 1)"
            )
    finally:
        db.close()


LEGACY_INLINE_BASE_SQL = """
DROP TRIGGER IF EXISTS messages_fts_insert;
DROP TRIGGER IF EXISTS messages_fts_delete;
DROP TRIGGER IF EXISTS messages_fts_update;
DROP TRIGGER IF EXISTS messages_fts_trigram_insert;
DROP TRIGGER IF EXISTS messages_fts_trigram_delete;
DROP TRIGGER IF EXISTS messages_fts_trigram_update;
DROP TABLE IF EXISTS messages_fts;
DROP TABLE IF EXISTS messages_fts_trigram;
DROP VIEW IF EXISTS messages_fts_source;
CREATE VIRTUAL TABLE messages_fts USING fts5(content);
CREATE TRIGGER messages_fts_insert AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, content) VALUES (
        new.id,
        COALESCE(new.content, '') || ' ' || COALESCE(new.tool_name, '') || ' ' || COALESCE(new.tool_calls, '')
    );
END;
CREATE TRIGGER messages_fts_delete AFTER DELETE ON messages BEGIN
    DELETE FROM messages_fts WHERE rowid = old.id;
END;
CREATE TRIGGER messages_fts_update AFTER UPDATE ON messages BEGIN
    DELETE FROM messages_fts WHERE rowid = old.id;
    INSERT INTO messages_fts(rowid, content) VALUES (
        new.id,
        COALESCE(new.content, '') || ' ' || COALESCE(new.tool_name, '') || ' ' || COALESCE(new.tool_calls, '')
    );
END;
INSERT INTO messages_fts(rowid, content)
SELECT id,
       COALESCE(content, '') || ' ' || COALESCE(tool_name, '') || ' ' || COALESCE(tool_calls, '')
FROM messages;
"""

LEGACY_INLINE_TRIGRAM_SQL = """
CREATE VIRTUAL TABLE messages_fts_trigram USING fts5(content, tokenize='trigram');
CREATE TRIGGER messages_fts_trigram_insert AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts_trigram(rowid, content) VALUES (
        new.id,
        COALESCE(new.content, '') || ' ' || COALESCE(new.tool_name, '') || ' ' || COALESCE(new.tool_calls, '')
    );
END;
CREATE TRIGGER messages_fts_trigram_delete AFTER DELETE ON messages BEGIN
    DELETE FROM messages_fts_trigram WHERE rowid = old.id;
END;
CREATE TRIGGER messages_fts_trigram_update AFTER UPDATE ON messages BEGIN
    DELETE FROM messages_fts_trigram WHERE rowid = old.id;
    INSERT INTO messages_fts_trigram(rowid, content) VALUES (
        new.id,
        COALESCE(new.content, '') || ' ' || COALESCE(new.tool_name, '') || ' ' || COALESCE(new.tool_calls, '')
    );
END;
INSERT INTO messages_fts_trigram(rowid, content)
SELECT id,
       COALESCE(content, '') || ' ' || COALESCE(tool_name, '') || ' ' || COALESCE(tool_calls, '')
FROM messages;
"""

LEGACY_INLINE_BOTH_SQL = LEGACY_INLINE_BASE_SQL + LEGACY_INLINE_TRIGRAM_SQL


@pytest.mark.parametrize(
    "missing_trigger",
    (
        "messages_fts_insert",
        "messages_fts_delete",
        "messages_fts_update",
        "messages_fts_trigram_insert",
        "messages_fts_trigram_delete",
        "messages_fts_trigram_update",
    ),
)
def test_startup_repairs_inline_triggers_with_inline_protocol(tmp_path, missing_trigger):
    db_path = tmp_path / f"inline-trigger-{missing_trigger}.db"
    seeded = SessionDB(db_path=db_path)
    try:
        if not seeded._fts_enabled or not seeded._trigram_available:
            pytest.skip("SQLite build lacks FTS5 trigram support")
        seeded.create_session(session_id="s1", source="cli")
        seeded.append_message("s1", role="user", content="existing repair sentinel")
    finally:
        seeded.close()

    raw = sqlite3.connect(db_path)
    try:
        raw.executescript(LEGACY_INLINE_BOTH_SQL)
        raw.execute(f"DROP TRIGGER {missing_trigger}")
        raw.commit()
    finally:
        raw.close()

    repaired = SessionDB(db_path=db_path)
    try:
        assert repaired.fts_storage_mode() == "inline"
        inserted = repaired.append_message(
            "s1", role="assistant", content="insert repair needle"
        )
        repaired._conn.execute(
            "UPDATE messages SET content = ? WHERE id = ?",
            ("updated repair needle", inserted),
        )
        repaired._conn.commit()
        assert _search_projection(repaired, "insert") == []
        assert _search_projection(repaired, "updated")[0][0] == inserted

        repaired._conn.execute("DELETE FROM messages WHERE id = ?", (inserted,))
        repaired._conn.commit()
        assert _search_projection(repaired, "updated") == []
    finally:
        repaired.close()


def test_standalone_repair_rebuilds_inline_base_and_trigram(tmp_path):
    db_path = tmp_path / "inline-repair.db"
    seeded = SessionDB(db_path=db_path)
    try:
        if not seeded._fts_enabled or not seeded._trigram_available:
            pytest.skip("SQLite build lacks FTS5 trigram support")
        seeded.create_session(session_id="s1", source="cli")
        message_id = seeded.append_message(
            "s1",
            role="assistant",
            content="repair canonical body",
            tool_name="repair_tool_needle",
        )
    finally:
        seeded.close()

    raw = sqlite3.connect(db_path, isolation_level=None)
    try:
        raw.executescript(LEGACY_INLINE_BOTH_SQL)
        raw.execute("DELETE FROM messages_fts")
        raw.execute("DELETE FROM messages_fts_trigram")
        assert raw.execute(
            "SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH 'repair'"
        ).fetchone()[0] == 0

        raw.execute("BEGIN IMMEDIATE")
        assert _rebuild_fts_table_for_repair(raw, "messages_fts") is True
        assert _rebuild_fts_table_for_repair(raw, "messages_fts_trigram") is True
        raw.commit()

        base_ids = raw.execute(
            "SELECT rowid FROM messages_fts WHERE messages_fts MATCH 'repair_tool_needle'"
        ).fetchall()
        trigram_ids = raw.execute(
            "SELECT rowid FROM messages_fts_trigram "
            "WHERE messages_fts_trigram MATCH 'canonical'"
        ).fetchall()
        assert base_ids == [(message_id,)]
        assert trigram_ids == [(message_id,)]
    finally:
        raw.close()


def _search_projection(db: SessionDB, query: str):
    return [
        (row["id"], row["snippet"], row["tool_name"], row["role"])
        for row in db.search_messages(query, include_inactive=True)
    ]


def test_explicit_migration_preserves_legacy_search_and_write_semantics(tmp_path):
    db_path = tmp_path / "legacy.db"
    seeded = SessionDB(db_path=db_path)
    try:
        if not seeded._fts_enabled or not seeded._trigram_available:
            pytest.skip("SQLite build lacks FTS5 trigram support")
        seeded.create_session(session_id="s1", source="cli")
        body_id = seeded.append_message(
            "s1", role="user", content="legacy alpha phrase 中文迁移测试"
        )
        tool_id = seeded.append_message(
            "s1",
            role="assistant",
            content="legacy tool body",
            tool_name="legacy_beta_tool",
            tool_calls=[{"name": "legacy_gamma_call"}],
        )
    finally:
        seeded.close()

    raw = sqlite3.connect(db_path)
    try:
        raw.executescript(LEGACY_INLINE_BOTH_SQL)
        raw.commit()
    finally:
        raw.close()

    db = SessionDB(db_path=db_path)
    try:
        # Startup remains bounded: opening a legacy database does not perform
        # the potentially multi-GiB rebuild automatically.
        assert "content='messages_fts_source'" not in _schema_sql(db, "messages_fts")
        assert db.get_meta("fts_storage_revision") is None

        queries = (
            "legacy",
            "alpha",
            "legacy_beta_tool",
            "legacy_gamma_call",
            "迁移测试",
        )
        before = {query: _search_projection(db, query) for query in queries}
        assert before["alpha"][0][0] == body_id
        assert before["legacy_beta_tool"][0][0] == tool_id

        report = db.migrate_fts_to_external_content()
        assert report["previous_mode"] == "inline"
        assert report["final_mode"] == "external"
        assert report["migrated_tables"] == [
            "messages_fts",
            "messages_fts_trigram",
        ]
        assert report["noop"] is False
        assert "content='messages_fts_source'" in _schema_sql(db, "messages_fts")
        assert db.get_meta("fts_storage_revision") == "external-content-v1"

        after = {query: _search_projection(db, query) for query in queries}
        assert after == before

        # Validate index/content consistency and the external-content trigger
        # delete protocol after ordinary message UPDATE and DELETE operations.
        db._conn.execute(
            "INSERT INTO messages_fts(messages_fts, rank) "
            "VALUES('integrity-check', 1)"
        )
        db._conn.execute(
            "UPDATE messages SET content = ? WHERE id = ?",
            ("replacement delta phrase", body_id),
        )
        db._conn.commit()
        assert _search_projection(db, "alpha") == []
        assert _search_projection(db, "delta")[0][0] == body_id

        db._conn.execute("DELETE FROM messages WHERE id = ?", (tool_id,))
        db._conn.commit()
        assert _search_projection(db, "legacy_beta_tool") == []
        db._conn.execute(
            "INSERT INTO messages_fts(messages_fts, rank) "
            "VALUES('integrity-check', 1)"
        )

        second = db.migrate_fts_to_external_content()
        assert second["noop"] is True
        assert second["migrated_tables"] == []
    finally:
        db.close()


def test_concurrent_migrators_serialize_schema_discovery_after_write_lock(tmp_path):
    db_path = tmp_path / "concurrent-migration.db"
    seeded = SessionDB(db_path=db_path)
    try:
        if not seeded._fts_enabled or not seeded._trigram_available:
            pytest.skip("SQLite build lacks FTS5 trigram support")
        seeded.create_session(session_id="s1", source="cli")
        seeded.append_message("s1", role="user", content="concurrent migration sentinel")
    finally:
        seeded.close()

    raw = sqlite3.connect(db_path)
    try:
        raw.executescript(LEGACY_INLINE_BOTH_SQL)
        raw.commit()
    finally:
        raw.close()

    connections = [SessionDB(db_path=db_path), SessionDB(db_path=db_path)]
    barrier = threading.Barrier(3)
    results = []
    errors = []
    result_lock = threading.Lock()

    def migrate(db):
        barrier.wait()
        try:
            report = db.migrate_fts_to_external_content()
            with result_lock:
                results.append(report)
        except Exception as exc:
            with result_lock:
                errors.append(exc)

    threads = [threading.Thread(target=migrate, args=(db,)) for db in connections]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=10)

    try:
        assert all(not thread.is_alive() for thread in threads)
        assert errors == []
        assert sorted(report["noop"] for report in results) == [False, True]
        migrated = next(report for report in results if not report["noop"])
        noop = next(report for report in results if report["noop"])
        assert migrated["previous_mode"] == "inline"
        assert noop["previous_mode"] == "external"
        assert connections[0].fts_storage_mode() == "external"
        assert _search_projection(connections[0], "sentinel")
    finally:
        for db in connections:
            db.close()


def test_migration_failure_after_swap_rolls_back_both_inline_indexes(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "rollback.db"
    seeded = SessionDB(db_path=db_path)
    try:
        if not seeded._fts_enabled or not seeded._trigram_available:
            pytest.skip("SQLite build lacks FTS5 trigram support")
        seeded.create_session(session_id="s1", source="cli")
        message_id = seeded.append_message(
            "s1", role="user", content="rollback sentinel needle"
        )
    finally:
        seeded.close()

    raw = sqlite3.connect(db_path)
    try:
        raw.executescript(LEGACY_INLINE_BOTH_SQL)
        raw.commit()
    finally:
        raw.close()

    db = SessionDB(db_path=db_path)
    original_validate = SessionDB._validate_external_fts_table

    def fail_after_swap(cursor, table_name):
        if table_name == "messages_fts":
            raise RuntimeError("injected final validation failure")
        return original_validate(cursor, table_name)

    monkeypatch.setattr(
        SessionDB,
        "_validate_external_fts_table",
        staticmethod(fail_after_swap),
    )
    try:
        before_base_sql = _schema_sql(db, "messages_fts")
        before_trigram_sql = _schema_sql(db, "messages_fts_trigram")
        with pytest.raises(RuntimeError, match="injected final validation failure"):
            db.migrate_fts_to_external_content()

        assert _schema_sql(db, "messages_fts") == before_base_sql
        assert _schema_sql(db, "messages_fts_trigram") == before_trigram_sql
        assert "content='messages_fts_source'" not in before_base_sql
        assert "content='messages_fts_source'" not in before_trigram_sql
        assert db.get_meta("fts_storage_revision") is None
        assert _search_projection(db, "sentinel")[0][0] == message_id
        leaked = db._conn.execute(
            "SELECT name FROM sqlite_master WHERE name LIKE '%__external_new%'"
        ).fetchall()
        assert leaked == []

        # The original triggers remain writable after rollback.
        appended = db.append_message(
            "s1", role="assistant", content="rollback survivor term"
        )
        assert _search_projection(db, "survivor")[0][0] == appended
    finally:
        db.close()


def test_read_only_connection_rejects_external_fts_migration(tmp_path):
    db_path = tmp_path / "read-only.db"
    writable = SessionDB(db_path=db_path)
    writable.close()

    read_only = SessionDB(db_path=db_path, read_only=True)
    try:
        with pytest.raises(sqlite3.OperationalError, match="requires a writable"):
            read_only.migrate_fts_to_external_content()
    finally:
        read_only.close()


def test_external_fts_ignores_non_searchable_message_updates(tmp_path):
    db = SessionDB(db_path=tmp_path / "metadata-update.db")
    try:
        if not db._fts_enabled:
            pytest.skip("SQLite build has no FTS5")
        db.create_session(session_id="s1", source="cli")
        message_id = db.append_message(
            "s1", role="user", content="metadata update sentinel"
        )

        before = db._conn.total_changes
        db._conn.execute(
            "UPDATE messages SET active = 0, compacted = 1 WHERE id = ?",
            (message_id,),
        )
        metadata_delta = db._conn.total_changes - before

        # Only the messages row changes. External FTS triggers are scoped to
        # id/content/tool fields and therefore do not rewrite either index.
        assert metadata_delta == 1
        db._conn.execute(
            "INSERT INTO messages_fts(messages_fts, rank) "
            "VALUES('integrity-check', 1)"
        )
        if db._trigram_available:
            db._conn.execute(
                "INSERT INTO messages_fts_trigram(messages_fts_trigram, rank) "
                "VALUES('integrity-check', 1)"
            )
        assert _search_projection(db, "sentinel")[0][0] == message_id
    finally:
        db.close()


@pytest.mark.live_system_guard_bypass
def test_process_kill_rolls_back_staged_fts_migration(tmp_path):
    db_path = tmp_path / "crash-recovery.db"
    marker_path = tmp_path / "staged-index-validated"

    seeded = SessionDB(db_path=db_path)
    try:
        if not seeded._fts_enabled or not seeded._trigram_available:
            pytest.skip("SQLite build lacks FTS5 trigram support")
        seeded.create_session(session_id="s1", source="cli")
        message_id = seeded.append_message(
            "s1", role="user", content="crash rollback sentinel"
        )
    finally:
        seeded.close()

    raw = sqlite3.connect(db_path)
    try:
        raw.executescript(LEGACY_INLINE_BOTH_SQL)
        raw.commit()
    finally:
        raw.close()

    repo_root = Path(__file__).resolve().parents[1]
    child_code = f"""
import sys
import time
from pathlib import Path

sys.path.insert(0, {str(repo_root)!r})
from hermes_state import SessionDB

marker = Path({str(marker_path)!r})
original_validate = SessionDB._validate_external_fts_table

def pause_after_staged_validation(cursor, table_name):
    original_validate(cursor, table_name)
    if '__external_new_' in table_name:
        marker.write_text(table_name)
        time.sleep(60)

SessionDB._validate_external_fts_table = staticmethod(pause_after_staged_validation)
db = SessionDB(db_path=Path({str(db_path)!r}))
db.migrate_fts_to_external_content()
"""
    proc = subprocess.Popen(
        [sys.executable, "-c", child_code],
        cwd=str(tmp_path),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 10
        while (
            not marker_path.exists()
            and proc.poll() is None
            and time.monotonic() < deadline
        ):
            time.sleep(0.02)

        if not marker_path.exists():
            stdout, stderr = proc.communicate(timeout=5)
            pytest.fail(
                "child did not reach staged-index validation before exiting: "
                f"returncode={proc.returncode}, stdout={stdout!r}, stderr={stderr!r}"
            )

        proc.kill()
        proc.wait(timeout=5)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)

    recovered = sqlite3.connect(db_path)
    try:
        assert recovered.execute("PRAGMA quick_check").fetchone()[0] == "ok"
        base_sql = recovered.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'messages_fts'"
        ).fetchone()[0]
        assert "content='messages_fts_source'" not in base_sql
        assert recovered.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE name LIKE '%__external_new%'"
        ).fetchone()[0] == 0
    finally:
        recovered.close()

    reopened = SessionDB(db_path=db_path)
    try:
        assert reopened.get_meta("fts_storage_revision") is None
        assert _search_projection(reopened, "sentinel")[0][0] == message_id
        assert reopened._conn is not None
        assert reopened._fts_trigger_count(reopened._conn.cursor()) == 6
    finally:
        reopened.close()


def test_mixed_mode_migrates_only_inline_trigram(tmp_path):
    db_path = tmp_path / "mixed-mode.db"
    seeded = SessionDB(db_path=db_path)
    try:
        if not seeded._fts_enabled or not seeded._trigram_available:
            pytest.skip("SQLite build lacks FTS5 trigram support")
        seeded.create_session(session_id="s1", source="cli")
        seeded.append_message("s1", role="user", content="mixed mode sentinel")
    finally:
        seeded.close()

    raw = sqlite3.connect(db_path)
    try:
        raw.executescript(
            """
            DROP TRIGGER messages_fts_trigram_insert;
            DROP TRIGGER messages_fts_trigram_delete;
            DROP TRIGGER messages_fts_trigram_update;
            DROP TABLE messages_fts_trigram;
            CREATE VIRTUAL TABLE messages_fts_trigram USING fts5(content, tokenize='trigram');
            CREATE TRIGGER messages_fts_trigram_insert AFTER INSERT ON messages BEGIN
              INSERT INTO messages_fts_trigram(rowid, content)
              VALUES (new.id, COALESCE(new.content, '') || ' ' || COALESCE(new.tool_name, '') || ' ' || COALESCE(new.tool_calls, ''));
            END;
            CREATE TRIGGER messages_fts_trigram_delete AFTER DELETE ON messages BEGIN
              DELETE FROM messages_fts_trigram WHERE rowid = old.id;
            END;
            CREATE TRIGGER messages_fts_trigram_update AFTER UPDATE OF id, content, tool_name, tool_calls ON messages BEGIN
              DELETE FROM messages_fts_trigram WHERE rowid = old.id;
              INSERT INTO messages_fts_trigram(rowid, content)
              VALUES (new.id, COALESCE(new.content, '') || ' ' || COALESCE(new.tool_name, '') || ' ' || COALESCE(new.tool_calls, ''));
            END;
            INSERT INTO messages_fts_trigram(rowid, content)
            SELECT id, COALESCE(content, '') || ' ' || COALESCE(tool_name, '') || ' ' || COALESCE(tool_calls, '') FROM messages;
            """
        )
        raw.commit()
    finally:
        raw.close()

    maintenance = SessionDB.open_for_fts_maintenance(db_path)
    try:
        assert maintenance.fts_storage_mode() == "mixed"
        report = maintenance.migrate_fts_to_external_content()
        assert report["previous_mode"] == "mixed"
        assert report["migrated_tables"] == ["messages_fts_trigram"]
        assert maintenance.fts_storage_mode() == "external"
        maintenance._conn.execute(
            "INSERT INTO messages_fts(messages_fts, rank) VALUES('integrity-check', 1)"
        )
        maintenance._conn.execute(
            "INSERT INTO messages_fts_trigram(messages_fts_trigram, rank) "
            "VALUES('integrity-check', 1)"
        )
    finally:
        maintenance.close()


def test_maintenance_open_skips_schema_and_metadata_repair(tmp_path):
    db_path = tmp_path / "maintenance-open.db"
    seeded = SessionDB(db_path=db_path)
    try:
        if not seeded._fts_enabled or not seeded._trigram_available:
            pytest.skip("SQLite build lacks FTS5 trigram support")
    finally:
        seeded.close()

    raw = sqlite3.connect(db_path)
    try:
        raw.executescript(LEGACY_INLINE_BOTH_SQL)
        raw.execute("DROP TRIGGER messages_fts_delete")
        raw.execute("DELETE FROM state_meta WHERE key = 'fts_storage_revision'")
        raw.commit()
    finally:
        raw.close()

    maintenance = SessionDB.open_for_fts_maintenance(db_path)
    try:
        missing = maintenance._conn.execute(
            "SELECT 1 FROM sqlite_master "
            "WHERE type = 'trigger' AND name = 'messages_fts_delete'"
        ).fetchone()
        revision = maintenance._conn.execute(
            "SELECT value FROM state_meta WHERE key = 'fts_storage_revision'"
        ).fetchone()
        assert missing is None
        assert revision is None
    finally:
        maintenance.close()


def test_external_noop_validates_and_restores_storage_revision(tmp_path):
    db_path = tmp_path / "external-noop-revision.db"
    seeded = SessionDB(db_path=db_path)
    try:
        if not seeded._fts_enabled:
            pytest.skip("SQLite build lacks FTS5")
    finally:
        seeded.close()

    raw = sqlite3.connect(db_path)
    try:
        raw.execute("DELETE FROM state_meta WHERE key = 'fts_storage_revision'")
        raw.commit()
    finally:
        raw.close()

    maintenance = SessionDB.open_for_fts_maintenance(db_path)
    try:
        report = maintenance.migrate_fts_to_external_content()
        assert report["noop"] is False
        assert report["schema_changed"] is False
        assert report["metadata_changed"] is True
        revision = maintenance._conn.execute(
            "SELECT value FROM state_meta WHERE key = 'fts_storage_revision'"
        ).fetchone()
        assert revision[0] == "external-content-v1"
    finally:
        maintenance.close()


def test_external_maintenance_repairs_missing_trigger_and_reports_change(tmp_path):
    db_path = tmp_path / "external-missing-trigger.db"
    seeded = SessionDB(db_path=db_path)
    try:
        if not seeded._fts_enabled:
            pytest.skip("SQLite build lacks FTS5")
    finally:
        seeded.close()

    raw = sqlite3.connect(db_path)
    try:
        raw.execute("DROP TRIGGER messages_fts_insert")
        raw.commit()
    finally:
        raw.close()

    maintenance = SessionDB.open_for_fts_maintenance(db_path)
    try:
        report = maintenance.migrate_fts_to_external_content()
        assert report["noop"] is False
        assert report["schema_changed"] is True
        assert report["metadata_changed"] is False
        assert maintenance._conn.execute(
            "SELECT 1 FROM sqlite_master "
            "WHERE type = 'trigger' AND name = 'messages_fts_insert'"
        ).fetchone()
    finally:
        maintenance.close()


def test_external_maintenance_repairs_wrong_present_trigger_definitions(tmp_path):
    db_path = tmp_path / "external-wrong-triggers.db"
    seeded = SessionDB(db_path=db_path)
    try:
        if not seeded._fts_enabled:
            pytest.skip("SQLite build lacks FTS5")
        seeded.create_session(session_id="s1", source="cli")
        message_id = seeded.append_message(
            "s1", role="user", content="before trigger repair"
        )
    finally:
        seeded.close()

    raw = sqlite3.connect(db_path)
    try:
        raw.executescript(
            """
            DROP TRIGGER messages_fts_delete;
            DROP TRIGGER messages_fts_update;
            CREATE TRIGGER messages_fts_delete AFTER DELETE ON messages BEGIN
              DELETE FROM messages_fts WHERE rowid = old.id;
            END;
            CREATE TRIGGER messages_fts_update AFTER UPDATE ON messages BEGIN
              DELETE FROM messages_fts WHERE rowid = old.id;
              INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content);
            END;
            """
        )
        raw.commit()
    finally:
        raw.close()

    maintenance = SessionDB.open_for_fts_maintenance(db_path)
    try:
        report = maintenance.migrate_fts_to_external_content()
        assert report["noop"] is False
        assert report["schema_changed"] is True
        maintenance._conn.execute(
            "UPDATE messages SET content = ? WHERE id = ?",
            ("after trigger repair", message_id),
        )
        maintenance._conn.commit()
        maintenance._conn.execute(
            "INSERT INTO messages_fts(messages_fts, rank) "
            "VALUES('integrity-check', 1)"
        )
    finally:
        maintenance.close()


def test_migration_refuses_unrelated_caller_transaction(tmp_path):
    db = SessionDB(db_path=tmp_path / "unrelated-transaction.db")
    try:
        if not db._fts_enabled:
            pytest.skip("SQLite build lacks FTS5")
        db._conn.execute("BEGIN IMMEDIATE")
        with pytest.raises(sqlite3.OperationalError, match="unrelated transaction"):
            db.migrate_fts_to_external_content()
        assert db._conn.in_transaction is True
        db._conn.rollback()
    finally:
        db.close()


def test_external_maintenance_repairs_trigger_with_changed_literal(tmp_path):
    db_path = tmp_path / "external-wrong-literal-trigger.db"
    seeded = SessionDB(db_path=db_path)
    try:
        if not seeded._fts_enabled:
            pytest.skip("SQLite build lacks FTS5")
        seeded.create_session(session_id="s1", source="cli")
    finally:
        seeded.close()

    canonical = SessionDB._fts_trigger_sql(
        "messages_fts", external=True
    )["messages_fts_insert"]
    wrong = canonical.replace(" || ' ' || ", " || '' || ")
    assert wrong != canonical
    assert SessionDB._normalize_trigger_sql(wrong) != SessionDB._normalize_trigger_sql(
        canonical
    )

    raw = sqlite3.connect(db_path)
    try:
        raw.execute("DROP TRIGGER messages_fts_insert")
        raw.execute(wrong)
        raw.commit()
    finally:
        raw.close()

    maintenance = SessionDB.open_for_fts_maintenance(db_path)
    try:
        report = maintenance.migrate_fts_to_external_content()
        assert report["noop"] is False
        assert report["schema_changed"] is True
    finally:
        maintenance.close()

    verified = SessionDB(db_path=db_path)
    try:
        message_id = verified.append_message(
            "s1",
            role="assistant",
            content="alpha",
            tool_name="beta",
        )
        assert any(hit["id"] == message_id for hit in verified.search_messages("alpha"))
        assert any(hit["id"] == message_id for hit in verified.search_messages("beta"))
        assert not verified.search_messages("alphabeta")
    finally:
        verified.close()
