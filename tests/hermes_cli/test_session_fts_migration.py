import os
import shutil
import sqlite3
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_state import SessionDB
from hermes_cli.session_fts import (
    SessionFtsMaintenanceError,
    configure_migrate_fts_parser,
    run_external_fts_migration,
)


LEGACY_INLINE_BOTH_SQL = """
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
CREATE VIRTUAL TABLE messages_fts_trigram USING fts5(content, tokenize='trigram');

CREATE TRIGGER messages_fts_insert AFTER INSERT ON messages BEGIN
  INSERT INTO messages_fts(rowid, content)
  VALUES (new.id, COALESCE(new.content, '') || ' ' || COALESCE(new.tool_name, '') || ' ' || COALESCE(new.tool_calls, ''));
END;
CREATE TRIGGER messages_fts_delete AFTER DELETE ON messages BEGIN
  DELETE FROM messages_fts WHERE rowid = old.id;
END;
CREATE TRIGGER messages_fts_update AFTER UPDATE OF id, content, tool_name, tool_calls ON messages BEGIN
  DELETE FROM messages_fts WHERE rowid = old.id;
  INSERT INTO messages_fts(rowid, content)
  VALUES (new.id, COALESCE(new.content, '') || ' ' || COALESCE(new.tool_name, '') || ' ' || COALESCE(new.tool_calls, ''));
END;

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

INSERT INTO messages_fts(rowid, content)
SELECT id, COALESCE(content, '') || ' ' || COALESCE(tool_name, '') || ' ' || COALESCE(tool_calls, '') FROM messages;
INSERT INTO messages_fts_trigram(rowid, content)
SELECT id, COALESCE(content, '') || ' ' || COALESCE(tool_name, '') || ' ' || COALESCE(tool_calls, '') FROM messages;
"""


def _options(**overrides):
    values = {
        "check_only": False,
        "writers_stopped": False,
        "unsafe_no_backup": False,
        "yes": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _disk_usage(free=100 * 1024**3):
    return shutil._ntuple_diskusage(total=200 * 1024**3, used=0, free=free)


@pytest.fixture
def legacy_inline_db(tmp_path):
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    try:
        if not db._fts_enabled or not db._trigram_available:
            pytest.skip("SQLite build lacks FTS5 trigram support")
        db.create_session(session_id="s1", source="cli")
        db.append_message("s1", role="user", content="maintenance migration sentinel")
    finally:
        db.close()

    raw = sqlite3.connect(db_path)
    try:
        raw.executescript(LEGACY_INLINE_BOTH_SQL)
        raw.commit()
    finally:
        raw.close()
    return db_path


def test_parser_exposes_explicit_safety_flags():
    import argparse

    parser = argparse.ArgumentParser()
    configure_migrate_fts_parser(parser)
    args = parser.parse_args(
        ["--check-only", "--writers-stopped", "--unsafe-no-backup", "--yes"]
    )
    assert args.check_only is True
    assert args.writers_stopped is True
    assert args.unsafe_no_backup is True
    assert args.yes is True


@pytest.mark.parametrize(
    ("table_name", "sql", "expected"),
    [
        (
            "messages_fts",
            'CREATE VIRTUAL TABLE "messages_fts" USING fts5( content, '
            'content = "messages_fts_source", content_rowid = "id" )',
            "external",
        ),
        (
            "messages_fts",
            "CREATE VIRTUAL TABLE messages_fts USING fts5("
            "content, content='messages', content_rowid='id')",
            "unsupported",
        ),
        (
            "messages_fts",
            "CREATE VIRTUAL TABLE messages_fts USING fts5("
            "content, content='messages_fts_source', content_rowid='rowid')",
            "unsupported",
        ),
        (
            "messages_fts_trigram",
            "CREATE VIRTUAL TABLE messages_fts_trigram USING fts5("
            "content, content='messages_fts_source', content_rowid='id', "
            "tokenize='unicode61')",
            "unsupported",
        ),
        (
            "messages_fts",
            "CREATE TABLE messages_fts(content TEXT)",
            "unsupported",
        ),
    ],
)
def test_canonical_fts_schema_classifier_is_structural(table_name, sql, expected):
    assert SessionDB._classify_fts_table_sql(table_name, sql) == expected


@pytest.mark.parametrize(
    ("object_type", "sql", "expected"),
    [
        (None, None, "missing"),
        ("table", "CREATE TABLE messages_fts_source(id, content)", "unsupported"),
        (
            "view",
            'CREATE VIEW "messages_fts_source" AS SELECT "id", '
            "COALESCE(\"content\", '') || ' ' || "
            "COALESCE(\"tool_name\", '') || ' ' || "
            "COALESCE(\"tool_calls\", '') AS \"content\" FROM \"messages\";",
            "canonical",
        ),
        (
            "view",
            "CREATE VIEW messages_fts_source AS "
            "SELECT id, 'WRONG' AS content FROM messages",
            "unsupported",
        ),
    ],
)
def test_fts_source_schema_classifier_is_structural(object_type, sql, expected):
    assert SessionDB._classify_fts_source_sql(object_type, sql) == expected


@pytest.mark.parametrize(
    "source_sql",
    [
        "CREATE VIEW messages_fts_source AS "
        "SELECT id, 'WRONG' AS content FROM messages",
        "CREATE TABLE messages_fts_source(id INTEGER, content TEXT)",
    ],
)
def test_unsupported_fts_source_objects_are_preserved_and_refused(
    legacy_inline_db, source_sql
):
    raw = sqlite3.connect(legacy_inline_db)
    try:
        raw.execute(source_sql)
        raw.commit()
        original = raw.execute(
            "SELECT type, sql FROM sqlite_master WHERE name = 'messages_fts_source'"
        ).fetchone()
    finally:
        raw.close()

    report = run_external_fts_migration(
        _options(check_only=True),
        db_path=legacy_inline_db,
        disk_usage_fn=lambda _path: _disk_usage(),
    )
    assert report["storage_mode"] == "unsupported"
    with pytest.raises(SessionFtsMaintenanceError, match="unsupported"):
        run_external_fts_migration(
            _options(writers_stopped=True),
            db_path=legacy_inline_db,
            disk_usage_fn=lambda _path: _disk_usage(),
        )
    assert not list(legacy_inline_db.parent.glob("*.pre-fts-external-*.bak"))

    maintenance = SessionDB.open_for_fts_maintenance(legacy_inline_db)
    try:
        with pytest.raises(sqlite3.OperationalError, match="messages_fts_source"):
            maintenance.migrate_fts_to_external_content()
    finally:
        maintenance.close()

    raw = sqlite3.connect(legacy_inline_db)
    try:
        assert raw.execute(
            "SELECT type, sql FROM sqlite_master WHERE name = 'messages_fts_source'"
        ).fetchone() == original
    finally:
        raw.close()


def test_canonical_quoted_source_view_is_accepted_and_retained(legacy_inline_db):
    raw = sqlite3.connect(legacy_inline_db)
    try:
        raw.execute(
            'CREATE VIEW "messages_fts_source" AS SELECT "id", '
            "COALESCE(\"content\", '') || ' ' || "
            "COALESCE(\"tool_name\", '') || ' ' || "
            "COALESCE(\"tool_calls\", '') AS \"content\" FROM \"messages\""
        )
        raw.commit()
    finally:
        raw.close()

    check = run_external_fts_migration(
        _options(check_only=True),
        db_path=legacy_inline_db,
        disk_usage_fn=lambda _path: _disk_usage(),
    )
    assert check["storage_mode"] == "inline"
    migrated = run_external_fts_migration(
        _options(writers_stopped=True),
        db_path=legacy_inline_db,
        disk_usage_fn=lambda _path: _disk_usage(),
    )
    assert migrated["final_mode"] == "external"


def test_external_tables_require_canonical_source_view(tmp_path):
    db_path = tmp_path / "external-missing-source.db"
    seeded = SessionDB(db_path=db_path)
    try:
        if not seeded._fts_enabled:
            pytest.skip("SQLite build lacks FTS5")
    finally:
        seeded.close()
    raw = sqlite3.connect(db_path)
    try:
        raw.execute("DROP VIEW messages_fts_source")
        raw.commit()
    finally:
        raw.close()

    report = run_external_fts_migration(
        _options(check_only=True),
        db_path=db_path,
        disk_usage_fn=lambda _path: _disk_usage(),
    )
    assert report["storage_mode"] == "unsupported"
    maintenance = SessionDB.open_for_fts_maintenance(db_path)
    try:
        with pytest.raises(sqlite3.OperationalError, match="messages_fts_source"):
            maintenance.migrate_fts_to_external_content()
    finally:
        maintenance.close()


def test_check_only_uses_encoded_uri_for_special_paths(tmp_path):
    special_parent = tmp_path / "parent?#"
    special_parent.mkdir()
    db_path = special_parent / "state?#.db"
    db = SessionDB(db_path=db_path)
    try:
        if not db._fts_enabled:
            pytest.skip("SQLite build lacks FTS5")
    finally:
        db.close()

    report = run_external_fts_migration(
        _options(check_only=True),
        db_path=db_path,
        disk_usage_fn=lambda _path: _disk_usage(),
    )

    assert report["storage_mode"] in {"external", "base-only-external"}
    assert not (special_parent / "state").exists()


def test_ordinary_canonical_tables_are_refused_by_command_and_core(tmp_path):
    db_path = tmp_path / "ordinary-canonical.db"
    seeded = SessionDB(db_path=db_path)
    try:
        if not seeded._fts_enabled:
            pytest.skip("SQLite build lacks FTS5")
    finally:
        seeded.close()

    raw = sqlite3.connect(db_path)
    try:
        raw.executescript(
            """
            DROP TRIGGER IF EXISTS messages_fts_insert;
            DROP TRIGGER IF EXISTS messages_fts_delete;
            DROP TRIGGER IF EXISTS messages_fts_update;
            DROP TRIGGER IF EXISTS messages_fts_trigram_insert;
            DROP TRIGGER IF EXISTS messages_fts_trigram_delete;
            DROP TRIGGER IF EXISTS messages_fts_trigram_update;
            DROP TABLE IF EXISTS messages_fts;
            DROP TABLE IF EXISTS messages_fts_trigram;
            DROP VIEW IF EXISTS messages_fts_source;
            CREATE TABLE messages_fts(content TEXT);
            CREATE TABLE messages_fts_trigram(content TEXT);
            INSERT INTO messages_fts VALUES('preserve me');
            """
        )
        raw.commit()
    finally:
        raw.close()

    report = run_external_fts_migration(
        _options(check_only=True),
        db_path=db_path,
        disk_usage_fn=lambda _path: _disk_usage(),
    )
    assert report["storage_mode"] == "unsupported"

    with pytest.raises(SessionFtsMaintenanceError, match="unsupported"):
        run_external_fts_migration(
            _options(writers_stopped=True, unsafe_no_backup=True),
            db_path=db_path,
            disk_usage_fn=lambda _path: _disk_usage(),
        )

    maintenance = SessionDB.open_for_fts_maintenance(db_path)
    try:
        with pytest.raises(sqlite3.OperationalError, match="unsupported canonical"):
            maintenance.migrate_fts_to_external_content()
    finally:
        maintenance.close()

    verify = sqlite3.connect(db_path)
    try:
        assert verify.execute("SELECT content FROM messages_fts").fetchone()[0] == (
            "preserve me"
        )
    finally:
        verify.close()


def test_check_only_reports_requirements_without_mutation_or_backup(legacy_inline_db):
    report = run_external_fts_migration(
        _options(check_only=True),
        db_path=legacy_inline_db,
        disk_usage_fn=lambda _path: _disk_usage(),
    )

    assert report["check_only"] is True
    assert report["storage_mode"] == "inline"
    assert report["space_ok"] is True
    assert report["backup_path"] is None
    assert list(legacy_inline_db.parent.glob("*.pre-fts-external-*.bak")) == []

    db = SessionDB(db_path=legacy_inline_db, read_only=True)
    try:
        assert db.fts_storage_mode() == "inline"
    finally:
        db.close()


def test_check_only_wal_preserves_logical_database(legacy_inline_db):
    writer = sqlite3.connect(legacy_inline_db, isolation_level=None)
    try:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("PRAGMA wal_autocheckpoint=0")
        writer.execute("BEGIN IMMEDIATE")
        writer.execute(
            "INSERT INTO state_meta(key, value) VALUES('wal_preflight_sentinel', 'kept') "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value"
        )
        writer.commit()
        before_schema = writer.execute("PRAGMA schema_version").fetchone()[0]
        before_value = writer.execute(
            "SELECT value FROM state_meta WHERE key = 'wal_preflight_sentinel'"
        ).fetchone()[0]

        report = run_external_fts_migration(
            _options(check_only=True),
            db_path=legacy_inline_db,
            disk_usage_fn=lambda _path: _disk_usage(),
        )

        assert report["check_only"] is True
        assert writer.execute("PRAGMA schema_version").fetchone()[0] == before_schema
        assert writer.execute(
            "SELECT value FROM state_meta WHERE key = 'wal_preflight_sentinel'"
        ).fetchone()[0] == before_value
    finally:
        writer.close()

    assert list(legacy_inline_db.parent.glob("*.pre-fts-external-*.bak")) == []


def test_partial_fts_schema_is_reported_and_refused(legacy_inline_db):
    raw = sqlite3.connect(legacy_inline_db)
    try:
        raw.executescript(
            """
            DROP TRIGGER messages_fts_trigram_insert;
            DROP TRIGGER messages_fts_trigram_delete;
            DROP TRIGGER messages_fts_trigram_update;
            DROP TABLE messages_fts_trigram;
            """
        )
        raw.commit()
    finally:
        raw.close()

    report = run_external_fts_migration(
        _options(check_only=True),
        db_path=legacy_inline_db,
        disk_usage_fn=lambda _path: _disk_usage(),
    )
    assert report["storage_mode"] == "base-only-inline"

    with pytest.raises(SessionFtsMaintenanceError, match="both FTS5 tables"):
        run_external_fts_migration(
            _options(writers_stopped=True),
            db_path=legacy_inline_db,
            disk_usage_fn=lambda _path: _disk_usage(),
        )
    assert list(legacy_inline_db.parent.glob("*.pre-fts-external-*.bak")) == []


def test_base_only_migrates_and_noops_when_trigram_is_unavailable(
    legacy_inline_db, monkeypatch
):
    import hermes_cli.session_fts as session_fts

    raw = sqlite3.connect(legacy_inline_db)
    try:
        raw.executescript(
            """
            DROP TRIGGER messages_fts_trigram_insert;
            DROP TRIGGER messages_fts_trigram_delete;
            DROP TRIGGER messages_fts_trigram_update;
            DROP TABLE messages_fts_trigram;
            """
        )
        raw.commit()
    finally:
        raw.close()

    monkeypatch.setattr(session_fts, "_runtime_supports_trigram", lambda: False)
    monkeypatch.setattr(
        SessionDB,
        "_sqlite_supports_trigram",
        lambda _self, _cursor: False,
    )

    first = run_external_fts_migration(
        _options(writers_stopped=True, unsafe_no_backup=True),
        db_path=legacy_inline_db,
        disk_usage_fn=lambda _path: _disk_usage(),
    )
    assert first["noop"] is False
    assert first["migrated_tables"] == ["messages_fts"]

    second = run_external_fts_migration(
        _options(writers_stopped=True, unsafe_no_backup=True),
        db_path=legacy_inline_db,
        disk_usage_fn=lambda _path: _disk_usage(),
    )
    assert second["noop"] is True
    assert second["storage_revision"] == "external-content-v1"


def test_actual_migration_requires_writer_stop_attestation(legacy_inline_db):
    with pytest.raises(SessionFtsMaintenanceError, match="--writers-stopped"):
        run_external_fts_migration(
            _options(),
            db_path=legacy_inline_db,
            disk_usage_fn=lambda _path: _disk_usage(),
        )
    assert list(legacy_inline_db.parent.glob("*.pre-fts-external-*.bak")) == []


def test_cli_refusal_exits_nonzero(legacy_inline_db):
    env = os.environ.copy()
    env["HERMES_HOME"] = str(legacy_inline_db.parent)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "hermes_cli.main",
            "sessions",
            "migrate-fts",
            "--yes",
        ],
        cwd=Path(__file__).parents[2],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "--writers-stopped" in result.stdout + result.stderr
    assert list(legacy_inline_db.parent.glob("*.pre-fts-external-*.bak")) == []


def test_cli_successfully_migrates_isolated_database(legacy_inline_db):
    env = os.environ.copy()
    env["HERMES_HOME"] = str(legacy_inline_db.parent)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "hermes_cli.main",
            "sessions",
            "migrate-fts",
            "--writers-stopped",
            "--unsafe-no-backup",
            "--yes",
        ],
        cwd=Path(__file__).parents[2],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "FTS migration completed and validated" in result.stdout
    migrated = SessionDB(db_path=legacy_inline_db, read_only=True)
    try:
        assert migrated.fts_storage_mode() == "external"
    finally:
        migrated.close()


def test_lock_contention_is_reported_cleanly(legacy_inline_db):
    blocker = sqlite3.connect(legacy_inline_db, isolation_level=None)
    blocker.execute("BEGIN IMMEDIATE")
    try:
        with pytest.raises(SessionFtsMaintenanceError, match="database is locked"):
            run_external_fts_migration(
                _options(writers_stopped=True, unsafe_no_backup=True),
                db_path=legacy_inline_db,
                disk_usage_fn=lambda _path: _disk_usage(),
            )
    finally:
        blocker.rollback()
        blocker.close()

    assert list(legacy_inline_db.parent.glob("*.pre-fts-external-*.bak")) == []


def test_disk_usage_failure_is_normalized(legacy_inline_db):
    def fail_disk_usage(_path):
        raise OSError("injected disk usage failure")

    with pytest.raises(SessionFtsMaintenanceError, match="FTS preflight failed"):
        run_external_fts_migration(
            _options(writers_stopped=True),
            db_path=legacy_inline_db,
            disk_usage_fn=fail_disk_usage,
        )


def test_verified_backup_runs_while_migration_writer_lock_is_held(
    legacy_inline_db, monkeypatch
):
    import hermes_cli.session_fts as session_fts

    real_backup = session_fts._create_verified_backup
    observed = []

    def verify_lock_then_backup(source, destination):
        contender = sqlite3.connect(source, timeout=0, isolation_level=None)
        try:
            try:
                contender.execute("BEGIN IMMEDIATE")
            except sqlite3.OperationalError as exc:
                observed.append("locked" in str(exc).lower())
            else:
                observed.append(False)
                contender.rollback()
        finally:
            contender.close()
        real_backup(source, destination)

    monkeypatch.setattr(session_fts, "_create_verified_backup", verify_lock_then_backup)
    run_external_fts_migration(
        _options(writers_stopped=True),
        db_path=legacy_inline_db,
        disk_usage_fn=lambda _path: _disk_usage(),
    )
    assert observed == [True]


def test_sqlite_tmpdir_overrides_python_tempdir(tmp_path, monkeypatch):
    import hermes_cli.session_fts as session_fts

    sqlite_tmp = tmp_path / "sqlite-tmp"
    python_tmp = tmp_path / "python-tmp"
    sqlite_tmp.mkdir()
    python_tmp.mkdir()
    monkeypatch.setenv("SQLITE_TMPDIR", str(sqlite_tmp))
    monkeypatch.setattr(session_fts.tempfile, "gettempdir", lambda: str(python_tmp))

    assert session_fts._sqlite_temp_path() == sqlite_tmp.resolve()


def test_eof_confirmation_cancels_without_changes(legacy_inline_db):
    def eof(_prompt):
        raise EOFError

    report = run_external_fts_migration(
        _options(writers_stopped=True, yes=False),
        db_path=legacy_inline_db,
        input_fn=eof,
        disk_usage_fn=lambda _path: _disk_usage(),
    )
    assert report["cancelled"] is True
    assert list(legacy_inline_db.parent.glob("*.pre-fts-external-*.bak")) == []


def test_insufficient_space_refuses_before_backup(legacy_inline_db):
    with pytest.raises(SessionFtsMaintenanceError, match="free space"):
        run_external_fts_migration(
            _options(writers_stopped=True),
            db_path=legacy_inline_db,
            disk_usage_fn=lambda _path: _disk_usage(free=1),
        )
    assert list(legacy_inline_db.parent.glob("*.pre-fts-external-*.bak")) == []


def test_post_backup_space_drop_preserves_backup_and_refuses_migration(
    legacy_inline_db,
):
    def disk_usage_after_backup(_path):
        backup_exists = bool(
            list(legacy_inline_db.parent.glob("*.pre-fts-external-*.bak"))
        )
        return _disk_usage(free=1 if backup_exists else 100 * 1024**3)

    with pytest.raises(
        SessionFtsMaintenanceError,
        match="post-backup migration requirement",
    ):
        run_external_fts_migration(
            _options(writers_stopped=True),
            db_path=legacy_inline_db,
            disk_usage_fn=disk_usage_after_backup,
        )

    backups = list(legacy_inline_db.parent.glob("*.pre-fts-external-*.bak"))
    assert len(backups) == 1
    untouched = SessionDB(db_path=legacy_inline_db, read_only=True)
    try:
        assert untouched.fts_storage_mode() == "inline"
    finally:
        untouched.close()


def test_verified_backup_then_migration_preserves_search(legacy_inline_db):
    report = run_external_fts_migration(
        _options(writers_stopped=True),
        db_path=legacy_inline_db,
        disk_usage_fn=lambda _path: _disk_usage(),
    )

    backup_path = Path(report["backup_path"])
    assert backup_path.exists()
    assert stat.S_IMODE(backup_path.stat().st_mode) == 0o600
    backup = sqlite3.connect(backup_path)
    try:
        assert backup.execute("PRAGMA quick_check").fetchone()[0] == "ok"
        backup_sql = backup.execute(
            "SELECT sql FROM sqlite_master WHERE name='messages_fts'"
        ).fetchone()[0]
        assert "content='messages_fts_source'" not in backup_sql
    finally:
        backup.close()

    db = SessionDB(db_path=legacy_inline_db)
    try:
        assert db.fts_storage_mode() == "external"
        hits = db.search_messages("sentinel")
        assert [row["session_id"] for row in hits] == ["s1"]
        assert db._conn.execute("PRAGMA quick_check").fetchone()[0] == "ok"
    finally:
        db.close()


def test_backup_fsync_failure_removes_partial_file(legacy_inline_db, monkeypatch):
    import hermes_cli.session_fts as session_fts

    destination = legacy_inline_db.with_name("forced-fsync-failure.bak")

    def fail_fsync(_fd):
        raise OSError("injected fsync failure")

    monkeypatch.setattr(session_fts.os, "fsync", fail_fsync)
    with pytest.raises(OSError, match="injected fsync failure"):
        session_fts._create_verified_backup(legacy_inline_db, destination)
    assert destination.exists() is False


def test_backup_fsyncs_file_and_parent_directory(legacy_inline_db, monkeypatch):
    import hermes_cli.session_fts as session_fts

    destination = legacy_inline_db.with_name("durable-backup.bak")
    calls = []
    monkeypatch.setattr(session_fts.os, "fsync", lambda fd: calls.append(fd))

    session_fts._create_verified_backup(legacy_inline_db, destination)

    assert destination.exists()
    expected_calls = 1 if os.name == "nt" else 2
    assert len(calls) == expected_calls


def test_backup_refuses_to_overwrite_existing_destination(legacy_inline_db):
    import hermes_cli.session_fts as session_fts

    destination = legacy_inline_db.with_name("existing-backup.bak")
    destination.write_bytes(b"do not overwrite")

    with pytest.raises(SessionFtsMaintenanceError, match="refusing to overwrite"):
        session_fts._create_verified_backup(legacy_inline_db, destination)
    assert destination.read_bytes() == b"do not overwrite"


def test_backup_io_failure_is_clean_and_does_not_migrate(
    legacy_inline_db, monkeypatch
):
    import hermes_cli.session_fts as session_fts

    def fail_backup(_source, _destination):
        raise OSError("injected backup failure")

    monkeypatch.setattr(session_fts, "_create_verified_backup", fail_backup)
    with pytest.raises(SessionFtsMaintenanceError, match="verified backup failed"):
        run_external_fts_migration(
            _options(writers_stopped=True),
            db_path=legacy_inline_db,
            disk_usage_fn=lambda _path: _disk_usage(),
        )

    db = SessionDB(db_path=legacy_inline_db, read_only=True)
    try:
        assert db.fts_storage_mode() == "inline"
    finally:
        db.close()

    writer = sqlite3.connect(legacy_inline_db, timeout=0, isolation_level=None)
    try:
        writer.execute("BEGIN IMMEDIATE")
        writer.rollback()
    finally:
        writer.close()


def test_migration_failure_preserves_verified_backup(
    legacy_inline_db, monkeypatch
):
    def fail_migration(_db):
        raise sqlite3.OperationalError("injected migration failure")

    monkeypatch.setattr(SessionDB, "migrate_fts_to_external_content", fail_migration)

    with pytest.raises(
        SessionFtsMaintenanceError,
        match="backup preserved at",
    ):
        run_external_fts_migration(
            _options(writers_stopped=True),
            db_path=legacy_inline_db,
            disk_usage_fn=lambda _path: _disk_usage(),
        )

    backups = list(legacy_inline_db.parent.glob("*.pre-fts-external-*.bak"))
    assert len(backups) == 1
    backup = sqlite3.connect(backups[0])
    try:
        assert backup.execute("PRAGMA quick_check").fetchone()[0] == "ok"
    finally:
        backup.close()


def test_unsafe_no_backup_is_explicit_and_skips_backup(legacy_inline_db):
    report = run_external_fts_migration(
        _options(writers_stopped=True, unsafe_no_backup=True),
        db_path=legacy_inline_db,
        disk_usage_fn=lambda _path: _disk_usage(),
    )
    assert report["backup_path"] is None
    assert list(legacy_inline_db.parent.glob("*.pre-fts-external-*.bak")) == []


def test_declined_confirmation_leaves_inline_database_untouched(legacy_inline_db):
    report = run_external_fts_migration(
        _options(writers_stopped=True, yes=False),
        db_path=legacy_inline_db,
        input_fn=lambda _prompt: "no",
        disk_usage_fn=lambda _path: _disk_usage(),
    )
    assert report["cancelled"] is True
    assert list(legacy_inline_db.parent.glob("*.pre-fts-external-*.bak")) == []

    db = SessionDB(db_path=legacy_inline_db, read_only=True)
    try:
        assert db.fts_storage_mode() == "inline"
    finally:
        db.close()


def test_console_check_only_uses_guarded_maintenance_api(
    legacy_inline_db, monkeypatch
):
    import hermes_cli.session_fts as session_fts
    from hermes_cli.console_engine import HermesConsoleEngine

    monkeypatch.setattr(session_fts, "DEFAULT_DB_PATH", legacy_inline_db)
    engine = HermesConsoleEngine()
    pending = engine.execute("sessions migrate-fts --check-only")
    assert pending.status == "confirm_required"

    result = engine.execute("sessions migrate-fts --check-only", confirmed=True)
    assert result.status == "ok"
    assert "FTS storage mode: inline" in result.output
    assert list(legacy_inline_db.parent.glob("*.pre-fts-external-*.bak")) == []


def test_gateway_process_cannot_run_mutating_migration(legacy_inline_db, monkeypatch):
    monkeypatch.setenv("_HERMES_GATEWAY", "1")
    with pytest.raises(SessionFtsMaintenanceError, match="outside the gateway"):
        run_external_fts_migration(
            _options(writers_stopped=True),
            db_path=legacy_inline_db,
            disk_usage_fn=lambda _path: _disk_usage(),
        )
