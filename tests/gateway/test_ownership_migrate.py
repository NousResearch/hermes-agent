"""#47 Phase 2 — sandbox tests for ``ownership_migrate.py``.

Every test runs on a private ``tmp_path`` DB; the production DB is never
touched. The tool must: create the two ledger tables behind a backup, verify
rowcounts + integrity, be idempotent, refuse a non-WAL DB, and roll back to a
state that matches the backup exactly.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from ownership_migrate import (
    BACKUP_SUFFIX,
    LEDGER_TABLES,
    Abort,
    _user_tables,
    cmd_apply,
    cmd_rollback,
    cmd_status,
    main,
)


# --- helpers ----------------------------------------------------------------

def _wal_db(path: Path, *, rows: int = 1) -> Path:
    """A minimal board-shaped DB in WAL mode with real rows to preserve."""
    conn = sqlite3.connect(str(path))
    try:
        assert conn.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
        conn.execute("CREATE TABLE tasks (id TEXT PRIMARY KEY, title TEXT)")
        conn.execute("CREATE TABLE task_events (id INTEGER PRIMARY KEY, task_id TEXT, kind TEXT)")
        for i in range(rows):
            conn.execute("INSERT INTO tasks VALUES (?, ?)", (f"t{i}", f"card {i}"))
            conn.execute("INSERT INTO task_events (task_id, kind) VALUES (?, 'created')", (f"t{i}",))
        conn.commit()
    finally:
        conn.close()
    return path


def _delete_db(path: Path) -> Path:
    """A rollback-journal DB (journal_mode must NOT be wal)."""
    conn = sqlite3.connect(str(path))
    try:
        assert conn.execute("PRAGMA journal_mode=DELETE").fetchone()[0] == "delete"
        conn.execute("CREATE TABLE tasks (id TEXT PRIMARY KEY)")
        conn.execute("INSERT INTO tasks VALUES ('t1')")
        conn.commit()
    finally:
        conn.close()
    return path


def _tables(path: Path) -> list[str]:
    conn = sqlite3.connect(str(path))
    try:
        return _user_tables(conn)
    finally:
        conn.close()


def _rowcount(path: Path, table: str) -> int:
    conn = sqlite3.connect(str(path))
    try:
        return int(conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
    finally:
        conn.close()


def _journal(path: Path) -> str:
    conn = sqlite3.connect(str(path))
    try:
        return str(conn.execute("PRAGMA journal_mode").fetchone()[0]).lower()
    finally:
        conn.close()


def _backups(path: Path) -> list[Path]:
    return sorted(path.parent.glob(f"{path.name}{BACKUP_SUFFIX}*.bak"))


# --- status -----------------------------------------------------------------

def test_status_reports_missing_ledger_on_wal_db(tmp_path):
    db = _wal_db(tmp_path / "kanban.db")
    res = cmd_status(db)
    assert res["wal"] is True
    assert res["journal_mode"] == "wal"
    assert res["ledger_tables_present"] == []
    assert res["schema_ok"] is None          # nothing to validate yet
    assert res["schema_problems"] == []
    assert res["integrity"] == ["ok"]
    assert res["backup_count"] == 0


def test_status_reports_ready_after_apply(tmp_path):
    db = _wal_db(tmp_path / "kanban.db")
    cmd_apply(db)
    res = cmd_status(db)
    assert res["ledger_tables_present"] == list(LEDGER_TABLES)
    assert res["schema_ok"] is True
    assert res["schema_problems"] == []
    assert res["ledger_rowcounts"] == {t: 0 for t in LEDGER_TABLES}
    assert res["backup_count"] == 1


# --- apply ------------------------------------------------------------------

def test_apply_creates_ledger_tables_with_backup_and_verifies(tmp_path):
    db = _wal_db(tmp_path / "kanban.db", rows=2)
    res = cmd_apply(db)
    assert res["already_applied"] is False
    assert res["created"] == list(LEDGER_TABLES)
    assert res["rowcounts_unchanged"] is True
    assert res["integrity"] == ["ok"]
    backup = Path(res["backup"])
    assert backup.exists()
    # The board data survived untouched.
    assert _rowcount(db, "tasks") == 2
    assert _rowcount(db, "task_events") == 2
    for table in LEDGER_TABLES:
        assert _rowcount(db, table) == 0
    # The backup is a pre-apply snapshot: it must NOT carry the ledger tables.
    assert set(LEDGER_TABLES).isdisjoint(_tables(backup))
    assert _rowcount(backup, "tasks") == 2


def test_apply_is_idempotent(tmp_path):
    db = _wal_db(tmp_path / "kanban.db")
    first = cmd_apply(db)
    second = cmd_apply(db)
    assert first["already_applied"] is False
    assert second["already_applied"] is True
    assert second["created"] == []
    assert second["backup"] is None
    assert len(_backups(db)) == 1          # no second backup on a no-op re-run
    assert _rowcount(db, "tasks") == 1
    assert set(LEDGER_TABLES) <= set(_tables(db))


def test_apply_dry_run_writes_nothing(tmp_path):
    db = _wal_db(tmp_path / "kanban.db")
    res = cmd_apply(db, dry_run=True)
    assert res["dry_run"] is True
    assert res["would_create"] == list(LEDGER_TABLES)
    assert set(LEDGER_TABLES).isdisjoint(_tables(db))
    assert _backups(db) == []
    assert _rowcount(db, "tasks") == 1


def test_apply_aborts_when_not_wal(tmp_path):
    db = _delete_db(tmp_path / "kanban.db")
    with pytest.raises(Abort, match="not 'wal'"):
        cmd_apply(db)
    assert set(LEDGER_TABLES).isdisjoint(_tables(db))
    assert _backups(db) == []
    assert main(["--db", str(db), "apply"]) == 2


def test_apply_aborts_on_incompatible_existing_schema(tmp_path):
    db = _wal_db(tmp_path / "kanban.db")
    conn = sqlite3.connect(str(db))
    try:
        conn.execute("CREATE TABLE task_ownership (scope_key TEXT)")   # wrong shape
        conn.commit()
    finally:
        conn.close()
    with pytest.raises(Abort, match="incompatible ledger schema"):
        cmd_apply(db)
    assert _backups(db) == []              # aborted before any backup


def test_apply_preserves_existing_rowcounts(tmp_path):
    db = _wal_db(tmp_path / "kanban.db", rows=5)
    before = {t: _rowcount(db, t) for t in _tables(db)}
    cmd_apply(db)
    after = {t: _rowcount(db, t) for t in _tables(db) if t not in LEDGER_TABLES}
    assert before == after


# --- rollback ---------------------------------------------------------------

def test_rollback_restores_backup_and_drops_tables(tmp_path):
    db = _wal_db(tmp_path / "kanban.db", rows=3)
    applied = cmd_apply(db)
    backup = Path(applied["backup"])
    rolled = cmd_rollback(db)
    assert rolled["matches_backup"] is True
    assert rolled["dropped"] == list(LEDGER_TABLES)
    assert set(LEDGER_TABLES).isdisjoint(_tables(db))
    assert _rowcount(db, "tasks") == 3
    assert _rowcount(db, "task_events") == 3
    assert _journal(db) == "wal"
    assert rolled["integrity"] == ["ok"]
    assert backup.exists()                 # the backup is kept for a second restore


def test_rollback_with_explicit_backup_path_discards_post_apply_writes(tmp_path):
    db = _wal_db(tmp_path / "kanban.db", rows=2)
    backup = Path(cmd_apply(db)["backup"])
    conn = sqlite3.connect(str(db))
    try:
        conn.execute("INSERT INTO tasks VALUES ('tx', 'transient')")
        conn.commit()
    finally:
        conn.close()
    assert _rowcount(db, "tasks") == 3
    rolled = cmd_rollback(db, backup=backup)
    assert rolled["matches_backup"] is True
    assert _rowcount(db, "tasks") == 2     # the post-apply row is gone


def test_rollback_aborts_without_backup(tmp_path):
    db = _wal_db(tmp_path / "kanban.db")
    with pytest.raises(Abort, match="no backup"):
        cmd_rollback(db)
    assert main(["--db", str(db), "rollback"]) == 2


def test_rollback_aborts_when_not_wal(tmp_path):
    db = _delete_db(tmp_path / "kanban.db")
    with pytest.raises(Abort, match="not 'wal'"):
        cmd_rollback(db)


def test_rollback_aborts_when_backup_is_missing_file(tmp_path):
    db = _wal_db(tmp_path / "kanban.db")
    with pytest.raises(Abort, match="no backup"):
        cmd_rollback(db, backup=tmp_path / "does-not-exist.bak")


# --- CLI exit codes ---------------------------------------------------------

def test_main_apply_status_exit_zero_and_json(tmp_path, capsys):
    db = _wal_db(tmp_path / "kanban.db")
    assert main(["--db", str(db), "apply"]) == 0
    out = capsys.readouterr().out
    assert '"ok": true' in out
    assert '"already_applied": false' in out
    assert main(["--db", str(db), "status"]) == 0
    out = capsys.readouterr().out
    assert '"ok": true' in out
    assert '"schema_ok": true' in out


def test_main_apply_missing_db_aborts(tmp_path):
    db = tmp_path / "absent.db"
    assert main(["--db", str(db), "apply"]) == 2
    assert main(["--db", str(db), "status"]) == 2
