"""External SQLite backups must preserve committed WAL records (#132705)."""

from __future__ import annotations

import sqlite3
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from hermes_cli import backup


def test_external_sqlite_backup_keeps_committed_wal_rows(tmp_path, monkeypatch):
    """Raw-copying ``cache.db`` while excluding ``cache.db-wal`` drops committed rows.

    Reproduce with an open writer (WAL not checkpointed), then archive through the
    real external path and assert the restored DB still has the committed row.
    """
    hermes_root = tmp_path / ".hermes"
    hermes_root.mkdir()
    # Minimal HERMES_HOME content so the backup is not empty of in-home files.
    (hermes_root / "config.yaml").write_text("model:\n  default: test\n", encoding="utf-8")

    external = tmp_path / "external"
    external.mkdir()
    db_path = external / "cache.db"

    writer = sqlite3.connect(db_path)
    writer.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)")
    writer.commit()
    writer.execute("PRAGMA journal_mode=WAL")
    writer.execute("PRAGMA wal_autocheckpoint=0")
    writer.execute("INSERT INTO t(v) VALUES ('committed-in-wal')")
    writer.commit()
    # Keep the writer open so the WAL stays live (the bug class under test).
    assert (external / "cache.db-wal").exists()

    out_zip = tmp_path / "backup.zip"
    monkeypatch.setattr(backup, "_collect_external_entries", lambda: (
        [(db_path, "_external/fixture/cache.db")],
        [],
    ))
    monkeypatch.setattr(backup, "display_hermes_home", lambda: str(hermes_root))

    ok = backup._run_backup_locked(SimpleNamespace(output=str(out_zip), keep=0), hermes_root)
    assert ok is True
    assert out_zip.exists()

    with zipfile.ZipFile(out_zip) as zf:
        names = zf.namelist()
        assert "_external/fixture/cache.db" in names
        assert not any(n.endswith(".db-wal") for n in names)
        extracted = tmp_path / "restored.db"
        extracted.write_bytes(zf.read("_external/fixture/cache.db"))

    reader = sqlite3.connect(extracted)
    rows = reader.execute("SELECT v FROM t").fetchall()
    reader.close()
    writer.close()
    assert rows == [("committed-in-wal",)]


def test_looks_like_sqlite_db_detects_extensionless(tmp_path):
    path = tmp_path / "cognee_db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE t (id INTEGER)")
    conn.commit()
    conn.close()
    assert backup._looks_like_sqlite_db(path) is True
    assert backup._looks_like_sqlite_db(tmp_path / "notes.json") is False
