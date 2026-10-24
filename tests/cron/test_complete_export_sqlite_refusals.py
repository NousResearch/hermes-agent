"""Actual SQLite setup-refusal preservation; no substituted connection or backend."""
from __future__ import annotations

from pathlib import Path
import sqlite3

import pytest

import hermes_cron_complete_export as export
from cron import executions


def _business_snapshot(path: Path) -> tuple[bytes, tuple[int, int]]:
    stat = path.stat()
    return path.read_bytes(), (stat.st_dev, stat.st_ino)


def test_native_length_limit_overflow_refusal_preserves_orphan_wal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "profile"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    path = home / "cron" / "executions.db"
    monkeypatch.setattr(executions, "EXECUTIONS_FILE", path)
    row = executions.create_execution("opaque-refusal-job", source="synthetic-refusal")
    # Produce real WAL pages with the stock writer, then let SQLite itself retain
    # that generation on close. No reader or writer remains to suppress checkpoints.
    with sqlite3.connect(path, isolation_level=None) as holder:
        holder.setconfig(sqlite3.SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE, True)
        assert holder.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        holder.execute("PRAGMA wal_autocheckpoint=0")
        assert executions.mark_execution_running(row["id"])["status"] == "running"
        assert holder.execute("SELECT status FROM executions WHERE id=?", (row["id"],)).fetchone()[0] == "running"
    holder.close()
    wal = Path(str(path) + "-wal")
    assert wal.is_file() and wal.stat().st_size > 32
    before = {str(item): _business_snapshot(item) for item in (path, wal)}
    # This valid public budget reaches the real native setlimit overflow path.
    # It allocates no budget-sized object and does not replace the SQLite connection.
    with pytest.raises(export.CronCaptureRefused, match="cron_capture_sqlite_row_limit_unavailable"):
        with export.open_existing_cron_complete_capture(
            budget=export.CronCaptureBudget(max_bytes=1 << 70)
        ) as capture:
            tuple(capture.iter_records())
    after = {str(item): _business_snapshot(item) for item in (path, wal)}
    assert after == before
