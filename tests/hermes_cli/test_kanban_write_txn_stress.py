"""Concurrency stress test for the kanban write-transaction duration cap.

Run with:
    pytest tests/hermes_cli/test_kanban_write_txn_stress.py -v -s
"""

from __future__ import annotations

import concurrent.futures
import os
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


def _make_home(tmp_path: Path) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    os.environ["HERMES_HOME"] = str(home)
    kb.init_db()
    return home


def _create_task_in_subprocess(db_path: str, title: str) -> str:
    """Run create_task in a fresh Python process using the given DB."""
    code = (
        "import sqlite3, sys\n"
        "sys.path.insert(0, '/home/hunter/hermes/hermes-agent')\n"
        "from hermes_cli import kanban_db as kb\n"
        "conn = kb.connect()\n"
        "tid = kb.create_task(conn, title=sys.argv[2])\n"
        "print(tid)\n"
        "conn.close()\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, db_path, title],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"subprocess failed: {result.stderr.strip() or result.stdout.strip()}"
        )
    return result.stdout.strip()


def test_20_concurrent_creates_all_succeed(tmp_path):
    """Spawn ~20 parallel kanban create processes against a scratch board.

    Assert all succeed, no .corrupt.* produced, and integrity_check OK after.
    """
    home = _make_home(tmp_path)
    db_path = str(home / "kanban.db")
    titles = [f"stress-{i:02d}" for i in range(20)]

    with concurrent.futures.ProcessPoolExecutor(max_workers=20) as exe:
        futures = [exe.submit(_create_task_in_subprocess, db_path, t)
                   for t in titles]
        ids = [f.result() for f in concurrent.futures.as_completed(futures)]

    assert len(ids) == 20, f"expected 20 ids, got {len(ids)}"
    assert len(set(ids)) == 20, f"duplicate ids: {ids}"

    # No corrupt backups produced
    corrupt_files = list(home.glob("kanban.db.corrupt.*"))
    assert not corrupt_files, f"corrupt backups found: {corrupt_files}"

    # integrity_check passes
    conn = kb.connect()
    try:
        row = conn.execute("PRAGMA integrity_check").fetchone()
        assert row and (row[0] or "").lower() == "ok", (
            f"integrity_check failed: {row[0] if row else 'no row'}"
        )
        count = conn.execute(
            "SELECT COUNT(*) FROM tasks WHERE title LIKE 'stress-%'"
        ).fetchone()[0]
        assert count == 20, f"expected 20 stress tasks, got {count}"
    finally:
        conn.close()


def test_write_txn_cap_interrupts_long_txn(tmp_path, monkeypatch):
    """A transaction that runs longer than the write-txn cap is aborted.

    Verifies the duration cap prevents a single writer from wedging the board.
    Uses a recursive CTE to generate enough VM opcodes that the progress
    handler fires within the 1-second cap.
    """
    home = _make_home(tmp_path)
    monkeypatch.setenv("HERMES_KANBAN_WRITE_TXN_TIMEOUT_SECONDS", "1")
    conn = kb.connect()
    try:
        with pytest.raises(sqlite3.OperationalError) as exc_info:
            with kb.write_txn(conn):
                conn.execute(
                    "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM c WHERE x < 10000000) "
                    "SELECT max(x) FROM c"
                )
        msg = str(exc_info.value).lower()
        assert "interrupted" in msg, f"expected interrupted, got: {msg}"
    finally:
        conn.close()


def test_killed_writer_does_not_quarantine_on_restart(tmp_path, monkeypatch):
    """Simulate a killed writer mid-transaction and verify the next connect()
    does NOT trip the corruption quarantine (wal_checkpoint recovery should
    clear the transient state).
    """
    home = _make_home(tmp_path)
    db_path = home / "kanban.db"
    import signal

    code = (
        "import sqlite3, time, sys\n"
        "conn = sqlite3.connect(sys.argv[1], isolation_level=None)\n"
        "conn.execute('PRAGMA journal_mode=WAL')\n"
        "conn.execute('BEGIN IMMEDIATE')\n"
        "conn.execute('UPDATE tasks SET title = title')\n"
        "time.sleep(60)\n"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", code, str(db_path)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    time.sleep(1)
    proc.send_signal(signal.SIGKILL)
    proc.wait()

    conn2 = kb.connect()
    try:
        row = conn2.execute("PRAGMA integrity_check").fetchone()
        assert row and (row[0] or "").lower() == "ok", (
            f"integrity_check failed after killed writer: {row[0] if row else 'no row'}"
        )
    finally:
        conn2.close()

    corrupt_files = list(home.glob("kanban.db.corrupt.*"))
    assert not corrupt_files, f"unexpected corrupt backups: {corrupt_files}"


def test_write_txn_cap_default_is_60_seconds(tmp_path):
    """Default cap is 60 s when env is unset."""
    home = _make_home(tmp_path)
    # Ensure env is absent
    os.environ.pop("HERMES_KANBAN_WRITE_TXN_TIMEOUT_SECONDS", None)
    assert kb._resolve_write_txn_timeout_seconds() == 60.0


def test_write_txn_cap_env_override(tmp_path, monkeypatch):
    """Env HERMES_KANBAN_WRITE_TXN_TIMEOUT_SECONDS overrides default."""
    home = _make_home(tmp_path)
    monkeypatch.setenv("HERMES_KANBAN_WRITE_TXN_TIMEOUT_SECONDS", "120")
    assert kb._resolve_write_txn_timeout_seconds() == 120.0


def test_write_txn_cap_zero_disables(tmp_path, monkeypatch):
    """Setting the env to 0 disables the cap."""
    home = _make_home(tmp_path)
    monkeypatch.setenv("HERMES_KANBAN_WRITE_TXN_TIMEOUT_SECONDS", "0")
    assert kb._resolve_write_txn_timeout_seconds() == 0.0


def test_write_txn_cap_negative_disables(tmp_path, monkeypatch):
    """Setting the env to a negative value disables the cap."""
    home = _make_home(tmp_path)
    monkeypatch.setenv("HERMES_KANBAN_WRITE_TXN_TIMEOUT_SECONDS", "-5")
    assert kb._resolve_write_txn_timeout_seconds() == 0.0


def test_write_txn_cap_invalid_string_falls_back(tmp_path, monkeypatch):
    """Non-numeric env value falls back to default."""
    home = _make_home(tmp_path)
    monkeypatch.setenv("HERMES_KANBAN_WRITE_TXN_TIMEOUT_SECONDS", "abc")
    assert kb._resolve_write_txn_timeout_seconds() == 60.0
