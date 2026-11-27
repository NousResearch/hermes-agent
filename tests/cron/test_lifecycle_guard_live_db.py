"""Regression tests for raw lifecycle-guard reads of live SQLite files."""

from __future__ import annotations

import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import pytest

from cron import lifecycle_guard
from hermes_cli.sqlite_safe_read import connect_tracked


@pytest.mark.parametrize("suffix", ["", "-wal", "-shm"])
def test_referenced_script_read_refuses_live_sqlite_connection(tmp_path, monkeypatch, suffix):
    db = tmp_path / "state.db"
    conn = connect_tracked(db)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("CREATE TABLE sample (value)")

        def must_not_read(*_args, **_kwargs):
            raise AssertionError("raw-opened a live SQLite database or sidecar")

        monkeypatch.setattr(lifecycle_guard, "_read_referenced_script_isolated", must_not_read)
        text, unsafe = lifecycle_guard._read_referenced_script(db.with_name(db.name + suffix))
        assert text is None
        assert unsafe is True
    finally:
        conn.close()


def test_stalled_script_read_does_not_block_an_unrelated_sqlite_connection(tmp_path, monkeypatch):
    script = tmp_path / "script.sh"
    script.write_text("echo ok\n", encoding="utf-8")
    entered, release = Event(), Event()
    read = lifecycle_guard._read_referenced_script_isolated

    def stall_read(*args, **kwargs):
        entered.set()
        assert release.wait(10), "test did not release the stalled reader"
        return read(*args, **kwargs)

    def open_ledger():
        conn = connect_tracked(tmp_path / "state.db")
        try:
            return conn.execute("SELECT 1").fetchone()
        finally:
            conn.close()

    monkeypatch.setattr(lifecycle_guard, "_read_referenced_script_isolated", stall_read)
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending_read = pool.submit(lifecycle_guard._read_referenced_script, script)
        try:
            assert entered.wait(5)
            assert pool.submit(open_ledger).result(timeout=2) == (1,)
            assert not pending_read.done()
        finally:
            release.set()
        assert pending_read.result(timeout=5) == ("echo ok\n", False)


@pytest.mark.platforms("macos", "linux")
def test_isolated_read_does_not_drop_locks_when_connection_opens_after_check(tmp_path, monkeypatch):
    db = tmp_path / "state.db"
    initial = connect_tracked(db)
    initial.execute("CREATE TABLE sample (value)")
    initial.close()
    entered, release = Event(), Event()
    read = lifecycle_guard._read_referenced_script_isolated

    def pause_after_live_check(*args, **kwargs):
        entered.set()
        assert release.wait(10)
        return read(*args, **kwargs)

    monkeypatch.setattr(lifecycle_guard, "_read_referenced_script_isolated", pause_after_live_check)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending_read = pool.submit(lifecycle_guard._read_referenced_script, db)
        conn = None
        try:
            assert entered.wait(5)
            conn = connect_tracked(db, timeout=0)
            conn.execute("BEGIN EXCLUSIVE")
            release.set()
            text, unsafe = pending_read.result(timeout=5)
            assert not unsafe and text is not None and text.startswith("SQLite format 3")
            subprocess.run(
                [sys.executable, "-c", """
import sqlite3, sys
with sqlite3.connect(sys.argv[1], timeout=0) as conn:
    try:
        conn.execute('BEGIN IMMEDIATE')
    except sqlite3.OperationalError as exc:
        assert exc.sqlite_errorcode == sqlite3.SQLITE_BUSY
    else:
        raise AssertionError('raw reader dropped the parent process SQLite lock')
""", str(db)], check=True, capture_output=True, timeout=5,
            )
        finally:
            release.set()
            if conn is not None:
                conn.close()


def test_referenced_script_worker_crash_blocks_scan(tmp_path, monkeypatch):
    script = tmp_path / "script.sh"
    script.write_text("echo ok\n", encoding="utf-8")
    worker = tmp_path / "crashing_worker.py"
    worker.write_text('raise RuntimeError("reader crashed")\n', encoding="utf-8")
    monkeypatch.setattr(lifecycle_guard, "__file__", str(worker))

    with pytest.raises(lifecycle_guard.GatewayLifecycleBlocked, match=script.name):
        lifecycle_guard.check_gateway_lifecycle(f'bash "{script}"')


def test_referenced_script_read_preserves_normal_script_behavior(tmp_path):
    script = tmp_path / "script.sh"
    script.write_text("echo ok\n", encoding="utf-8")
    text, unsafe = lifecycle_guard._read_referenced_script(script)

    assert text == "echo ok\n"
    assert unsafe is False


def test_invalid_referenced_script_path_is_still_tolerated():
    text, unsafe = lifecycle_guard._read_referenced_script(
        Path("/tmp/hermes\x00binary")
    )
    assert text is None
    assert unsafe is False
