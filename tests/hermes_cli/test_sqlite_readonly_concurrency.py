"""Real-process checks for SQLite reader locks and crash recovery."""
from __future__ import annotations

from contextlib import closing
from pathlib import Path
from queue import Queue
import sqlite3
import subprocess
import sys
import threading
import os
from concurrent.futures import ThreadPoolExecutor

import hermes_cli.sqlite_safe_read as safe_read

import pytest

from hermes_cli.sqlite_safe_read import (
    connect_tracked, has_live_connection, open_sqlite_readonly,
    read_header_bytes_preopen,
)


_TRY_WRITE = """
import sqlite3, sys
conn = sqlite3.connect(sys.argv[1], timeout=0, isolation_level=None)
try:
    conn.execute('BEGIN EXCLUSIVE')
    conn.execute("INSERT INTO sample VALUES ('external')")
    conn.execute('COMMIT')
    print('COMMITTED')
except sqlite3.OperationalError as error:
    if error.sqlite_errorcode != sqlite3.SQLITE_BUSY:
        raise
    print('BLOCKED')
finally:
    conn.close()
"""


_INTERRUPTED_WRITE = """
import sqlite3, sys
conn = sqlite3.connect(sys.argv[1], isolation_level=None)
conn.execute('PRAGMA synchronous=FULL')
conn.execute('PRAGMA cache_size=1')
conn.execute('BEGIN IMMEDIATE')
conn.executemany('INSERT INTO sample VALUES (?)', [('unfinished' + 'x' * 8192,)] * 200)
print('UNCOMMITTED', flush=True)
sys.stdin.read()
"""


def _make_db(path: Path, mode: str) -> None:
    with closing(sqlite3.connect(path)) as conn:
        conn.execute(f"PRAGMA journal_mode={mode}")
        conn.execute("CREATE TABLE sample(value TEXT)")
        conn.execute("INSERT INTO sample VALUES ('committed')")
        conn.commit()


def _external_write(path: Path) -> str:
    completed = subprocess.run(
        [sys.executable, "-c", _TRY_WRITE, str(path)],
        capture_output=True, text=True, check=True, timeout=30,
    )
    return completed.stdout.strip()


def _assert_reader_guard(tmp_path) -> None:
    path = tmp_path / "guard#literal.db"
    _make_db(path, "DELETE")
    with closing(open_sqlite_readonly(path, isolation_level=None)) as reader:
        reader.execute("BEGIN")
        assert reader.execute("SELECT value FROM sample").fetchall() == [("committed",)]
        assert read_header_bytes_preopen(path, length=16) is None
        # Another process must not enter the exclusive transaction after the probe.
        assert _external_write(path) == "BLOCKED"
    assert _external_write(path) == "COMMITTED"


@pytest.mark.platforms("linux")
def test_guarded_reader_keeps_external_writer_out_on_linux(tmp_path):
    _assert_reader_guard(tmp_path)


@pytest.mark.platforms("macos")
def test_guarded_reader_keeps_external_writer_out_on_macos(tmp_path):
    _assert_reader_guard(tmp_path)


@pytest.mark.platforms("windows")
def test_guarded_reader_keeps_external_writer_out_on_windows(tmp_path):
    _assert_reader_guard(tmp_path)


@pytest.mark.parametrize("mode", ["DELETE", "WAL"])
def test_killed_spilled_writer_obeys_readonly_recovery_contract(tmp_path, mode):
    # Force dirty pages to disk before the kill, not merely into a page cache.
    path = tmp_path / "crash.db"
    _make_db(path, mode)
    proc = subprocess.Popen(
        [sys.executable, "-c", _INTERRUPTED_WRITE, str(path)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True,
    )
    ready = Queue()
    stdout = proc.stdout
    assert stdout is not None
    thread = threading.Thread(target=lambda: ready.put(stdout.readline()), daemon=True)
    thread.start()
    try:
        marker = ready.get(timeout=30).strip()
        if marker != "UNCOMMITTED":
            proc.wait(timeout=30)
            diagnostic = proc.stderr.read() if proc.stderr is not None else ""
            pytest.fail(f"Interrupted writer never entered its transaction: {diagnostic}")
        proc.kill()
        proc.wait(timeout=30)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=30)
        thread.join(timeout=30)
        for pipe in (proc.stdin, proc.stdout, proc.stderr):
            if pipe is not None:
                pipe.close()
    assert not thread.is_alive()
    if mode == "DELETE":
        journal = Path(str(path) + "-journal")
        assert journal.stat().st_size > 512
        # Read-only access must not silently recover by obtaining write access.
        with pytest.raises(sqlite3.OperationalError) as observed:
            with closing(open_sqlite_readonly(path)) as reader:
                reader.execute("SELECT value FROM sample").fetchall()
        assert observed.value.sqlite_errorcode == sqlite3.SQLITE_READONLY_ROLLBACK
        assert not has_live_connection(path)
        # SQLite itself recovers the hot journal on an explicit writable open.
        with closing(connect_tracked(path)) as recovery:
            assert recovery.execute("SELECT value FROM sample").fetchall() == [("committed",)]
    else:
        assert Path(str(path) + "-wal").stat().st_size > 32
    with closing(open_sqlite_readonly(path)) as reader:
        assert reader.execute("SELECT value FROM sample").fetchall() == [("committed",)]
        assert reader.execute("PRAGMA integrity_check").fetchall() == [("ok",)]


@pytest.mark.platforms("linux")
def test_repeated_threaded_lifecycle_has_no_fd_or_registry_leak(tmp_path):
    path = tmp_path / "stress.db"
    _make_db(path, "DELETE")
    baseline_registry = dict(safe_read._live_connections)
    baseline_fds = len(os.listdir("/proc/self/fd"))
    barrier = threading.Barrier(8)

    def worker():
        barrier.wait(timeout=30)
        for _ in range(100):
            conn = open_sqlite_readonly(path)
            try:
                assert has_live_connection(path)
                assert read_header_bytes_preopen(path) is None
                assert conn.execute("SELECT count(*) FROM sample").fetchone() == (1,)
            finally:
                conn.close()
                conn.close()

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(worker) for _ in range(8)]
        for future in futures:
            future.result(timeout=60)
    assert safe_read._live_connections == baseline_registry
    assert len(os.listdir("/proc/self/fd")) == baseline_fds
    print(f"STRESS opens=800 threads=8 fd_before={baseline_fds} fd_after={len(os.listdir('/proc/self/fd'))} registry_restored=True")


@pytest.mark.platforms("linux")
def test_factory_exceptions_do_not_leak_fds_or_registry(tmp_path):
    path = tmp_path / "factory.db"
    _make_db(path, "DELETE")
    baseline_registry = dict(safe_read._live_connections)
    baseline_fds = len(os.listdir("/proc/self/fd"))

    class FailingFactory(sqlite3.Connection):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            raise RuntimeError("factory failed after open")

    try:
        for _ in range(100):
            with pytest.raises(RuntimeError, match="factory failed after open"):
                open_sqlite_readonly(path, factory=FailingFactory)
        assert safe_read._live_connections == baseline_registry
        assert len(os.listdir("/proc/self/fd")) == baseline_fds
    finally:
        # Clean up cyclic SQLite objects without masking the immediate leak.
        import gc
        gc.collect()


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("alias_kind", ["duplicate", "symlink", "hardlink"])
def test_alias_probe_preserves_reader_lock_and_accounting(tmp_path, alias_kind):
    path = tmp_path / "aliases.db"
    _make_db(path, "DELETE")
    alias = tmp_path / "alias.db"
    if alias_kind == "duplicate":
        alias = path.parent / "." / path.name
    elif alias_kind == "symlink":
        alias.symlink_to(path)
    else:
        os.link(path, alias)
    baseline = dict(safe_read._live_connections)
    with closing(open_sqlite_readonly(path, isolation_level=None)) as reader:
        reader.execute("BEGIN")
        reader.execute("SELECT * FROM sample").fetchall()
        assert _external_write(path) == "BLOCKED"
        # Probe from another thread while the real external writer retries.
        with ThreadPoolExecutor(max_workers=2) as pool:
            probes = pool.submit(lambda: [read_header_bytes_preopen(alias) for _ in range(100)])
            writes = pool.submit(lambda: [_external_write(path) for _ in range(5)])
            observed_probes = probes.result(timeout=30)
            observed_writes = writes.result(timeout=30)
        assert observed_writes == ["BLOCKED"] * 5, (alias_kind, observed_writes)
        assert observed_probes == [None] * 100
        assert has_live_connection(alias)
        with closing(open_sqlite_readonly(alias)):
            assert sum(safe_read._live_connections.values()) == sum(baseline.values()) + 2
        assert sum(safe_read._live_connections.values()) == sum(baseline.values()) + 1
    assert safe_read._live_connections == baseline
    assert _external_write(path) == "COMMITTED"
