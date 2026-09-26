"""Header probes retire only after SQLite releases the inode, not at process exit."""

from contextlib import closing as close_connection, contextmanager
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import threading

import psutil
import pytest

from hermes_cli import profile_lifecycle, profiles
from hermes_cli.sqlite_safe_read import connect_tracked, has_live_connection
from hermes_state import SessionDB
from hermes_state_dbfile import _pread_db_header


def _hold_databases(path, pipe):
    databases = []
    try:
        for index in range(2):
            db = SessionDB(db_path=path)
            databases.append(db)
            db.create_session(f"child-{index}", source="cli")
            assert db.get_session(f"child-{index}") is not None
        while True:
            pipe.send({"live": has_live_connection(path), "pid": os.getpid()})
            if not pipe.poll(30):
                raise TimeoutError("parent did not release child databases")
            command = pipe.recv()
            if command == "exit":
                return
            assert command == "close"
            databases.pop().close()
    finally:
        for db in databases:
            db.close()
        pipe.close()


@contextmanager
def _external_databases(path):
    ctx = multiprocessing.get_context("spawn")
    parent, child_pipe = ctx.Pipe()
    child = ctx.Process(target=_hold_databases, args=(path, child_pipe))
    child.start()
    child_pipe.close()
    try:
        yield child, parent
    finally:
        if child.is_alive():
            parent.send("exit")
        child.join(15)
        if child.is_alive():
            child.terminate()
            child.join(5)
        parent.close()
        assert child.exitcode == 0


def _receive(pipe):
    assert pipe.poll(30), "child did not acknowledge database lifecycle"
    return pipe.recv()


@pytest.mark.platforms("posix")
def test_closed_child_allows_profile_delete_but_live_sibling_refuses(tmp_path, monkeypatch):
    root = tmp_path / "isolated-hermes"
    root.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(root))
    # No service manager, live backend or multiplexer is part of this DB test.
    for name in ("_cleanup_gateway_service", "_maybe_unregister_gateway_service",
                 "_maybe_register_gateway_service", "_stop_bot_desktop", "_notify_multiplexer"):
        monkeypatch.setattr(profiles, name, lambda *a, **k: None)
    monkeypatch.setattr(profiles, "_profile_bound_backend_pids", lambda *a, **k: [])
    monkeypatch.setattr(profile_lifecycle, "_PROFILE_DB_RELEASE_TIMEOUT_SECONDS", 0)

    # A -> B -> A includes a fresh incarnation at the reused pathname.
    for name in ("alpha", "beta", "alpha"):
        home = profiles.create_profile(name, no_alias=True, no_skills=True)
        with _external_databases(home / "state.db") as (child, pipe):
            assert _receive(pipe)["live"]
            pipe.send("close")
            assert _receive(pipe)["live"], "one SQLite sibling still owns the file"
            assert child.pid in profile_lifecycle.external_profile_file_holders(home)
            with pytest.raises(RuntimeError, match="external process"):
                profiles.delete_profile(name, yes=True)
            assert home.is_dir() and not profiles.profile_home_is_tombstoned(home)
            pipe.send("close")
            assert not _receive(pipe)["live"]
            assert child.is_alive()
            # Real public delete must return success, not just remove the tree
            # and report a post-remove identity settlement failure.
            assert profiles.delete_profile(name, yes=True) == home
            assert not home.exists() and child.is_alive()
            assert not [f for f in psutil.Process(child.pid).open_files()
                        if Path(f.path).is_relative_to(home)]


def _foreign_exclusive(path):
    result = subprocess.run(
        [sys.executable, "-c", """
import sqlite3, sys
conn = sqlite3.connect(sys.argv[1], timeout=0, isolation_level=None)
try:
    conn.execute('BEGIN EXCLUSIVE')
    print('acquired')
except sqlite3.OperationalError as exc:
    assert 'locked' in str(exc), exc
    print('blocked')
finally:
    conn.close()
""", str(path)], capture_output=True, text=True, timeout=30, check=True,
    )
    return result.stdout.strip() == "acquired"


@pytest.mark.platforms("posix")
def test_probe_retirement_preserves_alias_and_deferred_close_locks(tmp_path, monkeypatch):
    path = tmp_path / "state.db"
    alias = tmp_path / "alias.db"
    owner = connect_tracked(path, isolation_level=None)
    owner.execute("CREATE TABLE items(value)")
    owner.executemany("INSERT INTO items VALUES (?)", [(1,), (2,), (3,)])
    os.link(path, alias)
    sibling = connect_tracked(alias, isolation_level=None)
    cursor = None
    try:
        assert _pread_db_header(path, 16) == b"SQLite format 3\x00"
        sibling.execute("BEGIN IMMEDIATE")
        owner.close()
        assert not has_live_connection(path) and has_live_connection(alias)
        assert not _foreign_exclusive(path), "probe close cancelled the alias's SQLite locks"
        # Replacement retires the old cached probe, but must keep its inode's
        # locks intact until the alias owner is actually gone.
        path.unlink()
        replacement = connect_tracked(path, isolation_level=None)
        try:
            replacement.execute("CREATE TABLE replacement(value)")
            assert _pread_db_header(path, 16) == b"SQLite format 3\x00"
        finally:
            replacement.close()
        assert not _foreign_exclusive(alias), "retired probe cancelled the replaced inode's locks"
        sibling.rollback()
        cursor = sibling.execute("SELECT * FROM items")
        assert cursor.fetchone() == (1,)
        sibling.close()  # sqlite3_close_v2 defers the fd close until cursor finalization.
        assert not has_live_connection(alias)
        assert not _foreign_exclusive(alias), "probe close cancelled a deferred SQLite close's locks"
        # CPython refuses cursor.close() after Connection.close(); releasing
        # the cursor finalizes the outstanding statement and its zombie DB.
        cursor = None
        # An offline read must not leave another process-lifetime probe behind.
        assert _pread_db_header(path, 16) == b"SQLite format 3\x00"
        assert _foreign_exclusive(path) and _foreign_exclusive(alias)
        identities = {(p.stat().st_dev, p.stat().st_ino) for p in (path, alias)}
        for opened in psutil.Process().open_files():
            stat = os.stat(opened.path)  # psutil reports fd=-1 on macOS
            assert (stat.st_dev, stat.st_ino) not in identities, "closed database still has a probe fd"
    finally:
        cursor = None
        sibling.close()
        owner.close()

    # "recovery" is session_recovery's own opener of the SessionDB-initialized output it
    # rebuilds: every in-process opener, not only SessionDB's, must survive probe cleanup.
    for opener in ("writer", "readiness", "doctor", "recovery"):
        _assert_concurrent_opener_keeps_locks(path, monkeypatch, opener)


def _assert_concurrent_opener_keeps_locks(path, monkeypatch, opener):
    # Stop the real cleanup AFTER descriptor inventory but before closing the
    # probe. Both a new writer and the production readiness reader must wait.
    import sqlite3
    import hermes_state_dbfile as dbfile
    from gateway.readiness import _probe_state_db
    from hermes_cli.doctor_state import _session_count
    from hermes_cli import session_recovery, sqlite_safe_read

    with close_connection(connect_tracked(path, isolation_level=None)) as seed:
        seed.execute("CREATE TABLE IF NOT EXISTS sessions(id TEXT)")
    closing = connect_tracked(path, check_same_thread=False)
    assert _pread_db_header(path, 16) == b"SQLite format 3\x00"
    probe_fd = dbfile._HEADER_PROBE_FDS[str(path)][0]
    entered, release, attempting, locked, finish_reader = (threading.Event() for _ in range(5))
    new_connections, errors, results = [], [], []
    real_close, real_connect = os.close, sqlite3.connect
    lifecycle_lock = sqlite_safe_read._live_lock

    class ObservedLock:
        def __enter__(self):
            if not lifecycle_lock.acquire(blocking=False):
                # A tracked opener has reached the contended lock. This is an
                # event barrier, not a timing guess about thread scheduling.
                attempting.set()
                lifecycle_lock.acquire()
            return self

        def __exit__(self, *args):
            lifecycle_lock.release()

    class PausedReaderConnection(sqlite3.Connection):
        def execute(self, sql, *args, **kwargs):
            cursor = super().execute(sql, *args, **kwargs)
            if sql in ("SELECT name FROM sqlite_master LIMIT 1", "SELECT COUNT(*) FROM sessions"):
                # Pause the real schema read after SQLite takes SHARED, before
                # fetchone finalizes it. Do not synthesize a writer transaction.
                locked.set()
                attempting.set()
                assert finish_reader.wait(15)
            return cursor

    def reader_connect(*args, **kwargs):
        kwargs["factory"] = PausedReaderConnection
        return real_connect(*args, **kwargs)

    def close_probe(fd):
        if fd == probe_fd:
            entered.set()
            assert release.wait(15)
        return real_close(fd)

    def close_owner():
        try:
            closing.close()
        except BaseException as exc:
            errors.append(exc)

    def open_successor():
        try:
            if opener == "readiness":
                results.append(_probe_state_db(path.parent))
            elif opener == "doctor":
                results.append(_session_count(path))
            elif opener == "recovery":
                # A same-thread connection: hold the write lock, then close it here.
                with close_connection(session_recovery._connect(path)) as conn:
                    conn.execute("BEGIN IMMEDIATE")
                    locked.set()
                    attempting.set()
                    assert finish_reader.wait(15)
                    conn.execute("ROLLBACK")
            else:
                conn = connect_tracked(path, check_same_thread=False, isolation_level=None)
                new_connections.append(conn)
                conn.execute("BEGIN IMMEDIATE")
                locked.set()
                attempting.set()
        except BaseException as exc:
            errors.append(exc)

    with monkeypatch.context() as patcher:
        patcher.setattr(sqlite_safe_read, "_live_lock", ObservedLock())
        patcher.setattr(dbfile.os, "close", close_probe)
        if opener in ("readiness", "doctor"):
            patcher.setattr(sqlite3, "connect", reader_connect)
        closer = threading.Thread(target=close_owner)
        successor = threading.Thread(target=open_successor)
        closer.start()
        try:
            assert entered.wait(10)
            successor.start()
            assert attempting.wait(10), errors
            if locked.is_set():
                assert not _foreign_exclusive(path), "successor never held its SQLite lock"
            release.set()
            closer.join(10)
            assert not closer.is_alive() and not errors, errors
            assert locked.wait(10), errors
            assert not _foreign_exclusive(path), f"{opener} lost its locks to old probe cleanup"
        finally:
            release.set()
            finish_reader.set()
            closer.join(15)
            if successor.ident is not None:
                successor.join(15)
            for conn in new_connections:
                conn.close()
            closing.close()
        assert not successor.is_alive() and not errors, errors
        if opener in ("readiness", "doctor"):
            assert results == ([{"status": "ok"}] if opener == "readiness" else [0])
    assert not has_live_connection(path) and _foreign_exclusive(path)


@pytest.mark.platforms("posix")
def test_descriptor_scan_skips_unprobed_closes_and_never_blocks_tracked_opens(tmp_path, monkeypatch):
    """Probe release enumerates every descriptor, so it must be off the connection hot path:
    closing a database that has no probe never scans, and the scan never runs under the
    lifecycle lock that every tracked open needs."""
    state, other = tmp_path / "state.db", tmp_path / "kanban.db"
    owner = connect_tracked(state, check_same_thread=False, isolation_level=None)
    owner.execute("CREATE TABLE items(value)")
    assert _pread_db_header(state, 16) == b"SQLite format 3\x00"  # the live owner's probe
    scans, scanning, resume, pause = [], threading.Event(), threading.Event(), threading.Event()
    real_listdir = os.listdir

    def listdir(directory):
        if os.fspath(directory) in ("/proc/self/fd", "/dev/fd"):
            scans.append(directory)
            if pause.is_set():
                scanning.set()
                assert resume.wait(15)
        return real_listdir(directory)

    monkeypatch.setattr(os, "listdir", listdir)
    for _ in range(5):
        with close_connection(connect_tracked(other)) as conn:
            conn.execute("SELECT 1")
    assert not scans, "closing a database without a header probe enumerated every descriptor"

    opened = []

    def open_other():
        with close_connection(connect_tracked(other, check_same_thread=False)) as conn:
            opened.append(conn.execute("SELECT 1").fetchone())

    pause.set()
    closer = threading.Thread(target=owner.close)  # the probe owner's last close scans
    opener = threading.Thread(target=open_other)
    closer.start()
    try:
        assert scanning.wait(10), "the probe owner's last close never looked for other holders"
        opener.start()
        opener.join(10)
        assert opened == [(1,)], "a tracked open waited behind another close's descriptor scan"
    finally:
        resume.set()
        closer.join(15)
        if opener.ident is not None:
            opener.join(15)
        owner.close()
    assert not closer.is_alive() and not has_live_connection(state)
