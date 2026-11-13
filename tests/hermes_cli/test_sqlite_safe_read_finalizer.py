"""A tracked connection collected without close() releases its registry entry (#149291).

Only ``close()`` used to untrack, so a dropped connection left its path "live" for the life of the
process and every ``offline_file_access`` on it was refused. Each tracked connection now carries a
release object that SQLite frees only after closing the files, so the entry is released once the
descriptors are gone and never before, on any thread, without taking a lock during collection.
Each connection must still be untracked exactly once, whether it is closed, collected, or both.
"""

import gc
from collections import deque
import os
import sqlite3
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

import hermes_agent.hermes_cli.sqlite_safe_read as sr
from hermes_agent.hermes_cli.sqlite_safe_read import (
    LiveConnectionError,
    connect_tracked,
    has_live_connection,
    offline_file_access,
)


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(sr, "_live_connections", {})
    monkeypatch.setattr(sr, "_released", deque(), raising=False)
    path = tmp_path / "state.db"
    sqlite3.connect(path).close()
    return path


def _offline_allowed(path) -> bool:
    try:
        with offline_file_access(path):
            return True
    except LiveConnectionError:
        return False


def _open_and_drop(path, **kwargs) -> None:
    conn = connect_tracked(str(path), tracking_path=path, **kwargs)
    conn.execute("SELECT 1")


def test_collected_connection_releases_its_entry(db):
    _open_and_drop(db)
    gc.collect()
    assert not has_live_connection(db)
    assert _offline_allowed(db)


def test_collecting_one_of_two_keeps_the_path_live(db):
    kept = connect_tracked(str(db), tracking_path=db)
    _open_and_drop(db)
    gc.collect()
    assert has_live_connection(db), "the open connection's locks are still at stake"
    assert not _offline_allowed(db)
    kept.close()
    assert not has_live_connection(db)


def test_close_then_collect_untracks_once(db):
    other = connect_tracked(str(db), tracking_path=db)
    closed = connect_tracked(str(db), tracking_path=db)
    closed.close()
    del closed
    gc.collect()
    assert has_live_connection(db), "a second untrack would have released the other connection"
    other.close()
    assert not has_live_connection(db)


def test_retrofitted_connection_is_released_when_collected(db):
    class OwnFactory(sqlite3.Connection):
        pass

    def opener(path, **kwargs):
        kwargs["factory"] = OwnFactory  # discards the tracking factory, forcing the retrofit
        return sqlite3.connect(path, **kwargs)

    _open_and_drop(db, connect_fn=opener)
    gc.collect()
    assert not has_live_connection(db)


def test_failed_release_registration_leaves_no_entry(db, monkeypatch):
    class NoFunctions(sqlite3.Connection):
        def create_function(self, *args, **kwargs):
            raise sqlite3.ProgrammingError("cannot register")

    with pytest.raises(sqlite3.ProgrammingError):
        connect_tracked(str(db), tracking_path=db, factory=NoFunctions)
    assert not has_live_connection(db)


def _fd_paths():
    if sys.platform == "linux":
        for fd in os.listdir("/proc/self/fd"):
            try:
                yield os.readlink(f"/proc/self/fd/{fd}")
            except OSError:
                pass
    else:  # macOS: no /proc, but F_GETPATH names an open descriptor's file
        import fcntl
        for fd in os.listdir("/dev/fd"):
            try:
                yield fcntl.fcntl(int(fd), fcntl.F_GETPATH, bytes(1024)).rstrip(b"\0").decode()
            except OSError:
                pass


def _db_fds(path) -> int:
    target = os.path.realpath(path)
    return sum(1 for name in _fd_paths() if name.startswith(target))


def _wal_db(db) -> None:
    setup = sqlite3.connect(db)
    setup.execute("PRAGMA journal_mode=WAL")
    setup.execute("CREATE TABLE t(x)")
    setup.close()


def _open_writer(db):
    conn = connect_tracked(str(db), tracking_path=db)
    conn.execute("BEGIN IMMEDIATE")
    conn.execute("INSERT INTO t VALUES (1)")
    return conn


fd_listing = pytest.mark.skipif(sys.platform not in ("linux", "darwin"), reason="needs an fd listing")


@fd_listing
@pytest.mark.parametrize("how", ["collected", "collected-on-another-thread", "closed"])
def test_release_object_dies_only_after_the_files_close(db, monkeypatch, how):
    _wal_db(db)
    seen = []
    original = sr._ReleaseOnClose.__del__

    def spy(self):
        seen.append(_db_fds(db))
        original(self)

    monkeypatch.setattr(sr._ReleaseOnClose, "__del__", spy)
    if how == "closed":
        _open_writer(db).close()
    elif how == "collected":
        _open_writer(db)
        gc.collect()
    else:
        box = []
        worker = threading.Thread(target=lambda: box.append(_open_writer(db)))
        worker.start()
        worker.join()
        box.clear()  # collected here, where check_same_thread forbids close()
        gc.collect()
    assert seen == [0], "the entry could be released while a descriptor was still open"
    assert not has_live_connection(db)


@fd_listing
def test_probe_refused_until_descriptor_closed(db, monkeypatch):
    _wal_db(db)
    key = sr._key(db)
    released = []
    original = sr._track_key

    def spy(k, delta=1):
        original(k, delta)
        if k == key and k not in sr._live_connections:
            released.append(_db_fds(db))

    monkeypatch.setattr(sr, "_track_key", spy)
    _open_writer(db)
    gc.collect()
    assert has_live_connection(db) is False
    assert released == [0], "the entry was released while the descriptor was still open"


_WRITE_LOCK_CHILD = """import sqlite3, sys
c = sqlite3.connect(sys.argv[1], timeout=0.5)
try:
    c.execute("BEGIN IMMEDIATE")
    print("ACQUIRED")
except sqlite3.OperationalError:
    print("LOCKED")
"""


def test_cross_thread_drop_closes_and_releases(db):
    # A thread-affine connection can't be closed off its owner thread, so SQLite's own
    # close-on-collect must still run: its write lock goes away and so does the entry.
    _wal_db(db)
    box = []
    worker = threading.Thread(target=lambda: box.append(_open_writer(db)))
    worker.start()
    worker.join()
    box.clear()
    gc.collect()
    child = subprocess.run([sys.executable, "-c", _WRITE_LOCK_CHILD, str(db)],
                           capture_output=True, text=True, check=True)
    assert child.stdout.strip() == "ACQUIRED"
    assert not has_live_connection(db)


def test_collection_never_waits_for_the_guard(db, tmp_path):
    # Collection can run while its thread holds any lock (a logging handler's, for one), so it
    # must not wait for _live_lock while another thread holds it and may need that lock.
    gc.disable()
    try:
        conn = connect_tracked(str(db), tracking_path=db)
        conn.self_ref = conn  # left for the cycle collector
        del conn
        holding, done = threading.Event(), threading.Event()

        def hold():
            with offline_file_access(tmp_path / "other.db"):
                holding.set()
                done.wait(10)

        holder = threading.Thread(target=hold)
        holder.start()
        assert holding.wait(5)
        collector = threading.Thread(target=gc.collect)
        collector.start()
        collector.join(5)
        blocked = collector.is_alive()
        done.set()
        holder.join(5)
        collector.join(5)
    finally:
        gc.enable()
    assert not blocked, "collection waited for _live_lock"
    assert not has_live_connection(db)


def test_close_with_an_unfinished_cursor_stays_live_until_the_cursor_goes(db):
    # sqlite3_close_v2 keeps the files (and their locks) open while a statement is unfinished.
    _wal_db(db)
    filler = sqlite3.connect(db)
    filler.executemany("INSERT INTO t VALUES (?)", [(i,) for i in range(3)])
    filler.commit()
    filler.close()
    conn = connect_tracked(str(db), tracking_path=db)
    cursor = conn.execute("SELECT x FROM t")
    cursor.fetchone()
    conn.close()
    assert has_live_connection(db), "the files are still open behind the unfinished cursor"
    del cursor
    gc.collect()
    assert not has_live_connection(db)


def test_a_factory_that_drops_the_release_function_is_refused(db):
    class Silent(sqlite3.Connection):
        def create_function(self, *args, **kwargs):
            return None

    with pytest.raises(sqlite3.ProgrammingError):
        connect_tracked(str(db), tracking_path=db, factory=Silent)
    assert not has_live_connection(db)


def test_connect_tracked_runs_no_sql(db):
    # Any statement makes SQLite read the schema, so a malformed one would fail the connect
    # before the repair that opens through it could run.
    statements = []

    class Traced(sqlite3.Connection):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.set_trace_callback(statements.append)

    connect_tracked(str(db), tracking_path=db, factory=Traced).close()
    assert statements == []

    setup = sqlite3.connect(db)
    setup.execute("CREATE TABLE t(x)")
    setup.execute("PRAGMA writable_schema=ON")
    setup.execute("INSERT INTO sqlite_master SELECT * FROM sqlite_master WHERE name='t'")
    setup.commit()
    setup.close()
    connect_tracked(str(db), tracking_path=db).close()  # malformed schema: still opens


_EXCLUSIVE_HOLDER = """import sqlite3, sys
c = sqlite3.connect(sys.argv[1], isolation_level=None)
c.execute("CREATE TABLE t(x)")
c.execute("BEGIN EXCLUSIVE")
c.execute("INSERT INTO t VALUES (1)")
print("HELD", flush=True)
sys.stdin.read()
"""


def test_connect_does_not_wait_on_a_locked_database(db):
    # A schema read inside connect_tracked would wait out the busy timeout under _live_lock.
    holder = subprocess.Popen([sys.executable, "-c", _EXCLUSIVE_HOLDER, str(db)],
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "HELD"
        start = time.monotonic()
        connect_tracked(str(db), tracking_path=db, timeout=2).close()
        elapsed = time.monotonic() - start
    finally:
        holder.stdin.close()
        holder.wait(10)
        holder.stdout.close()
    assert elapsed < 0.5


def test_connection_in_a_cycle_is_released_by_the_collector(db):
    gc.disable()
    try:
        conn = connect_tracked(str(db), tracking_path=db)
        conn.self_ref = conn  # only the cycle collector can free it
        del conn
        assert has_live_connection(db)
        gc.collect()
    finally:
        gc.enable()
    assert not has_live_connection(db)


def test_exit_with_a_leaked_connection_is_clean(tmp_path):
    # At exit a daemon thread may be frozen holding _live_lock while leaked connections are
    # collected: exit must neither wait for it nor report errors from the release object.
    db = tmp_path / "state.db"
    sqlite3.connect(db).close()
    script = textwrap.dedent("""
        import gc, sqlite3, sys, threading, time
        import hermes_cli.sqlite_safe_read as sr
        gc.disable()
        db, other = sys.argv[1], sys.argv[2]
        class Holder: pass
        h = Holder(); h.conn = sr.connect_tracked(db, tracking_path=db); h.me = h
        del h
        held = threading.Event()
        def worker():
            with sr.offline_file_access(other):
                held.set(); time.sleep(60)
        threading.Thread(target=worker, daemon=True).start()
        held.wait()
    """)
    root = Path(sr.__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-c", script, str(db), str(tmp_path / "other.db")],
        cwd=root, timeout=30, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Exception ignored" not in result.stderr
