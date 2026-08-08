"""Tests for the SessionDB read-path split (per-thread read-only connections).

The gateway shares ONE SessionDB across every agent, so recall/browse reads
used to queue behind writer flushes on self._lock — a measured production
convoy (a 0.2s FTS query stretched to 112s while 6-8 concurrent turns
flushed tool results). These tests pin the new contract: reads run on a
per-thread read-only connection under WAL, never touch self._lock, and fall
back to the legacy locked path when WAL or the read connection is missing.
"""

import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

import hermes_state
from hermes_state import SessionDB


@pytest.fixture()
def db(tmp_path):
    d = SessionDB(db_path=tmp_path / "state.db")
    d.create_session(session_id="s1", source="cli", model="m")
    d.append_message("s1", role="user", content="hello graphiti world")
    d.append_message("s1", role="assistant", content="the neo4j daemon is healthy")
    yield d
    d.close()


@pytest.mark.requires_wal
def test_read_conn_is_per_thread(db):
    conns = {}

    def grab(key):
        with db._read_ctx() as conn:
            conns[key] = conn

    t1 = threading.Thread(target=grab, args=(1,))
    t2 = threading.Thread(target=grab, args=(2,))
    t1.start(); t2.start(); t1.join(); t2.join()
    assert conns[1] is not None and conns[2] is not None
    assert conns[1] is not conns[2]


def test_read_conn_reused_within_thread(db):
    with db._read_ctx() as first:
        pass
    with db._read_ctx() as second:
        pass
    assert first is second


@pytest.mark.requires_wal
def test_reads_do_not_take_writer_lock(db):
    """Reads must complete while another thread holds self._lock."""
    acquired = db._lock.acquire()
    assert acquired
    try:
        done = {}

        def reader():
            done["session"] = db.get_session("s1")
            done["search"] = db.search_messages("graphiti", limit=10)
            done["messages"] = db.get_messages("s1")

        t = threading.Thread(target=reader)
        t.start()
        t.join(timeout=5.0)
        assert not t.is_alive(), "read path blocked on writer lock"
        assert done["session"]["id"] == "s1"
        assert any("graphiti" in (m.get("snippet") or "") for m in done["search"])
        assert len(done["messages"]) == 2
    finally:
        db._lock.release()




def test_read_your_writes(db):
    """A fresh committed write must be visible to the read connection."""
    db.append_message("s1", role="user", content="zanzibar checkpoint")
    rows = db.search_messages("zanzibar", limit=5)
    assert rows, "committed write invisible to read connection"




def test_non_wal_uses_locked_path(db):
    db._wal_active = False
    with db._read_ctx() as conn:
        assert conn is db._conn
    # And queries still work via the legacy path.
    assert db.get_session("s1")["id"] == "s1"


@pytest.mark.requires_wal
def test_read_conn_open_failure_marks_thread(db, monkeypatch, tmp_path):
    """A failed read-conn open must not retry per query; fallback still works."""
    import sqlite3 as _sqlite3

    calls = {"n": 0}
    real_connect = _sqlite3.connect

    def failing_connect(*a, **k):
        if a and isinstance(a[0], str) and a[0].startswith("file:") and "mode=ro" in a[0]:
            calls["n"] += 1
            raise _sqlite3.OperationalError("simulated open failure")
        return real_connect(*a, **k)

    fresh = SessionDB(db_path=tmp_path / "state2.db")
    try:
        fresh.create_session(session_id="x", source="cli", model="m")
        monkeypatch.setattr("hermes_state.sqlite3.connect", failing_connect)
        assert fresh.get_session("x")["id"] == "x"
        assert fresh.get_session("x")["id"] == "x"
        assert calls["n"] == 1, "open failure should be remembered per thread"
    finally:
        fresh.close()


@pytest.mark.requires_wal
def test_anchored_view_and_around_use_read_path(db):
    msgs = db.get_messages("s1")
    anchor = msgs[0]["id"]
    acquired = db._lock.acquire()
    try:
        done = {}

        def reader():
            done["around"] = db.get_messages_around("s1", anchor, window=2)
            done["view"] = db.get_anchored_view("s1", anchor, window=2, bookend=1)

        t = threading.Thread(target=reader)
        t.start(); t.join(timeout=5.0)
        assert not t.is_alive(), "anchored reads blocked on writer lock"
        assert done["around"]["window"]
        assert done["view"]["window"]
    finally:
        db._lock.release()


@pytest.mark.requires_wal
def test_finished_read_threads_are_reaped_during_db_lifetime(db):
    """Historical workers must not retain real SQLite connections."""
    opened = []

    def read_once():
        with db._read_ctx() as conn:
            assert conn.execute("SELECT 1").fetchone()[0] == 1
            opened.append(conn)

    for _ in range(40):
        worker = threading.Thread(target=read_once)
        worker.start()
        worker.join(timeout=10)
        assert not worker.is_alive()

    # The main thread's first read reaps the final finished worker before it
    # opens its own connection. Every earlier worker was reaped by its successor.
    assert db.get_session("s1")["id"] == "s1"
    assert len(db._read_conns) == 1
    for conn in opened:
        with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
            conn.execute("SELECT 1")


@pytest.mark.requires_wal
def test_cached_reader_reaps_finished_worker(db):
    """Reaping runs on reuse too, not only when a new connection registers."""
    with db._read_ctx() as cached:
        pass

    holder = {}

    def read_once():
        with db._read_ctx() as conn:
            holder["conn"] = conn

    worker = threading.Thread(target=read_once)
    worker.start()
    worker.join(timeout=10)
    assert not worker.is_alive()
    assert len(db._read_conns) == 2

    with db._read_ctx() as reused:
        assert reused is cached
    assert len(db._read_conns) == 1
    with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
        holder["conn"].execute("SELECT 1")


@pytest.mark.requires_wal
def test_dead_readers_are_reaped_before_next_connect(db, monkeypatch):
    """A full fd budget must be relieved before attempting another open."""
    def open_once():
        with db._read_ctx() as conn:
            return conn

    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(open_once).result(timeout=10)
    assert len(db._read_conns) == 1

    real_connect = hermes_state._connect_tracked_db

    def fd_limited_connect(*args, **kwargs):
        if db._read_conns:
            raise sqlite3.OperationalError("too many open files")
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(hermes_state, "_connect_tracked_db", fd_limited_connect)
    with ThreadPoolExecutor(max_workers=1) as pool:
        assert pool.submit(open_once).result(timeout=10) is not None
    with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
        first.execute("SELECT 1")


@pytest.mark.requires_wal
def test_close_waits_for_in_flight_read_lease(db):
    entered = threading.Event()
    release = threading.Event()
    holder = {}

    def reader():
        with db._read_ctx() as conn:
            holder["conn"] = conn
            entered.set()
            assert release.wait(timeout=10)
            assert conn.execute("SELECT 1").fetchone()[0] == 1

    with ThreadPoolExecutor(max_workers=2) as pool:
        reader_future = pool.submit(reader)
        assert entered.wait(timeout=10)
        close_future = pool.submit(db.close)
        deadline = time.monotonic() + 5
        while not db._read_conns_closed and time.monotonic() < deadline:
            time.sleep(0.001)
        assert db._read_conns_closed
        assert not close_future.done()
        release.set()
        reader_future.result(timeout=10)
        close_future.result(timeout=10)

    with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
        holder["conn"].execute("SELECT 1")


@pytest.mark.requires_wal
def test_close_waits_for_reader_open_already_in_progress(db, monkeypatch):
    real_connect = hermes_state._connect_tracked_db
    open_started = threading.Event()
    finish_open = threading.Event()
    close_started = threading.Event()
    holder = {}

    def blocked_connect(*args, **kwargs):
        conn = real_connect(*args, **kwargs)
        holder["conn"] = conn
        open_started.set()
        assert finish_open.wait(timeout=10)
        return conn

    monkeypatch.setattr(hermes_state, "_connect_tracked_db", blocked_connect)

    def open_reader():
        with db._read_ctx() as conn:
            assert conn.execute("SELECT 1").fetchone()[0] == 1
            return conn

    def close_db():
        close_started.set()
        db.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        open_future = pool.submit(open_reader)
        assert open_started.wait(timeout=10)
        close_future = pool.submit(close_db)
        assert close_started.wait(timeout=10)
        assert not close_future.done()
        finish_open.set()
        assert open_future.result(timeout=10) is holder["conn"]
        close_future.result(timeout=10)

    with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
        holder["conn"].execute("SELECT 1")


def test_close_from_read_context_fails_fast_without_poisoning_db(db):
    with db._read_ctx():
        with pytest.raises(RuntimeError, match="inside a read context"):
            db.close()
    assert db.get_session("s1")["id"] == "s1"


@pytest.mark.requires_wal
def test_session_resume_reads_do_not_take_writer_lock(db):
    """session.resume's three read paths must not convoy behind writer flushes.

    get_messages_as_conversation / get_resume_conversations /
    get_ancestor_display_prefix are the hottest reads in the file — every
    resume across the gateway, CLI, and ACP adapter goes through one of
    them — so they must use the same per-thread read-only connection as
    get_messages, not the legacy self._lock path.
    """
    db.create_session(session_id="parent1", source="cli", model="m")
    db.append_message("parent1", role="user", content="parent turn")
    db.append_message("parent1", role="assistant", content="parent reply")
    db.create_session(session_id="child1", source="cli", model="m", parent_session_id="parent1")
    db.append_message("child1", role="user", content="child turn")
    db.append_message("child1", role="assistant", content="child reply")

    acquired = db._lock.acquire()
    try:
        done = {}

        def reader():
            done["conversation"] = db.get_messages_as_conversation("s1")
            done["resume"] = db.get_resume_conversations("child1")
            done["ancestor_prefix"] = db.get_ancestor_display_prefix("child1")

        t = threading.Thread(target=reader)
        t.start(); t.join(timeout=5.0)
        assert not t.is_alive(), "session resume reads blocked on writer lock"
        assert len(done["conversation"]) == 2
        model_history, display_history = done["resume"]
        assert len(model_history) == 2
        assert len(display_history) == 4
        assert len(done["ancestor_prefix"]) == 2
    finally:
        db._lock.release()
