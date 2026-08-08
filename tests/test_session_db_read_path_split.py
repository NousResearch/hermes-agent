"""Tests for the SessionDB read-path split (bounded pooled readers).

The gateway shares ONE SessionDB across every agent, so recall/browse reads
used to queue behind writer flushes on self._lock — a measured production
convoy (a 0.2s FTS query stretched to 112s while 6-8 concurrent turns
flushed tool results). These tests pin the new contract: reads run on a
leased read-only connection under WAL, never touch self._lock when pool capacity
is available, and fall back to the legacy locked path when WAL or a reader is
unavailable.
"""

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from hermes_state import SessionDB


@pytest.fixture()
def db(tmp_path):
    d = SessionDB(db_path=tmp_path / "state.db")
    d.create_session(session_id="s1", source="cli", model="m")
    d.append_message("s1", role="user", content="hello graphiti world")
    d.append_message("s1", role="assistant", content="the neo4j daemon is healthy")
    yield d
    d.close()


def test_read_pool_reuses_connections_across_short_lived_threads(db):
    """Completed executor threads must not leave one reader each behind."""
    db._wal_active = True
    seen = []

    def read_once():
        assert db.get_session("s1")["id"] == "s1"
        with db._read_pool_lock:
            seen.append(len(db._read_pool_all))

    for _ in range(db._READ_POOL_MAX * 4):
        t = threading.Thread(target=read_once)
        t.start()
        t.join()

    assert max(seen) <= db._READ_POOL_MAX
    assert len(db._read_pool_all) <= db._READ_POOL_MAX
    assert len(db._read_pool_idle) == len(db._read_pool_all)


def test_read_pool_leases_distinct_connections_concurrently(db):
    db._wal_active = True
    barrier = threading.Barrier(2)
    conns = []

    def lease():
        with db._read_ctx() as conn:
            conns.append(conn)
            barrier.wait(timeout=5)

    t1 = threading.Thread(target=lease)
    t2 = threading.Thread(target=lease)
    t1.start(); t2.start(); t1.join(); t2.join()

    assert len(conns) == 2
    assert conns[0] is not conns[1]
    assert len(db._read_pool_all) == 2
    assert len(db._read_pool_idle) == 2


def test_read_pool_saturation_falls_back_without_opening_extra_connection(db):
    db._wal_active = True
    db._READ_POOL_MAX = 1
    entered = threading.Event()
    release = threading.Event()

    def hold_reader():
        with db._read_ctx():
            entered.set()
            release.wait(timeout=5)

    holder = threading.Thread(target=hold_reader)
    holder.start()
    assert entered.wait(timeout=5)

    assert db._acquire_read_conn() is None
    assert len(db._read_pool_all) == 1

    release.set()
    holder.join(timeout=5)
    assert not holder.is_alive()
    assert len(db._read_pool_all) == 1
    assert len(db._read_pool_idle) == 1


def test_read_pool_discards_broken_connection(db):
    db._wal_active = True
    conn = db._acquire_read_conn()
    assert conn is not None
    db._release_read_conn(conn, broken=True)

    assert conn not in db._read_pool_all
    assert conn not in db._read_pool_idle
    replacement = db._acquire_read_conn()
    assert replacement is not None
    assert replacement is not conn
    db._release_read_conn(replacement)


def test_close_closes_active_reader_when_returned(db):
    db._wal_active = True
    conn = db._acquire_read_conn()
    assert conn is not None

    db.close()
    assert db._read_pool_closed
    assert conn in db._read_pool_all

    db._release_read_conn(conn)
    assert conn not in db._read_pool_all
    assert conn not in db._read_pool_idle
    with pytest.raises(Exception):
        conn.execute("SELECT 1")


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
    assert db._acquire_read_conn() is None
    # And queries still work via the legacy path.
    assert db.get_session("s1")["id"] == "s1"


@pytest.mark.requires_wal
def test_read_conn_open_failure_marks_thread(db, monkeypatch, tmp_path):
    """A failed pooled-reader open falls back without retaining a connection."""
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
        assert calls["n"] == 2
        assert not fresh._read_pool_all
    finally:
        fresh.close()


def test_non_sql_exception_returns_reader_to_pool(db):
    db._wal_active = True
    with pytest.raises(ValueError):
        with db._read_ctx():
            raise ValueError("caller failure")

    assert len(db._read_pool_all) == 1
    assert len(db._read_pool_idle) == 1


def test_non_sql_exceptions_do_not_exhaust_pool(db):
    db._wal_active = True

    def fail_once(_):
        with pytest.raises(ValueError):
            with db._read_ctx():
                raise ValueError("caller failure")

    with ThreadPoolExecutor(max_workers=db._READ_POOL_MAX) as executor:
        list(executor.map(fail_once, range(db._READ_POOL_MAX * 4)))

    assert len(db._read_pool_all) <= db._READ_POOL_MAX
    assert len(db._read_pool_idle) == len(db._read_pool_all)


def test_reader_initialization_failure_closes_unpublished_connection(db, monkeypatch):
    db._wal_active = True
    import sqlite3
    real_pragmas = __import__("hermes_state").apply_database_pragmas

    def failing_pragmas(conn, *, db_label):
        real_pragmas(conn, db_label=db_label)
        raise sqlite3.DatabaseError("simulated post-connect initialization failure")

    monkeypatch.setattr("hermes_state.apply_database_pragmas", failing_pragmas)
    assert db.get_session("s1")["id"] == "s1"
    assert not db._read_pool_all
    assert not db._read_pool_idle


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
