"""Tests for the SessionDB read-path split (per-thread read-only connections).

The gateway shares ONE SessionDB across every agent, so recall/browse reads
used to queue behind writer flushes on self._lock — a measured production
convoy (a 0.2s FTS query stretched to 112s while 6-8 concurrent turns
flushed tool results). These tests pin the new contract: reads run on a
per-thread read-only connection under WAL, never touch self._lock, and fall
back to the legacy locked path when WAL or the read connection is missing.
"""

import threading

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


@pytest.mark.requires_wal
def test_read_conn_is_per_thread(db):
    conns = {}

    def grab(key):
        conns[key] = db._get_read_conn()

    t1 = threading.Thread(target=grab, args=(1,))
    t2 = threading.Thread(target=grab, args=(2,))
    t1.start(); t2.start(); t1.join(); t2.join()
    assert conns[1] is not None and conns[2] is not None
    assert conns[1] is not conns[2]


def test_read_conn_reused_within_thread(db):
    assert db._get_read_conn() is db._get_read_conn()


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
    assert db._get_read_conn() is None
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


# ── Bounded read-conn cache (fd leak guard) ──────────────────────────────────


@pytest.mark.requires_wal
def test_read_conn_cache_bound_evicts_idle_first_wave(db):
    """The cache bound must close stale conns even when their owners never
    read again — the EVICTOR closes them, not the idle owner threads.

    Regression for the reviewer finding that generation eviction must
    establish a real bound on live descriptors: wave-1 threads stay idle
    (no owner-side close), so only the evictor's close can release their
    fds, and _read_conns must not grow past the bound.
    """
    db._MAX_READ_CONNS = 4
    wave1 = {}

    def grab(key):
        wave1[key] = db._get_read_conn()

    ts = [threading.Thread(target=grab, args=(i,)) for i in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    wave1_conns = [wave1[i] for i in range(4)]
    assert all(c is not None for c in wave1_conns)
    assert len(db._read_conns) == 4
    assert db._read_gen == 0

    # Second wave of NEW threads forces eviction while wave 1 stays idle.
    wave2 = {}

    def grab2(key):
        wave2[key] = db._get_read_conn()

    ts2 = [threading.Thread(target=grab2, args=(10 + i,)) for i in range(2)]
    for t in ts2:
        t.start()
    for t in ts2:
        t.join()

    # Bound held; generation bumped; wave-1 conns closed BY THE EVICTOR
    # (their owners never re-read, so the owner-side discard never ran).
    assert db._read_gen == 1
    assert len(db._read_conns) <= 4
    for c in wave1_conns:
        assert getattr(c, "_hermes_read_closed", False), (
            "idle wave-1 conn not closed by eviction"
        )
    for c in wave2.values():
        assert c is not None
        assert not getattr(c, "_hermes_read_closed", True)

    # A wave-1 thread reading again lazily reopens a FRESH connection and
    # reads still work.
    reopened = {}

    def reread():
        reopened["conn"] = db._get_read_conn()
        reopened["session"] = db.get_session("s1")

    t = threading.Thread(target=reread)
    t.start(); t.join(timeout=5.0)
    assert reopened["conn"] is not None
    assert reopened["conn"] is not wave1_conns[0]
    assert not getattr(reopened["conn"], "_hermes_read_closed", True)
    assert reopened["session"]["id"] == "s1"


@pytest.mark.requires_wal
def test_eviction_waits_for_inflight_read(db):
    """The evictor must not close a connection another thread is reading on.

    Every query holds the per-conn RLock (see _read_ctx); holding that lock
    here simulates an in-flight read and the evictor must block on it, not
    close mid-read.
    """
    db._MAX_READ_CONNS = 1
    owner_conn = db._get_read_conn()
    assert owner_conn is not None

    in_flight = threading.Event()
    release = threading.Event()

    def busy_reader():
        owner_conn._hermes_read_lock.acquire()
        try:
            in_flight.set()
            release.wait(5.0)
        finally:
            owner_conn._hermes_read_lock.release()

    reader = threading.Thread(target=busy_reader)
    reader.start()
    assert in_flight.wait(5.0)

    evicted = {"done": False}

    def evictor():
        db._get_read_conn()  # second registration -> overflow -> eviction
        evicted["done"] = True

    t = threading.Thread(target=evictor)
    t.start()
    t.join(timeout=0.3)
    assert not evicted["done"], "evictor closed a conn with a read in flight"

    release.set()
    t.join(timeout=5.0)
    reader.join(timeout=5.0)
    assert evicted["done"]
    # The owner's conn was closed by the evictor only after the read ended.
    assert getattr(owner_conn, "_hermes_read_closed", False)


@pytest.mark.requires_wal
def test_failed_close_stays_registered_and_tracked(db, monkeypatch):
    """A failed close must not drop a live fd from either registry.

    Reviewer finding: unregistering before a successful close lets a live fd
    become unreachable (close() can no longer reach it) and untracked (the
    byte-probe guard then runs against an open database). The conn must stay
    in _read_conns and in the tracking registry until close() succeeds.
    """
    import sqlite3

    from hermes_cli.sqlite_safe_read import has_live_connection

    db._MAX_READ_CONNS = 1
    conn = db._get_read_conn()
    assert conn is not None
    assert has_live_connection(db.db_path)

    real_close = conn.close
    state = {"n": 0}

    def flaky_close():
        state["n"] += 1
        if state["n"] == 1:
            raise sqlite3.ProgrammingError("simulated close failure")
        real_close()

    monkeypatch.setattr(conn, "close", flaky_close)

    def other_open():
        return db._get_read_conn()

    t = threading.Thread(target=other_open)
    t.start(); t.join()

    # Eviction attempted to close *conn*; the simulated failure kept it
    # registered and tracked.
    assert state["n"] == 1
    assert conn in db._read_conns, "failed close must stay reachable by close()"
    assert has_live_connection(db.db_path), "failed close must not untrack"

    # The next successful close (shutdown drain) removes and untracks it.
    monkeypatch.undo()
    db.close()
    assert db._read_conns == set()
    assert not has_live_connection(db.db_path)


@pytest.mark.requires_wal
def test_close_drains_read_conns_and_untracks(db):
    """close() must reach every per-thread read conn (including idle owners),
    release its fds, and clear the tracking registry."""
    from hermes_cli.sqlite_safe_read import has_live_connection

    conns = {}

    def grab(key):
        conns[key] = db._get_read_conn()

    ts = [threading.Thread(target=grab, args=(i,)) for i in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert len(db._read_conns) == 4
    assert has_live_connection(db.db_path)

    db.close()
    assert db._read_conns == set()
    assert db._read_conns_closed is True
    for c in conns.values():
        assert getattr(c, "_hermes_read_closed", False)
    assert not has_live_connection(db.db_path)

    # A read after close must not reopen a read connection.
    assert db._get_read_conn() is None
