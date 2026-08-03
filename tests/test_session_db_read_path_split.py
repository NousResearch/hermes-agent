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


# ── #75269: finished-thread read connections must be reaped at runtime ──
#
# These tests force ``_wal_active = True`` so the per-thread read path runs
# regardless of whether the linked SQLite build actually enables WAL (some
# runtimes fall back to journal_mode=DELETE). The reaping contract is
# platform-independent: it only depends on the connection lifecycle, not on
# WAL semantics, so forcing the flag exercises the real production path.


def _force_read_path(d):
    """Activate the per-thread read-only split on runtimes where WAL is off."""
    d._wal_active = True


def test_finished_read_threads_do_not_accumulate_conns(tmp_path):
    """A long-lived SessionDB must not retain a read conn per historical worker.

    Reproduces #75269: sequential worker threads each open one read connection
    and exit. Without runtime reaping the strong ``_read_conns`` set grows
    without bound; with reaping it stays bounded by live reader concurrency.
    """
    import gc

    d = SessionDB(db_path=tmp_path / "state.db")
    try:
        d.create_session(session_id="s1", source="cli", model="m")
        _force_read_path(d)

        def read_once():
            assert d._get_read_conn() is not None

        for _ in range(40):
            w = threading.Thread(target=read_once)
            w.start()
            w.join(timeout=10)
            assert not w.is_alive()

        gc.collect()
        retained = len(d._read_conns)
        assert retained < 12, (
            f"finished worker threads retained {retained} read connections; "
            "descriptor usage must be bounded by live reader concurrency (#75269)"
        )
    finally:
        d.close()


def test_reaped_connection_is_actually_closed(tmp_path):
    """A connection whose owning thread exited must be closed on reap."""
    import sqlite3 as _sqlite3

    d = SessionDB(db_path=tmp_path / "state.db")
    try:
        d.create_session(session_id="s1", source="cli", model="m")
        _force_read_path(d)

        holder = {}

        def worker_open():
            holder["conn"] = d._get_read_conn()

        w = threading.Thread(target=worker_open)
        w.start(); w.join(timeout=10)
        assert not w.is_alive()
        victim = holder["conn"]
        assert victim is not None

        # Trigger a reap by registering another connection from a fresh thread.
        def worker_reap():
            assert d._get_read_conn() is not None

        r = threading.Thread(target=worker_reap)
        r.start(); r.join(timeout=10)
        assert not r.is_alive()

        assert victim not in d._read_conns, "dead-thread connection was not reaped"
        with pytest.raises(_sqlite3.ProgrammingError):
            victim.execute("SELECT 1")
    finally:
        d.close()


def test_live_thread_connection_not_reaped(tmp_path):
    """A connection whose owner thread is still alive must survive a reap."""
    import threading as _t

    d = SessionDB(db_path=tmp_path / "state.db")
    try:
        d.create_session(session_id="s1", source="cli", model="m")
        _force_read_path(d)

        ready = _t.Event()
        release = _t.Event()
        live_conn = {}

        def hold():
            live_conn["c"] = d._get_read_conn()
            ready.set()
            release.wait(timeout=10)

        keeper = _t.Thread(target=hold)
        keeper.start()
        assert ready.wait(timeout=10)
        conn = live_conn["c"]
        assert conn is not None

        # Register from other threads, which triggers reaping sweeps.
        for _ in range(5):
            w = _t.Thread(target=lambda: d._get_read_conn())
            w.start(); w.join(timeout=10)

        assert keeper.is_alive(), "live reader thread died unexpectedly"
        assert conn in d._read_conns, "live-thread connection was wrongly reaped"

        release.set()
        keeper.join(timeout=10)
    finally:
        d.close()


def test_reaping_is_thread_safe_under_concurrency(tmp_path):
    """Concurrent readers + reaping must not raise."""
    import time

    d = SessionDB(db_path=tmp_path / "state.db")
    try:
        d.create_session(session_id="s1", source="cli", model="m")
        _force_read_path(d)
        d.append_message("s1", role="user", content="concurrency smoke")

        errors = []

        def reader():
            try:
                for _ in range(20):
                    d.get_session("s1")
                    d._get_read_conn()  # may register + reap
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=reader) for _ in range(8)]
        for i, t in enumerate(threads):
            t.start()
            if i % 2:  # stagger so threads exit at different times
                time.sleep(0.001)
        for t in threads:
            t.join(timeout=20)
        assert not errors, f"concurrent read/reap raised: {errors!r}"
        assert not any(t.is_alive() for t in threads)
    finally:
        d.close()
