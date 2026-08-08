"""Tests for the SessionDB read-path split (per-thread read-only connections).

The gateway shares ONE SessionDB across every agent, so recall/browse reads
used to queue behind writer flushes on self._lock — a measured production
convoy (a 0.2s FTS query stretched to 112s while 6-8 concurrent turns
flushed tool results). These tests pin the new contract: reads run on a
per-thread read-only connection under WAL, never touch self._lock, and fall
back to the legacy locked path when WAL or the read connection is missing.
"""

import os
import sqlite3
import threading

import pytest

import hermes_state
from hermes_state import SessionDB


def _count_open_fds():
    for fd_dir in ("/proc/self/fd", "/dev/fd"):
        try:
            with os.scandir(fd_dir) as entries:
                return sum(1 for _ in entries)
        except OSError:
            continue
    pytest.skip("process fd directory is unavailable")


def _start_checked_thread(target):
    failures = []

    def checked_target():
        try:
            target()
        except BaseException as exc:
            failures.append(exc)

    thread = threading.Thread(target=checked_target)
    thread.start()
    return thread, failures


def _join_checked_thread(thread, failures, *, message="worker did not finish"):
    thread.join(timeout=5.0)
    assert not thread.is_alive(), message
    if failures:
        raise failures[0]


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


def test_close_closes_read_connection_owned_by_another_thread(db):
    """close() must close the fd, not only forget the tracked connection."""
    db._wal_active = True
    opened = threading.Event()
    closed = threading.Event()
    outcome = {}

    def reader():
        conn = db._get_read_conn()
        assert conn is not None
        opened.set()
        closed.wait(timeout=5.0)
        outcome["cached_after_close"] = db._get_read_conn()
        try:
            conn.execute("SELECT 1").fetchone()
        except Exception as exc:  # inspected by the parent thread
            outcome["error"] = exc
        else:
            outcome["usable_after_close"] = True

    thread, failures = _start_checked_thread(reader)
    assert opened.wait(timeout=5.0), "reader did not open its connection"

    db.close()
    closed.set()
    _join_checked_thread(thread, failures)

    error = outcome.get("error")
    assert isinstance(error, sqlite3.ProgrammingError)
    assert "closed" in str(error).lower()
    assert outcome["cached_after_close"] is None
    assert "usable_after_close" not in outcome
    db.close()


def test_close_waits_for_active_read_operation(db, monkeypatch):
    """close() must drain an active reader before closing its connection."""
    db._wal_active = True
    operation_started = threading.Event()
    allow_operation_to_finish = threading.Event()
    close_returned = threading.Event()
    outcome = {}
    real_connect = hermes_state._connect_tracked_db

    class ControllableConnection:
        def __init__(self, conn):
            object.__setattr__(self, "_conn", conn)
            object.__setattr__(self, "close_calls", 0)

        def __getattr__(self, name):
            return getattr(self._conn, name)

        def __setattr__(self, name, value):
            if name in {"_conn", "close_calls"}:
                object.__setattr__(self, name, value)
            else:
                setattr(self._conn, name, value)

        def execute(self, sql, *args, **kwargs):
            if sql == "SELECT 1":
                operation_started.set()
                assert allow_operation_to_finish.wait(timeout=5.0)
            return self._conn.execute(sql, *args, **kwargs)

        def close(self):
            self.close_calls += 1
            self._conn.close()

    def controllable_connect(*args, **kwargs):
        conn = ControllableConnection(real_connect(*args, **kwargs))
        outcome["conn"] = conn
        return conn

    monkeypatch.setattr(hermes_state, "_connect_tracked_db", controllable_connect)

    def reader():
        with db._read_ctx() as conn:
            outcome["value"] = conn.execute("SELECT 1").fetchone()[0]

    reader_thread, reader_failures = _start_checked_thread(reader)
    assert operation_started.wait(timeout=5.0), "reader did not start its query"

    def closer():
        db.close()
        close_returned.set()

    close_thread, close_failures = _start_checked_thread(closer)
    try:
        with db._read_conns_lock:
            assert db._read_conns_closed
            assert outcome["conn"].close_calls == 0
            assert not close_returned.is_set()
    finally:
        allow_operation_to_finish.set()

    _join_checked_thread(reader_thread, reader_failures)
    _join_checked_thread(close_thread, close_failures, message="close() deadlocked")

    assert outcome["value"] == 1
    assert outcome["conn"].close_calls == 1
    assert close_returned.is_set()


def test_read_ctx_exception_releases_active_operation(db):
    """An exception inside a read context must not strand close() waiting."""
    db._wal_active = True

    with pytest.raises(RuntimeError, match="simulated read failure"):
        with db._read_ctx():
            raise RuntimeError("simulated read failure")

    with db._read_conns_lock:
        assert not db._read_conns_active


def test_reaper_skips_dead_owner_connection_with_active_operation(db):
    """Dead-owner reaping must never close a connection marked active."""

    class RecordingConnection:
        def __init__(self):
            self.close_calls = 0

        def close(self):
            self.close_calls += 1

    owner = threading.Thread(target=lambda: None)
    owner.start()
    owner.join(timeout=5.0)
    assert not owner.is_alive()
    conn = RecordingConnection()

    with db._read_conns_lock:
        db._read_conns[conn] = owner
        db._read_conns_active[conn] = 1
        db._reap_dead_read_conns_locked()
        assert conn.close_calls == 0
        assert conn in db._read_conns

        db._read_conns_active.pop(conn)
        db._reap_dead_read_conns_locked()
        assert conn.close_calls == 1
        assert conn not in db._read_conns


def test_read_conn_open_racing_close_is_closed(db, monkeypatch):
    """A reader opened during close must not escape the final drain."""
    db._wal_active = True
    opened = threading.Event()
    resume_open = threading.Event()
    outcome = {}
    real_connect = hermes_state._connect_tracked_db

    def blocking_connect(*args, **kwargs):
        conn = real_connect(*args, **kwargs)
        outcome["conn"] = conn
        opened.set()
        resume_open.wait(timeout=5.0)
        return conn

    monkeypatch.setattr(hermes_state, "_connect_tracked_db", blocking_connect)

    def reader():
        outcome["result"] = db._get_read_conn()

    thread, reader_failures = _start_checked_thread(reader)
    assert opened.wait(timeout=5.0), "reader did not reach the open race"

    close_returned = threading.Event()

    def closer():
        db.close()
        close_returned.set()

    close_thread, close_failures = _start_checked_thread(closer)
    with db._read_conns_lock:
        assert db._read_conns_closed
        assert not close_returned.is_set()
    resume_open.set()
    _join_checked_thread(thread, reader_failures)
    _join_checked_thread(close_thread, close_failures, message="close() deadlocked")

    assert close_returned.is_set()
    assert outcome["result"] is None
    assert not db._read_conns
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        outcome["conn"].execute("SELECT 1")


def test_unexpected_read_conn_setup_error_releases_opening_accounting(
    db, monkeypatch
):
    """A non-SQLite setup error must not strand close() on the Condition."""
    db._wal_active = True
    opened = {}
    real_connect = hermes_state._connect_tracked_db

    def recording_connect(*args, **kwargs):
        conn = real_connect(*args, **kwargs)
        opened["conn"] = conn
        return conn

    def unexpected_pragma_error(*_args, **_kwargs):
        raise RuntimeError("unexpected pragma setup failure")

    monkeypatch.setattr(hermes_state, "_connect_tracked_db", recording_connect)
    monkeypatch.setattr(
        hermes_state,
        "apply_database_pragmas",
        unexpected_pragma_error,
    )

    try:
        with pytest.raises(RuntimeError, match="unexpected pragma setup failure"):
            db._get_read_conn()

        with db._read_conns_lock:
            assert db._read_conns_opening == 0
        assert not db._read_conns
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            opened["conn"].execute("SELECT 1")
    finally:
        # Keep this regression test cleanup-safe if the accounting bug is
        # reintroduced, so the fixture's close() reports a failure instead of
        # hanging the entire test process forever.
        with db._read_conns_lock:
            if db._read_conns_opening:
                db._read_conns_opening = 0
                db._read_conns_lock.notify_all()
        conn = opened.get("conn")
        if conn is not None:
            try:
                conn.close()
            except sqlite3.Error:
                pass


def test_short_lived_reader_connections_are_reaped(db):
    """Completed reader threads must not accumulate on a long-lived DB."""
    db._wal_active = True
    stable_conn = db._get_read_conn()
    assert stable_conn is not None

    for _ in range(30):
        thread, failures = _start_checked_thread(lambda: db.get_session("s1"))
        _join_checked_thread(thread, failures)

    assert len(db._read_conns) <= 2
    assert stable_conn in db._read_conns
    assert db._read_conns[stable_conn] is threading.current_thread()
    assert db._get_read_conn() is stable_conn
    assert stable_conn.execute("SELECT 1").fetchone()[0] == 1


def test_short_lived_readers_do_not_leak_process_fds(db):
    """The retained SQLite descriptors stay bounded across completed threads."""
    db._wal_active = True
    before = _count_open_fds()

    for _ in range(30):
        thread, failures = _start_checked_thread(lambda: db.get_session("s1"))
        _join_checked_thread(thread, failures)

    after = _count_open_fds()
    assert after <= before + 6, f"process fds grew from {before} to {after}"


def test_close_logs_read_conn_error_and_retries_idempotently(db, caplog):
    """A failed read close stays tracked without making close() newly raise."""

    class FlakyConnection:
        def __init__(self):
            self.close_calls = 0

        def close(self):
            self.close_calls += 1
            if self.close_calls == 1:
                raise RuntimeError("simulated read close failure")

    conn = FlakyConnection()
    with db._read_conns_lock:
        db._read_conns[conn] = threading.current_thread()

    with caplog.at_level("WARNING"):
        db.close()

    assert conn.close_calls == 1
    assert conn in db._read_conns
    assert "failed to close a read-only state.db connection" in caplog.text

    db.close()
    assert conn.close_calls == 2
    assert conn not in db._read_conns
    db.close()
