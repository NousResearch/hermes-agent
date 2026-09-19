"""Shared-writer fallback coverage for SessionDB pure-read helpers.

``_read_ctx()`` normally serves WAL reads from a bounded pool.  When WAL is
unavailable, however, it deliberately falls back to the shared writer under
``self._lock``.  These helpers used to bypass that fallback and could call the
same sqlite3.Connection concurrently.
"""

from __future__ import annotations

from contextlib import contextmanager
import sqlite3
import threading
from typing import Any, Callable

import pytest

from agent.session_activity import ActivityProvenance
from hermes_state import SessionDB


_WORKERS = 8
_BARRIER_TIMEOUT_S = 5.0
_JOIN_TIMEOUT_S = 10.0


class _RendezvousWriterLock:
    """Make all fallback callers arrive before serializing the shared writer."""

    def __init__(self, workers: int):
        self._arrivals = threading.Barrier(workers, timeout=_BARRIER_TIMEOUT_S)
        self._lock = threading.Lock()
        self.entered = threading.Event()

    def __enter__(self):
        self.entered.set()
        self._arrivals.wait()
        self._lock.acquire()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self._lock.release()
        return False


class _SerializedCursor:
    """Serialize use of a native cursor returned by the shared writer."""

    def __init__(self, cursor: Any, delegate_lock: threading.Lock):
        self._cursor = cursor
        self._delegate_lock = delegate_lock

    def _call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        with self._delegate_lock:
            result = getattr(self._cursor, method)(*args, **kwargs)
        # sqlite3 cursor execute methods return the native cursor itself; do
        # not let that escape this serializing proxy.
        return self if result is self._cursor else result

    def execute(self, *args: Any, **kwargs: Any) -> _SerializedCursor:
        return self._call("execute", *args, **kwargs)

    def executemany(self, *args: Any, **kwargs: Any) -> _SerializedCursor:
        return self._call("executemany", *args, **kwargs)

    def executescript(self, *args: Any, **kwargs: Any) -> _SerializedCursor:
        return self._call("executescript", *args, **kwargs)

    def fetchone(self) -> Any:
        return self._call("fetchone")

    def fetchmany(self, *args: Any, **kwargs: Any) -> Any:
        return self._call("fetchmany", *args, **kwargs)

    def fetchall(self) -> Any:
        return self._call("fetchall")

    def close(self) -> Any:
        return self._call("close")

    def setinputsizes(self, *args: Any, **kwargs: Any) -> Any:
        return self._call("setinputsizes", *args, **kwargs)

    def setoutputsize(self, *args: Any, **kwargs: Any) -> Any:
        return self._call("setoutputsize", *args, **kwargs)

    def __iter__(self) -> _SerializedCursor:
        return self

    def __next__(self) -> Any:
        return self._call("__next__")


class _SharedWriterExecuteProbe:
    """Record overlapping execute entries without concurrent SQLite access."""

    def __init__(self, conn: Any, writer_lock: _RendezvousWriterLock, workers: int):
        self._conn = conn
        self._writer_lock = writer_lock
        self._direct_callers = threading.Barrier(workers, timeout=_BARRIER_TIMEOUT_S)
        self._activity_lock = threading.Lock()
        self._delegate_lock = threading.Lock()
        self._active = 0
        self.max_concurrent_execute = 0

    def execute(self, *args: Any, **kwargs: Any):
        with self._activity_lock:
            self._active += 1
            self.max_concurrent_execute = max(self.max_concurrent_execute, self._active)
        try:
            # A pre-fix direct helper never enters the fallback lock, so all
            # worker calls rendezvous here and make the unsafe overlap
            # deterministic.  A fixed helper has already entered that lock
            # before reaching execute(), so it must not wait for blocked peers.
            if not self._writer_lock.entered.is_set():
                self._direct_callers.wait()
        finally:
            with self._activity_lock:
                self._active -= 1

        # Keep the temp database itself safe even while demonstrating that the
        # old method body reached its shared writer execute concurrently. The
        # returned cursor is also proxied because fetches step SQLite too.
        with self._delegate_lock:
            cursor = self._conn.execute(*args, **kwargs)
        return _SerializedCursor(cursor, self._delegate_lock)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._conn, name)


@pytest.fixture
def db(tmp_path):
    session_db = SessionDB(db_path=tmp_path / "state.db")
    yield session_db
    session_db.close()


def _assert_shared_writer_fallback_is_serialized(
    db: SessionDB, call: Callable[[], Any], expected: Any
) -> None:
    """Run *call* concurrently while forcing ``_read_ctx`` writer fallback."""
    original_conn = db._conn
    original_lock = db._lock
    original_wal_active = db._wal_active
    writer_lock = _RendezvousWriterLock(_WORKERS)
    probe = _SharedWriterExecuteProbe(original_conn, writer_lock, _WORKERS)
    start = threading.Barrier(_WORKERS, timeout=_BARRIER_TIMEOUT_S)
    results: list[Any] = []
    errors: list[BaseException] = []
    results_lock = threading.Lock()

    def worker() -> None:
        try:
            start.wait()
            result = call()
            with results_lock:
                results.append(result)
        except BaseException as exc:
            with results_lock:
                errors.append(exc)

    db._wal_active = False
    db._lock = writer_lock
    db._conn = probe
    threads = [threading.Thread(target=worker) for _ in range(_WORKERS)]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=_JOIN_TIMEOUT_S)

        assert not [thread for thread in threads if thread.is_alive()]
        assert errors == []
        assert results == [expected] * _WORKERS
        assert probe.max_concurrent_execute == 1, (
            "shared writer execute overlapped "
            f"{probe.max_concurrent_execute} callers; the helper must enter "
            "_read_ctx() before executing its read"
        )
        assert writer_lock.entered.is_set()
    finally:
        db._conn = original_conn
        db._lock = original_lock
        db._wal_active = original_wal_active


def _assert_checked_out_reader_avoids_shared_writer(
    db: SessionDB, call: Callable[[], Any], expected: Any
) -> None:
    """Run *call* concurrently and reject delegate reads on the writer."""
    original_conn = db._conn
    original_lock = db._lock
    original_read_ctx = db._read_ctx
    writer_lock = _RendezvousWriterLock(_WORKERS)
    probe = _SharedWriterExecuteProbe(original_conn, writer_lock, _WORKERS)
    start = threading.Barrier(_WORKERS, timeout=_BARRIER_TIMEOUT_S)
    results: list[Any] = []
    errors: list[BaseException] = []
    results_lock = threading.Lock()

    # The forced non-WAL fallback below intentionally aliases ``conn`` and
    # ``self._conn``. Use independent read-only checkouts here so this test
    # also proves that a helper does not re-dereference the shared writer.
    @contextmanager
    def checked_out_read_ctx():
        conn = sqlite3.connect(
            f"file:{db.db_path}?mode=ro",
            uri=True,
            check_same_thread=False,
            timeout=5.0,
            isolation_level=None,
        )
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def worker() -> None:
        try:
            start.wait()
            result = call()
            with results_lock:
                results.append(result)
        except BaseException as exc:
            with results_lock:
                errors.append(exc)

    db._lock = writer_lock
    db._conn = probe
    db._read_ctx = checked_out_read_ctx
    threads = [threading.Thread(target=worker) for _ in range(_WORKERS)]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=_JOIN_TIMEOUT_S)

        assert not [thread for thread in threads if thread.is_alive()]
        assert errors == []
        assert results == [expected] * _WORKERS
        assert probe.max_concurrent_execute == 0, (
            "checked-out reader bypassed by shared writer execute; "
            "delegate traversal must use the connection from _read_ctx()"
        )
        assert not writer_lock.entered.is_set()
    finally:
        db._read_ctx = original_read_ctx
        db._conn = original_conn
        db._lock = original_lock


def _helper_call(db: SessionDB, helper_name: str) -> Callable[[], Any]:
    session_id = "pooled-reader-fallback"
    if helper_name == "get_compression_lock_holder":
        return lambda: db.get_compression_lock_holder(session_id)
    if helper_name == "clear_session_activity_labels":
        db.create_session(session_id, source="test")

        def clear_labels(conn):
            conn.execute(
                "UPDATE sessions SET last_activity_description = ?, "
                "last_activity_provenance = ? WHERE id = ?",
                ("", ActivityProvenance.UNKNOWN.value, session_id),
            )

        db._execute_write(clear_labels)
        return lambda: db.clear_session_activity_labels(session_id)
    if helper_name == "get_handoff_state":
        return lambda: db.get_handoff_state(session_id)
    if helper_name == "list_pending_handoffs":
        return db.list_pending_handoffs
    if helper_name == "get_session_delete_targets":
        db.create_session(session_id, source="test")
        db.create_session(
            "pooled-reader-delegate",
            source="test",
            parent_session_id=session_id,
            model_config={"_delegate_from": session_id},
        )
        return lambda: db.get_session_delete_targets(session_id)
    raise AssertionError(f"unknown helper: {helper_name}")


@pytest.mark.parametrize(
    ("helper_name", "expected"),
    [
        ("get_compression_lock_holder", None),
        ("clear_session_activity_labels", None),
        ("get_handoff_state", None),
        ("list_pending_handoffs", []),
        (
            "get_session_delete_targets",
            ["pooled-reader-fallback", "pooled-reader-delegate"],
        ),
    ],
)
def test_pure_read_helpers_serialize_shared_writer_fallback(db, helper_name, expected):
    """Every direct pure read must use the non-WAL shared-writer fallback."""
    _assert_shared_writer_fallback_is_serialized(
        db, _helper_call(db, helper_name), expected
    )


def test_get_session_delete_targets_uses_checked_out_reader(db):
    """The delegate walk must stay on the connection `_read_ctx` checked out."""
    _assert_checked_out_reader_avoids_shared_writer(
        db,
        _helper_call(db, "get_session_delete_targets"),
        ["pooled-reader-fallback", "pooled-reader-delegate"],
    )
