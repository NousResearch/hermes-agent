"""The SessionDB read path must not leak one connection per (SessionDB x thread).

``_get_read_conn`` used to cache a read-only connection in ``threading.local()``
and pin it in a strong set (``_read_conns``) that was only ever drained by
``close()``. Starlette dispatches sync routes on anyio worker threads, so a
SessionDB that is never closed -- the dashboard's module-global ``_db`` and
the per-session ``session_db`` handles -- gained a connection, and a file
descriptor, for every worker thread that ever served a read. In production
that walked into the 256 soft ``RLIMIT_NOFILE`` a service manager hands the
process, after which every request failed with ``OSError`` EMFILE while the
process stayed alive, so the supervisor's restart-on-exit never fired.

Worse, those connections were opened WITHOUT ``check_same_thread=False`` (both
writer opens pass it), so ``close()`` on them raised ``ProgrammingError`` from
a different thread and the bare ``except Exception: pass`` hid it -- leaving
``hermes_cli.sqlite_safe_read``'s registry permanently over-counted as well.

The contract pinned here: reads borrow from a BOUNDED pool, connections are
returned and reused, surplus connections are closed rather than dropped, and
``close()`` actually closes them from whatever thread it runs on.

These assert on the pool/registry counts, never on ``lsof``: SQLite's unix VFS
parks a closed descriptor on a per-inode reuse list while any connection still
holds POSIX locks on that inode, so raw descriptor counts lag the real
connection count and make such assertions flaky.
"""

import threading

import pytest

from hermes_state import SessionDB


def _live_count(path) -> int:
    """Live-connection count the tracking registry holds for *path*."""
    import hermes_cli.sqlite_safe_read as mod

    with mod._live_lock:
        return mod._live_connections.get(mod._key(path), 0)


@pytest.fixture()
def db(tmp_path):
    d = SessionDB(db_path=tmp_path / "state.db")
    d.create_session(session_id="s1", source="cli", model="m")
    d.append_message("s1", role="user", content="hello graphiti world")
    d.append_message("s1", role="assistant", content="the neo4j daemon is healthy")
    yield d
    d.close()


def _read(db):
    db.get_session("s1")
    db.search_messages("graphiti", limit=5)
    db.get_messages("s1")


@pytest.mark.requires_wal
def test_read_pool_is_bounded_across_many_threads(db):
    """150 short-lived reader threads must not pin 150 connections."""
    maxsize = db._read_pool.maxsize
    assert maxsize > 0, "read pool must be bounded"

    for _ in range(6):
        threads = [threading.Thread(target=_read, args=(db,)) for _ in range(25)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert db._read_pool.qsize() <= maxsize

    # The pre-fix code held 151 connections here (150 readers + main thread).
    assert db._read_pool.qsize() <= maxsize
    # +1 for the writer connection SessionDB always holds.
    assert _live_count(db.db_path) <= maxsize + 1


@pytest.mark.requires_wal
def test_read_conn_returned_to_pool_and_reused(db):
    """Sequential reads on one thread reuse a pooled connection, not a new one."""
    with db._read_ctx() as conn:
        first = conn
    assert db._read_pool.qsize() >= 1, "connection was not returned to the pool"
    with db._read_ctx() as conn:
        assert conn is first, "pooled connection was not reused"


@pytest.mark.requires_wal
def test_pooled_conn_is_usable_from_another_thread(db):
    """A pooled connection is handed between threads, so it must not be
    bound to its creating thread (check_same_thread=False)."""
    with db._read_ctx() as conn:
        borrowed = conn

    errors = []

    def use_it():
        try:
            borrowed.execute("SELECT 1").fetchone()
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    t = threading.Thread(target=use_it)
    t.start()
    t.join()
    assert not errors, f"pooled connection unusable off-thread: {errors}"


@pytest.mark.requires_wal
def test_close_drains_pool_from_a_foreign_thread(tmp_path):
    """close() must actually close pooled connections, including ones opened
    on threads that have since exited -- the swallowed ProgrammingError."""
    d = SessionDB(db_path=tmp_path / "state2.db")
    d.create_session(session_id="s1", source="cli", model="m")

    # Populate the pool from a worker thread, then let that thread die.
    t = threading.Thread(target=lambda: d.get_session("s1"))
    t.start()
    t.join()
    assert d._read_pool.qsize() >= 1

    d.close()
    assert d._read_pool.qsize() == 0
    # Registry back to zero proves the closes succeeded rather than raising
    # ProgrammingError into a bare except.
    assert _live_count(d.db_path) == 0


@pytest.mark.requires_wal
def test_reader_after_close_does_not_repopulate_pool(db):
    """A read racing close() must close its connection, not refill the pool."""
    db.close()
    assert db._read_pool.qsize() == 0
    # A read arriving after the drain must not open-and-requeue a connection
    # that nothing will ever close again.
    with db._read_ctx():
        pass
    assert db._read_pool.qsize() == 0


def test_reads_are_still_correct_under_concurrency(db):
    """Pooling must not corrupt results when threads share connections."""
    results = []
    errors = []

    def reader():
        try:
            results.append(db.get_session("s1")["id"])
            results.append(len(db.get_messages("s1")))
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=reader) for _ in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, f"concurrent reads failed: {errors}"
    assert results.count("s1") == 12
    assert results.count(2) == 12


@pytest.mark.requires_wal
def test_read_open_failure_backs_off_but_recovers(db):
    """A failed read-only open must not permanently demote the read path.

    The first version of this fix used a sticky instance-wide boolean
    (``_read_open_failed``). Its likeliest trigger is transient fd pressure --
    EMFILE, the very condition this pool exists to prevent -- and because the
    gateway shares ONE SessionDB across every agent, a single blip would have
    convoyed every subsequent reader behind the writer lock for the life of
    the process. The stamp must expire.
    """
    import time as _time

    from hermes_state import _READ_OPEN_RETRY_SECONDS

    baseline = db._get_read_conn()
    assert baseline is not None, "baseline read open should succeed"
    db._close_read_conn(baseline)

    db._read_open_failed_at = _time.monotonic()
    assert db._get_read_conn() is None, "should back off immediately after a failure"

    db._read_open_failed_at = _time.monotonic() - (_READ_OPEN_RETRY_SECONDS + 1)
    recovered = db._get_read_conn()
    assert recovered is not None, "read path must self-heal once the window expires"
    db._close_read_conn(recovered)


@pytest.mark.requires_wal
def test_checkout_seam_is_the_single_acquisition_point(db):
    """``_read_ctx`` must acquire via ``_checkout_read_conn`` and nothing else.

    If a future edit re-inlines the pool checkout into ``_read_ctx``, patching
    ``_get_read_conn`` silently exercises nothing whenever the pool is warm --
    which is exactly how the writer-lock fallback test below would rot into a
    no-op without failing.
    """
    calls = []
    original = db._checkout_read_conn

    def _spy():
        calls.append(1)
        return original()

    db._checkout_read_conn = _spy
    try:
        with db._read_ctx():
            pass
    finally:
        db._checkout_read_conn = original
    assert calls, "_read_ctx must route acquisition through _checkout_read_conn"


def test_fallback_to_locked_writer_when_read_conn_unavailable(db, monkeypatch):
    """With no read connection available, reads still work under self._lock.

    Patched at the acquisition SEAM rather than at ``_get_read_conn``: the
    pool is consulted first, so a patched ``_get_read_conn`` is never reached
    while the pool holds a connection and this test would pass while
    exercising nothing.
    """
    monkeypatch.setattr(db, "_checkout_read_conn", lambda: None)
    assert db.get_session("s1")["id"] == "s1"
    assert db.search_messages("graphiti", limit=5)
