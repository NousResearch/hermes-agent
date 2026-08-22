"""Regression: the gateway delivery ledger must close every SQLite connection.

Sibling of the cron execution-ledger leak (#69567 / PR #69594). The ledger used
``with _connect() as conn:`` where ``sqlite3.Connection.__exit__`` commits or
rolls back but never closes, leaking the db/-wal/-shm file descriptors on every
call until a long-running gateway exhausts ``RLIMIT_NOFILE``. ``record_obligation``
runs on every outbound final response, so this is the highest-frequency leaker of
the set. These tests fail if the deterministic ``close()`` is ever removed again.
"""

import sqlite3

import pytest

from gateway import delivery_ledger as dl


class _TrackingConnection:
    """Delegates to a real sqlite3.Connection while recording close() calls.

    sqlite3.Connection is a static C type: it has no per-instance __dict__ and
    its methods can't be monkeypatched, so open/close tracking is done via a
    delegating wrapper returned in place of the real connection.
    """

    def __init__(self, real, closed_ids):
        object.__setattr__(self, "_real", real)
        object.__setattr__(self, "_closed_ids", closed_ids)

    def close(self):
        self._closed_ids.append(id(self._real))
        self._real.close()

    def __enter__(self):
        self._real.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        return self._real.__exit__(exc_type, exc, tb)

    def __getattr__(self, name):
        return getattr(self._real, name)

    def __setattr__(self, name, value):
        setattr(self._real, name, value)


def _point_ledger(monkeypatch, tmp_path):
    monkeypatch.setattr(dl, "_db_path", lambda: tmp_path / "state.db")
    return dl


def _track_connections(monkeypatch):
    opened, closed = [], []
    real_connect = sqlite3.connect

    def tracking_connect(*args, **kwargs):
        conn = real_connect(*args, **kwargs)
        opened.append(id(conn))
        return _TrackingConnection(conn, closed)

    monkeypatch.setattr(dl.sqlite3, "connect", tracking_connect)
    return opened, closed


def test_ledger_operations_close_every_connection(monkeypatch, tmp_path):
    """Every public ledger operation must close the connection it opened."""
    _point_ledger(monkeypatch, tmp_path)
    opened, closed = _track_connections(monkeypatch)

    oid = dl.compute_obligation_id("sess", "msg", "content")
    dl.record_obligation(
        obligation_id=oid, session_key="sess", platform="telegram",
        chat_id="123", thread_id=None, content="hello",
    )
    dl.mark_attempting(oid)
    dl.mark_delivered(oid)
    dl.sweep_recoverable()
    dl.debug_rows()

    assert opened, "expected at least one connection to be opened"
    assert len(opened) == len(closed)
    assert set(opened) == set(closed)


def test_survives_transient_state_db_write_contention(monkeypatch, tmp_path):
    """Transient lock contention on state.db must be ridden out — the
    ledger survives a brief hold by a competing writer instead of
    raising ``database is locked`` mid-turn.

    Repro pattern: a sibling Hermes process (CLI turn append, older
    install mid-FTS-maintenance) holds the WAL write lock long enough
    that the ledger's previous implicit-transaction commit would have
    surfaced as an OperationalError. The shared
    ``state_db_begin_immediate`` primitive now waits it out via the same
    jitter schedule ``SessionDB._execute_write`` uses.
    """
    import threading
    _point_ledger(monkeypatch, tmp_path)

    # Force the WAL into existence so the contention is real.
    # Let ``_connect()`` create the schema with the right columns; we just
    # need to force WAL so a second opener actually has to fight the lock.
    dl._connect().close()
    seed = sqlite3.connect(str(tmp_path / "state.db"), timeout=10, isolation_level=None)
    try:
        seed.execute("PRAGMA journal_mode=WAL")
    finally:
        seed.close()

    competitor_holder = []
    release = threading.Event()

    def _hold_lock():
        c = sqlite3.connect(str(tmp_path / "state.db"), timeout=10, isolation_level=None)
        competitor_holder.append(c)
        c.execute("BEGIN IMMEDIATE")
        release.wait()
        try:
            c.execute("COMMIT")
        except Exception:
            try:
                c.execute("ROLLBACK")
            except Exception:
                pass
        c.close()

    competitor = threading.Thread(target=_hold_lock, daemon=True)
    competitor.start()

    deadline = __import__("time").monotonic() + 2.0
    while not competitor_holder:
        if __import__("time").monotonic() > deadline:
            pytest.fail("competitor never acquired the lock")
        __import__("time").sleep(0.01)
    __import__("time").sleep(0.05)

    # Schedule the competitor to release after 250 ms — well within the
    # primitive's 20 s default patience budget but long enough to force
    # at least one retry cycle.
    def _release_after():
        __import__("time").sleep(0.25)
        release.set()
    threading.Thread(target=_release_after, daemon=True).start()

    # record_obligation must succeed by riding out the hold.
    dl.record_obligation(
        obligation_id="obl_contended",
        session_key="sess",
        platform="telegram",
        chat_id="123",
        thread_id=None,
        content="contended write",
    )

    # Confirm via debug_rows that the row landed.
    import json as _json
    parsed = _json.loads(dl.debug_rows())
    found = [r for r in parsed if r["id"] == "obl_contended"]
    assert found, parsed

    competitor.join(timeout=2.0)


