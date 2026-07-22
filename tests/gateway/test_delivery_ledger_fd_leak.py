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


def test_early_return_still_closes_connection(monkeypatch, tmp_path):
    """A no-op update (no matching row) must still open and close exactly once."""
    _point_ledger(monkeypatch, tmp_path)
    opened, closed = _track_connections(monkeypatch)

    dl.mark_delivered("does-not-exist")

    assert len(opened) == 1
    assert len(closed) == 1


def test_exception_during_operation_still_closes_connection(monkeypatch, tmp_path):
    """A failing statement inside the transaction must roll back and close."""
    _point_ledger(monkeypatch, tmp_path)
    opened, closed = _track_connections(monkeypatch)

    with pytest.raises(sqlite3.IntegrityError):
        with dl._transaction() as conn:
            # Missing NOT NULL columns -> constraint failure inside the block.
            conn.execute(
                "INSERT INTO delivery_obligations (obligation_id) VALUES ('x')"
            )

    assert len(opened) == 1
    assert len(closed) == 1


def test_schema_init_failure_still_closes_connection(monkeypatch, tmp_path):
    """A PRAGMA/DDL failure after connect() must still close the connection."""
    _point_ledger(monkeypatch, tmp_path)
    opened, closed = [], []
    real_connect = sqlite3.connect

    class _FailingSchemaConnection(_TrackingConnection):
        def execute(self, sql, *args, **kwargs):
            if "CREATE TABLE" in sql:
                raise sqlite3.OperationalError("simulated schema init failure")
            return self._real.execute(sql, *args, **kwargs)

    def tracking_connect(*args, **kwargs):
        conn = real_connect(*args, **kwargs)
        opened.append(id(conn))
        return _FailingSchemaConnection(conn, closed)

    monkeypatch.setattr(dl.sqlite3, "connect", tracking_connect)

    with pytest.raises(sqlite3.OperationalError):
        with dl._transaction():
            pass

    assert len(opened) == 1
    assert len(closed) == 1
