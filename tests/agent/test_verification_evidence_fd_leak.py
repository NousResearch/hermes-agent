"""Regression: the verification-evidence ledger must close every connection.

Sibling of the cron execution-ledger leak (#69567 / PR #69594). The evidence
ledger used ``with _connect() as conn:`` where the connection context manager
commits/rolls back but never closes, leaking the db/-wal/-shm file descriptors
on every recorded terminal result, workspace edit, and status read. These tests
fail if the deterministic ``close()`` is ever removed again.
"""

import sqlite3

import pytest

from agent import verification_evidence as ve


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
    monkeypatch.setattr(ve, "_db_path", lambda: tmp_path / "verification_evidence.db")
    return ve


def _track_connections(monkeypatch):
    opened, closed = [], []
    real_connect = sqlite3.connect

    def tracking_connect(*args, **kwargs):
        conn = real_connect(*args, **kwargs)
        opened.append(id(conn))
        return _TrackingConnection(conn, closed)

    monkeypatch.setattr(ve.sqlite3, "connect", tracking_connect)
    return opened, closed


def _python_project(root):
    (root / "pyproject.toml").write_text("[tool.pytest.ini_options]\n")


def test_ledger_operations_close_every_connection(monkeypatch, tmp_path):
    """Recording, editing, and status reads must close every connection opened."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    _point_ledger(monkeypatch, tmp_path)
    _python_project(tmp_path)
    opened, closed = _track_connections(monkeypatch)

    ve.record_terminal_result(
        command="python -m pytest tests/test_calc.py::test_even -q",
        cwd=tmp_path, session_id="s1", exit_code=0, output="1 passed",
    )
    ve.verification_status(session_id="s1", cwd=tmp_path)
    ve.mark_workspace_edited(session_id="s1", cwd=tmp_path, paths=["mod.py"])

    assert opened, "expected at least one connection to be opened"
    assert len(opened) == len(closed)
    assert set(opened) == set(closed)


def test_exception_during_operation_still_closes_connection(monkeypatch, tmp_path):
    """A failing statement inside the transaction must roll back and close."""
    _point_ledger(monkeypatch, tmp_path)
    opened, closed = _track_connections(monkeypatch)

    with pytest.raises(sqlite3.IntegrityError):
        with ve._transaction() as conn:
            # Missing NOT NULL columns -> constraint failure inside the block.
            conn.execute("INSERT INTO verification_events (id) VALUES (1)")

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

    monkeypatch.setattr(ve.sqlite3, "connect", tracking_connect)

    with pytest.raises(sqlite3.OperationalError):
        with ve._transaction():
            pass

    assert len(opened) == 1
    assert len(closed) == 1
