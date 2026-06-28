"""write_txn BUSY-retry behaviour for the shared sqlite_util helper.

Mirrors the kanban_db-specific tests but targets the shared primitive that
projects_db (and any future consumer) imports.  No real DB is touched: a
fake connection records and replays scripted boundary outcomes.
"""

import sqlite3

import pytest

from hermes_cli import sqlite_util


class _FakeConn:
    """Records execute() calls and replays a scripted result per SQL statement."""

    def __init__(self, script):
        self._script = {k: list(v) for k, v in script.items()}
        self.calls = []

    def execute(self, sql, *args):
        self.calls.append(sql)
        key = sql.strip().split()[0].upper()
        outcomes = self._script.get(key)
        if outcomes:
            outcome = outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
        return None

    def count(self, prefix):
        prefix = prefix.upper()
        return sum(1 for c in self.calls if c.strip().upper().startswith(prefix))


def _busy():
    return sqlite3.OperationalError("database is locked")


def _other():
    return sqlite3.OperationalError("no such table: projects")


def test_transient_busy_at_begin_is_absorbed():
    conn = _FakeConn({"BEGIN": [_busy(), None]})
    with sqlite_util.write_txn(conn):
        pass
    assert conn.count("BEGIN") == 2
    assert conn.count("COMMIT") == 1


def test_transient_busy_at_commit_is_absorbed():
    conn = _FakeConn({"COMMIT": [_busy(), None]})
    with sqlite_util.write_txn(conn):
        pass
    assert conn.count("COMMIT") == 2


def test_non_busy_error_is_not_retried():
    conn = _FakeConn({"BEGIN": [_other()]})
    with pytest.raises(sqlite3.OperationalError, match="no such table"):
        with sqlite_util.write_txn(conn):
            pass
    assert conn.count("BEGIN") == 1


def test_persistent_busy_is_bounded():
    conn = _FakeConn({"BEGIN": [_busy()] * 50})
    with pytest.raises(sqlite3.OperationalError, match="database is locked"):
        with sqlite_util.write_txn(conn):
            pass
    assert conn.count("BEGIN") <= sqlite_util._BUSY_MAX_RETRIES + 1


def test_body_is_not_replayed_on_commit_retry():
    conn = _FakeConn({"COMMIT": [_busy(), None]})
    body_runs = 0
    with sqlite_util.write_txn(conn):
        body_runs += 1
    assert body_runs == 1


def test_persistent_busy_at_commit_rolls_back():
    conn = _FakeConn({"COMMIT": [_busy()] * 50})
    with pytest.raises(sqlite3.OperationalError, match="database is locked"):
        with sqlite_util.write_txn(conn):
            pass
    assert conn.count("ROLLBACK") == 1


def test_retry_sleep_respects_jitter_bounds(monkeypatch):
    slept = []
    monkeypatch.setattr(sqlite_util.time, "sleep", lambda s: slept.append(s))
    conn = _FakeConn({"BEGIN": [_busy(), _busy(), None]})
    with sqlite_util.write_txn(conn):
        pass
    assert slept
    assert all(s >= sqlite_util._BUSY_RETRY_MIN_S for s in slept)
    assert all(s <= sqlite_util._BUSY_RETRY_MAX_S for s in slept)


def test_clean_path_commits_once():
    conn = _FakeConn({})
    with sqlite_util.write_txn(conn):
        pass
    assert conn.count("BEGIN") == 1
    assert conn.count("COMMIT") == 1
