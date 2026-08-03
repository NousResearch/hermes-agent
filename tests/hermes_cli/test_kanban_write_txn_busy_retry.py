"""write_txn BUSY-retry behaviour.

These tests target the transaction boundary (BEGIN IMMEDIATE / COMMIT) only.
On unmodified main write_txn has no application-level retry, so the
"transient BUSY is absorbed" and "persistent BUSY is bounded" cases fail until
the fix lands. No real DB is touched: a fake connection records and replays
scripted boundary outcomes.
"""

import sqlite3

import pytest

from hermes_cli import kanban_db as kb


class _FakeConn:
    """Records execute() calls and replays a scripted result per SQL statement.

    script maps an uppercased SQL prefix to a list of outcomes consumed in
    order. An outcome is either an Exception (raised) or None (success).
    """

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
    return sqlite3.OperationalError("no such table: tasks")


@pytest.fixture(autouse=True)
def _no_file_check(monkeypatch):
    # Isolate the boundary behaviour from the post-commit invariant.
    monkeypatch.setattr(kb, "_check_file_length_invariant", lambda conn: None)


def test_retry_sleep_respects_floor(monkeypatch):
    # The jitter has a floor so a retry can't busy-spin back into the collision.
    slept = []
    monkeypatch.setattr(kb.time, "sleep", lambda s: slept.append(s))
    conn = _FakeConn({"BEGIN": [_busy(), _busy(), None]})
    with kb.write_txn(conn):
        pass
    assert slept
    assert all(s >= kb._BUSY_RETRY_MIN_S for s in slept)
    assert all(s <= kb._BUSY_RETRY_MAX_S for s in slept)


def test_transient_busy_at_begin_is_absorbed():
    conn = _FakeConn({"BEGIN": [_busy(), None]})
    with kb.write_txn(conn):
        pass
    assert conn.count("BEGIN") == 2
    assert conn.count("COMMIT") == 1


def test_transient_busy_at_commit_is_absorbed():
    conn = _FakeConn({"COMMIT": [_busy(), None]})
    with kb.write_txn(conn):
        pass
    assert conn.count("COMMIT") == 2


def test_non_busy_operational_error_is_not_retried():
    conn = _FakeConn({"BEGIN": [_other()]})
    with pytest.raises(sqlite3.OperationalError, match="no such table"):
        with kb.write_txn(conn):
            pass
    assert conn.count("BEGIN") == 1


def test_persistent_busy_is_bounded_and_reraises():
    conn = _FakeConn({"BEGIN": [_busy()] * 50})
    with pytest.raises(sqlite3.OperationalError, match="database is locked"):
        with kb.write_txn(conn):
            pass
    # Bounded: a finite number of attempts, not 50.
    assert conn.count("BEGIN") < 50


def test_body_is_not_replayed_on_commit_retry():
    conn = _FakeConn({"COMMIT": [_busy(), None]})
    body_runs = 0
    with kb.write_txn(conn):
        body_runs += 1
    assert body_runs == 1


def test_clean_path_commits_once():
    conn = _FakeConn({})
    with kb.write_txn(conn):
        pass
    assert conn.count("BEGIN") == 1


def test_persistent_busy_at_commit_rolls_back():
    # Exhausted COMMIT leaves the txn open; write_txn must ROLLBACK before
    # re-raising so the connection isn't poisoned for the next transaction.
    conn = _FakeConn({"COMMIT": [_busy()] * 50})
    with pytest.raises(sqlite3.OperationalError, match="database is locked"):
        with kb.write_txn(conn):
            pass
    assert conn.count("ROLLBACK") == 1
