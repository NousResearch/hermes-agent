"""Seam tests for the kanban_db_txn extraction (kanban_db.py god-file slice R2).

Proves the re-export seam: every name the godfile used to define in the
2731-2838 window now resolves through ``hermes_cli.kanban_db`` to the exact
same object defined in ``hermes_cli.kanban_db_txn``, so ~50 in-file call
sites, 11 external files, and monkeypatching tests (e.g.
test_kanban_write_txn_busy_retry.py which patches
``kb._check_file_length_invariant``) keep working with zero edits.
"""

import sqlite3

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_txn as kbtxn

REEXPORTED = [
    "_BUSY_MAX_RETRIES",
    "_BUSY_RETRY_MAX_S",
    "_BUSY_RETRY_MIN_S",
    "_check_file_length_invariant",
    "_execute_boundary_with_retry",
    "_is_busy_error",
    "write_txn",
]


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
    return sqlite3.OperationalError("no such table: tasks")


@pytest.fixture
def no_file_check(monkeypatch):
    # Isolate boundary behaviour from the post-commit invariant. Patches the
    # godfile attribute (exactly like test_kanban_write_txn_busy_retry.py),
    # which write_txn reaches through its lazy import — proving the seam.
    monkeypatch.setattr(kb, "_check_file_length_invariant", lambda conn: None)
    yield


def test_reexport_identity():
    """Every re-exported name is the identical object, not a shadow copy."""
    for name in REEXPORTED:
        assert getattr(kb, name) is getattr(kbtxn, name), name
        assert getattr(kb, name) is not None, name


def test_module_imports_do_not_shadow():
    assert kb.write_txn.__module__ == "hermes_cli.kanban_db_txn"
    assert kbtxn.write_txn.__module__ == "hermes_cli.kanban_db_txn"


def test_is_busy_error_classification():
    assert kbtxn._is_busy_error(sqlite3.OperationalError("database is locked"))
    assert kbtxn._is_busy_error(sqlite3.OperationalError("database is busy"))
    assert not kbtxn._is_busy_error(_other())
    assert not kbtxn._is_busy_error(ValueError("database is locked"))


def test_execute_boundary_with_retry_recovers_from_busy(monkeypatch):
    slept = []
    monkeypatch.setattr(kbtxn.time, "sleep", lambda s: slept.append(s))
    conn = _FakeConn({"BEGIN": [_busy(), _busy(), None]})
    kbtxn._execute_boundary_with_retry(conn, "BEGIN IMMEDIATE")
    assert conn.count("BEGIN") == 3
    assert len(slept) == 2
    assert all(kbtxn._BUSY_RETRY_MIN_S <= s <= kbtxn._BUSY_RETRY_MAX_S for s in slept)


def test_execute_boundary_with_retry_exhausts_bounded():
    conn = _FakeConn({"BEGIN": [_busy()] * 50})
    with pytest.raises(sqlite3.OperationalError, match="database is locked"):
        kbtxn._execute_boundary_with_retry(conn, "BEGIN IMMEDIATE")
    assert conn.count("BEGIN") == kbtxn._BUSY_MAX_RETRIES + 1


def test_execute_boundary_with_retry_reraises_non_busy_immediately():
    conn = _FakeConn({"BEGIN": [_other()]})
    with pytest.raises(sqlite3.OperationalError, match="no such table"):
        kbtxn._execute_boundary_with_retry(conn, "BEGIN IMMEDIATE")
    assert conn.count("BEGIN") == 1


def test_write_txn_commits(no_file_check):
    conn = _FakeConn({})
    with kb.write_txn(conn):
        pass
    assert conn.count("BEGIN IMMEDIATE") == 1
    assert conn.count("COMMIT") == 1
    assert conn.count("ROLLBACK") == 0


def test_write_txn_rolls_back_on_body_exception(no_file_check):
    conn = _FakeConn({})
    with pytest.raises(RuntimeError, match="boom"):
        with kb.write_txn(conn):
            raise RuntimeError("boom")
    assert conn.count("BEGIN IMMEDIATE") == 1
    assert conn.count("ROLLBACK") == 1
    assert conn.count("COMMIT") == 0


def test_write_txn_commit_exhaustion_rolls_back(no_file_check):
    # Exhausted COMMIT leaves the txn open; write_txn must ROLLBACK before
    # re-raising so the connection isn't poisoned for the next transaction.
    conn = _FakeConn({"COMMIT": [_busy()] * 50})
    with pytest.raises(sqlite3.OperationalError, match="database is locked"):
        with kb.write_txn(conn):
            pass
    assert conn.count("ROLLBACK") == 1


def test_write_txn_busy_at_begin_is_absorbed(no_file_check):
    conn = _FakeConn({"BEGIN": [_busy(), None]})
    with kb.write_txn(conn):
        pass
    assert conn.count("BEGIN") == 2
    assert conn.count("COMMIT") == 1


def test_torn_extend_raises_database_error(monkeypatch):
    """_check_file_length_invariant raises when header page count > file length."""

    class _HeaderConn:
        def execute(self, sql):
            return _HeaderRow()

    class _HeaderRow:
        def fetchone(self):
            return ("delete",)  # rollback journal — invariant enforced

    monkeypatch.setattr(
        "hermes_cli.sqlite_safe_read.file_length_matches_header", lambda conn: False
    )
    with pytest.raises(sqlite3.DatabaseError, match="torn-extend"):
        kb._check_file_length_invariant(_HeaderConn())


def test_delegated_mutation_guard_still_enforced(monkeypatch):
    """The lazy in-body import seam survived: write_txn still refuses children."""

    def _reject():
        raise PermissionError("delegate_task child contexts cannot mutate Kanban")

    monkeypatch.setattr(kb, "_assert_not_delegated_child_mutation", _reject)
    conn = _FakeConn({})
    with pytest.raises(PermissionError, match="delegate_task child"):
        with kb.write_txn(conn):
            pass
    assert conn.calls == []  # guard fires before any SQL is issued
