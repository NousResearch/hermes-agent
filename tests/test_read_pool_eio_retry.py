"""Pooled-read transient EIO: poison-eviction + read_execute retry (#100871).

On CoW / sparse-vhd backing stores (WSL2 ext4-on-vhdx, ZFS, APFS-CoW) a
pooled read connection can throw ``disk I/O error`` for one statement while
the DB is intact. Recycling that handle would repeat the EIO for every
later borrower; the fix closes it and retries the statement once on a fresh
connection.
"""

import sqlite3
import sys

sys.path.insert(0, r"C:\Users\salma\dev\hermes-agent")

import pytest

from hermes_state import SessionDB


@pytest.fixture()
def db(tmp_path):
    d = SessionDB(db_path=tmp_path / "state.db")
    d.create_session(session_id="s1", source="cli", model="test")
    yield d
    # Tests replace _checkout_read_conn with fakes that bypass the permit
    # accounting; a normal close() would then over-release the semaphore.
    # Restore the real method before closing so teardown is balanced.
    d._checkout_read_conn = type(d)._checkout_read_conn.__get__(d)
    d._close_read_conn = type(d)._close_read_conn.__get__(d)
    d.close()


def _poison(conn):
    """Make a real pooled connection's next execute raise transient EIO."""
    conn.execute = lambda *a, **k: (_ for _ in ()).throw(
        sqlite3.OperationalError("disk I/O error")
    )  # type: ignore[method-assign]
    return conn


def test_read_execute_retries_transient_eio(db):
    """First connection poisoned with EIO -> retry on a healthy connection
    returns the row; the retry is counted."""
    orig = db._checkout_read_conn
    state = {"first": True}

    def checkout():
        if state["first"]:
            state["first"] = False
            return _poison(orig())
        return orig()

    db._checkout_read_conn = checkout

    cursor = db.read_execute("SELECT 'ok' AS v")
    assert cursor.fetchone()[0] == "ok"
    assert db._read_ioerr_retries == 1, "retry must be counted"


def test_read_execute_propagates_deterministic_errors(db):
    """Non-EIO OperationalError (corruption, schema) propagates on attempt
    one — no retry, no masking."""
    orig = db._checkout_read_conn

    def boom(*a, **k):
        raise sqlite3.OperationalError("no such table: sessions")

    real = orig()
    real.execute = boom
    db._checkout_read_conn = lambda: real

    with pytest.raises(sqlite3.OperationalError, match="no such table"):
        db.read_execute("SELECT * FROM sessions")
    assert db._read_ioerr_retries == 0


def test_get_session_survives_transient_eio(db):
    """The 37-traceback crash site: get_session must return the row when
    the first pooled connection flakes with EIO."""
    orig = db._checkout_read_conn
    state = {"first": True}

    def checkout():
        if state["first"]:
            state["first"] = False
            return _poison(orig())
        return orig()

    db._checkout_read_conn = checkout
    row = db.get_session("s1")
    assert row is not None
    assert row["id"] == "s1"
    assert db._read_ioerr_retries == 1


def test_poisoned_connection_is_not_recycled(db):
    """The pool must not regain a connection that raised EIO — it is
    closed instead, so later borrowers don't inherit the flake."""
    orig = db._checkout_read_conn
    state = {"first": True}

    class _Tracking:
        def __init__(self):
            self.closed = []
            self.real = db._close_read_conn

        def __call__(self, conn):
            self.closed.append(conn)
            self.real(conn)

    tracker = _Tracking()

    def checkout():
        if state["first"]:
            state["first"] = False
            return _poison(orig())
        return orig()

    db._close_read_conn = tracker
    db._checkout_read_conn = checkout
    db.get_session("s1")

    assert tracker.closed, "the EIO-poisoned connection must be closed, not recycled"


def test_read_execute_double_eio_propagates(db):
    """A deterministic EIO (real I/O failure) fails on both attempts and
    propagates — the retry is once-only, never a loop."""
    orig = db._checkout_read_conn

    def checkout():
        return _poison(orig())

    db._checkout_read_conn = checkout

    with pytest.raises(sqlite3.OperationalError, match="disk I/O error"):
        db.read_execute("SELECT 1")
    assert db._read_ioerr_retries == 1, "exactly one retry, then propagate"
