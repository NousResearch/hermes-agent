"""SQLITE_CANTOPEN on a live writer connection self-heals with one reopen.

A long-lived gateway writer can, after an idle period or a WAL/SHM transition,
fail a write with ``SQLITE_CANTOPEN`` ("unable to open database file") even
though the file is healthy and a fresh connection opens fine -- observed as
``session_persistence_failed`` on interactive turns and as
``session_split_failed`` during compression. ``_execute_write`` now drops the
poisoned handle and retries once on a freshly opened writer, logging the full
traceback so the failure stays diagnosable.
"""

import sqlite3

import pytest

from hermes_state import SessionDB


class _CantopenOnce:
    """sqlite3.Connection proxy: raises CANTOPEN on the Nth execute, else delegates."""

    def __init__(self, real, fail_on):
        object.__setattr__(self, "_real", real)
        object.__setattr__(self, "_count", 0)
        object.__setattr__(self, "_fail_on", fail_on)

    def execute(self, *args, **kwargs):
        object.__setattr__(self, "_count", self._count + 1)
        if self._count == self._fail_on:
            raise sqlite3.OperationalError("unable to open database file")
        return object.__getattribute__(self, "_real").execute(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_real"), name)

    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, "_real"), name, value)


class _CantopenAlways:
    """Connection proxy whose every statement fails with SQLITE_CANTOPEN."""

    def __init__(self, real):
        object.__setattr__(self, "_real", real)

    def execute(self, *args, **kwargs):
        raise sqlite3.OperationalError("unable to open database file")

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_real"), name)

    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, "_real"), name, value)


@pytest.fixture
def db(tmp_path):
    d = SessionDB(db_path=tmp_path / "state.db")
    d._insert_session_row("sess", "cli")
    yield d
    d.close()


class TestCantopenReopen:
    def test_cantopen_on_begin_immediate_reopens_and_retries(self, db):
        real = db._conn
        db._conn = _CantopenOnce(real, fail_on=1)  # BEGIN IMMEDIATE
        rid = db.append_message("sess", "user", content="hello")
        assert isinstance(rid, int)
        assert db._conn is not real
        assert db._conn.execute("SELECT count(*) FROM messages").fetchone()[0] == 1

    def test_cantopen_inside_callback_reopens_and_retries(self, db):
        real = db._conn
        db._conn = _CantopenOnce(real, fail_on=2)  # first statement inside fn()
        db.append_message("sess", "user", content="hello")
        assert db._conn is not real
        assert db._conn.execute("SELECT count(*) FROM messages").fetchone()[0] == 1

    def test_persistent_cantopen_propagates_after_one_reopen(self, db):
        real_open = db._open_writer_conn
        db._open_writer_conn = lambda: _CantopenAlways(real_open())
        db._conn = _CantopenAlways(db._conn)
        with pytest.raises(sqlite3.OperationalError, match="unable to open database file"):
            db.append_message("sess", "user", content="hello")
