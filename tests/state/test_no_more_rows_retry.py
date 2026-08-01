"""Retry of transient 'no more rows available' engine errors (#74934 port).

Under dual gateway/agent WAL contention (FTS5 trigram sync holding the
write lock on large appends), the SQLite engine can raise a transient
'no more rows available' error. The exception CLASS varies with the
SQLite build — some surface it as ``sqlite3.InterfaceError``, which is a
sibling of ``DatabaseError`` (not a subclass) and therefore escaped both
existing retry branches in ``_execute_write`` on attempt 0, killing the
turn as ``session_persistence_failed`` while the identical write would
have succeeded milliseconds later.

The fix is message-scoped, not class-scoped: any ``sqlite3.Error`` whose
text contains 'no more rows available' retries within the existing
deadline/patience loop; every other error propagates untouched.
"""

import sqlite3

import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path, monkeypatch):
    # Keep retries fast: tiny jitter, short-but-sufficient patience.
    monkeypatch.setattr(SessionDB, "_WRITE_PATIENCE_S", 2.0)
    monkeypatch.setattr(SessionDB, "_WRITE_RETRY_MIN_S", 0.001)
    monkeypatch.setattr(SessionDB, "_WRITE_RETRY_MAX_S", 0.005)
    d = SessionDB(db_path=tmp_path / "state.db")
    yield d
    d.close()


class TestNoMoreRowsRetry:
    def test_transient_interface_error_is_retried_to_success(self, db):
        """InterfaceError('no more rows available') must be retried inside
        the deadline/patience loop and succeed once the contention clears."""
        calls = {"n": 0}

        def flaky(conn):
            calls["n"] += 1
            if calls["n"] <= 3:
                raise sqlite3.InterfaceError("no more rows available")
            conn.execute(
                "INSERT INTO state_meta (key, value) VALUES ('nmr', 'ok') "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value"
            )
            return "done"

        assert db._execute_write(flaky) == "done"
        assert calls["n"] == 4
        assert db.get_meta("nmr") == "ok"

    def test_unrelated_interface_error_propagates_immediately(self, db):
        """The catch-all is message-scoped: an InterfaceError with any other
        text must escape on the first attempt, not be swallowed/retried."""
        calls = {"n": 0}

        def broken(conn):
            calls["n"] += 1
            raise sqlite3.InterfaceError("bad parameter or other API misuse")

        with pytest.raises(sqlite3.InterfaceError, match="bad parameter"):
            db._execute_write(broken)
        assert calls["n"] == 1

    def test_no_more_rows_via_database_error_is_retried(self, db):
        """Some builds raise the same transient message through the generic
        DatabaseError class — it must ride the same retry loop instead of
        being misrouted into the FTS-corruption rebuild path."""
        calls = {"n": 0}

        def flaky(conn):
            calls["n"] += 1
            if calls["n"] <= 2:
                raise sqlite3.DatabaseError("no more rows available")
            return "ok"

        assert db._execute_write(flaky) == "ok"
        assert calls["n"] == 3

    def test_exhausted_patience_propagates_the_transient_error(self, db, monkeypatch):
        """If contention never clears within the patience budget, the
        original error must surface rather than looping forever."""
        monkeypatch.setattr(SessionDB, "_WRITE_PATIENCE_S", 0.05)

        def always(conn):
            raise sqlite3.InterfaceError("no more rows available")

        with pytest.raises(sqlite3.InterfaceError, match="no more rows"):
            db._execute_write(always)
