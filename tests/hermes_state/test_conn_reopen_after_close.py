"""Regression: writes after a concurrent close() must not raise NoneType.

Repro of the live bug seen on a very long WebUI session: ``close()`` (process
shutdown / WAL-shrink / restart hook) sets ``self._conn = None``; a late flush
thread (run_agent's session-DB mirror) then reached ``self._conn.execute(...)``
and crashed with ``'NoneType' object has no attribute 'execute'``, dropping that
message's DB mirror (the JSON store still had it — no user-visible loss — but the
searchable DB copy diverged). The fix: ``_execute_write`` re-opens a nulled
connection under the lock before BEGIN IMMEDIATE.
"""

import sqlite3

import pytest

from hermes_state import SessionDB


def _make_db(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("s1", "test")
    return db


def test_append_after_close_reopens_and_persists(tmp_path):
    db = _make_db(tmp_path)
    db.append_message(session_id="s1", role="user", content="before")
    db.close()
    assert db._conn is None  # close nulled the handle

    # Late flush after close must NOT raise; it should re-open and write.
    mid = db.append_message(session_id="s1", role="assistant", content="after")
    assert isinstance(mid, int) and mid > 0
    assert db._conn is not None

    rows = db.get_messages("s1")
    assert [r["content"] for r in rows] == ["before", "after"]


def test_readonly_db_does_not_silently_reopen(tmp_path):
    SessionDB(db_path=tmp_path / "state.db").create_session("s1", "test")
    ro = SessionDB(db_path=tmp_path / "state.db", read_only=True)
    ro.close()
    with pytest.raises(sqlite3.ProgrammingError):
        ro._ensure_conn_locked()
