"""FTS5 ``flush`` is a 3.44 command. Older SQLite raises SQL logic error for it.

That text is also what genuine shadow-table damage raises, so the probe must
skip the command on old SQLite instead of treating the error as corruption.
See #120567.
"""

import sqlite3
import uuid
from pathlib import Path

import pytest

import hermes_state_repair
from hermes_state import SessionDB
from hermes_state_repair import _db_opens_cleanly


class _RejectFlush:
    """Stand in for SQLite < 3.44: the flush command is an unknown config write."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self.flush_calls = 0

    def execute(self, sql, *args, **kwargs):
        if isinstance(sql, str) and "values('flush')" in sql.lower():
            self.flush_calls += 1
            raise sqlite3.OperationalError("SQL logic error")
        return self._conn.execute(sql, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._conn, name)


def _healthy_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    sid = db.create_session(session_id=str(uuid.uuid4()), source="cli")
    db.append_message(sid, role="user", content="hello world")
    db.close()
    return db_path


def _connect_rejecting_flush(real):
    def connect(path, **kwargs):
        return _RejectFlush(real(path, **kwargs))

    return connect


def test_pre_344_unknown_flush_is_not_reported_as_corruption(tmp_path, monkeypatch):
    db_path = _healthy_db(tmp_path)
    monkeypatch.setattr(sqlite3, "sqlite_version_info", (3, 40, 0))
    monkeypatch.setattr(
        hermes_state_repair,
        "_connect_repair_durable",
        _connect_rejecting_flush(hermes_state_repair._connect_repair_durable),
    )

    assert _db_opens_cleanly(db_path) is None


def test_modern_sqlite_still_issues_flush(tmp_path, monkeypatch):
    if sqlite3.sqlite_version_info < (3, 44, 0):
        pytest.skip("this interpreter's SQLite predates the flush command")

    db_path = _healthy_db(tmp_path)
    seen = {"n": 0}
    real = hermes_state_repair._connect_repair_durable

    class _CountFlush(_RejectFlush):
        def execute(self, sql, *args, **kwargs):
            if isinstance(sql, str) and "values('flush')" in sql.lower():
                seen["n"] += 1
                raise sqlite3.OperationalError("SQL logic error")
            return self._conn.execute(sql, *args, **kwargs)

    def connect(path, **kwargs):
        return _CountFlush(real(path, **kwargs))

    monkeypatch.setattr(hermes_state_repair, "_connect_repair_durable", connect)
    reason = _db_opens_cleanly(db_path)

    assert seen["n"] >= 1
    assert reason is not None and "SQL logic error" in reason
