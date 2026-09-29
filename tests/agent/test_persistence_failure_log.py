"""The persistence-failure log line must carry SQLite's own provenance (#RIC-103).

``Session DB append_message failed: database or disk is full`` was logged as ``str(e)``
alone, and that text is produced with the filesystem healthy too — a per-connection
``max_page_count`` ceiling, a temp file SQLite could not create, a heap limit. The operator
copy built on that string ("free some space") then points at a disk that is not full.

These tests raise a REAL ``SQLITE_FULL`` from a real connection (a page-count ceiling, on a
filesystem with free space) and assert the log line the operator reads names the exception
class, the SQLite result code, and the database path.
"""

from __future__ import annotations

import logging
import sqlite3
from types import SimpleNamespace

import pytest

from agent.session_persistence import _db_flush_failed


def _real_sqlite_full(tmp_path) -> sqlite3.OperationalError:
    """SQLITE_FULL raised by the engine, with free space on the filesystem."""
    conn = sqlite3.connect(tmp_path / "state.db")
    conn.execute("CREATE TABLE t(id INTEGER PRIMARY KEY, body TEXT)")
    conn.executemany("INSERT INTO t(body) VALUES (?)", [("x" * 4096,) for _ in range(50)])
    conn.commit()
    conn.execute("PRAGMA max_page_count = 30")  # ceiling, not free space
    with pytest.raises(sqlite3.OperationalError) as excinfo:
        conn.executemany("INSERT INTO t(body) VALUES (?)", [("y" * 4096,) for _ in range(200)])
        conn.commit()
    conn.close()
    return excinfo.value


def _agent(db_path):
    return SimpleNamespace(
        session_id="s1",
        _db_flush_scan_prefix="stale",
        _last_persistence_error_cause=None,
        _session_db=SimpleNamespace(db_path=db_path),
        _compression_adoption_failed=False,
    )


def test_failed_flush_logs_result_code_and_db_path(tmp_path, caplog):
    err = _real_sqlite_full(tmp_path)
    assert str(err) == "database or disk is full"  # the misleading prose
    db_path = tmp_path / "state.db"
    agent = _agent(db_path)

    with caplog.at_level(logging.WARNING, logger="run_agent"):
        assert _db_flush_failed(agent, err, [], adoption_budget=0) is False

    logged = caplog.text
    assert "database or disk is full" in logged
    assert "SQLITE_FULL" in logged, "the result code name must be in the log line"
    assert "sqlite_errorcode=13" in logged
    assert str(db_path) in logged, "the failing database path must be in the log line"
    assert "cause=disk" in logged
    # The turn-end explanation still reads the same bucket; only the logged detail changed.
    assert agent._last_persistence_error_cause == "disk"
    assert agent._db_flush_scan_prefix is None


def test_failed_flush_log_line_survives_an_agent_without_a_session_db(caplog):
    """Defensive: the log line must not be the thing that raises inside a failure path."""
    agent = SimpleNamespace(
        session_id="s1", _db_flush_scan_prefix=None, _last_persistence_error_cause=None,
        _compression_adoption_failed=False,
    )
    with caplog.at_level(logging.WARNING, logger="run_agent"):
        assert _db_flush_failed(agent, RuntimeError("boom"), [], adoption_budget=0) is False
    assert "db=unknown" in caplog.text
    assert "cause=unknown" in caplog.text
