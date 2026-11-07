"""The message storage-state read and the insights reads wait as long as main's did for another process's lock.

Main ran them on the store's writer connection, whose lock wait is 1 s: on a DELETE-journal store another process's
commit makes them fail after that second. The insights reads of a read-only handle (the CLI and the dashboard) ran
on main's read-only connection, which waits 5 s. Each case runs main's read and the backend's against a store that a
separate process holds locked for longer than the first wait and shorter than the second.
"""

from __future__ import annotations

import importlib.util
import sqlite3
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import pytest
import hermes_yaml as yaml

from hermes_constants import get_hermes_home
from persistence.conversation.messages import NewMessage
from persistence.conversation.sessions import NewSession
from persistence.core.errors import StoreContention
from persistence.sqlite import SQLiteBackend
from tests.persistence_sqlite._legacy_store import legacy_import, legacy_store, oracle_tree
from tests.persistence_sqlite.conftest import profile_home, store_for

__all__ = ["legacy_store"]

_HOLD_S = 3.0  # past the writer connection's 1 s wait, inside the read-only connection's 5 s
_REFUSED_BY_S = 2.5

_HOLDER = """
import sqlite3, sys, time
conn = sqlite3.connect(sys.argv[1], isolation_level=None)
conn.execute("BEGIN EXCLUSIVE")
print("held", flush=True)
time.sleep(float(sys.argv[2]))
conn.execute("COMMIT")
"""

_STORAGE_STATE_SQL = "SELECT session_id, active, compacted FROM messages WHERE id = ?"


@pytest.fixture(autouse=True)
def _delete_journal():
    (get_hermes_home() / "config.yaml").write_text(
        yaml.safe_dump({"database": {"journal_mode": "delete"}}), encoding="utf-8")


@contextmanager
def _locked(path: Path):
    """``path`` held under an exclusive lock by another process for ``_HOLD_S``."""
    holder = subprocess.Popen([sys.executable, "-c", _HOLDER, str(path), str(_HOLD_S)],
                              stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "held"
        yield
    finally:
        assert holder.wait(timeout=30) == 0


def _timed(read):
    """``(result or raised exception, seconds)`` of one read."""
    started = time.monotonic()
    try:
        result = read()
    except Exception as exc:  # the outcome under test
        result = exc
    return result, time.monotonic() - started


def _main_insights_engine():
    """Main's ``agent.insights.InsightsEngine``, loaded from the pinned checkout."""
    spec = importlib.util.spec_from_file_location("_main_agent_insights", oracle_tree() / "agent" / "insights.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.InsightsEngine


def _main_store(tmp_path: Path, *, message: bool = True):
    """Main's writer store over a fresh file holding one session (and one message); returns (db, message id)."""
    SessionDB = legacy_import("hermes_state").SessionDB
    db = SessionDB(db_path=tmp_path / "main" / "state.db")
    assert db._conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "delete"
    db.create_session("s1", source="cli")
    message_id = db.append_message("s1", role="user", content="hello") if message else None
    return db, message_id


@pytest.fixture
def backend_store(scope_a):
    profile_home(scope_a.name)
    backend = SQLiteBackend({scope_a.hermes_home_key: store_for(scope_a)})
    try:
        store = backend.open(scope_a)
        store.sessions.create(NewSession("s1", "cli"))
        yield store, store.messages.append("s1", NewMessage("user", "hello"))
    finally:
        backend.close()


def test_message_storage_state_waits_one_second_as_main(tmp_path, legacy_store, backend_store, scope_a):
    db, main_id = _main_store(tmp_path)
    try:
        def main_read():
            with db._lock:
                return db._conn.execute(_STORAGE_STATE_SQL, (main_id,)).fetchone()
        with _locked(db.db_path):
            main_outcome, main_s = _timed(main_read)
    finally:
        db.close()
    assert isinstance(main_outcome, sqlite3.OperationalError) and main_s < _REFUSED_BY_S

    store, message_id = backend_store
    with _locked(store_for(scope_a)):
        outcome, seconds = _timed(lambda: store.messages.message_storage_state(message_id))
    assert isinstance(outcome, StoreContention) and seconds < _REFUSED_BY_S


def test_insights_on_a_writer_store_wait_one_second_as_main(tmp_path, legacy_store, backend_store, scope_a):
    db, _ = _main_store(tmp_path)
    try:
        engine = _main_insights_engine()(db)
        with _locked(db.db_path):
            main_outcome, main_s = _timed(lambda: engine._query("_GET_SESSIONS", 0.0, None))
    finally:
        db.close()
    assert isinstance(main_outcome, sqlite3.OperationalError) and main_s < _REFUSED_BY_S

    store, _ = backend_store
    analytics = store.usage_analytics
    assert [row["id"] for row in analytics.insight_sessions(started_since=0.0)] == ["s1"]
    with _locked(store_for(scope_a)):
        outcome, seconds = _timed(lambda: analytics.insight_sessions(started_since=0.0))
    assert isinstance(outcome, StoreContention) and seconds < _REFUSED_BY_S


def test_insights_on_a_read_only_store_keep_main_read_only_wait(tmp_path, legacy_store, backend_store, scope_a):
    main_writer, _ = _main_store(tmp_path, message=False)
    main_writer.close()
    db = legacy_import("hermes_state").SessionDB(db_path=tmp_path / "main" / "state.db", read_only=True)
    try:
        engine = _main_insights_engine()(db)
        with _locked(db.db_path):
            main_outcome, _ = _timed(lambda: engine._query("_GET_SESSIONS", 0.0, None))
    finally:
        db.close()
    assert [row["id"] for row in main_outcome] == ["s1"]

    reader = SQLiteBackend({scope_a.hermes_home_key: store_for(scope_a)}, read_only=True)
    try:
        analytics = reader.open(scope_a).usage_analytics
        assert [row["id"] for row in analytics.insight_sessions(started_since=0.0)] == ["s1"]
        with _locked(store_for(scope_a)):
            outcome, _ = _timed(lambda: analytics.insight_sessions(started_since=0.0))
    finally:
        reader.close()
    assert [row["id"] for row in outcome] == ["s1"]
