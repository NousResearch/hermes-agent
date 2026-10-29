"""Exercise restore write deferral against real SQLite and native compressors."""

from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from pathlib import Path
import sqlite3
import sys
import uuid

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from agent.context_compressor import ContextCompressor
from agent.context_compressor_state import defer_compressor_state_writes
from agent.model_metadata import get_model_context_length
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from hermes_state import SessionDB


@pytest.fixture
def session_compressors(tmp_path):
    token = set_hermes_home_override(tmp_path)
    db = SessionDB(db_path=tmp_path / "sessions.db")
    ids = [uuid.uuid4().hex, uuid.uuid4().hex]
    engines = []
    for session_id in ids:
        db.create_session(session_id, source="cli")
        engine = ContextCompressor(model="gpt-4.1", provider="openai", quiet_mode=True)
        engine.bind_session_state(db, session_id)
        engines.append(engine)
    try:
        yield db, ids, engines
    finally:
        for engine in engines:
            engine.bind_session_state()
        db.close()
        reset_hermes_home_override(token)


def test_failed_restore_discards_native_model_reset_writes(session_compressors):
    db, ids, engines = session_compressors
    first = engines[0]
    db.set_compression_fallback_streak(ids[0], 1)
    first._load_fallback_compression_streak()
    closed_reader = sqlite3.connect(":memory:")
    closed_reader.close()
    with pytest.raises(sqlite3.ProgrammingError):
        with defer_compressor_state_writes(first):
            first.update_model("gpt-4.1-mini", get_model_context_length("gpt-4.1-mini"), provider="openai")
            closed_reader.execute("SELECT 1")
    assert db.get_compression_fallback_streak(ids[0]) == 1
    with defer_compressor_state_writes(first):
        first.update_model("gpt-4.1", get_model_context_length("gpt-4.1"), provider="openai")
    assert db.get_compression_fallback_streak(ids[0]) == 0


def test_other_sessions_and_context_copied_threads_write_independently(session_compressors):
    db, ids, engines = session_compressors
    first, second = engines
    db.set_compression_fallback_streak(ids[0], 1)
    with defer_compressor_state_writes(first):
        assert first._durable_write("set_compression_fallback_streak", "restore", 0)
        with ThreadPoolExecutor(max_workers=1) as pool:
            assert pool.submit(copy_context().run, first._durable_write, "set_compression_fallback_streak", "worker", 2).result(timeout=30)
        assert second._durable_write("set_compression_fallback_streak", "other session", 1)
        assert db.get_compression_fallback_streak(ids[0]) == 2
        assert db.get_compression_fallback_streak(ids[1]) == 1
    assert db.get_compression_fallback_streak(ids[0]) == 0
    assert db.get_compression_fallback_streak(ids[1]) == 1


def test_deferred_native_write_failure_is_visible(session_compressors):
    db, _ids, engines = session_compressors
    with db._lock:
        db._conn.execute('PRAGMA query_only = ON')
    try:
        with pytest.raises(sqlite3.OperationalError, match='readonly'):
            with defer_compressor_state_writes(engines[0]):
                assert engines[0]._durable_write('set_compression_fallback_streak', 'restore', 0)
    finally:
        with db._lock:
            db._conn.execute('PRAGMA query_only = OFF')
