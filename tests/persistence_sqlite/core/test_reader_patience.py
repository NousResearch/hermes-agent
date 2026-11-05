"""The read-only reader's patience: how long its open and each of its reads wait for a store another connection holds
exclusively, before the read raises ``StoreContention``. Each wait is compared with a raw read-only SQLite connection
given the same timeout and running one statement under the same hold, measured right after it: SQLite can wait longer
than the timeout it is given before it reports the store locked."""

from __future__ import annotations

import sqlite3
import time
from contextlib import closing

import pytest

from persistence import StoreContention
from persistence_assembly import open_persistence, open_persistence_reader, open_store_admin, profile_scope
from tests.fixtures.storage_holder import read_busy_timeout_seam
from tests.fixtures.storage_seed import seed_session
from tests.persistence_sqlite.conftest import profile_home


def _store(name: str):
    home = profile_home(name)
    with open_persistence(profile_scope(home=home), eager=True) as store:
        seed_session(store, "s", "cli")
    return home / "state.db"


def _held(path):
    return open_store_admin(path, read_only=False).exclusive_repair_guard()


def _raw(path, timeout: float):
    return closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=timeout))


def _raw_read(conn):
    return conn.execute("SELECT 1 FROM messages LIMIT 1").fetchall()


def _waited(read) -> float:
    start = time.monotonic()
    with pytest.raises(StoreContention):
        read()
    return time.monotonic() - start


def _raw_waited(conn) -> float:
    start = time.monotonic()
    with pytest.raises(sqlite3.OperationalError, match="locked"):
        _raw_read(conn)
    return time.monotonic() - start


def _about(waited: float, patience_s: float) -> bool:
    return patience_s * 0.7 <= waited < patience_s * 1.3


@pytest.mark.parametrize("patience_s", [1.0, 2.0])
def test_the_open_waits_the_patience_the_reader_is_given(patience_s):
    path = _store("alpha")
    with _held(path), open_persistence_reader(path, patience_s=patience_s) as reader, _raw(path, patience_s) as raw:
        waited = _waited(reader.sessions.first_session_started_at)
        raw_waited = _raw_waited(raw)
    assert _about(waited, raw_waited), (waited, raw_waited)


def test_each_read_waits_the_patience_the_reader_is_given():
    path = _store("alpha")
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("PRAGMA journal_mode=DELETE")
    with open_persistence_reader(path, patience_s=0.5) as reader, _raw(path, 0.5) as raw:
        assert reader.sessions.first_session_started_at() is not None
        _raw_read(raw)
        with _held(path):
            message_waited = _waited(reader.messages.has_any_message)
            message_raw = _raw_waited(raw)
            session_waited = _waited(reader.sessions.first_session_started_at)
            session_raw = _raw_waited(raw)
        assert _about(message_waited, message_raw) and _about(session_waited, session_raw), (
            message_waited, message_raw, session_waited, session_raw)
        assert reader.messages.has_any_message() is False


def test_without_a_patience_the_reader_waits_the_backend_read_wait(monkeypatch):
    monkeypatch.setattr(*read_busy_timeout_seam(), 1.0)
    path = _store("alpha")
    with _held(path), open_persistence_reader(path) as reader, _raw(path, 1.0) as raw:
        waited = _waited(reader.sessions.first_session_started_at)
        raw_waited = _raw_waited(raw)
    assert _about(waited, raw_waited), (waited, raw_waited)
