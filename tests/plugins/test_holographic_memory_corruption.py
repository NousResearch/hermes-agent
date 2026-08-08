import json
import sqlite3

import pytest

from plugins.memory.holographic import HolographicMemoryProvider
from plugins.memory.holographic.store import MemoryStore


def test_memory_store_quarantines_corrupt_database_with_shared_registry(tmp_path):
    db_path = tmp_path / "memory_store.db"
    db_path.write_bytes(b"not a sqlite database")
    (tmp_path / "memory_store.db-wal").write_bytes(b"stale wal")
    (tmp_path / "memory_store.db-shm").write_bytes(b"stale shm")

    store = MemoryStore(db_path=db_path)

    try:
        assert db_path.exists()
        assert (tmp_path / "memory_store.db.corrupt").read_bytes() == b"not a sqlite database"
        wal = tmp_path / "memory_store.db-wal"
        shm = tmp_path / "memory_store.db-shm"
        if wal.exists():
            assert wal.read_bytes() != b"stale wal"
        if shm.exists():
            assert shm.read_bytes() != b"stale shm"
        assert store._entry["ready"] is True
        assert MemoryStore._shared[store._key] is store._entry

        store.add_fact("Hermes remembers the moonlit path")
        assert store.list_facts()[0]["content"] == "Hermes remembers the moonlit path"
        sqlite3.connect(db_path).execute("SELECT COUNT(*) FROM facts").fetchone()
    finally:
        store.close()


def test_memory_store_reuses_recovered_shared_connection(tmp_path):
    db_path = tmp_path / "memory_store.db"
    db_path.write_bytes(b"not a sqlite database")

    first = MemoryStore(db_path=db_path)
    second = MemoryStore(db_path=db_path)
    try:
        assert first._entry is second._entry
        first.add_fact("shared recovery works")
        assert second.list_facts()[0]["content"] == "shared recovery works"
    finally:
        first.close()
        second.close()


def test_memory_store_does_not_quarantine_transient_database_errors(tmp_path, monkeypatch):
    db_path = tmp_path / "memory_store.db"
    db_path.write_bytes(b"valid user data")

    def raise_locked(self):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(MemoryStore, "_init_db", raise_locked)

    with pytest.raises(sqlite3.OperationalError, match="database is locked"):
        MemoryStore(db_path=db_path)

    assert db_path.read_bytes() == b"valid user data"
    assert not (tmp_path / "memory_store.db.corrupt").exists()


def test_fact_store_returns_clear_error_when_provider_uninitialized():
    provider = HolographicMemoryProvider(config={})

    result = json.loads(provider.handle_tool_call("fact_store", {"action": "list"}))

    assert "holographic memory is unavailable" in result["error"]
    assert "configured database" in result["error"]
    assert "~/.hermes" not in result["error"]
    assert "NoneType" not in result["error"]


def test_fact_feedback_returns_clear_error_when_provider_uninitialized():
    provider = HolographicMemoryProvider(config={})

    result = json.loads(provider.handle_tool_call("fact_feedback", {"action": "helpful", "fact_id": 1}))

    assert "holographic memory is unavailable" in result["error"]
    assert "configured database" in result["error"]
    assert "~/.hermes" not in result["error"]
    assert "NoneType" not in result["error"]


def test_system_prompt_reports_unavailable_memory_when_uninitialized():
    provider = HolographicMemoryProvider(config={})
    prompt = provider.system_prompt_block()

    assert "Unavailable" in prompt
    assert "fact storage failed to initialize" in prompt
