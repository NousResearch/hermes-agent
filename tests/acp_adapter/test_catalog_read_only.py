"""Browsing ACP history is not an execution/storage owner."""
import sqlite3

from acp_adapter.session import SessionManager
from persistence_assembly import open_persistence, profile_scope
from tests.fixtures.storage_seed import seed_messages, seed_session


def test_empty_catalog_does_not_initialize_storage(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    manager = SessionManager()
    assert manager.list_sessions() == []
    assert not (tmp_path / "state.db").exists()
    assert manager._db_instance is None


def test_catalog_reads_committed_history_without_schema_writes(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = tmp_path / "state.db"
    with open_persistence(profile_scope(home=tmp_path)) as store:
        seed_session(store, "catalog", "acp", model="fixture", model_config={"cwd": str(tmp_path)})
        seed_messages(store, "catalog", {"role": "user", "content": "catalog input"})
    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA journal_mode=DELETE")
    before = path.read_bytes()
    manager = SessionManager()
    rows = manager.list_sessions(cwd=str(tmp_path))
    assert [row["session_id"] for row in rows] == ["catalog"]
    assert path.read_bytes() == before
    assert manager._db_instance is None
