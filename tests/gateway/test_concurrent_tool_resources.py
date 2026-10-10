"""Physical concurrent worker resource lifetimes are independent of batch futures."""
from concurrent.futures import ThreadPoolExecutor
import sqlite3
import threading

from agent.chat_execution_context import bind_chat_execution
from gateway.chat_execution_recovery import ChatExecutionStore


def test_resource_handle_stays_unresolved_until_physical_worker_finishes(tmp_path):
    store = ChatExecutionStore(tmp_path)
    assert store.start("key", 1, "session")
    assert store.worker_started("key", 1)
    resource_id = "resource-1"
    assert store.reserve_concurrent_tool_resource("key", 1, resource_id)
    assert store.start_concurrent_tool_resource("key", 1, resource_id, "session")
    assert store.inspect_resources("key")[0]["state"] == "started"
    assert store.finish_concurrent_tool_resource("key", 1, resource_id, "session")
    assert store.inspect_resources("key")[0]["state"] == "finished"


def test_reservation_requires_started_latest_parent_and_rejects_duplicate(tmp_path):
    store = ChatExecutionStore(tmp_path)
    assert store.reserve("key", 0, 0) == 1
    assert not store.reserve_concurrent_tool_resource("key", 1, "r")
    assert store.start("key", 2, "session")
    assert store.worker_started("key", 2)
    assert store.reserve_concurrent_tool_resource("key", 2, "r")
    assert not store.reserve_concurrent_tool_resource("key", 2, "r")


def test_store_migrates_resource_table_on_existing_home(tmp_path):
    db = sqlite3.connect(tmp_path / "chat-execution.sqlite3")
    db.execute("CREATE TABLE executions (session_key TEXT NOT NULL, generation INTEGER NOT NULL, session_id TEXT NOT NULL, worker TEXT NOT NULL, worker_started INTEGER NOT NULL, effects TEXT NOT NULL, tool_observed INTEGER NOT NULL, PRIMARY KEY(session_key,generation))")
    db.commit()
    db.close()
    ChatExecutionStore(tmp_path)
    db = sqlite3.connect(tmp_path / "chat-execution.sqlite3")
    assert db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='concurrent_tool_resources'").fetchone()
    db.close()
