"""Concurrent journal openers must not race schema migration or lose old rows."""
from concurrent.futures import ThreadPoolExecutor
import sqlite3
from threading import Barrier

import pytest

from gateway.chat_execution_recovery import ChatExecutionStore


@pytest.mark.parametrize("legacy", [False, True])
def test_concurrent_initialization_preserves_journal(tmp_path, legacy):
    path = tmp_path / "chat-execution.sqlite3"
    if legacy:
        with sqlite3.connect(path) as db:
            db.execute("""CREATE TABLE executions (
                session_key TEXT NOT NULL, generation INTEGER NOT NULL,
                session_id TEXT NOT NULL, worker TEXT NOT NULL, worker_started INTEGER NOT NULL,
                effects TEXT NOT NULL, tool_observed INTEGER NOT NULL,
                PRIMARY KEY(session_key, generation))""")
            db.execute("INSERT INTO executions VALUES ('chat', 7, 'session', 'pending', 1, 'unknown', 1)")
    ready = Barrier(16)

    def open_store():
        ready.wait(timeout=15)
        return ChatExecutionStore(tmp_path)

    with ThreadPoolExecutor(max_workers=16) as pool:
        stores = list(pool.map(lambda _: open_store(), range(16)))
    assert all(store.latest_generation("chat") == (7 if legacy else 0) for store in stores)
    with sqlite3.connect(path) as db:
        columns = [row[1] for row in db.execute("PRAGMA table_info(executions)")]
        assert columns.count("worker_pid") == 1
        assert columns.count("worker_start_time") == 1
        if legacy:
            assert db.execute("SELECT * FROM executions").fetchall() == [
                ("chat", 7, "session", "pending", 1, "unknown", 1, None, None)
            ]
    assert ChatExecutionStore(tmp_path).latest_generation("chat") == (7 if legacy else 0)
