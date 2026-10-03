"""Current-chat visibility is independent of the foreground agent slot."""
from types import SimpleNamespace
import threading
from queue import Queue

import pytest


@pytest.mark.parametrize("kind, expected", [
    ("idle", "No"), ("starting", "Yes"), ("child", "Yes"),
    ("other_child", "No"), ("completed", "No"), ("pending", "Yes"),
    ("process", "Yes"), ("server", "No"), ("other_process", "No"),
    ("queue", "Yes"), ("inflight", "Yes"), ("unavailable", "Unknown"),
    ("delivery_gap", "Unknown"), ("delivered_process", "No"),
    ("other_pending", "No"), ("batch", "Yes"),
])
def test_current_chat_work_snapshot(kind, expected, tmp_path, monkeypatch):
    from gateway import slash_commands_status as status
    from tools import async_delegation as delegation
    from tools.process_registry import process_registry

    monkeypatch.setattr(delegation, "list_async_delegations", lambda: [
        {"session_key": "other" if kind == "other_child" else "chat",
         "status": "running" if kind in {"child", "other_child"} else "completed"}
    ])
    process = SimpleNamespace(session_key="other" if kind == "other_process" else "chat",
                              notify_on_complete=kind != "server", exited=kind in {
                                  "inflight", "delivery_gap", "delivered_process"},
                              id="proc", started_at=1)
    monkeypatch.setattr(process_registry, "_running", {"proc": process} if kind in {
        "process", "server", "other_process", "inflight", "delivery_gap", "delivered_process"} else {})
    monkeypatch.setattr(process_registry, "_finished", {})
    queue = Queue()
    if kind in {"pending", "other_pending"}:
        queue.put({"session_key": "other" if kind == "other_pending" else "chat", "type": "completion"})
    monkeypatch.setattr(process_registry, "completion_queue", queue)
    runner = SimpleNamespace(
        _running_agents={"chat": object()} if kind == "starting" else {},
        _completion_delivery_lock=threading.Lock(),
        _completion_deliveries_inflight={("completion", "proc", 1)} if kind == "inflight" else set(),
        _completion_deliveries_delivered={("completion", "proc", 1)} if kind == "delivered_process" else set(),
        _completion_notification_batches={("chat",): [object()]} if kind == "batch" else {},
    )
    if kind == "unavailable":
        monkeypatch.setattr(delegation, "list_async_delegations", lambda: 1 / 0)
    assert status._chat_work_state(runner, "chat", kind == "queue", tmp_path) == expected
    assert queue.qsize() == (1 if kind in {"pending", "other_pending"} else 0)  # never drains


def test_durable_pending_completion_is_read_only_and_chat_scoped(tmp_path, monkeypatch):
    import sqlite3
    from gateway import slash_commands_status as status
    from tools import async_delegation as delegation
    from tools.process_registry import process_registry

    monkeypatch.setattr(delegation, "list_async_delegations", lambda: [])
    monkeypatch.setattr(process_registry, "_running", {})
    monkeypatch.setattr(process_registry, "_finished", {})
    monkeypatch.setattr(process_registry, "completion_queue", Queue())
    runner = SimpleNamespace(_running_agents={}, _completion_notification_batches={})
    db = sqlite3.connect(tmp_path / "state.db")
    db.execute("CREATE TABLE async_delegations (origin_session TEXT, delivery_state TEXT, state TEXT)")
    db.execute("INSERT INTO async_delegations VALUES ('chat', 'pending', 'completed')")
    db.commit()
    assert status._chat_work_state(runner, "chat", 0, tmp_path) == "Yes"
    assert status._chat_work_state(runner, "other", 0, tmp_path) == "No"
    db.execute("UPDATE async_delegations SET delivery_state='delivered'")
    db.commit()
    assert status._chat_work_state(runner, "chat", 0, tmp_path) == "No"
    db.close()
