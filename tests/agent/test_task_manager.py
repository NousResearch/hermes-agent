"""Tests for agent.task_manager."""

import pytest

from agent.task_manager import TaskManager


def test_create_task(tmp_path):
    manager = TaskManager(storage_path=tmp_path / "tasks.json")

    task = manager.create(
        title="Write structured task manager",
        kind="local",
        assignee="local",
        metadata={"phase": "d1"},
    )

    assert task["title"] == "Write structured task manager"
    assert task["status"] == "pending"
    assert task["kind"] == "local"
    assert task["assignee"] == "local"
    assert task["metadata"] == {"phase": "d1"}
    assert task["task_id"]
    assert task["created_at"] == task["updated_at"]


def test_update_status_valid(tmp_path):
    manager = TaskManager(storage_path=tmp_path / "tasks.json")
    task = manager.create(title="Run task")

    running = manager.update(task["task_id"], status="running")
    completed = manager.update(
        task["task_id"],
        status="completed",
        result_summary="finished",
        metadata={"ok": True},
    )

    assert running["status"] == "running"
    assert completed["status"] == "completed"
    assert completed["result_summary"] == "finished"
    assert completed["metadata"] == {"ok": True}
    assert completed["updated_at"] != completed["created_at"]


def test_update_status_invalid(tmp_path):
    manager = TaskManager(storage_path=tmp_path / "tasks.json")
    task = manager.create(title="Skip running")

    with pytest.raises(ValueError, match="Invalid status transition"):
        manager.update(task["task_id"], status="completed")


def test_list_tasks(tmp_path):
    manager = TaskManager(storage_path=tmp_path / "tasks.json")
    pending_task = manager.create(title="Pending local task", kind="local")
    delegated_task = manager.create(title="Delegated task", kind="delegated")
    manager.update(delegated_task["task_id"], status="running")

    pending = manager.list(status="pending")
    delegated = manager.list(kind="delegated")

    assert [task["task_id"] for task in pending] == [pending_task["task_id"]]
    assert [task["task_id"] for task in delegated] == [delegated_task["task_id"]]


def test_get_task(tmp_path):
    manager = TaskManager(storage_path=tmp_path / "tasks.json")
    created = manager.create(title="Lookup task")

    fetched = manager.get(created["task_id"])

    assert fetched == created
    assert manager.get("missing-task") is None


def test_cancel_task(tmp_path):
    manager = TaskManager(storage_path=tmp_path / "tasks.json")
    task = manager.create(title="Cancel me")

    cancelled = manager.cancel(task["task_id"])

    assert cancelled["status"] == "cancelled"


def test_persistence(tmp_path):
    storage_path = tmp_path / "state" / "tasks.json"
    manager = TaskManager(storage_path=storage_path)
    created = manager.create(
        title="Persisted task",
        kind="system",
        assignee="system",
        metadata={"persisted": True},
    )

    reloaded = TaskManager(storage_path=storage_path)
    fetched = reloaded.get(created["task_id"])

    assert fetched == created


def test_multiple_tasks_no_data_leak(tmp_path):
    manager_a = TaskManager(storage_path=tmp_path / "a" / "tasks.json")
    manager_b = TaskManager(storage_path=tmp_path / "b" / "tasks.json")

    task_a = manager_a.create(title="Task A", metadata={"owner": "a"})
    task_b = manager_b.create(title="Task B", metadata={"owner": "b"})

    task_a["metadata"]["owner"] = "mutated"

    assert [task["title"] for task in manager_a.list()] == ["Task A"]
    assert [task["title"] for task in manager_b.list()] == ["Task B"]
    assert manager_a.get(task_a["task_id"])["metadata"] == {"owner": "a"}
    assert manager_b.get(task_b["task_id"])["metadata"] == {"owner": "b"}
