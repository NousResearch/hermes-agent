from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.task_store import TaskStatus, TaskStore


class RecordingHooks:
    def __init__(self, callback=None, fail_on: set[str] | None = None):
        self.callback = callback
        self.fail_on = set(fail_on or [])
        self.events: list[tuple[str, dict]] = []

    def emit(self, event_name: str, payload: dict) -> None:
        self.events.append((event_name, payload))
        if self.callback is not None:
            self.callback(event_name, payload)
        if event_name in self.fail_on:
            raise RuntimeError(f"boom: {event_name}")


class SequenceRegistry:
    def __init__(self, *poll_responses: dict):
        self._responses = list(poll_responses)

    def poll(self, session_id: str) -> dict:
        assert session_id == "proc-1"
        assert self._responses
        return self._responses.pop(0)


@pytest.fixture()
def store(tmp_path: Path) -> TaskStore:
    return TaskStore(root_dir=tmp_path)


def test_lifecycle_hooks_fire_once_in_order_after_persistence(tmp_path: Path) -> None:
    observed_paths: list[Path] = []

    def callback(event_name: str, payload: dict) -> None:
        task_path = tmp_path / f"{payload['task_id']}.json"
        observed_paths.append(task_path)
        assert task_path.exists()
        persisted = json.loads(task_path.read_text(encoding="utf-8"))
        assert persisted["id"] == payload["task_id"]

    hooks = RecordingHooks(callback=callback)
    store = TaskStore(root_dir=tmp_path, hooks=hooks)

    task = store.create_task(goal="ship it")
    store.transition_task(task.id, TaskStatus.queued)
    store.transition_task(task.id, TaskStatus.running)
    store.record_result(task.id, status=TaskStatus.completed, result={"ok": True}, summary="done")

    assert observed_paths == [tmp_path / f"{task.id}.json"] * 3
    assert [name for name, _ in hooks.events] == [
        "task.created",
        "task.started",
        "task.completed",
    ]


def test_hook_failures_do_not_corrupt_task_state(tmp_path: Path) -> None:
    hooks = RecordingHooks(fail_on={"task.started", "task.failed"})
    store = TaskStore(root_dir=tmp_path, hooks=hooks)

    task = store.create_task(goal="break safely")
    store.transition_task(task.id, TaskStatus.queued)
    started = store.transition_task(task.id, TaskStatus.running)
    failed = store.record_result(task.id, status=TaskStatus.failed, error="boom", exit_code=23)

    assert started.execution.status == TaskStatus.running
    assert failed.execution.status == TaskStatus.failed
    persisted = store.require_task(task.id)
    assert persisted.execution.status == TaskStatus.failed
    assert persisted.execution.last_error == "boom"
    assert persisted.execution.exit_code == 23
    assert [name for name, _ in hooks.events] == [
        "task.created",
        "task.started",
        "task.failed",
    ]


def test_hook_payloads_redact_sensitive_values(tmp_path: Path) -> None:
    hooks = RecordingHooks()
    store = TaskStore(root_dir=tmp_path, hooks=hooks)

    task = store.create_task(
        goal="use secrets",
        metadata={"token": "meta-secret", "nested": {"password": "meta-pass"}},
        permissions={"api_key": "perm-secret"},
        resolved_inputs={"auth": {"secret": "input-secret"}},
        launch_spec={"env": {"OPENAI_API_KEY": "launch-secret"}},
    )

    _, payload = hooks.events[0]
    task_payload = payload["task"]

    assert task_payload["metadata"]["token"] == "[REDACTED]"
    assert task_payload["metadata"]["nested"]["password"] == "[REDACTED]"
    assert task_payload["permissions"]["api_key"] == "[REDACTED]"
    assert task_payload["resolved_inputs"]["auth"]["secret"] == "[REDACTED]"
    assert task_payload["launch_spec"]["env"]["OPENAI_API_KEY"] == "[REDACTED]"

    persisted = store.require_task(task.id)
    assert persisted.metadata["token"] == "meta-secret"
    assert persisted.permissions["api_key"] == "perm-secret"
    assert persisted.resolved_inputs["auth"]["secret"] == "input-secret"
    assert persisted.launch_spec["env"]["OPENAI_API_KEY"] == "launch-secret"


def test_retry_requested_hook_fires_once_when_continuation_enters_retry_state(tmp_path: Path) -> None:
    hooks = RecordingHooks()
    store = TaskStore(root_dir=tmp_path, hooks=hooks)

    task = store.create_task(goal="retry me")

    store.update_continuation(task.id, status="pending")
    store.update_continuation(task.id, status="retry_requested")
    store.update_continuation(task.id, status="retry_requested")
    store.prepare_for_retry(task.id)

    retry_events = [(name, payload) for name, payload in hooks.events if name == "task.retry_requested"]
    assert len(retry_events) == 1
    assert retry_events[0][1]["task"]["execution"]["continuation"]["status"] == "pending"
    assert store.require_task(task.id).execution.status == TaskStatus.draft


def test_reconcile_task_emits_started_completed_and_reconciled_hooks_in_order(tmp_path: Path) -> None:
    hooks = RecordingHooks()
    store = TaskStore(root_dir=tmp_path, hooks=hooks)

    task = store.create_task(goal="watch process")
    store.attach_process(task.id, process_session_id="proc-1", process_command="python worker.py")

    registry = SequenceRegistry(
        {"status": "running"},
        {"status": "exited", "exit_code": 0, "output_preview": "all done"},
    )

    running = store.reconcile_task(task.id, process_registry=registry)
    completed = store.reconcile_task(task.id, process_registry=registry)

    assert running.execution.status == TaskStatus.running
    assert completed.execution.status == TaskStatus.completed
    assert [name for name, _ in hooks.events] == [
        "task.created",
        "task.started",
        "task.reconciled",
        "task.completed",
        "task.reconciled",
    ]
    assert hooks.events[-1][1]["task"]["execution"]["status"] == "completed"
