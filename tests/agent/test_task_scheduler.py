from __future__ import annotations

from pathlib import Path

import pytest

from agent.task_store import TaskStatus, TaskStore
from agent.task_scheduler import AtlasTaskScheduler


class RecordingLauncher:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, task, launch_spec):
        self.calls.append({"task_id": task.id, "launch_spec": dict(launch_spec)})
        return {
            "process_session_id": f"session-{task.id}",
            "process_task_id": f"process-{task.id}",
            "process_command": launch_spec.get("command") or f"launch {task.id}",
        }


@pytest.fixture()
def task_store(tmp_path: Path) -> TaskStore:
    return TaskStore(root_dir=tmp_path)


def test_scheduler_launches_runnable_tasks_in_dependency_order_and_records_metadata(task_store: TaskStore) -> None:
    first = task_store.create_task(goal="first", launch_spec={"command": "run first"}, runtime_mode="task")
    second = task_store.create_task(goal="second", blockedBy=[first.id], launch_spec={"command": "run second"})
    launcher = RecordingLauncher()
    scheduler = AtlasTaskScheduler(
        task_store=task_store,
        launcher=launcher,
        owner="atlas",
        agent_name="Atlas-lite",
        model="gpt-test",
        runtime_mode="execution_supervisor",
    )

    first_result = scheduler.run_once(max_launches=2)

    assert first_result["launched_task_ids"] == [first.id]
    assert [call["task_id"] for call in launcher.calls] == [first.id]
    first_record = task_store.require_task(first.id)
    assert first_record.owner == "atlas"
    assert first_record.metadata["agent"] == "Atlas-lite"
    assert first_record.metadata["model"] == "gpt-test"
    assert first_record.runtime_mode == "task"
    assert first_record.execution.status == TaskStatus.queued
    assert first_record.execution.process_session_id == f"session-{first.id}"
    assert launcher.calls[0]["launch_spec"]["owner"] == "atlas"
    assert launcher.calls[0]["launch_spec"]["agent"] == "Atlas-lite"
    assert launcher.calls[0]["launch_spec"]["model"] == "gpt-test"
    assert launcher.calls[0]["launch_spec"]["runtime_mode"] == "task"

    task_store.transition_task(first.id, TaskStatus.running)
    task_store.record_result(first.id, status=TaskStatus.completed, result={"ok": True}, summary="done", exit_code=0)

    second_result = scheduler.run_once(max_launches=2)

    assert second_result["launched_task_ids"] == [second.id]
    assert [call["task_id"] for call in launcher.calls] == [first.id, second.id]
    second_record = task_store.require_task(second.id)
    assert second_record.execution.process_session_id == f"session-{second.id}"


def test_scheduler_serializes_tasks_that_share_a_thread_id(task_store: TaskStore) -> None:
    first = task_store.create_task(goal="first", threadID="thread-1", launch_spec={"command": "run first"})
    second = task_store.create_task(goal="second", threadID="thread-1", launch_spec={"command": "run second"})
    launcher = RecordingLauncher()
    scheduler = AtlasTaskScheduler(task_store=task_store, launcher=launcher)

    initial = scheduler.run_once(max_launches=2)

    assert initial["launched_task_ids"] == [first.id]
    assert [call["task_id"] for call in launcher.calls] == [first.id]

    task_store.transition_task(first.id, TaskStatus.running)
    task_store.record_result(first.id, status=TaskStatus.completed, result={"ok": True}, summary="done", exit_code=0)

    follow_up = scheduler.run_once(max_launches=2)

    assert follow_up["launched_task_ids"] == [second.id]
    assert [call["task_id"] for call in launcher.calls] == [first.id, second.id]


def test_scheduler_keeps_downstream_tasks_blocked_when_dependency_failed(task_store: TaskStore) -> None:
    upstream = task_store.create_task(goal="upstream", launch_spec={"command": "run upstream"})
    downstream = task_store.create_task(goal="downstream", blockedBy=[upstream.id], launch_spec={"command": "run downstream"})
    launcher = RecordingLauncher()
    scheduler = AtlasTaskScheduler(task_store=task_store, launcher=launcher)

    scheduler.run_once(max_launches=1)
    task_store.record_result(upstream.id, status=TaskStatus.failed, result={"ok": False}, summary="boom", error="boom", exit_code=1)

    result = scheduler.run_once(max_launches=2)
    status = scheduler.status()

    assert result["launched_task_ids"] == []
    assert [call["task_id"] for call in launcher.calls] == [upstream.id]
    assert downstream.id in status["blocked_tasks"]
    blocked = status["blocked_tasks"][downstream.id]
    assert blocked["reason"] == "dependency_failed"
    assert blocked["dependency_ids"] == [upstream.id]


def test_scheduler_prepares_retry_requested_tasks_before_relaunch(task_store: TaskStore) -> None:
    task = task_store.create_task(goal="retry me", launch_spec={"command": "run retry", "background": True})
    task_store.attach_process(task.id, process_session_id="old-session", process_command="run retry")
    task_store.transition_task(task.id, TaskStatus.running)
    task_store.record_result(
        task.id,
        status=TaskStatus.failed,
        result={"stale": True},
        summary="old summary",
        error="old error",
        exit_code=9,
    )
    task_store.update_continuation(task.id, status="retry_requested", attempt_count=2)

    launcher = RecordingLauncher()
    scheduler = AtlasTaskScheduler(task_store=task_store, launcher=launcher)

    result = scheduler.run_once(max_launches=1)

    assert result["prepared_retry_task_ids"] == [task.id]
    assert result["launched_task_ids"] == [task.id]
    reloaded = task_store.require_task(task.id)
    assert reloaded.execution.status == TaskStatus.queued
    assert reloaded.execution.result is None
    assert reloaded.execution.exit_code is None
    assert reloaded.execution.last_error is None
    assert reloaded.summary is None
    assert reloaded.execution.started_at is None
    assert reloaded.execution.finished_at is None
    assert reloaded.execution.process_session_id == f"session-{task.id}"


def test_scheduler_dry_run_reports_next_launch_candidates(task_store: TaskStore) -> None:
    first = task_store.create_task(goal="first", launch_spec={"command": "run first"})
    task_store.create_task(goal="second", blockedBy=[first.id], launch_spec={"command": "run second"})
    launcher = RecordingLauncher()
    scheduler = AtlasTaskScheduler(task_store=task_store, launcher=launcher)

    preview = scheduler.dry_run(max_launches=2)

    assert preview["launchable_task_ids"] == [first.id]
    assert launcher.calls == []


def test_retry_requested_continuation_does_not_relaunch_after_success(task_store: TaskStore) -> None:
    task = task_store.create_task(goal="retry once", launch_spec={"command": "run retry"})
    task_store.attach_process(task.id, process_session_id="old-session", process_command="run retry")
    task_store.transition_task(task.id, TaskStatus.running)
    task_store.record_result(task.id, status=TaskStatus.failed, result={"stale": True}, summary="old", error="old", exit_code=1)
    task_store.update_continuation(task.id, status="retry_requested", attempt_count=1)

    launcher = RecordingLauncher()
    scheduler = AtlasTaskScheduler(task_store=task_store, launcher=launcher)

    first = scheduler.run_once(max_launches=1)
    assert first["launched_task_ids"] == [task.id]
    relaunched = task_store.require_task(task.id)
    assert relaunched.execution.continuation.get("status") is None

    task_store.transition_task(task.id, TaskStatus.running)
    task_store.record_result(task.id, status=TaskStatus.completed, result={"ok": True}, summary="done", exit_code=0)

    second = scheduler.run_once(max_launches=1)
    assert second["prepared_retry_task_ids"] == []
    assert second["launched_task_ids"] == []
    assert [call["task_id"] for call in launcher.calls] == [task.id]
