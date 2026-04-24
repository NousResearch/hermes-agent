from __future__ import annotations

from types import SimpleNamespace

from agent.task_store import TaskStatus, TaskStore
from run_agent import AIAgent


def _make_agent(*, session_id: str = "session-top", depth: int = 0) -> AIAgent:
    agent = AIAgent.__new__(AIAgent)
    agent.session_id = session_id
    agent.runtime_activation_state = {"runtime_mode": "execution_supervisor"}
    agent._supervisor_task_snapshot = []
    agent._delegate_depth = depth
    agent.model = "gpt-test"
    agent.provider = "openai"
    return agent


def test_execution_supervisor_note_launches_one_runnable_task_and_reports_counts(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_TASK_STORE_DIR", str(tmp_path / "task-store"))
    store = TaskStore(root_dir=tmp_path / "task-store")
    runnable = store.create_task(
        goal="launch me",
        owner_session_id="session-top",
        launch_spec={"runner": "delegate", "background": True, "command": "delegate launch"},
    )
    blocked = store.create_task(
        goal="blocked",
        owner_session_id="session-top",
        blockedBy=[runnable.id],
        launch_spec={"runner": "delegate", "background": True},
    )
    running = store.create_task(
        goal="running",
        owner_session_id="session-top",
        launch_spec={"runner": "delegate", "background": True},
    )
    store.transition_task(running.id, TaskStatus.queued)
    store.transition_task(running.id, TaskStatus.running)
    completed = store.create_task(
        goal="done",
        owner_session_id="session-top",
        launch_spec={"runner": "delegate", "background": True},
    )
    store.transition_task(completed.id, TaskStatus.queued)
    store.transition_task(completed.id, TaskStatus.running)
    store.record_result(completed.id, status=TaskStatus.completed, result={"ok": True}, summary="done", exit_code=0)

    launched = []

    def _launcher(self, record, launch_spec, *, store):
        launched.append({"task_id": record.id, "launch_spec": dict(launch_spec)})
        return {
            "process_session_id": f"proc-{record.id}",
            "process_command": launch_spec.get("command") or record.goal,
            "background": True,
        }

    monkeypatch.setattr(AIAgent, "_launch_execution_supervisor_task", _launcher, raising=False)

    note = _make_agent()._build_execution_supervisor_note()

    assert "runnable=1" in note
    assert "blocked=1" in note
    assert "running=1" in note
    assert "completed=1" in note
    assert runnable.id in note
    assert "launched_task_ids=" in note
    assert launched == [{"task_id": runnable.id, "launch_spec": launched[0]["launch_spec"]}]
    assert store.require_task(runnable.id).execution.status == TaskStatus.queued
    assert store.require_task(blocked.id).execution.status == TaskStatus.draft


def test_execution_supervisor_note_reports_status_only_for_delegated_children(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_TASK_STORE_DIR", str(tmp_path / "task-store"))
    store = TaskStore(root_dir=tmp_path / "task-store")
    task = store.create_task(
        goal="child task",
        owner_session_id="session-child",
        launch_spec={"runner": "delegate", "background": True},
    )

    def _launcher(self, record, launch_spec, *, store):
        raise AssertionError("delegated children must not auto-launch supervisor tasks")

    monkeypatch.setattr(AIAgent, "_launch_execution_supervisor_task", _launcher, raising=False)

    note = _make_agent(session_id="session-child", depth=1)._build_execution_supervisor_note()

    assert "auto_launch=disabled" in note
    assert "delegate_depth" in note
    assert task.id in note
    assert store.require_task(task.id).execution.status == TaskStatus.draft


def test_execution_supervisor_note_reports_launch_failures_without_crashing(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_TASK_STORE_DIR", str(tmp_path / "task-store"))
    store = TaskStore(root_dir=tmp_path / "task-store")
    task = store.create_task(
        goal="boom",
        owner_session_id="session-top",
        launch_spec={"runner": "delegate", "background": True},
    )

    def _launcher(self, record, launch_spec, *, store):
        raise RuntimeError("launcher exploded")

    monkeypatch.setattr(AIAgent, "_launch_execution_supervisor_task", _launcher, raising=False)

    note = _make_agent()._build_execution_supervisor_note()

    assert "launch_error=launcher exploded" in note
    assert task.id in note
    assert store.require_task(task.id).execution.status == TaskStatus.draft
