"""Tests for the minimal frontdesk control-loop runtime."""

import json
import threading

import pytest

from agent.control_plane import Intent, Recommendation
from agent.orchestration_runtime import OrchestrationRuntime
from agent.task_registry import (
    FRONTDESK_RUNNING_WORKER,
    FRONTDESK_WORKER_DONE_PENDING_REVIEW,
    STAGE_DONE,
    STAGE_QUEUED,
    STAGE_RUNNING,
    STATUS_CANCELLED,
    STATUS_RUNNING,
)
from agent.worker_lanes import CancelToken, ThreadWorkerLane, WorkerSpec, WorkerStatus


def test_frontdesk_worker_decision_creates_task_and_starts_registered_worker_lane():
    runtime = OrchestrationRuntime.create()
    entered = threading.Event()

    def runner(spec: WorkerSpec, token: CancelToken):  # noqa: ARG001
        entered.set()
        return f"done:{spec.goal}"

    runtime.worker_registry.register(ThreadWorkerLane(runner=runner, name="thread"))

    result = runtime.handle_frontdesk_input(
        "draft a report.md with the audit",
        frontdesk_mode_active=True,
        session_key="s1",
        source_surface="gateway",
    )

    assert result.decision.intent is Intent.NEW_TASK_WORKER
    assert result.decision.recommendation is Recommendation.WORKER_LANE
    assert result.action == "worker_started"
    assert result.task_id is not None
    assert result.worker_id is not None
    task = runtime.task_registry.get_task(result.task_id)
    assert task is not None
    assert task.status == STATUS_RUNNING
    assert task.frontdesk_state == FRONTDESK_RUNNING_WORKER
    assert task.worker_stage == STAGE_RUNNING
    assert task.active_worker_id == result.worker_id
    assert task.worker_kind == "thread"
    assert entered.wait(2.0)
    assert runtime.worker_registry.wait(result.worker_id, timeout=2.0)
    worker_result = runtime.worker_registry.result(result.worker_id)
    assert worker_result is not None
    assert worker_result.status == WorkerStatus.DONE


def test_frontdesk_worker_completion_retains_result_for_review():
    runtime = OrchestrationRuntime.create()

    def runner(spec: WorkerSpec, token: CancelToken):  # noqa: ARG001
        return {
            "summary": f"done:{spec.goal}",
            "artifacts": [{"path": "report.md"}],
            "tests": {"pytest": "passed"},
            "raw_process": object(),
        }

    runtime.worker_registry.register(ThreadWorkerLane(runner=runner, name="thread"))
    started = runtime.handle_frontdesk_input(
        "draft a report.md with the audit",
        frontdesk_mode_active=True,
        session_key="s1",
        source_surface="gateway",
    )

    assert started.task_id is not None
    assert started.worker_id is not None
    assert runtime.worker_registry.wait(started.worker_id, timeout=2.0)

    attached = runtime.collect_worker_results()
    assert [r["worker_id"] for r in attached] == [started.worker_id]

    task = runtime.task_registry.get_task(started.task_id)
    assert task is not None
    assert task.result is not None
    assert task.result["worker_id"] == started.worker_id
    assert task.result["task_id"] == started.task_id
    assert task.result["status"] == "succeeded"
    assert task.result["summary"].startswith("done:draft")
    assert task.result["artifacts"] == [{"path": "report.md"}]
    assert task.result["tests"] == {"pytest": "passed"}
    assert "raw_process" not in task.result
    assert task.result["review_status"] == "pending_review"
    assert task.status == STATUS_RUNNING
    assert task.frontdesk_state == FRONTDESK_WORKER_DONE_PENDING_REVIEW
    assert task.worker_stage == STAGE_DONE
    assert task.reviewer_stage == STAGE_QUEUED
    json.dumps(task.to_dict(), allow_nan=False)


@pytest.mark.parametrize(
    "followup_text",
    [
        "중국집은 없나",
        "빠니니를 파는 곳도 찾아보고 있어야지",
        "이 조건도 넣어줘",
    ],
)
def test_korean_followup_attaches_to_active_worker_task(followup_text):
    runtime = OrchestrationRuntime.create()
    entered = threading.Event()
    release = threading.Event()

    def runner(spec: WorkerSpec, token: CancelToken):  # noqa: ARG001
        entered.set()
        release.wait(2.0)
        token.raise_if_cancelled()
        return "done"

    lane = ThreadWorkerLane(runner=runner, name="thread")
    runtime.worker_registry.register(lane)
    started = runtime.handle_frontdesk_input(
        "워커 레인에 배당해서 이 회귀를 조사해줘",
        frontdesk_mode_active=True,
        session_key="s1",
        source_surface="gateway",
    )
    assert started.action == "worker_started"
    assert started.task_id is not None
    assert started.worker_id is not None
    assert entered.wait(2.0)

    try:
        result = runtime.handle_frontdesk_input(
            followup_text,
            frontdesk_mode_active=True,
            session_key="s1",
            source_surface="gateway",
        )

        assert result.action == "followup_attached"
        assert result.task_id == started.task_id
        assert result.worker_id == started.worker_id
        assert result.message.startswith("control: follow-up attached")
        task = runtime.task_registry.get_task(started.task_id)
        assert task is not None
        assert [item.text for item in task.pending_followups] == [followup_text]
        assert task.pending_followups[0].session_key == "s1"
        assert task.pending_followups[0].task_hint == started.task_id
        assert [item.text for item in lane.followups(started.worker_id)] == [followup_text]
    finally:
        runtime.worker_registry.cancel(started.worker_id)
        release.set()
        runtime.worker_registry.wait(started.worker_id, timeout=2.0)


def test_followup_does_not_swallow_stop_or_explicit_status():
    runtime = OrchestrationRuntime.create()
    entered = threading.Event()
    release = threading.Event()

    def runner(spec: WorkerSpec, token: CancelToken):  # noqa: ARG001
        entered.set()
        release.wait(2.0)
        token.raise_if_cancelled()
        return "done"

    runtime.worker_registry.register(ThreadWorkerLane(runner=runner, name="thread"))
    started = runtime.handle_frontdesk_input(
        "워커 레인에 배당해서 이 회귀를 조사해줘",
        frontdesk_mode_active=True,
        session_key="s1",
    )
    assert started.task_id is not None
    assert started.worker_id is not None
    assert entered.wait(2.0)

    natural_question = runtime.handle_frontdesk_input(
        "지금 뭐 하고 있어?",
        frontdesk_mode_active=True,
        session_key="s1",
    )
    assert natural_question.action == "followup_attached"

    status = runtime.handle_frontdesk_input(
        "/status",
        frontdesk_mode_active=True,
        session_key="s1",
    )
    assert status.action == "status"
    task = runtime.task_registry.get_task(started.task_id)
    assert task is not None
    assert [item.text for item in task.pending_followups] == ["지금 뭐 하고 있어?"]

    stopped = runtime.handle_frontdesk_input(
        "멈춰",
        frontdesk_mode_active=True,
        session_key="s1",
    )
    assert stopped.action == "stopped"
    assert stopped.cancelled_tasks == 1
    assert [item.text for item in task.pending_followups] == ["지금 뭐 하고 있어?"]
    release.set()
    assert runtime.worker_registry.wait(started.worker_id, timeout=2.0)


def test_followup_without_active_worker_falls_through_honestly():
    runtime = OrchestrationRuntime.create()
    runtime.worker_registry.register(ThreadWorkerLane(runner=lambda spec, token: "ok", name="thread"))

    result = runtime.handle_frontdesk_input(
        "이 조건도 넣어줘",
        frontdesk_mode_active=True,
        session_key="s1",
    )

    assert result.action == "main"
    assert result.message == "route: main"
    assert result.task_id is None
    assert result.worker_id is None
    assert runtime.task_registry.list_tasks(session_key="s1") == []


def test_session_scoped_explicit_status_does_not_leak_other_session_workers():
    runtime = OrchestrationRuntime.create()
    release = threading.Event()

    def runner(spec: WorkerSpec, token: CancelToken):  # noqa: ARG001
        release.wait(2.0)
        token.raise_if_cancelled()
        return "done"

    runtime.worker_registry.register(ThreadWorkerLane(runner=runner, name="thread"))
    s1 = runtime.handle_frontdesk_input(
        "워커 레인에 배당해서 s1 작업을 조사해줘",
        frontdesk_mode_active=True,
        session_key="s1",
    )
    s2 = runtime.handle_frontdesk_input(
        "워커 레인에 배당해서 s2 작업을 조사해줘",
        frontdesk_mode_active=True,
        session_key="s2",
    )
    assert s1.worker_id is not None
    assert s2.worker_id is not None

    try:
        status = runtime.handle_frontdesk_input(
            "/status",
            frontdesk_mode_active=True,
            session_key="s1",
        )

        assert status.action == "status"
        assert s1.worker_id in status.message
        assert s2.worker_id not in status.message
        assert "s2 작업" not in status.message
        assert "not registered" not in status.message
    finally:
        runtime.worker_registry.cancel(s1.worker_id)
        runtime.worker_registry.cancel(s2.worker_id)
        release.set()
        runtime.worker_registry.wait(s1.worker_id, timeout=2.0)
        runtime.worker_registry.wait(s2.worker_id, timeout=2.0)


def test_session_scoped_stop_does_not_cancel_other_session_workers():
    runtime = OrchestrationRuntime.create()
    release = threading.Event()

    def runner(spec: WorkerSpec, token: CancelToken):  # noqa: ARG001
        release.wait(2.0)
        token.raise_if_cancelled()
        return "done"

    runtime.worker_registry.register(ThreadWorkerLane(runner=runner, name="thread"))
    s1 = runtime.handle_frontdesk_input(
        "워커 레인에 배당해서 s1 작업을 조사해줘",
        frontdesk_mode_active=True,
        session_key="s1",
    )
    s2 = runtime.handle_frontdesk_input(
        "워커 레인에 배당해서 s2 작업을 조사해줘",
        frontdesk_mode_active=True,
        session_key="s2",
    )
    assert s1.worker_id is not None
    assert s2.worker_id is not None

    stopped = runtime.handle_frontdesk_input(
        "멈춰",
        frontdesk_mode_active=True,
        session_key="s1",
    )

    assert stopped.action == "stopped"
    assert stopped.cancelled_tasks == 1
    assert stopped.cancelled_workers == 1
    assert runtime.worker_registry.status(s1.worker_id).cancel_requested is True
    assert runtime.worker_registry.status(s2.worker_id).cancel_requested is False

    runtime.worker_registry.cancel(s2.worker_id)
    release.set()
    runtime.worker_registry.wait(s1.worker_id, timeout=2.0)
    runtime.worker_registry.wait(s2.worker_id, timeout=2.0)


def test_frontdesk_status_returns_local_overview_without_starting_worker():
    runtime = OrchestrationRuntime.create()
    runtime.task_registry.create_task("existing", session_key="s1", status=STATUS_RUNNING)

    result = runtime.handle_frontdesk_input(
        "/status",
        frontdesk_mode_active=True,
        session_key="s1",
    )

    assert result.decision.intent is Intent.STATUS
    assert result.action == "status"
    assert "Tasks:" in result.message or "Active tasks:" in result.message
    assert "existing" in result.message
    assert result.task_id is None
    assert result.worker_id is None


def test_frontdesk_status_shows_available_lane_when_idle():
    runtime = OrchestrationRuntime.create()
    runtime.worker_registry.register(ThreadWorkerLane(runner=lambda spec, token: "ok", name="main"))

    result = runtime.handle_frontdesk_input(
        "/status",
        frontdesk_mode_active=True,
        session_key="s1",
    )

    assert result.decision.intent is Intent.STATUS
    assert result.action == "status"
    assert "Available worker lanes: main" in result.message


def test_frontdesk_stop_cancels_active_workers_and_tasks_without_replay():
    runtime = OrchestrationRuntime.create()
    release = threading.Event()

    def runner(spec: WorkerSpec, token: CancelToken):  # noqa: ARG001
        release.wait(2.0)
        token.raise_if_cancelled()
        return "should not matter"

    runtime.worker_registry.register(ThreadWorkerLane(runner=runner, name="thread"))
    started = runtime.handle_frontdesk_input(
        "draft a report.md with the audit",
        frontdesk_mode_active=True,
        session_key="s1",
    )
    assert started.worker_id is not None
    assert started.task_id is not None

    stopped = runtime.handle_frontdesk_input("그만", frontdesk_mode_active=True, session_key="s1")

    assert stopped.decision.intent is Intent.STOP
    assert stopped.action == "stopped"
    assert stopped.cancelled_tasks == 1
    assert stopped.cancelled_workers == 1
    task = runtime.task_registry.get_task(started.task_id)
    assert task is not None
    assert task.status == STATUS_CANCELLED
    release.set()
    assert runtime.worker_registry.wait(started.worker_id, timeout=2.0)
    assert runtime.worker_registry.status(started.worker_id).cancel_requested is True


def test_frontdesk_steer_calls_callback_when_main_is_in_flight():
    runtime = OrchestrationRuntime.create()
    steers = []

    def accept(text: str) -> bool:
        steers.append(text)
        return True

    result = runtime.handle_frontdesk_input(
        "also update the config file",
        frontdesk_mode_active=True,
        main_in_flight=True,
        steer_callback=accept,
    )

    assert result.decision.intent is Intent.STEER
    assert result.action == "steered"
    assert steers == ["also update the config file"]
    assert result.message.startswith("control: steered")


def test_frontdesk_steer_false_callback_falls_back_to_main():
    runtime = OrchestrationRuntime.create()
    steers = []

    result = runtime.handle_frontdesk_input(
        "also update the config file",
        frontdesk_mode_active=True,
        main_in_flight=True,
        steer_callback=lambda text: steers.append(text) and False,
    )

    assert result.decision.intent is Intent.STEER
    assert result.action == "main"
    assert result.message != "control: steered active main turn"
    assert steers == ["also update the config file"]


def test_frontdesk_steer_raising_callback_falls_back_to_main():
    runtime = OrchestrationRuntime.create()

    def reject(_text: str) -> bool:
        raise RuntimeError("not ready")

    result = runtime.handle_frontdesk_input(
        "also update the config file",
        frontdesk_mode_active=True,
        main_in_flight=True,
        steer_callback=reject,
    )

    assert result.decision.intent is Intent.STEER
    assert result.action == "main"
    assert result.message != "control: steered active main turn"


def test_frontdesk_steer_without_running_main_falls_back_to_main():
    runtime = OrchestrationRuntime.create()

    result = runtime.handle_frontdesk_input(
        "also update the config file",
        frontdesk_mode_active=True,
        main_in_flight=False,
    )

    assert result.decision.intent is Intent.STEER
    assert result.action == "main"
    assert "no active main turn" in result.message
