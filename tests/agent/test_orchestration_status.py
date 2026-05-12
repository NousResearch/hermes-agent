import copy
import json
import threading
import time

from agent.orchestration_status import (
    OrchestrationSnapshot,
    OrchestrationStatusFormatter,
    build_snapshot,
    format_agents,
    format_overview,
    format_tasks,
    looks_like_orchestration_status_query,
)
from agent.pending_turn_queue import PendingTurnItem
from agent.task_registry import (
    STATUS_BLOCKED,
    STATUS_CANCELLED,
    STATUS_DONE,
    STATUS_ERROR,
    STATUS_RUNNING,
    TaskRegistry,
)
from agent.worker_lanes import (
    CancelToken,
    ThreadWorkerLane,
    WorkerLaneRegistry,
    WorkerSpec,
    WorkerStatus,
    link_worker_to_task,
)

TIMEOUT = 2.0


def _gated_runner():
    entered = threading.Event()
    release = threading.Event()

    def runner(spec, token: CancelToken):  # noqa: ARG001
        entered.set()
        release.wait(TIMEOUT)
        token.raise_if_cancelled()
        return "done"

    return runner, entered, release


def test_empty_snapshot_and_empty_formatting_are_graceful():
    snap = build_snapshot()
    assert snap.tasks == []
    assert snap.workers == []
    assert snap.counts["tasks_total"] == 0
    assert snap.counts["workers_total"] == 0
    assert snap.warnings == []
    assert format_tasks(snap) == "No active tasks are currently registered."
    assert format_agents(snap) == "No active workers are currently registered."
    assert format_overview(snap) == "No active tasks or workers are currently registered."
    json.dumps(snap.to_dict(), allow_nan=False)


def test_task_snapshot_and_formatting_includes_counts_worker_and_state():
    reg = TaskRegistry()
    task = reg.create_task("Draft the MM market research report", session_key="s1", status=STATUS_RUNNING)
    reg.assign_worker(task.task_id, "worker-1", worker_kind="claude_code")
    reg.attach_followup(task.task_id, PendingTurnItem(text="include CELMoD", session_key="s1", raw=object()))
    reg.add_note(task.task_id, "correction: exclude old pathway")
    reg.attach_artifact(task.task_id, {"path": "report.md", "kind": "markdown"})
    reg.create_task("Need Woo decision", session_key="s1", status=STATUS_BLOCKED)
    reg.create_task("Already done", session_key="s1", status=STATUS_DONE)

    snap = build_snapshot(reg, session_key="s1")
    assert snap.counts["tasks_total"] == 3
    assert snap.counts["tasks_active"] == 2
    assert snap.counts["tasks_blocked"] == 1
    assert snap.counts["followups_pending"] == 1
    json.dumps(snap.to_dict(), allow_nan=False)

    text = format_tasks(snap)
    assert "Tasks:" in text
    assert task.task_id in text
    assert "Draft the MM market research" in text
    assert "worker: worker-1 (claude_code)" in text
    assert "1 follow-up queued" in text
    assert "1 note" in text
    assert "needs you" in text


def test_format_tasks_registry_defaults_to_active_tasks_only_and_session_scope():
    reg = TaskRegistry()
    active_s1 = reg.create_task("active one", session_key="s1", status=STATUS_RUNNING)
    reg.create_task("done one", session_key="s1", status=STATUS_DONE)
    reg.create_task("active other session", session_key="s2", status=STATUS_RUNNING)

    text = format_tasks(reg, session_key="s1")
    assert "Active tasks:" in text
    assert active_s1.task_id in text
    assert "done one" not in text
    assert "active other session" not in text


def test_worker_registry_snapshot_formats_running_done_error_cancelled_and_cancel_requested():
    wlr = WorkerLaneRegistry()

    # running + cancel requested
    runner, entered, release = _gated_runner()
    running_lane = ThreadWorkerLane(runner=runner, name="running")
    wlr.register(running_lane)
    running_handle = wlr.start(
        WorkerSpec(goal="long running implementation", task_id="task-run", lane="running")
    )
    assert entered.wait(TIMEOUT)
    assert wlr.cancel(running_handle.worker_id) is True

    # done
    done_lane = ThreadWorkerLane(runner=lambda spec, token: "finished", name="done")
    wlr.register(done_lane)
    done = done_lane.start(WorkerSpec(goal="write final answer", task_id="task-done", lane="done"))
    assert done_lane.wait(done.worker_id, timeout=TIMEOUT)

    # error
    def boom(spec, token):  # noqa: ARG001
        raise RuntimeError("boom")

    error_lane = ThreadWorkerLane(runner=boom, name="error")
    wlr.register(error_lane)
    err = error_lane.start(WorkerSpec(goal="explode", task_id="task-error", lane="error"))
    assert error_lane.wait(err.worker_id, timeout=TIMEOUT)

    # cancelled terminal
    release.set()
    assert running_lane.wait(running_handle.worker_id, timeout=TIMEOUT)

    snap = build_snapshot(worker_registry=wlr)
    assert snap.counts["workers_total"] == 3
    statuses = {w["worker_id"]: w["status"] for w in snap.workers}
    assert statuses[done.worker_id] == WorkerStatus.DONE
    assert statuses[err.worker_id] == WorkerStatus.ERROR
    assert statuses[running_handle.worker_id] == WorkerStatus.CANCELLED
    json.dumps(snap.to_dict(), allow_nan=False)

    text = format_agents(snap, compact=False)
    assert done.worker_id in text and 'result="finished"' in text
    assert err.worker_id in text and 'error="RuntimeError: boom"' in text
    assert running_handle.worker_id in text and "cancel requested" in text


def test_overview_cross_links_task_and_worker_and_warns_on_mismatches():
    reg = TaskRegistry()
    task = reg.create_task("linked work", session_key="s1", status=STATUS_RUNNING)
    lane = ThreadWorkerLane(runner=lambda spec, token: "ok", name="linked")
    wlr = WorkerLaneRegistry()
    wlr.register(lane)
    handle = wlr.start(WorkerSpec(goal="linked work", task_id=task.task_id, lane="linked"))
    link_worker_to_task(reg, task.task_id, handle)
    assert lane.wait(handle.worker_id, timeout=TIMEOUT)

    snap = build_snapshot(reg, wlr, session_key="s1")
    text = format_overview(snap)
    assert "Orchestration status" in text
    assert handle.worker_id in text
    assert f"worker: {handle.worker_id}" in text
    assert "Workers:" in text
    assert snap.warnings == []

    reg.assign_worker(task.task_id, "missing-worker")
    warning_snap = build_snapshot(reg, wlr, session_key="s1")
    assert warning_snap.warnings
    assert "missing-worker" in format_overview(warning_snap)


def test_snapshot_does_not_touch_pending_turn_raw():
    class RawBomb:
        def __deepcopy__(self, memo):  # pragma: no cover - should never be called
            raise AssertionError("raw was deep-copied")

        def __getattribute__(self, name):
            if name not in {"__class__", "__deepcopy__", "__repr__"}:
                raise AssertionError("raw was inspected")
            return object.__getattribute__(self, name)

    reg = TaskRegistry()
    task = reg.create_task("raw-safe task", session_key="s1")
    item = PendingTurnItem(text="follow-up", session_key="s1", raw=RawBomb())
    reg.attach_followup(task.task_id, item)

    snap = build_snapshot(reg, session_key="s1")
    assert snap.counts["followups_pending"] == 1
    json.dumps(snap.to_dict(), allow_nan=False)
    copy.deepcopy(snap.to_dict())
    assert "1 follow-up queued" in format_tasks(snap)


def test_worker_with_non_json_safe_metadata_degrades_without_raising():
    lane = ThreadWorkerLane(runner=lambda spec, token: "ok")
    handle = lane.start(WorkerSpec(goal="metadata unsafe", task_id="task-x", metadata={"ok": "yes"}))
    # Mutate after validation to simulate a future/foreign lane with unsafe metadata.
    lane._workers[handle.worker_id].spec.metadata["bad"] = object()  # noqa: SLF001
    assert lane.wait(handle.worker_id, timeout=TIMEOUT)

    text = format_agents(lane)
    assert handle.worker_id in text
    # Goal lookup through snapshot may be dropped, but formatting must not raise.


def test_natural_language_status_query_helper_korean_and_english():
    for text in (
        "지금 뭐 하고 있어?",
        "돌고 있는 작업 있어?",
        "에이전트 뭐 돌아가?",
        "내가 봐야 하는 거 있어?",
        "what are you working on?",
        "any agents running?",
        "show tasks",
        "anything blocked?",
    ):
        assert looks_like_orchestration_status_query(text), text
    for text in ("부대찌개 칼로리 알려줘", "이것도 반영해줘", "stop", ""):
        assert not looks_like_orchestration_status_query(text), text


def test_formatter_facade_matches_module_functions():
    reg = TaskRegistry()
    reg.create_task("facade", session_key="s1", status=STATUS_ERROR)
    snap = OrchestrationStatusFormatter.snapshot(reg, session_key="s1")
    assert isinstance(snap, OrchestrationSnapshot)
    assert OrchestrationStatusFormatter.format_tasks(snap) == format_tasks(snap)
    assert OrchestrationStatusFormatter.format_agents(snap) == format_agents(snap)
    assert OrchestrationStatusFormatter.format_overview(snap) == format_overview(snap)


def test_plain_task_dict_inputs_are_normalized_to_json_safe_snapshot():
    snap = build_snapshot([
        {
            "task_id": "task-dict",
            "status": STATUS_RUNNING,
            "goal": object(),
            "worker_id": object(),
            "worker_kind": object(),
            "session_key": object(),
            "followups": [object(), object()],
            "notes": {"nested": object()},
            "artifacts": object(),
            "active": object(),
            "blocked": object(),
            "error": object(),
        }
    ])

    task = snap.tasks[0]
    assert task["task_id"] == "task-dict"
    assert task["goal"] == ""
    assert task["worker_id"] is None
    assert task["worker_kind"] is None
    assert task["session_key"] is None
    assert task["followups"] == 2
    assert task["notes"] == 1
    assert task["artifacts"] == 0
    assert task["active"] is True
    json.dumps(snap.to_dict(), allow_nan=False)


def test_formatting_accepts_plain_worker_dicts_and_lists():
    workers = [
        {
            "worker_id": "worker-a",
            "status": WorkerStatus.RUNNING,
            "lane": "review",
            "task_id": "task-a",
            "goal": "review the implementation",
            "cancel_requested": False,
        }
    ]
    text = format_agents(workers)
    assert "worker-a [running] lane=review task=task-a" in text
    assert 'goal="review the implementation"' in text

    tasks = [
        {
            "task_id": "task-a",
            "status": STATUS_CANCELLED,
            "goal": "cancelled task",
            "active": False,
            "followups": 0,
            "notes": 0,
            "artifacts": 0,
            "blocked": False,
            "error": False,
        }
    ]
    assert "Tasks:" in format_tasks(tasks)
