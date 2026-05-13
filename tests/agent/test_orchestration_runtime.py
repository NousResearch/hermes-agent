import copy
import json
import threading

import pytest

from agent.control_plane import (
    Confidence,
    ControlPlaneDecision,
    Intent,
    Recommendation,
    Signal,
)
from agent.orchestration_runtime import (
    RUNTIME_ATTR,
    OrchestrationRuntime,
    advise_frontdesk_for_owner,
    format_runtime_agents,
    format_runtime_overview,
    format_runtime_tasks,
    get_or_create_orchestration_runtime,
    get_orchestration_runtime,
    set_orchestration_runtime,
)
from agent.orchestration_status import OrchestrationSnapshot
from agent.pending_turn_queue import PendingTurnItem
from agent.task_registry import STATUS_DONE, STATUS_RUNNING, TaskRegistry
from agent.worker_lanes import (
    CancelToken,
    ThreadWorkerLane,
    WorkerLaneRegistry,
    WorkerSpec,
    WorkerStatus,
    link_worker_to_task,
)

TIMEOUT = 2.0


class _DummyOwner:
    """Stand-in for a HermesCLI / gateway runner / session object."""


def _gated_runner():
    """A runner that blocks at ``release`` so its worker stays ``running``."""
    entered = threading.Event()
    release = threading.Event()

    def runner(spec, token: CancelToken):  # noqa: ARG001
        entered.set()
        release.wait(TIMEOUT)
        token.raise_if_cancelled()
        return "done"

    return runner, entered, release


# --------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------
def test_create_makes_fresh_empty_in_memory_registries():
    rt = OrchestrationRuntime.create()
    assert isinstance(rt.task_registry, TaskRegistry)
    assert isinstance(rt.worker_registry, WorkerLaneRegistry)
    assert len(rt.task_registry) == 0
    assert rt.task_registry.path is None  # in-memory only -- no file persistence
    assert rt.worker_registry.lane_names() == []
    # Two creates are fully independent.
    other = OrchestrationRuntime.create()
    assert other.task_registry is not rt.task_registry
    assert other.worker_registry is not rt.worker_registry


def test_explicit_registry_injection_is_used_as_is():
    tr = TaskRegistry()
    wr = WorkerLaneRegistry()
    rt = OrchestrationRuntime(task_registry=tr, worker_registry=wr)
    assert rt.task_registry is tr
    assert rt.worker_registry is wr
    tr.create_task("injected goal", session_key="s1", status=STATUS_RUNNING)
    assert "injected goal" in rt.format_tasks(session_key="s1")


# --------------------------------------------------------------------------
# Per-owner attach / fetch helpers (no global state)
# --------------------------------------------------------------------------
def test_get_set_get_or_create_helpers_on_dummy_owner():
    owner = _DummyOwner()
    assert get_orchestration_runtime(owner) is None  # nothing attached yet
    assert not hasattr(owner, RUNTIME_ATTR)

    created = get_or_create_orchestration_runtime(owner)
    assert isinstance(created, OrchestrationRuntime)
    assert getattr(owner, RUNTIME_ATTR) is created
    # Idempotent: a second get-or-create returns the same object.
    assert get_or_create_orchestration_runtime(owner) is created
    assert get_orchestration_runtime(owner) is created

    replacement = OrchestrationRuntime.create()
    assert set_orchestration_runtime(owner, replacement) is replacement
    assert get_orchestration_runtime(owner) is replacement
    assert get_or_create_orchestration_runtime(owner) is replacement


def test_set_orchestration_runtime_rejects_non_runtime():
    owner = _DummyOwner()
    with pytest.raises(TypeError):
        set_orchestration_runtime(owner, object())  # type: ignore[arg-type]
    assert not hasattr(owner, RUNTIME_ATTR)


def test_get_helpers_ignore_a_foreign_value_in_the_slot():
    owner = _DummyOwner()
    setattr(owner, RUNTIME_ATTR, "not a runtime")
    assert get_orchestration_runtime(owner) is None
    # get-or-create replaces the bad value with a real runtime.
    rt = get_or_create_orchestration_runtime(owner)
    assert isinstance(rt, OrchestrationRuntime)
    assert getattr(owner, RUNTIME_ATTR) is rt


def test_no_global_singleton_leakage_between_owners():
    a = _DummyOwner()
    b = _DummyOwner()
    ra = get_or_create_orchestration_runtime(a)
    rb = get_or_create_orchestration_runtime(b)
    assert ra is not rb
    assert ra.task_registry is not rb.task_registry
    assert ra.worker_registry is not rb.worker_registry

    ra.task_registry.create_task("only on a", session_key="s1", status=STATUS_RUNNING)
    assert "only on a" in format_runtime_tasks(a, session_key="s1")
    assert format_runtime_tasks(b, session_key="s1") == "No active tasks are currently registered."

    # A brand-new owner the helpers have never seen also stays empty.
    assert format_runtime_overview(_DummyOwner()) == "No active tasks or workers are currently registered."


# --------------------------------------------------------------------------
# Empty-state formatting is graceful
# --------------------------------------------------------------------------
def test_empty_runtime_formats_gracefully_and_snapshot_is_empty():
    rt = OrchestrationRuntime.create()
    assert rt.format_tasks() == "No active tasks are currently registered."
    assert rt.format_agents() == "No active workers are currently registered."
    assert rt.format_overview() == "No active tasks or workers are currently registered."
    snap = rt.snapshot()
    assert isinstance(snap, OrchestrationSnapshot)
    assert snap.tasks == []
    assert snap.workers == []
    assert snap.counts["tasks_total"] == 0
    assert snap.counts["workers_total"] == 0
    json.dumps(snap.to_dict(), allow_nan=False)


def test_format_runtime_helpers_create_empty_runtime_when_missing():
    owner = _DummyOwner()
    assert format_runtime_tasks(owner) == "No active tasks are currently registered."
    # The helper attached a real runtime so later work is visible through it.
    rt = get_orchestration_runtime(owner)
    assert isinstance(rt, OrchestrationRuntime)
    assert format_runtime_agents(owner) == "No active workers are currently registered."
    assert format_runtime_overview(owner) == "No active tasks or workers are currently registered."
    assert get_orchestration_runtime(owner) is rt  # still the same one


# --------------------------------------------------------------------------
# Injected task / worker state formats correctly
# --------------------------------------------------------------------------
def test_injected_task_and_worker_state_formats_through_runtime_and_helpers():
    rt = OrchestrationRuntime.create()
    task = rt.task_registry.create_task(
        "Phase 7 runtime wiring", session_key="s1", status=STATUS_RUNNING
    )
    rt.task_registry.create_task("already finished", session_key="s1", status=STATUS_DONE)

    runner, entered, release = _gated_runner()
    lane = ThreadWorkerLane(runner=runner, name="thread")
    rt.worker_registry.register(lane)
    handle = rt.worker_registry.start(
        WorkerSpec(goal="Phase 7 runtime wiring", task_id=task.task_id, lane="thread")
    )
    link_worker_to_task(rt.task_registry, task.task_id, handle, worker_kind="claude_code")
    assert entered.wait(TIMEOUT)

    try:
        # Attach this runtime to an owner and drive everything through the helpers.
        owner = _DummyOwner()
        set_orchestration_runtime(owner, rt)

        # /tasks-style board: active tasks only, worker linkage shown.
        tasks_text = format_runtime_tasks(owner, session_key="s1")
        assert "Active tasks:" in tasks_text
        assert task.task_id in tasks_text
        assert "Phase 7 runtime wiring" in tasks_text
        assert f"worker: {handle.worker_id} (claude_code)" in tasks_text
        assert "already finished" not in tasks_text  # active-only by default

        # /agents-style board: live worker status / lane / task / goal.
        agents_text = format_runtime_agents(owner)
        assert "Workers:" in agents_text or "Active workers:" in agents_text
        assert f"{handle.worker_id} [running]" in agents_text
        assert "lane=thread" in agents_text
        assert f"task={task.task_id}" in agents_text
        assert 'goal="Phase 7 runtime wiring"' in agents_text

        # Combined overview: cross-links the live worker status into the task block.
        overview_text = format_runtime_overview(owner, session_key="s1")
        assert "Orchestration status" in overview_text
        assert task.task_id in overview_text
        assert f"worker: {handle.worker_id} (claude_code) [running]" in overview_text
        assert handle.worker_id in overview_text

        snap = rt.snapshot(session_key="s1")
        assert snap.counts["tasks_total"] == 2
        assert snap.counts["tasks_active"] == 1
        assert snap.counts["workers_total"] == 1
        assert snap.counts["workers_running"] == 1
        assert snap.warnings == []  # task<->worker linkage is consistent
        json.dumps(snap.to_dict(), allow_nan=False)
    finally:
        release.set()
        lane.wait(handle.worker_id, timeout=TIMEOUT)


def test_runtime_snapshot_is_json_safe_with_a_followup_carrying_raw():
    """A follow-up with a non-JSON, non-copyable ``raw`` passthrough must not be
    inspected, serialised, or deep-copied anywhere on the runtime/snapshot path."""

    class RawBomb:
        def __deepcopy__(self, memo):  # pragma: no cover - must never be called
            raise AssertionError("raw was deep-copied")

        def __getattribute__(self, name):
            if name not in {"__class__", "__deepcopy__", "__repr__"}:
                raise AssertionError("raw was inspected")
            return object.__getattribute__(self, name)

    rt = OrchestrationRuntime.create()
    task = rt.task_registry.create_task("raw-safe task", session_key="s1", status=STATUS_RUNNING)
    rt.task_registry.attach_followup(
        task.task_id, PendingTurnItem(text="also include X", session_key="s1", raw=RawBomb())
    )

    snap = rt.snapshot(session_key="s1")
    assert snap.counts["followups_pending"] == 1
    json.dumps(snap.to_dict(), allow_nan=False)
    copy.deepcopy(snap.to_dict())

    owner = _DummyOwner()
    set_orchestration_runtime(owner, rt)
    text = format_runtime_tasks(owner, session_key="s1")
    assert "1 follow-up queued" in text


def test_runtime_format_tasks_is_session_scoped_and_active_only():
    rt = OrchestrationRuntime.create()
    a_active = rt.task_registry.create_task("active in s1", session_key="s1", status=STATUS_RUNNING)
    rt.task_registry.create_task("done in s1", session_key="s1", status=STATUS_DONE)
    rt.task_registry.create_task("active in s2", session_key="s2", status=STATUS_RUNNING)

    text_s1 = rt.format_tasks(session_key="s1")
    assert a_active.task_id in text_s1
    assert "done in s1" not in text_s1
    assert "active in s2" not in text_s1

    # No session filter: every active task across sessions, terminal ones dropped.
    text_all = rt.format_tasks()
    assert "active in s1" in text_all
    assert "active in s2" in text_all
    assert "done in s1" not in text_all


# --------------------------------------------------------------------------
# Frontdesk policy advisory (read-only pass-through)
# --------------------------------------------------------------------------
def test_advise_frontdesk_routes_a_research_artifact_request_to_worker_lane():
    rt = OrchestrationRuntime.create()
    decision = rt.advise_frontdesk(
        "investigate the regression and write a report.md with the findings",
        frontdesk_mode_active=True,
    )
    assert isinstance(decision, ControlPlaneDecision)
    assert decision.recommendation == Recommendation.WORKER_LANE
    assert decision.intent is Intent.NEW_TASK_WORKER
    assert decision.confidence == Confidence.HIGH
    assert Signal.RESEARCH in decision.signals
    assert Signal.ARTIFACT in decision.signals


def test_advise_frontdesk_keeps_short_status_query_on_main():
    rt = OrchestrationRuntime.create()
    decision = rt.advise_frontdesk("status?")
    assert decision.recommendation == Recommendation.MAIN
    assert not decision.should_delegate


def test_advise_frontdesk_does_not_mutate_runtime_state():
    rt = OrchestrationRuntime.create()
    before_tasks = len(rt.task_registry)
    before_lanes = list(rt.worker_registry.lane_names())
    rt.advise_frontdesk("delegate this in the background", frontdesk_mode_active=True)
    assert len(rt.task_registry) == before_tasks
    assert list(rt.worker_registry.lane_names()) == before_lanes


def test_advise_frontdesk_for_owner_creates_runtime_if_absent():
    owner = _DummyOwner()
    assert get_orchestration_runtime(owner) is None
    decision = advise_frontdesk_for_owner(
        owner, "draft a report.md with the audit", frontdesk_mode_active=True
    )
    assert decision.should_delegate
    rt = get_orchestration_runtime(owner)
    assert isinstance(rt, OrchestrationRuntime)
    assert advise_frontdesk_for_owner(owner, "status?").recommendation == Recommendation.MAIN


def test_advise_frontdesk_for_owner_does_not_inspect_other_owners():
    owner_a = _DummyOwner()
    owner_b = _DummyOwner()
    advise_frontdesk_for_owner(owner_a, "write a report.md")
    assert get_orchestration_runtime(owner_b) is None


def test_worker_with_non_json_safe_metadata_degrades_without_raising_through_runtime():
    rt = OrchestrationRuntime.create()
    lane = ThreadWorkerLane(runner=lambda spec, token: "ok")
    rt.worker_registry.register(lane)
    handle = rt.worker_registry.start(WorkerSpec(goal="metadata unsafe", task_id="task-x"))
    # Simulate a future/foreign lane carrying non-JSON-safe spec metadata.
    lane._workers[handle.worker_id].spec.metadata["bad"] = object()  # noqa: SLF001
    assert lane.wait(handle.worker_id, timeout=TIMEOUT)

    text = rt.format_agents()
    assert handle.worker_id in text
    # The done worker still shows up; goal lookup via snapshot may be dropped.
    statuses = {w["worker_id"]: w["status"] for w in rt.snapshot().workers}
    assert statuses[handle.worker_id] == WorkerStatus.DONE
