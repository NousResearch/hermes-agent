"""Tests for Phase 5 Worker Mapping — pure kanban_task -> TaskState + WorkerRegistryEntry.

8 tests covering:
* kanban_task_to_task_state (3 tests)
* worker_registry_from_kanban_tasks (3 tests)
* build_batch_inputs (1 test)
* derive_restrictions (1 test)
"""

from __future__ import annotations

from agent.executive.worker_mapping import (
    kanban_task_to_task_state,
    worker_registry_from_kanban_tasks,
    build_batch_inputs,
    compute_dispatch_fingerprint,
    derive_restrictions,
)
from agent.executive.types import RiskLevel, PolicyDecision


class _FakeTask:
    def __init__(self, id, status="ready", assignee="anthropic", skills=None,
                 consecutive_failures=0, started_at=None):
        self.id = id
        self.status = status
        self.assignee = assignee  # type: ignore[assignment]
        self.skills = skills
        self.consecutive_failures = consecutive_failures
        self.started_at = started_at


# ── kanban_task_to_task_state (3 tests) ────────────────────────


def test_map_status_ready():
    t = _FakeTask(id="t-1", status="ready")
    s = kanban_task_to_task_state(t)
    assert s["task_id"] == "t-1"
    assert s["state"] == "ready"
    assert s["last_worker_id"] == "anthropic"
    assert s["failure_count"] == 0


def test_map_status_running():
    t = _FakeTask(id="t-2", status="running", consecutive_failures=2, started_at=12345)
    s = kanban_task_to_task_state(t)
    assert s["state"] == "running"
    assert s["failure_count"] == 2
    assert s["started_at"] == 12345


def test_map_status_done():
    t = _FakeTask(id="t-3", status="done")
    s = kanban_task_to_task_state(t)
    assert s["state"] == "done"


# ── worker_registry_from_kanban_tasks (3 tests) ────────────────


def test_worker_registry_unique_assignees():
    tasks = [
        _FakeTask(id="t-1", assignee="anthropic"),
        _FakeTask(id="t-2", assignee="minimax"),
        _FakeTask(id="t-3", assignee="anthropic"),  # duplicate
    ]
    workers = worker_registry_from_kanban_tasks(tasks)
    assert len(workers) == 2
    ids = {w["worker_id"] for w in workers}
    assert ids == {"anthropic", "minimax"}


def test_worker_registry_skips_unassigned():
    tasks = [
        _FakeTask(id="t-1", assignee=None),
        _FakeTask(id="t-2", assignee="anthropic"),
    ]
    workers = worker_registry_from_kanban_tasks(tasks)
    assert len(workers) == 1
    assert workers[0]["worker_id"] == "anthropic"


def test_worker_registry_includes_per_task_skills():
    tasks = [
        _FakeTask(id="t-1", assignee="anthropic", skills=["research", "code"]),
    ]
    workers = worker_registry_from_kanban_tasks(tasks)
    assert workers[0]["capabilities"] == ["anthropic", "research", "code"]


# ── build_batch_inputs (1 test) ────────────────────────────────


def test_build_batch_inputs_preserves_order():
    tasks = [
        _FakeTask(id="t-2", assignee="minimax"),
        _FakeTask(id="t-1", assignee="anthropic"),
        _FakeTask(id="t-3", assignee="minimax"),
    ]
    task_ids = ("t-1", "t-2", "t-3")
    task_states, workers = build_batch_inputs(task_ids, kanban_tasks=tasks)
    assert [s["task_id"] for s in task_states] == ["t-1", "t-2", "t-3"]
    # Workers: unique assignees (anthropic, minimax) in any order.
    assert len(workers) == 2
    assert {w["worker_id"] for w in workers} == {"anthropic", "minimax"}


# ── derive_restrictions (1 test) ───────────────────────────────


def test_derive_restrictions_r6_does_not_add_no_external():
    pd = PolicyDecision(
        objective_id="obj-1",
        risk_level=RiskLevel.R6,
        allowed_actions=(),
        forbidden_actions=(),
        approval_required=True,
        warnings=(),
        approval_requirements=(),
        risk_score=0.9,
        risk_components={},
        created_at="2026-07-02T00:00:00Z",
        decision_fingerprint="d1",
    )
    restrictions = derive_restrictions(pd)
    # R6 with approval_required=True: no "no_external" added.
    assert "no_external" not in restrictions
    # And since approval_required=True, "autonomous_only" not added.
    assert "autonomous_only" not in restrictions


def test_compute_dispatch_fingerprint_stable():
    """Same inputs produce the same fingerprint."""
    fp1 = compute_dispatch_fingerprint(
        task_ids=("t-1", "t-2"),
        restrictions=frozenset({"a", "b"}),
        decision_fingerprint="d1",
        request_fingerprint="r1",
        kanban_apply_fingerprint="k1",
    )
    fp2 = compute_dispatch_fingerprint(
        task_ids=("t-2", "t-1"),  # order swapped
        restrictions=frozenset({"b", "a"}),  # order swapped
        decision_fingerprint="d1",
        request_fingerprint="r1",
        kanban_apply_fingerprint="k1",
    )
    # Sorted inputs -> same fingerprint.
    assert fp1 == fp2
    assert fp1.startswith("sha256:")
