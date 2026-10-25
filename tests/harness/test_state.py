"""State contracts: lock behavior and JSON round-trips."""

import pytest

from harness.state import (
    TERMINAL_STATUSES,
    FeatureLock,
    FeatureState,
    FeatureStatus,
    ScopeReason,
    ScopeRejected,
    Task,
    TaskStatus,
)


def _feature(fid="f1", task="t1"):
    return FeatureState(id=fid, task_id=task, name=fid)


def test_first_selection_needs_no_reason():
    lock = FeatureLock()
    assert lock.select(_feature()).id == "f1"
    assert lock.active is not None and lock.active.id == "f1"


def test_unjustified_switch_is_rejected():
    lock = FeatureLock()
    lock.select(_feature("f1"))
    with pytest.raises(ScopeRejected):
        lock.select(_feature("f2"), reason="because-i-said-so")


def test_justified_switch_records_transition():
    lock = FeatureLock()
    lock.select(_feature("f1"))
    lock.select(
        _feature("f2"), reason=ScopeReason.FEATURE_COMPLETE, evidence="tests pass"
    )
    assert lock.active is not None and lock.active.id == "f2"
    assert lock.transitions[-1]["reason"] == ScopeReason.FEATURE_COMPLETE


def test_task_round_trip_preserves_budget():
    task = Task(id="t1", goal="fix it", success_criteria=["tests pass"])
    clone = Task.from_dict(task.to_dict())
    assert clone.goal == "fix it"
    assert clone.budget.max_tool_calls == task.budget.max_tool_calls
    assert clone.status == TaskStatus.CREATED


def test_terminal_statuses_cover_failure_modes():
    assert {"COMPLETED", "FAILED", "BUDGET_EXHAUSTED", "STOPPED"} <= TERMINAL_STATUSES


def test_feature_status_defaults_to_pending():
    assert _feature().status == FeatureStatus.PENDING
