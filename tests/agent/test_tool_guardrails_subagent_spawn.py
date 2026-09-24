"""Tests for transactional subagent spawn accounting (#72550).

The guardrail reserves after normalisation and charges only after child
construction succeeds. This fixes four bugs:

1. Batch crosses the cap boundary
2. JSON-string batches are under-counted
3. Rejected batches consume budget
4. Failed/partial child construction consumes budget or leaks children
"""

import json
import threading
from types import SimpleNamespace

import pytest

from agent.tool_guardrails import (
    LoopCapConfig,
    ToolCallGuardrailConfig,
    ToolCallGuardrailController,
    _set_active_subagent_guardrail,
    reserve_subagent_spawns,
)


# ── helpers ────────────────────────────────────────────────────────────


def _controller(cap):
    ctrl = ToolCallGuardrailController(
        ToolCallGuardrailConfig(loop_caps=LoopCapConfig(max_subagents=cap))
    )
    _set_active_subagent_guardrail(ctrl)
    return ctrl


def _json_batch(n):
    return json.dumps([{"goal": f"inspect task {i}"} for i in range(n)])


def _commit(count):
    with reserve_subagent_spawns(count) as reservation:
        return reservation.commit()


# ── reservation/commit core ─────────────────────────────────────────────


def test_before_call_checks_cap_without_charging():
    """before_call blocks at cap but does NOT increment the counter."""
    ctrl = _controller(3)
    for i in range(3):
        assert ctrl.before_call("delegate_task", {"goal": str(i)}).action == "allow"
        assert ctrl._turn_subagent_count == 0  # still 0—not charged

    # Counter is still 0; the 4th call passes the cap check too.
    assert ctrl.before_call("delegate_task", {"goal": "4"}).action == "allow"


def test_commit_charges_after_normalisation():
    """A committed reservation increments the counter; before_call then blocks."""
    ctrl = _controller(3)
    _commit(3)
    assert ctrl._turn_subagent_count == 3
    decision = ctrl.before_call("delegate_task", {"goal": "x"})
    assert decision.action == "block"
    assert decision.code == "loop_subagent_cap"


def test_reservation_caps_at_remaining_budget():
    """A reservation never admits more than max_subagents."""
    ctrl = _controller(2)
    with reserve_subagent_spawns(60) as reservation:
        assert reservation.count == 2
        reservation.commit()
    assert ctrl._turn_subagent_count == 2


# ── bug 1: batch boundary ───────────────────────────────────────────────


def test_batch_does_not_cross_cap_boundary():
    """A batch does NOT let count exceed cap — commit caps it and returns charged count."""
    ctrl = _controller(2)
    charged = _commit(3)
    assert ctrl._turn_subagent_count == 2  # capped at 2, not 3
    assert charged == 2  # only 2 were charged


def test_remaining_budget_after_partial_commit():
    """After a capped commit, the budget is fully consumed."""
    ctrl = _controller(3)
    _commit(2)
    assert ctrl._turn_subagent_count == 2
    # 1 remaining
    _commit(2)  # only 1 more fits
    assert ctrl._turn_subagent_count == 3


# ── bug 2: JSON-string batches ──────────────────────────────────────────


def test_json_string_batch_commits_correct_count():
    """JSON-string tasks commit the actual parsed count, not 1."""
    ctrl = _controller(5)
    # delegate_tool would parse this into 3 tasks and commit 3
    _commit(3)
    assert ctrl._turn_subagent_count == 3


def test_json_string_batch_hits_cap():
    """A large JSON-string batch respects the cap."""
    ctrl = _controller(2)
    _commit(5)  # parsed from JSON, capped at 2
    assert ctrl._turn_subagent_count == 2
    assert ctrl.before_call("delegate_task", {"goal": "x"}).action == "block"


# ── bug 3: rejected batches ─────────────────────────────────────────────


def test_rejected_batch_does_not_consume_budget():
    """When delegate_tool rejects (no commit call), budget is untouched."""
    ctrl = _controller(3)
    # Simulate: oversized batch rejected by delegate_tool — no commit call
    # Budget unchanged
    assert ctrl.before_call("delegate_task", {"goal": "valid"}).action == "allow"
    _commit(1)
    assert ctrl._turn_subagent_count == 1
    # Still 2 more allowed
    assert ctrl.before_call("delegate_task", {"goal": "still_ok"}).action == "allow"
    _commit(1)
    _commit(1)


def test_oversized_then_valid_both_allowed():
    """Oversized (rejected) + valid call: the valid call still gets budget."""
    ctrl = _controller(5)
    # Simulate oversized call: before_call returns allow, but delegate_tool
    # rejects and never calls commit.  Budget unchanged.
    assert ctrl.before_call("delegate_task", {"tasks": [{"goal": str(i)} for i in range(60)]}).action == "allow"
    # No commit happened → budget still 0
    assert ctrl._turn_subagent_count == 0
    # Valid call proceeds
    assert ctrl.before_call("delegate_task", {"goal": "real"}).action == "allow"
    _commit(1)
    assert ctrl._turn_subagent_count == 1


# ── reset ────────────────────────────────────────────────────────────────


def test_reset_clears_count():
    ctrl = _controller(5)
    _commit(4)
    assert ctrl._turn_subagent_count == 4
    ctrl.reset_for_turn()
    assert ctrl._turn_subagent_count == 0
    assert ctrl.before_call("delegate_task", {"goal": "fresh"}).action == "allow"


# ── boundary: cap = 1 ───────────────────────────────────────────────────


def test_cap_one_blocks_after_first_commit():
    ctrl = _controller(1)
    assert ctrl.before_call("delegate_task", {"goal": "only"}).action == "allow"
    _commit(1)
    assert ctrl._turn_subagent_count == 1
    assert ctrl.before_call("delegate_task", {"goal": "nope"}).action == "block"


def test_fully_rejected_call_names_dropped_goals():
    ctrl = _controller(1)
    _commit(1)
    decision = ctrl.before_call("delegate_task", {
        "tasks": [{"goal": "audit logs safely"}, {"goal": "clean cache safely"}],
    })

    assert decision.action == "block"
    assert "audit logs safely" in decision.message
    assert "clean cache safely" in decision.message


# ── web_search untouched ─────────────────────────────────────────────────


def test_web_search_cap_still_works():
    """The reservation/commit change does not affect web_search loop cap."""
    ctrl = ToolCallGuardrailController(
        ToolCallGuardrailConfig(loop_caps=LoopCapConfig(max_web_searches=2))
    )
    _set_active_subagent_guardrail(ctrl)
    assert ctrl.before_call("web_search", {"query": "q1"}).action == "allow"
    assert ctrl.before_call("web_search", {"query": "q2"}).action == "allow"
    decision = ctrl.before_call("web_search", {"query": "q3"})
    assert decision.action == "block"
    assert decision.code == "loop_web_search_cap"



def test_commit_returns_correct_charged_count():
    """Reservation commit returns the number actually charged."""
    ctrl = _controller(3)
    assert _commit(1) == 1
    assert _commit(5) == 2  # only 2 remaining


def test_return_value_allows_caller_to_trim_tasks():
    """The caller can use the return value to skip excess child construction."""
    ctrl = _controller(2)
    task_list = list(range(5))
    with reserve_subagent_spawns(len(task_list)) as reservation:
        charged = reservation.count
        reservation.commit(charged)
    # Only create the charged number of children
    created = task_list[:charged]
    assert len(created) == 2
    assert len(created) < len(task_list)


def test_uncommitted_reservation_rolls_back_without_charging():
    ctrl = _controller(2)
    with reserve_subagent_spawns(2) as reservation:
        assert reservation.count == 2
        assert ctrl._turn_subagent_reserved_count == 2

    assert ctrl._turn_subagent_reserved_count == 0
    assert ctrl._turn_subagent_count == 0
    assert ctrl.before_call("delegate_task", {"goal": "budget remains"}).action == "allow"


def test_thread_local_reservations_isolate_parent_and_worker_budgets():
    parent = _controller(3)
    worker_result = []

    def run_worker():
        worker = _controller(4)
        worker_result.append((_commit(2), worker._turn_subagent_count))

    thread = threading.Thread(target=run_worker)
    thread.start()
    thread.join()
    assert worker_result == [(2, 2)]
    assert parent._turn_subagent_count == 0
    assert _commit(1) == 1
    assert parent._turn_subagent_count == 1


def _patch_delegate_runtime(monkeypatch):
    from tools import delegate_tool

    monkeypatch.setattr(delegate_tool, "_load_config", lambda: {})
    monkeypatch.setattr(delegate_tool, "_get_max_spawn_depth", lambda: 2)
    monkeypatch.setattr(delegate_tool, "_get_max_concurrent_children", lambda: 5)
    monkeypatch.setattr(delegate_tool, "_oneshot_spawn_budget", lambda *_args: None)
    monkeypatch.setattr(delegate_tool, "_announce_batch", lambda *_args: None)
    monkeypatch.setattr(delegate_tool, "_capture_origin", lambda: ("", "", None, None, False))
    monkeypatch.setattr(delegate_tool, "_resolve_delegation_credentials", lambda *_args, **_kwargs: {
        "model": "test-model", "provider": None, "base_url": None,
        "api_key": None, "api_mode": None, "command": None, "args": None,
    })
    monkeypatch.setattr(
        "tools.delegation_live_log.create_live_transcripts",
        lambda task_list, *_args, **_kwargs: (None, [None] * len(task_list), []),
    )
    return delegate_tool


class _BuiltChild:
    def __init__(self):
        self.tool_progress_callback = None
        self.closed = False

    def close(self):
        self.closed = True


def _parent():
    return SimpleNamespace(
        session_id="parent", _delegate_depth=0, _active_children=[],
        _active_children_lock=threading.Lock(),
    )


def test_delegate_construction_failure_rolls_back_budget_and_cleans_partial_child(monkeypatch):
    """Real delegate_task path: a partial batch failure charges zero and closes the built child."""
    from tools.delegate_tool_child_run import _attach_child

    delegate_tool = _patch_delegate_runtime(monkeypatch)
    ctrl = _controller(2)
    parent = _parent()
    partial = _BuiltChild()
    calls = 0

    def fail_second(**kwargs):
        nonlocal calls
        assert kwargs["defer_spawn_lifecycle"] is True
        calls += 1
        if calls == 1:
            _attach_child(kwargs["parent_agent"], partial)
            return partial
        raise ValueError("synthetic child construction failure")

    monkeypatch.setattr(delegate_tool, "_build_child_preserving_parent_tools", fail_second)
    failed = json.loads(delegate_tool.delegate_task(
        tasks=[{"goal": "inspect first child"}, {"goal": "inspect second child"}],
        parent_agent=parent,
    ))

    assert failed["error"] == "synthetic child construction failure"
    assert ctrl._turn_subagent_count == 0
    assert ctrl._turn_subagent_reserved_count == 0
    assert partial.closed is True
    assert parent._active_children == []

    monkeypatch.setattr(delegate_tool, "_build_child_preserving_parent_tools", lambda **_kwargs: _BuiltChild())
    monkeypatch.setattr(delegate_tool, "_run_batch", lambda batch, _background: json.dumps({
        "built_goals": [task["goal"] for task in batch.task_list],
    }))
    admitted = json.loads(delegate_tool.delegate_task(
        tasks=[{"goal": "retry first child"}, {"goal": "retry second child"}],
        parent_agent=parent,
    ))
    assert admitted["built_goals"] == ["retry first child", "retry second child"]
    assert ctrl._turn_subagent_count == 2


def test_json_batch_boundary_builds_and_charges_only_admitted_tasks(monkeypatch):
    delegate_tool = _patch_delegate_runtime(monkeypatch)
    ctrl = _controller(2)
    built = []

    def build(**kwargs):
        built.append(kwargs["goal"])
        return _BuiltChild()

    monkeypatch.setattr(delegate_tool, "_build_child_preserving_parent_tools", build)
    monkeypatch.setattr(delegate_tool, "_run_batch", lambda batch, _background: json.dumps({
        "goals": [task["goal"] for task in batch.task_list],
        "rejected_tasks": batch.rejected_tasks,
    }))
    result = json.loads(delegate_tool.delegate_task(
        tasks=_json_batch(3),  # type: ignore[arg-type] - production recovers JSON-string batches
        parent_agent=_parent(),
    ))

    assert built == ["inspect task 0", "inspect task 1"]
    assert result["goals"] == built
    assert [entry["goal"] for entry in result["rejected_tasks"]] == ["inspect task 2"]
    assert ctrl._turn_subagent_count == 2
    assert ctrl._turn_subagent_reserved_count == 0
