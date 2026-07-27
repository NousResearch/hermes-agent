"""Tests for subagent spawn reservation/commit pattern (#72550).

The guardrail now defers charging to ``commit_subagent_spawn()``, which
delegate_tool calls after normalisation (JSON-string recovery +
max_concurrent_children validation).  This fixes three bugs:

1. Batch crosses the cap boundary
2. JSON-string batches are under-counted
3. Rejected batches consume budget
"""

import json

import pytest

from agent.tool_guardrails import (
    LoopCapConfig,
    ToolCallGuardrailConfig,
    ToolCallGuardrailController,
    _set_active_subagent_guardrail,
    commit_subagent_spawn,
)


# ── helpers ────────────────────────────────────────────────────────────


def _controller(cap):
    ctrl = ToolCallGuardrailController(
        ToolCallGuardrailConfig(loop_caps=LoopCapConfig(max_subagents=cap))
    )
    _set_active_subagent_guardrail(ctrl)
    return ctrl


def _json_batch(n):
    return json.dumps([{"goal": str(i)} for i in range(n)])


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
    """commit_subagent_spawn increments the counter; before_call then blocks."""
    ctrl = _controller(3)
    commit_subagent_spawn(3)
    assert ctrl._turn_subagent_count == 3
    decision = ctrl.before_call("delegate_task", {"goal": "x"})
    assert decision.action == "block"
    assert decision.code == "loop_subagent_cap"


def test_commit_caps_at_remaining_budget():
    """commit_subagent_spawn never exceeds max_subagents."""
    ctrl = _controller(2)
    commit_subagent_spawn(60)  # oversized
    assert ctrl._turn_subagent_count == 2


# ── bug 1: batch boundary ───────────────────────────────────────────────


def test_batch_does_not_cross_cap_boundary():
    """A batch does NOT let count exceed cap — commit caps it."""
    ctrl = _controller(2)
    commit_subagent_spawn(3)
    assert ctrl._turn_subagent_count == 2  # capped at 2, not 3


def test_remaining_budget_after_partial_commit():
    """After a capped commit, the budget is fully consumed."""
    ctrl = _controller(3)
    commit_subagent_spawn(2)
    assert ctrl._turn_subagent_count == 2
    # 1 remaining
    commit_subagent_spawn(2)  # only 1 more fits
    assert ctrl._turn_subagent_count == 3


# ── bug 2: JSON-string batches ──────────────────────────────────────────


def test_json_string_batch_commits_correct_count():
    """JSON-string tasks commit the actual parsed count, not 1."""
    ctrl = _controller(5)
    # delegate_tool would parse this into 3 tasks and commit 3
    commit_subagent_spawn(3)
    assert ctrl._turn_subagent_count == 3


def test_json_string_batch_hits_cap():
    """A large JSON-string batch respects the cap."""
    ctrl = _controller(2)
    commit_subagent_spawn(5)  # parsed from JSON, capped at 2
    assert ctrl._turn_subagent_count == 2
    assert ctrl.before_call("delegate_task", {"goal": "x"}).action == "block"


# ── bug 3: rejected batches ─────────────────────────────────────────────


def test_rejected_batch_does_not_consume_budget():
    """When delegate_tool rejects (no commit call), budget is untouched."""
    ctrl = _controller(3)
    # Simulate: oversized batch rejected by delegate_tool — no commit call
    # Budget unchanged
    assert ctrl.before_call("delegate_task", {"goal": "valid"}).action == "allow"
    commit_subagent_spawn(1)
    assert ctrl._turn_subagent_count == 1
    # Still 2 more allowed
    assert ctrl.before_call("delegate_task", {"goal": "still_ok"}).action == "allow"
    commit_subagent_spawn(1)
    commit_subagent_spawn(1)


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
    commit_subagent_spawn(1)
    assert ctrl._turn_subagent_count == 1


# ── reset ────────────────────────────────────────────────────────────────


def test_reset_clears_count():
    ctrl = _controller(5)
    commit_subagent_spawn(4)
    assert ctrl._turn_subagent_count == 4
    ctrl.reset_for_turn()
    assert ctrl._turn_subagent_count == 0
    assert ctrl.before_call("delegate_task", {"goal": "fresh"}).action == "allow"


# ── boundary: cap = 1 ───────────────────────────────────────────────────


def test_cap_one_blocks_after_first_commit():
    ctrl = _controller(1)
    assert ctrl.before_call("delegate_task", {"goal": "only"}).action == "allow"
    commit_subagent_spawn(1)
    assert ctrl._turn_subagent_count == 1
    assert ctrl.before_call("delegate_task", {"goal": "nope"}).action == "block"


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
