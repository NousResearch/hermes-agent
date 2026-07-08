"""Pilot G3 — decompose_goal_to_subgoals consumes a pilot ExecutionContract.

The planner is a pure function: goal_text + GoalContract + execution_contract
dict -> list[PlannerSubgoal]. Pilot confirms the chain produces one
subgoal per success_criterion, gated by the evidence-pack prefix when
present.
"""

from __future__ import annotations

import pytest

from agent.executive.planner import decompose_goal_to_subgoals


class _StubGoalContract:
    """Minimal GoalContract stand-in for hermetic unit tests."""

    def __init__(self, goal_id: str, title: str, success_criteria):
        self.goal_id = goal_id
        self.title = title
        self.success_criteria = tuple(success_criteria)


def test_g3_one_subgoal_per_success_criterion():
    goal_contract = _StubGoalContract(
        goal_id="g-pilot-001",
        title="pilot canary",
        success_criteria=("alpha", "beta", "gamma"),
    )
    execution_contract = {
        "success_criteria": ["alpha", "beta", "gamma"],
        "approval_requirements": [],
        "hard_constraints": [],
        "soft_constraints": [],
        "budget": {"max_iterations": 3, "max_duration_minutes": 5},
        "evidence_pack_summary": "",  # pilot: no gating
    }
    subgoals = decompose_goal_to_subgoals(
        "pilot canary goal",
        goal_contract,
        execution_contract,
    )
    assert len(subgoals) == 3
    assert [s.title for s in subgoals] == ["alpha", "beta", "gamma"]


def test_g3_evidence_pack_gating_appends_marker():
    """When execution_contract.evidence_pack_summary starts with the
    gating prefix, every subgoal gets a [GATED: ...] marker.
    """
    goal_contract = _StubGoalContract(
        goal_id="g-pilot-002",
        title="pilot canary gated",
        success_criteria=("alpha",),
    )
    execution_contract = {
        "success_criteria": ["alpha"],
        "approval_requirements": [],
        "hard_constraints": [],
        "soft_constraints": [],
        "budget": {},
        "evidence_pack_summary": "[REQUIRES_HUMAN] needs review",
    }
    subgoals = decompose_goal_to_subgoals(
        "pilot canary goal gated",
        goal_contract,
        execution_contract,
    )
    assert len(subgoals) == 1
    assert "[GATED:" in subgoals[0].expected_output


def test_g3_empty_success_criteria_returns_empty():
    goal_contract = _StubGoalContract(
        goal_id="g-pilot-003",
        title="empty",
        success_criteria=(),
    )
    execution_contract = {
        "success_criteria": [],
        "approval_requirements": [],
        "hard_constraints": [],
        "soft_constraints": [],
        "budget": {},
    }
    assert decompose_goal_to_subgoals("pilot", goal_contract, execution_contract) == []


def test_g3_max_subgoals_caps_count():
    goal_contract = _StubGoalContract(
        goal_id="g-pilot-004",
        title="cap test",
        success_criteria=tuple(f"c{i}" for i in range(10)),
    )
    execution_contract = {
        "success_criteria": [f"c{i}" for i in range(10)],
        "approval_requirements": [],
        "hard_constraints": [],
        "soft_constraints": [],
        "budget": {},
    }
    subgoals = decompose_goal_to_subgoals(
        "pilot", goal_contract, execution_contract, max_subgoals=3
    )
    assert len(subgoals) == 3