"""Tests for Executive v2 Phase 3 Planner."""

from __future__ import annotations

import pytest

from agent.executive.planner import (
    compute_plan_fingerprint,
    decompose_goal_to_subgoals,
    map_subgoals_to_task_specs,
)
from agent.executive.types import PlannerSubgoal


def _make_contract(
    *,
    success_criteria: tuple[str, ...] = ("Information is gathered", "Sources cited"),
    approval_requirements: tuple[dict, ...] = (),
    hard: tuple[str, ...] = (),
    soft: tuple[str, ...] = (),
    risk_score: float = 0.1,
    max_iterations: int = 24,
    max_duration_minutes: int = 60,
) -> dict:
    return {
        "success_criteria": list(success_criteria),
        "approval_requirements": list(approval_requirements),
        "hard_constraints": list(hard),
        "soft_constraints": list(soft),
        "risk_score": risk_score,
        "budget": {
            "max_iterations": max_iterations,
            "max_duration_minutes": max_duration_minutes,
        },
    }


# ── Section 1: decompose_goal_to_subgoals (6 tests) ────────────────

def test_decompose_research_objective():
    subgoals = decompose_goal_to_subgoals(
        "OBJECTIVE: investigate X",
        goal_contract=None,
        execution_contract=_make_contract(
            success_criteria=("Investiga información sobre X",),
        ),
    )
    assert len(subgoals) == 1
    sg = subgoals[0]
    assert sg.id == "sg-0"
    assert sg.intent == "RESEARCH"
    assert sg.risk_level == "low"
    assert sg.approval_required is False


def test_decompose_strategic_objective_high_risk():
    subgoals = decompose_goal_to_subgoals(
        "OBJECTIVE: consigue bancarizar productos",
        goal_contract=None,
        execution_contract=_make_contract(
            success_criteria=("Consigue bancarizar los productos Onion",),
            risk_score=0.85,
            approval_requirements=(
                {"gate": "EXPLICIT_STRATEGIC", "approver": "user", "ttl_hours": 24},
            ),
        ),
        risk_score=0.85,
    )
    assert len(subgoals) == 1
    sg = subgoals[0]
    assert sg.intent == "STRATEGIC"
    assert sg.risk_level == "high"
    assert sg.approval_required is True


def test_decompose_build_objective():
    subgoals = decompose_goal_to_subgoals(
        "OBJECTIVE: implementa una API REST",
        goal_contract=None,
        execution_contract=_make_contract(
            success_criteria=("Implementa la API REST completamente",),
        ),
    )
    assert len(subgoals) == 1
    assert subgoals[0].intent == "BUILD"


def test_decompose_empty_success_criteria_returns_empty_list():
    subgoals = decompose_goal_to_subgoals(
        "OBJECTIVE: x",
        goal_contract=None,
        execution_contract=_make_contract(success_criteria=()),
    )
    assert subgoals == []


def test_decompose_truncates_to_max_subgoals():
    subgoals = decompose_goal_to_subgoals(
        "OBJECTIVE: x",
        goal_contract=None,
        execution_contract=_make_contract(
            success_criteria=tuple(f"Criterion {i}" for i in range(10)),
        ),
        max_subgoals=8,
    )
    # Test reorders; just check count.
    assert len(subgoals) <= 8


def test_decompose_invariant_under_repeated_call():
    contract = _make_contract(
        success_criteria=("a", "b", "c"),
        risk_score=0.3,
    )
    r1 = decompose_goal_to_subgoals("t", None, contract, risk_score=0.3)
    r2 = decompose_goal_to_subgoals("t", None, contract, risk_score=0.3)
    assert len(r1) == len(r2)
    # Note: created_at may differ between PlannerSubgoal instances if it
    # were used; we don't use it, so this should be deterministic.
    ids1 = [s.id for s in r1]
    ids2 = [s.id for s in r2]
    assert ids1 == ids2


# ── Section 2: map_subgoals_to_task_specs (4 tests) ───────────────

def test_map_subgoals_preserves_order():
    subgoals = [
        PlannerSubgoal(
            id="sg-0", title="A", intent="RESEARCH", constraints=(),
            expected_output="a", risk_level="low", approval_required=False,
            estimated_iterations=1, timeout_seconds=60, source_criterion_index=0,
        ),
        PlannerSubgoal(
            id="sg-1", title="B", intent="BUILD", constraints=(),
            expected_output="b", risk_level="low", approval_required=False,
            estimated_iterations=1, timeout_seconds=60, source_criterion_index=1,
        ),
    ]
    specs = map_subgoals_to_task_specs(subgoals)
    assert len(specs) == 2
    assert specs[0].description == "A"
    assert specs[1].description == "B"


def test_map_task_ids_empty():
    subgoals = [
        PlannerSubgoal(
            id="sg-0", title="A", intent="RESEARCH", constraints=(),
            expected_output="a", risk_level="low", approval_required=False,
            estimated_iterations=1, timeout_seconds=60, source_criterion_index=0,
        ),
    ]
    specs = map_subgoals_to_task_specs(subgoals)
    # Phase 3 = preview; task_id is empty.
    assert all(s.task_id == "" for s in specs)


def test_map_dependencies_empty():
    subgoals = [
        PlannerSubgoal(
            id="sg-0", title="A", intent="RESEARCH", constraints=(),
            expected_output="a", risk_level="low", approval_required=False,
            estimated_iterations=1, timeout_seconds=60, source_criterion_index=0,
        ),
    ]
    specs = map_subgoals_to_task_specs(subgoals)
    # Phase 3 = linear; dependencies are empty.
    assert all(len(s.dependencies) == 0 for s in specs)


def test_map_invariant_under_repeated_call():
    subgoals = [
        PlannerSubgoal(
            id="sg-0", title="A", intent="RESEARCH", constraints=(),
            expected_output="a", risk_level="low", approval_required=False,
            estimated_iterations=1, timeout_seconds=60, source_criterion_index=0,
        ),
    ]
    s1 = map_subgoals_to_task_specs(subgoals)
    s2 = map_subgoals_to_task_specs(subgoals)
    assert [s.description for s in s1] == [s.description for s in s2]


# ── Section 3: compute_plan_fingerprint (2 tests) ────────────────

def test_plan_fingerprint_stable():
    subgoals = [
        PlannerSubgoal(
            id="sg-0", title="A", intent="RESEARCH", constraints=(),
            expected_output="a", risk_level="low", approval_required=False,
            estimated_iterations=1, timeout_seconds=60, source_criterion_index=0,
        ),
    ]
    specs = map_subgoals_to_task_specs(subgoals)
    fp1 = compute_plan_fingerprint("oid-1", subgoals, specs)
    fp2 = compute_plan_fingerprint("oid-1", subgoals, specs)
    assert fp1 == fp2
    assert len(fp1) == 64


def test_plan_fingerprint_changes_with_subgoals():
    s1 = [
        PlannerSubgoal(
            id="sg-0", title="A", intent="RESEARCH", constraints=(),
            expected_output="a", risk_level="low", approval_required=False,
            estimated_iterations=1, timeout_seconds=60, source_criterion_index=0,
        ),
    ]
    s2 = [
        PlannerSubgoal(
            id="sg-0", title="B", intent="RESEARCH", constraints=(),
            expected_output="b", risk_level="low", approval_required=False,
            estimated_iterations=1, timeout_seconds=60, source_criterion_index=0,
        ),
    ]
    fp1 = compute_plan_fingerprint("oid-1", s1, map_subgoals_to_task_specs(s1))
    fp2 = compute_plan_fingerprint("oid-1", s2, map_subgoals_to_task_specs(s2))
    assert fp1 != fp2


# ── Section 4: classification helpers (2 tests) ───────────────────

def test_classify_intent_research():
    from agent.executive.planner import _classify_intent
    assert _classify_intent("investigate new technology") == "RESEARCH"
    assert _classify_intent("research the market") == "RESEARCH"


def test_classify_intent_strategic_tie_breaker():
    from agent.executive.planner import _classify_intent
    # "consigue" is in STRATEGIC keywords.
    assert _classify_intent("consigue bancarizar productos") == "STRATEGIC"
