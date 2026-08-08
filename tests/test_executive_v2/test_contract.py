"""Tests for Executive v2 contract builder and fingerprint."""

from __future__ import annotations

import json

import pytest

from agent.executive.contract import (
    COMPLEXITY_BUDGET,
    build_execution_contract_v1,
    build_knowledge_summary_text,
    compute_risk_components,
)
from agent.executive.types import (
    BudgetPolicy,
    CapabilityCandidate,
    CapabilityDiscovery,
    ClassifiedObjective,
    Complexity,
    ExecutionContractV1,
    GoalClass,
    NormalizedObjective,
    RiskComponents,
    RiskProfile,
)


def _make_normalized(**overrides) -> NormalizedObjective:
    defaults = dict(
        objective_id="oid-1",
        goal_class=GoalClass.STRATEGIC,
        constraints=("forbidden:stripe",),
        success_criteria=("Sub-goal is delivered",),
        human_constraints=(),
        approval_requirements=(),
        risk_profile=RiskProfile.HIGH,
        estimated_complexity=Complexity.L,
        knowledge_requirements=("memory:global",),
        execution_requirements={},
        created_at="2026-01-01T00:00:00Z",
        created_by="u-1",
    )
    defaults.update(overrides)
    return NormalizedObjective(**defaults)


def _make_classified(**overrides) -> ClassifiedObjective:
    defaults = dict(
        goal_class=GoalClass.STRATEGIC,
        risk_profile=RiskProfile.HIGH,
        estimated_complexity=Complexity.L,
        rationale="test",
        signal_tokens=(),
    )
    defaults.update(overrides)
    return ClassifiedObjective(**defaults)


def _make_discovery() -> CapabilityDiscovery:
    return CapabilityDiscovery(
        objective_id="oid-1",
        discovered_at="2026-01-01T00:00:00Z",
        candidates=(
            CapabilityCandidate(
                kind="tool", id="agent.tools.example", name="example",
                source_path="/x", description="d", keywords=(),
                match_score=0.8, match_reasons=(),
            ),
        ),
        reuse_decision="reuse", rationale="match", gaps=(),
        p0_query_duration_ms=10, p1_query_duration_ms=5,
    )


def test_contract_schema_version_is_1_0():
    c = build_execution_contract_v1(
        _make_normalized(), _make_classified(), _make_discovery(), user_id="u-1"
    )
    assert c.contract_version == "1.0"


def test_contract_required_fields_present():
    c = build_execution_contract_v1(
        _make_normalized(), _make_classified(), _make_discovery(), user_id="u-1"
    )
    assert c.objective_id == "oid-1"
    assert c.created_by == "u-1"
    assert c.goal_id is None  # Phase 1: not wired
    assert c.planner_inputs_preferred_workflow is None
    assert c.planner_inputs_preferred_role is None
    assert c.scheduler_hints_deadline is None


def test_contract_fingerprint_is_stable():
    n = _make_normalized()
    c1 = build_execution_contract_v1(n, _make_classified(), _make_discovery(), user_id="u-1")
    c2 = build_execution_contract_v1(n, _make_classified(), _make_discovery(), user_id="u-1")
    # Different contract_id (uuid4), but same fingerprint (deterministic from objective).
    assert c1.fingerprint == c2.fingerprint


def test_contract_risk_components_total_bounded_0_to_1():
    n = _make_normalized()
    classified = _make_classified()
    c = build_execution_contract_v1(n, classified, _make_discovery(), user_id="u-1")
    assert 0.0 <= c.risk_score <= 1.0
    # STRATEGIC + HIGH risk + L complexity -> high risk_score.
    assert c.risk_score > 0.0


def test_contract_approval_requirements_derived_from_risk():
    n = _make_normalized(risk_profile=RiskProfile.HIGH, estimated_complexity=Complexity.L)
    c = build_execution_contract_v1(
        n, _make_classified(risk_profile=RiskProfile.HIGH, estimated_complexity=Complexity.L),
        _make_discovery(), user_id="u-1",
    )
    gates = [ar["gate"] for ar in c.approval_requirements]
    assert "HIGH_RISK_DRAFT" in gates


def test_contract_budget_by_complexity():
    n = _make_normalized(estimated_complexity=Complexity.XL)
    c = build_execution_contract_v1(
        n, _make_classified(estimated_complexity=Complexity.XL),
        _make_discovery(), user_id="u-1",
    )
    assert c.budget["policy"] == "strict"
    assert c.budget["max_cost_usd"] == 2000.0


def test_contract_does_not_include_planner_inputs_filled_phase1():
    c = build_execution_contract_v1(
        _make_normalized(), _make_classified(), _make_discovery(), user_id="u-1"
    )
    assert c.planner_inputs_sub_goals == ()
    assert c.planner_inputs_preferred_workflow is None
    assert c.planner_inputs_preferred_role is None
    assert c.required_workflows == ()


def test_contract_serializable_to_json():
    c = build_execution_contract_v1(
        _make_normalized(), _make_classified(), _make_discovery(), user_id="u-1"
    )
    j = json.dumps(c.__dict__, default=str)
    parsed = json.loads(j)
    assert parsed["contract_version"] == "1.0"
    assert parsed["objective_id"] == "oid-1"


def test_compute_risk_components_financial_high():
    n = _make_normalized(constraints=("banking", "payment"))
    rc = compute_risk_components(_make_classified(), n)
    assert rc.financial == 1.0


def test_compute_risk_components_data_sensitivity_high():
    n = _make_normalized(constraints=("pii", "personal"))
    rc = compute_risk_components(_make_classified(), n)
    assert rc.data_sensitivity == 1.0


def test_build_knowledge_summary_text_empty():
    d = CapabilityDiscovery(
        objective_id="x", discovered_at="now", candidates=(),
        reuse_decision="generate", rationale="r", gaps=(),
        p0_query_duration_ms=0, p1_query_duration_ms=0,
    )
    text = build_knowledge_summary_text(d)
    assert "No P0/P1" in text
