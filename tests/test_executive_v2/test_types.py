"""Tests for Executive v2 types: immutability, serialization, fingerprint."""

from __future__ import annotations

import json

import pytest

from agent.executive.types import (
    CapabilityCandidate,
    Complexity,
    ExecutionContractV1,
    GoalClass,
    NormalizedObjective,
    ObjectiveState,
    ObjectiveStateData,
    RiskProfile,
    compute_contract_fingerprint,
    compute_fingerprint,
    new_uuid,
    now_iso8601,
    objective_archive_key,
    objective_key,
)


def test_types_dataclass_immutability_enforced():
    n = NormalizedObjective(
        objective_id="x",
        goal_class=GoalClass.BUILD,
        constraints=(),
        success_criteria=(),
        human_constraints=(),
        approval_requirements=(),
        risk_profile=RiskProfile.LOW,
        estimated_complexity=Complexity.XS,
        knowledge_requirements=(),
        execution_requirements={},
        created_at="2026-01-01",
        created_by="u",
    )
    with pytest.raises(Exception):
        n.goal_class = GoalClass.OTHER  # type: ignore[misc]


def test_types_json_serializable():
    s = ObjectiveStateData(
        objective_id="oid",
        state=ObjectiveState.DRAFT,
        objective_text="hello",
        constraints=[],
        user_id="u",
        created_at="2026-01-01",
    )
    d = s.to_dict()
    j = json.dumps(d, default=str)
    parsed = json.loads(j)
    assert parsed["objective_id"] == "oid"
    assert parsed["state"] == "DRAFT"


def test_types_roundtrip_via_json():
    s = ObjectiveStateData(
        objective_id="oid2",
        state=ObjectiveState.NORMALIZED,
        objective_text="x",
        constraints=["forbidden:stripe"],
        user_id="u",
        created_at="2026-01-01",
        fingerprint="abc",
    )
    parsed = ObjectiveStateData.from_dict(s.to_dict())
    assert parsed.objective_id == s.objective_id
    assert parsed.state == ObjectiveState.NORMALIZED
    assert parsed.fingerprint == "abc"


def test_compute_fingerprint_is_stable_across_calls():
    fp1 = compute_fingerprint("text", ["a", "b"], "u", "2026-01-01T00:00:00Z")
    fp2 = compute_fingerprint("text", ["b", "a"], "u", "2026-01-01T00:00:00Z")
    assert fp1 == fp2
    assert len(fp1) == 64


def test_compute_fingerprint_changes_with_text():
    fp1 = compute_fingerprint("text-a", [], "u", "2026-01-01T00:00:00Z")
    fp2 = compute_fingerprint("text-b", [], "u", "2026-01-01T00:00:00Z")
    assert fp1 != fp2


def test_compute_contract_fingerprint_differs_from_seed():
    fp_obj = compute_fingerprint("t", [], "u", "2026-01-01T00:00:00Z")
    fp_contract = compute_contract_fingerprint("oid-123", fp_obj)
    assert fp_contract != fp_obj
    assert len(fp_contract) == 64


def test_state_storage_keys():
    assert objective_key("abc") == "objective:abc"
    assert objective_archive_key("abc") == "objective_archive:abc"
    assert objective_key("abc") != objective_archive_key("abc")


def test_now_iso8601_format():
    s = now_iso8601()
    # ISO 8601: YYYY-MM-DDTHH:MM:SS...
    assert "T" in s
    assert s.startswith("20")


def test_new_uuid_unique():
    assert new_uuid() != new_uuid()
    assert len(new_uuid()) == 36  # uuid4 standard


def test_execution_contract_v1_required_fields():
    contract = ExecutionContractV1(
        contract_version="1.0",
        contract_id="cid",
        objective_id="oid",
        goal_id=None,
        fingerprint="fp",
        required_capabilities=(),
        required_tools=(),
        required_skills=(),
        required_roles=(),
        required_workflows=(),
        required_providers=(),
        knowledge_summary_keys=(),
        knowledge_summary_text="",
        hard_constraints=(),
        soft_constraints=(),
        approval_requirements=(),
        risk_components={},
        risk_score=0.0,
        budget={},
        execution_strategy="sequential",
        rollback_strategy="manual",
        planner_inputs_sub_goals=(),
        planner_inputs_success_criteria=(),
        planner_inputs_hard_constraints=(),
        planner_inputs_soft_constraints=(),
        planner_inputs_preferred_workflow=None,
        planner_inputs_preferred_role=None,
        scheduler_hints_priority="medium",
        scheduler_hints_deadline=None,
        scheduler_hints_blocking_objectives=(),
        scheduler_hints_parallelism_allowed=False,
        success_criteria=(),
        verification_method="judge",
        verification_timeout_minutes=60,
        judge_model=None,
        evidence_required=True,
        created_at="2026-01-01",
        created_by="u",
    )
    assert contract.contract_version == "1.0"
    assert contract.execution_strategy == "sequential"
    assert contract.rollback_strategy == "manual"


def test_capability_candidate_immutable():
    c = CapabilityCandidate(
        kind="tool",
        id="x",
        name="x",
        source_path="/x",
        description="",
        keywords=(),
        match_score=0.5,
        match_reasons=(),
    )
    with pytest.raises(Exception):
        c.match_score = 0.9  # type: ignore[misc]
