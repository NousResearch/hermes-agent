"""Pilot G2 — build_execution_contract_v1 accepts a frozen EvidencePack.

The contract builder is a pure function. The pilot feeds it an
EvidencePack from the canary_b1 fixture and confirms the contract
carries the evidence_pack_ref / evidence_pack_summary / confidence /
freshness fields required by OBJ_DRYRUN_KD_KANBAN_PILOT_V1.

We use the *real* dataclasses (NormalizedObjective, ClassifiedObjective,
CapabilityCandidate, CapabilityDiscovery) so the contract builder's
duck-typing path is exercised end-to-end.
"""

from __future__ import annotations

import pytest

from agent.executive.contract import build_execution_contract_v1
from agent.executive.types import (
    CapabilityCandidate,
    CapabilityDiscovery,
    ClassifiedObjective,
    Complexity,
    ExecutionContractV1,
    GoalClass,
    NormalizedObjective,
    RiskProfile,
    now_iso8601,
)


def _make_inputs():
    normalized = NormalizedObjective(
        objective_id="obj-pilot-contract-001",
        goal_class=GoalClass.AUTOMATE,
        constraints=("forbidden:network",),
        success_criteria=(
            "EvidencePack fields propagate to ExecutionContract",
            "Contract fingerprint changes when summary changes",
        ),
        human_constraints=(),
        approval_requirements=(),
        risk_profile=RiskProfile.LOW,
        estimated_complexity=Complexity.XS,
        knowledge_requirements=(),
        execution_requirements={},
        created_at=now_iso8601(),
        created_by="pilot-user",
    )
    classified = ClassifiedObjective(
        goal_class=GoalClass.AUTOMATE,
        risk_profile=RiskProfile.LOW,
        estimated_complexity=Complexity.XS,
        rationale="pilot; hermetic dryrun",
        signal_tokens=(),
    )
    candidate = CapabilityCandidate(
        kind="skill",
        id="test.candidate.skill",
        name="pilot-skill",
        source_path="agent.executive.contract",
        description="synthetic canary candidate",
        keywords=("pilot",),
        match_score=0.9,
        match_reasons=("canary",),
    )
    discovered = CapabilityDiscovery(
        objective_id="obj-pilot-contract-001",
        discovered_at=now_iso8601(),
        candidates=(candidate,),
        reuse_decision="reuse",
        rationale="synthetic canary candidate",
        gaps=(),
        p0_query_duration_ms=0,
        p1_query_duration_ms=0,
    )
    return normalized, classified, discovered


def test_g2_contract_without_evidence_has_empty_summary():
    """When evidence_pack=None, the contract must keep the pre-wiring
    defaults (per build_execution_contract_v1 docstring)."""
    n, c, d = _make_inputs()
    pre = build_execution_contract_v1(n, c, d, user_id="pilot-user")
    assert isinstance(pre, ExecutionContractV1)
    assert pre.evidence_pack_summary == ""
    assert pre.evidence_pack_ref is None
    assert pre.evidence_pack_confidence == 0.0
    assert pre.evidence_pack_freshness == 0.0


def test_g2_contract_with_frozen_evidence_carries_knowledge_fields(
    frozen_evidence_pack,
):
    n, c, d = _make_inputs()
    contract = build_execution_contract_v1(
        n, c, d, user_id="pilot-user", evidence_pack=frozen_evidence_pack
    )
    # Pilot asserts these are populated when evidence_pack is supplied.
    assert contract.evidence_pack_ref is not None
    assert contract.evidence_pack_ref.startswith(
        "objective_knowledge_discovery:obj-pilot-contract-001"
    ), f"unexpected evidence_pack_ref: {contract.evidence_pack_ref!r}"
    assert "[NO_EVIDENCE_YET]" in contract.evidence_pack_summary
    # overall_freshness_score=1.0 -> evidence_pack_freshness=1.0
    assert contract.evidence_pack_freshness == pytest.approx(1.0)


def test_g2_contract_summary_change_propagates_to_contract_dict(
    frozen_evidence_pack,
):
    """compute_contract_fingerprint derives its seed ONLY from
    (objective_id, normalized.fingerprint) — it intentionally does NOT
    consider the evidence_pack_summary. So we test the next-best
    invariant: mutating the evidence summary MUST mutate the
    evidence_pack_summary field of the resulting contract.
    """
    n, c, d = _make_inputs()
    a = build_execution_contract_v1(
        n, c, d, user_id="pilot-user", evidence_pack=frozen_evidence_pack
    )
    n2, c2, d2 = _make_inputs()
    mutated = frozen_evidence_pack
    mutated.summary_text = "MUTATED: pilot summary changed"
    b = build_execution_contract_v1(
        n2, c2, d2, user_id="pilot-user", evidence_pack=mutated
    )
    assert a.evidence_pack_summary != b.evidence_pack_summary
    assert "MUTATED:" in b.evidence_pack_summary
    assert "MUTATED:" not in a.evidence_pack_summary


def test_g2_contract_summary_prefix_can_trigger_human_review(
    frozen_evidence_pack,
):
    """A summary starting with [REQUIRES_HUMAN] must add a
    knowledge_review approval requirement."""
    frozen_evidence_pack.summary_text = "[REQUIRES_HUMAN] needs review"
    n, c, d = _make_inputs()
    contract = build_execution_contract_v1(
        n, c, d, user_id="pilot-user", evidence_pack=frozen_evidence_pack
    )
    gates = {ar.get("gate") for ar in contract.approval_requirements}
    assert "knowledge_review" in gates, (
        f"expected knowledge_review gate; got gates={gates!r}"
    )


def test_g2_contract_to_dict_round_trip_preserves_evidence_fields(
    frozen_evidence_pack,
):
    """When the contract is serialised to dict (the shape the engine
    stores in state.contract), the evidence_pack_* keys survive."""
    n, c, d = _make_inputs()
    contract = build_execution_contract_v1(
        n, c, d, user_id="pilot-user", evidence_pack=frozen_evidence_pack
    )
    d_contract = contract.__dict__
    assert "evidence_pack_ref" in d_contract
    assert "evidence_pack_summary" in d_contract
    assert "evidence_pack_confidence" in d_contract
    assert "evidence_pack_freshness" in d_contract