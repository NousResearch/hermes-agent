"""Fail-closed contracts for local specialist benchmark and promotion receipts."""

from __future__ import annotations

import hashlib
from dataclasses import replace

from agent.profile_benchmark import (
    CandidatePromotionGate,
    CandidateProposal,
    FrozenBenchmarkCaseSet,
    OperatorApproval,
)
from gateway.candidate_profile_requests import CandidateProfileRequests
from gateway.capability_registry import CapabilityRegistry, CapabilitySignature


SIGNATURE = CapabilitySignature(
    domain="market-data",
    actions=("audit", "read"),
    evidence_class="diagnostic-only",
    requested_permissions=("market-data:read",),
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _candidate(tmp_path, now: list[int]):
    db_path = tmp_path / "benchmark.db"
    requests = CandidateProfileRequests(db_path=db_path, clock=lambda: now[0])
    opened = requests.open_or_reuse(SIGNATURE, source_key="test:benchmark")
    assert opened.status == "candidate"
    proposal = CandidateProposal(
        candidate_id=opened.request_id,
        proposal_author="claude",
        sol_reviewer="sol",
        proposal_hash=_digest("proposal"),
        signature_hash=SIGNATURE.signature_hash,
        permissions_hash=SIGNATURE.permissions_hash,
    )
    cases = FrozenBenchmarkCaseSet(case_ids=(_digest("case-a"), _digest("case-b")))
    gate = CandidatePromotionGate(
        requests,
        CapabilityRegistry(db_path=db_path),
        clock=lambda: now[0],
    )
    return requests, gate, proposal, cases


def test_candidate_cannot_score_or_verify_its_own_proposal(tmp_path):
    now = [1_000]
    _, gate, proposal, cases = _candidate(tmp_path, now)

    benchmark_result, receipt = gate.benchmark(
        proposal, cases, scorer_model="claude", scores=(100, 100), scorer_available=True
    )

    assert benchmark_result.status == "rejected"
    assert receipt.status == "rejected"


def test_unavailable_scorer_fails_closed_without_benchmark_transition(tmp_path):
    now = [1_000]
    requests, gate, proposal, cases = _candidate(tmp_path, now)

    result, receipt = gate.benchmark(
        proposal, cases, scorer_model="benchmark-model", scores=(100, 100), scorer_available=False
    )

    assert result.status == "rejected"
    assert receipt.status == "unavailable"
    assert requests.lifecycle_snapshot(proposal.candidate_id).lifecycle_status == "candidate"


def test_case_set_or_proposal_hash_change_invalidates_benchmark_receipt(tmp_path):
    now = [1_000]
    _, gate, proposal, cases = _candidate(tmp_path, now)
    result, receipt = gate.benchmark(
        proposal, cases, scorer_model="benchmark-model", scores=(90, 90), scorer_available=True
    )
    assert result.status == "benchmarked"

    assert not gate.valid_benchmark(receipt, replace(proposal, proposal_hash=_digest("mutated")), cases)
    assert not gate.valid_benchmark(
        receipt,
        proposal,
        FrozenBenchmarkCaseSet(case_ids=(_digest("case-a"), _digest("mutated-case"))),
    )
    object.__setattr__(receipt, "case_set_hash", _digest("mutated-receipt-case-set"))
    assert not gate.valid_benchmark(receipt, proposal, cases)


def test_permission_delta_cannot_be_benchmarked_or_activated(tmp_path):
    now = [1_000]
    _, gate, proposal, cases = _candidate(tmp_path, now)
    expanded = replace(proposal, permissions_hash=_digest("market-data-read-and-write"))

    result, receipt = gate.benchmark(
        expanded, cases, scorer_model="benchmark-model", scores=(100, 100), scorer_available=True
    )

    assert result.status == "rejected"
    assert receipt.status == "rejected"


def test_verified_candidate_needs_independent_disposable_sandbox_and_approval(tmp_path):
    now = [1_000]
    requests, gate, proposal, cases = _candidate(tmp_path, now)
    benchmark_result, benchmark = gate.benchmark(
        proposal, cases, scorer_model="benchmark-model", scores=(90, 90), scorer_available=True
    )
    assert benchmark_result.status == "benchmarked"
    assert gate.open_disposable_sandbox(proposal, benchmark, sandbox_id="sandbox-1").status == "benchmarked"

    rejected, own_verification = gate.verify(
        proposal,
        benchmark,
        verifier_identity="benchmark-model",
        sandbox_id="sandbox-1",
        verifier_available=True,
    )
    assert rejected.status == "rejected"
    assert own_verification.status == "rejected"

    verified, verification = gate.verify(
        proposal,
        benchmark,
        verifier_identity="independent-verifier",
        sandbox_id="sandbox-1",
        verifier_available=True,
    )
    assert verified.status == "verified"
    assert gate.stage(proposal, benchmark, verification, None).status == "rejected"
    assert requests.lifecycle_snapshot(proposal.candidate_id).lifecycle_status == "verified"


def test_expired_receipts_and_missing_operator_approval_cannot_stage_or_activate(tmp_path):
    now = [1_000]
    _, gate, proposal, cases = _candidate(tmp_path, now)
    _, benchmark = gate.benchmark(
        proposal,
        cases,
        scorer_model="benchmark-model",
        scores=(90, 90),
        scorer_available=True,
        receipt_lifetime_seconds=1,
    )
    assert gate.open_disposable_sandbox(proposal, benchmark, sandbox_id="sandbox-1").status == "benchmarked"
    _, verification = gate.verify(
        proposal,
        benchmark,
        verifier_identity="independent-verifier",
        sandbox_id="sandbox-1",
        verifier_available=True,
    )
    now[0] += 2
    approval = OperatorApproval(
        candidate_id=proposal.candidate_id,
        approval_id="approval-1",
        operator_identity="operator",
        verification_result_hash=verification.result_hash,
        target_state="staged",
        approved=True,
        issued_at=now[0],
    )

    assert gate.stage(proposal, benchmark, verification, approval).status == "rejected"
    assert gate.activate(
        proposal, SIGNATURE, "market-data-specialist", benchmark, verification, approval
    ).status == "rejected"


def test_unattested_operator_identity_cannot_stage_or_activate_a_candidate(tmp_path):
    now = [1_000]
    requests, gate, proposal, cases = _candidate(tmp_path, now)
    _, benchmark = gate.benchmark(
        proposal, cases, scorer_model="benchmark-model", scores=(90, 90), scorer_available=True
    )
    assert gate.open_disposable_sandbox(proposal, benchmark, sandbox_id="sandbox-1").status == "benchmarked"
    _, verification = gate.verify(
        proposal,
        benchmark,
        verifier_identity="independent-verifier",
        sandbox_id="sandbox-1",
        verifier_available=True,
    )
    approval = OperatorApproval(
        candidate_id=proposal.candidate_id,
        approval_id="approval-1",
        operator_identity="operator",
        verification_result_hash=verification.result_hash,
        target_state="staged",
        approved=True,
        issued_at=now[0],
    )
    assert gate.record_operator_approval(approval) is False
    assert gate.stage(proposal, benchmark, verification, approval).status == "rejected"
    active_approval = OperatorApproval(
        candidate_id=proposal.candidate_id,
        approval_id="approval-2",
        operator_identity="operator",
        verification_result_hash=verification.result_hash,
        target_state="active",
        approved=True,
        issued_at=now[0],
    )
    active = gate.activate(
        proposal,
        SIGNATURE,
        "market-data-specialist",
        benchmark,
        verification,
        active_approval,
        expires_at=now[0] + 100,
    )

    assert active.status == "rejected"
    assert requests.lifecycle_snapshot(proposal.candidate_id).lifecycle_status == "verified"


def test_authenticated_allowlisted_operator_approval_is_durable_and_can_stage(tmp_path):
    now = [1_000]
    requests, gate, proposal, cases = _candidate(tmp_path, now)
    _, benchmark = gate.benchmark(
        proposal, cases, scorer_model="benchmark-model", scores=(90, 90), scorer_available=True
    )
    assert gate.open_disposable_sandbox(proposal, benchmark, sandbox_id="sandbox-1").status == "benchmarked"
    _, verification = gate.verify(
        proposal, benchmark, verifier_identity="independent-verifier", sandbox_id="sandbox-1", verifier_available=True
    )
    approval = OperatorApproval(
        candidate_id=proposal.candidate_id,
        approval_id="dashboard-approval-1",
        operator_identity="portal:operator-1",
        verification_result_hash=verification.result_hash,
        target_state="staged",
        approved=True,
        issued_at=now[0],
    )

    assert gate.record_operator_approval(
        approval, authenticated_operator_identity="portal:operator-1"
    ) is True
    assert gate.stage(proposal, benchmark, verification, approval).status == "staged"
    with gate._connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM specialist_operator_approvals").fetchone()[0] == 1
    assert requests.lifecycle_snapshot(proposal.candidate_id).lifecycle_status == "staged"


def test_arbitrary_operator_identity_never_creates_staged_or_active_authority(tmp_path):
    now = [1_000]
    requests, gate, proposal, cases = _candidate(tmp_path, now)
    _, benchmark = gate.benchmark(
        proposal, cases, scorer_model="benchmark-model", scores=(90, 90), scorer_available=True
    )
    assert gate.open_disposable_sandbox(proposal, benchmark, sandbox_id="sandbox-1").status == "benchmarked"
    _, verification = gate.verify(
        proposal, benchmark, verifier_identity="independent-verifier", sandbox_id="sandbox-1", verifier_available=True
    )
    forged = OperatorApproval(
        candidate_id=proposal.candidate_id,
        approval_id="arbitrary-string-is-not-authority",
        operator_identity="totally-unverified-identity",
        verification_result_hash=verification.result_hash,
        target_state="staged",
        approved=True,
        issued_at=now[0],
    )

    assert gate.record_operator_approval(forged) is False
    assert gate.stage(proposal, benchmark, verification, forged).status == "rejected"
    with gate._connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM specialist_operator_approvals").fetchone()[0] == 0
    assert requests.lifecycle_snapshot(proposal.candidate_id).lifecycle_status == "verified"


def test_constructed_receipts_or_approvals_cannot_bypass_durable_promotion_evidence(tmp_path):
    now = [1_000]
    requests, gate, proposal, cases = _candidate(tmp_path, now)
    _, benchmark = gate.benchmark(
        proposal, cases, scorer_model="benchmark-model", scores=(90, 90), scorer_available=True
    )
    assert gate.open_disposable_sandbox(proposal, benchmark, sandbox_id="sandbox-1").status == "benchmarked"
    _, verification = gate.verify(
        proposal, benchmark, verifier_identity="independent-verifier", sandbox_id="sandbox-1", verifier_available=True
    )
    forged_stage_approval = OperatorApproval(
        candidate_id=proposal.candidate_id,
        approval_id="forged-stage",
        operator_identity="operator",
        verification_result_hash=verification.result_hash,
        target_state="staged",
        approved=True,
        issued_at=now[0],
    )

    assert gate.stage(proposal, benchmark, verification, forged_stage_approval).status == "rejected"
    assert requests.lifecycle_snapshot(proposal.candidate_id).lifecycle_status == "verified"
    assert CapabilityRegistry(db_path=requests._db_path).resolve(SIGNATURE).status == "no_match"


def test_candidate_lifecycle_rejects_all_skips_and_reversals(tmp_path):
    now = [1_000]
    requests, gate, proposal, cases = _candidate(tmp_path, now)
    receipt_hash = _digest("legal-transition-receipt")
    illegal = (
        ("candidate", "verified"),
        ("candidate", "staged"),
        ("candidate", "active"),
        ("benchmarked", "candidate"),
        ("verified", "benchmarked"),
        ("staged", "verified"),
        ("active", "staged"),
    )
    for expected, next_status in illegal:
        try:
            requests.append_lifecycle_transition(
                proposal.candidate_id,
                expected_status=expected,
                next_status=next_status,
                reason_code="illegal_transition",
                receipt_hash=receipt_hash,
            )
        except ValueError:
            pass
        else:
            assert False, f"accepted illegal lifecycle edge {expected}->{next_status}"
    assert requests.lifecycle_snapshot(proposal.candidate_id).lifecycle_status == "candidate"
    assert CapabilityRegistry(db_path=requests._db_path).resolve(SIGNATURE).status == "no_match"
