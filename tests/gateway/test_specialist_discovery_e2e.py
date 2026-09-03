"""End-to-end local contracts for governed specialist discovery.

All inputs are synthetic opaque hashes.  These tests exercise no provider,
model, network, task worker, Discord adapter, or external write.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from agent.profile_benchmark import (
    CandidatePromotionGate,
    CandidateProposal,
    FrozenBenchmarkCaseSet,
    OperatorApproval,
)
from gateway.capability_registry import CapabilityRegistry, CapabilitySignature
from gateway.candidate_profile_requests import CandidateProfileRequests
from gateway.configured_board import configured_board_db_path
from gateway.specialist_handoff import HandoffSource, create_specialist_handoff
from gateway.specialist_routing import RouteKind, SpecialistRouteDecision
from gateway.status import specialist_discovery_status
from hermes_cli import kanban_db as kb


SIGNATURE = CapabilitySignature(
    domain="market-data",
    actions=("audit", "read"),
    evidence_class="diagnostic-only",
    requested_permissions=("market-data:read",),
)
BOARD = "exampleproject-burndown"


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _source(message_id: str) -> HandoffSource:
    return HandoffSource(
        # This is only a persisted subscription shape; the test never calls
        # an adapter or sends a notification.
        platform="discord",
        chat_id="local-no-send",
        chat_type="group",
        user_id="synthetic-operator",
        message_id=message_id,
    )


def _provision_test_profiles(home: Path) -> None:
    for profile in (
        "task-orchestrator",
        "burndown-patch-steward",
        "market-data-authority-auditor",
    ):
        (home / "profiles" / profile).mkdir(parents=True)


def _decision() -> SpecialistRouteDecision:
    return SpecialistRouteDecision(
        kind=RouteKind.SPECIALIST,
        profile="market-data-authority-auditor",
        confidence=0.99,
        reason="synthetic local no-match",
        title="Synthetic local capability audit",
    )


def test_no_match_stays_inert_without_authenticated_approval_and_exposes_recovery(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    _provision_test_profiles(home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = configured_board_db_path(BOARD)
    kb.init_db(db_path)
    registry = CapabilityRegistry(db_path=db_path)

    task = create_specialist_handoff(
        decision=_decision(),
        source=_source("no-match-to-active"),
        request="Synthetic local-only diagnostic evidence audit.",
        signature=SIGNATURE,
        registry=registry,
        board=BOARD,
    )

    assert task.ok, task.reason
    assert task.candidate_request_id
    assert task.candidate_status == "candidate"
    with kb.connect_closing(db_path) as conn:
        source_task = kb.get_task(conn, task.task_id)
    assert source_task is not None
    assert source_task.assignee == "task-orchestrator"

    now = [10_000]
    requests = CandidateProfileRequests(db_path=db_path, clock=lambda: now[0])
    proposal = CandidateProposal(
        candidate_id=task.candidate_request_id,
        proposal_author="claude",
        sol_reviewer="sol",
        proposal_hash=_hash("synthetic-proposal"),
        signature_hash=SIGNATURE.signature_hash,
        permissions_hash=SIGNATURE.permissions_hash,
    )
    gate = CandidatePromotionGate(requests, registry, clock=lambda: now[0])
    cases = FrozenBenchmarkCaseSet(case_ids=(_hash("case-1"), _hash("case-2")))

    assert registry.resolve(SIGNATURE).status == "no_match"

    benchmarked, benchmark = gate.benchmark(
        proposal, cases, scorer_model="independent-benchmark", scores=(90, 91), scorer_available=True
    )
    assert benchmarked.status == "benchmarked"
    assert gate.open_disposable_sandbox(proposal, benchmark, sandbox_id="synthetic-sandbox").status == "benchmarked"
    verified, verification = gate.verify(
        proposal,
        benchmark,
        verifier_identity="independent-verifier",
        sandbox_id="synthetic-sandbox",
        verifier_available=True,
    )
    assert verified.status == "verified"
    assert gate.stage(proposal, benchmark, verification, None).status == "rejected"
    assert gate.activate(proposal, SIGNATURE, "synthetic-specialist", benchmark, verification, None).status == "rejected"

    staged_approval = OperatorApproval(
        candidate_id=proposal.candidate_id,
        approval_id="synthetic-stage-approval",
        operator_identity="operator",
        verification_result_hash=verification.result_hash,
        target_state="staged",
        approved=True,
        issued_at=now[0],
    )
    assert gate.record_operator_approval(staged_approval) is False
    assert gate.stage(proposal, benchmark, verification, staged_approval).status == "rejected"
    assert gate.activate(proposal, SIGNATURE, "synthetic-specialist", benchmark, verification, None).status == "rejected"

    canary_result, canary = gate.run_local_no_send_canary(proposal, verification)
    assert canary_result.status == "rejected"
    assert canary is None
    assert registry.resolve(SIGNATURE).status == "no_match"
    active_approval = OperatorApproval(
        candidate_id=proposal.candidate_id,
        approval_id="synthetic-active-approval",
        operator_identity="operator",
        verification_result_hash=verification.result_hash,
        target_state="active",
        approved=True,
        issued_at=now[0],
    )
    assert gate.record_operator_approval(active_approval) is False
    active = gate.activate(
        proposal, SIGNATURE, "synthetic-specialist", benchmark, verification, active_approval
    )
    assert active.status == "rejected"
    assert registry.resolve(SIGNATURE).status == "no_match"

    before = specialist_discovery_status(task.candidate_request_id, db_path=db_path, now=now[0])
    after = specialist_discovery_status(task.candidate_request_id, db_path=db_path, now=now[0])
    assert after == before  # restart reconciliation is a pure durable read; no rerun.
    assert before["recovery_action"] == "task_orchestrator"
    by_stage = {row["stage"]: row for row in before["rows"]}
    assert by_stage["source_task"]["status"] == "triage"
    assert by_stage["candidate_request"]["status"] == "verified"
    assert by_stage["benchmark"]["receipt_hash"] == benchmark.result_hash
    assert by_stage["sandbox"]["status"] == "verified_shape"
    assert by_stage["verification"]["receipt_hash"] == verification.result_hash
    assert by_stage["canary"]["status"] == "not_run"
    assert by_stage["active_profile"]["status"] == "not_active"


def test_expired_or_revoked_profiles_stop_resolving_and_handoff_uses_safe_fallback(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    _provision_test_profiles(home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = configured_board_db_path(BOARD)
    kb.init_db(db_path)
    registry = CapabilityRegistry(db_path=db_path)
    registry.register_fixed_baseline(
        profile_id="market-data-authority-auditor", signature=SIGNATURE, expires_at=int(time.time()) - 1
    )
    assert registry.resolve(SIGNATURE).status == "no_match"

    registry.register_fixed_baseline(profile_id="market-data-authority-auditor", signature=SIGNATURE)
    assert registry.resolve(SIGNATURE).profile == "market-data-authority-auditor"
    registry.revoke(
        profile_id="market-data-authority-auditor", signature=SIGNATURE, reason_code="synthetic_rollback"
    )
    assert registry.resolve(SIGNATURE).status == "no_match"

    fallback = create_specialist_handoff(
        decision=_decision(),
        source=_source("revoked-safe-fallback"),
        request="Synthetic fallback after local revocation.",
        signature=SIGNATURE,
        registry=registry,
        board=BOARD,
    )
    assert fallback.ok, fallback.reason
    assert fallback.candidate_request_id
    with kb.connect_closing(db_path) as conn:
        source_task = kb.get_task(conn, fallback.task_id)
    assert source_task is not None
    assert source_task.assignee == "task-orchestrator"


def test_status_recovery_opens_sqlite_read_only_and_refuses_missing_or_uninitialized_db(tmp_path):
    db_path = tmp_path / "status.db"
    now = [10_000]
    requests = CandidateProfileRequests(db_path=db_path, clock=lambda: now[0])
    candidate = requests.open_or_reuse(SIGNATURE, source_key="local:read-only-status")
    assert candidate.status == "candidate"

    # Keep a standard WAL writer open so the reader takes the same fresh path
    # used during a concurrent revocation. The writer—not recovery—creates the
    # WAL; recovery must leave its bytes unchanged.
    writer = kb.connect(db_path)
    writer.execute("PRAGMA journal_mode=WAL")
    writer.execute("PRAGMA user_version = 1")
    writer.commit()

    # A fresh read-only SQLite connection may attach/create -shm while it
    # observes a live WAL; it must not mutate the database or WAL evidence.
    paths = [db_path, db_path.with_name(db_path.name + "-wal")]
    before = {path: path.read_bytes() if path.exists() else None for path in paths}
    status = specialist_discovery_status(candidate.request_id, db_path=db_path, now=now[0])
    after = {path: path.read_bytes() if path.exists() else None for path in paths}
    writer.close()

    assert status["recovery_action"] == "task_orchestrator"
    assert after == before

    missing = tmp_path / "missing.db"
    missing_status = specialist_discovery_status(candidate.request_id, db_path=missing, now=now[0])
    assert missing_status["recovery_action"] == "normal_triage"
    assert missing.exists() is False
    assert missing.with_name(missing.name + "-wal").exists() is False
    assert missing.with_name(missing.name + "-shm").exists() is False

    uninitialized = tmp_path / "uninitialized.db"
    uninitialized.write_bytes(b"not sqlite")
    original = uninitialized.read_bytes()
    uninitialized_status = specialist_discovery_status(candidate.request_id, db_path=uninitialized, now=now[0])
    assert uninitialized_status["recovery_action"] == "normal_triage"
    assert uninitialized.read_bytes() == original


def test_status_reader_sees_committed_wal_revocation_without_mutating_db_or_wal(tmp_path):
    db_path = tmp_path / "live-wal-status.db"
    now = int(time.time())
    candidate = CandidateProfileRequests(db_path=db_path, clock=lambda: now).open_or_reuse(
        SIGNATURE, source_key="local:wal-revocation"
    )
    assert candidate.status == "candidate"
    benchmark_hash = _hash("wal-benchmark")
    verification_hash = _hash("wal-verification")
    revocation_hash = _hash("wal-revocation")
    writer = kb.connect(db_path)
    try:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute(
            """
            INSERT INTO specialist_benchmark_receipts (
                result_hash, candidate_id, proposal_input_hash, proposal_author, sol_reviewer,
                signature_hash, permissions_hash, case_set_hash, scorer_model, scores_json,
                pass_threshold, status, issued_at, expires_at, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'passed', ?, ?, ?)
            """,
            (
                benchmark_hash, candidate.request_id, _hash("proposal"), "claude", "sol",
                SIGNATURE.signature_hash, SIGNATURE.permissions_hash, _hash("case-set"),
                "independent-benchmark", json.dumps([100]), 80, now, now + 3_600, now,
            ),
        )
        writer.execute(
            """
            INSERT INTO specialist_verification_receipts (
                result_hash, candidate_id, benchmark_result_hash, verifier_identity, sandbox_id,
                status, issued_at, expires_at, created_at
            ) VALUES (?, ?, ?, ?, ?, 'verified', ?, ?, ?)
            """,
            (verification_hash, candidate.request_id, benchmark_hash, "independent-verifier", "sandbox-1", now, now + 3_600, now),
        )
        writer.execute(
            """
            INSERT INTO capability_profiles (
                profile_id, signature_hash, permissions_hash,
                model_receipt_hash, verification_receipt_hash,
                domain, actions_json, evidence_class, requested_permissions_json,
                expires_at, status, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, 'active', ?)
            """,
            (
                "market-data-authority-auditor", SIGNATURE.signature_hash, SIGNATURE.permissions_hash,
                benchmark_hash, verification_hash, SIGNATURE.domain, json.dumps(SIGNATURE.actions),
                SIGNATURE.evidence_class, json.dumps(SIGNATURE.requested_permissions), now,
            ),
        )
        writer.commit()
        assert specialist_discovery_status(candidate.request_id, db_path=db_path, now=now)["recovery_action"] == "active_resolution"

        writer.execute(
            """
            INSERT INTO specialist_profile_revocations (
                revocation_hash, profile_id, signature_hash, reason_code, created_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (revocation_hash, "market-data-authority-auditor", SIGNATURE.signature_hash, "wal_rollback", now),
        )
        writer.commit()
        watched = [db_path, db_path.with_name(db_path.name + "-wal")]
        before = {path: path.read_bytes() if path.exists() else None for path in watched}
        status = specialist_discovery_status(candidate.request_id, db_path=db_path, now=now)
        after = {path: path.read_bytes() if path.exists() else None for path in watched}

        by_stage = {row["stage"]: row for row in status["rows"]}
        assert status["recovery_action"] == "task_orchestrator"
        assert by_stage["rollback"]["receipt_hash"] == revocation_hash
        assert by_stage["active_profile"]["status"] == "revoked"
        assert after == before
    finally:
        writer.close()
