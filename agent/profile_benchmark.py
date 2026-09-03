"""Local, receipt-bound specialist-candidate benchmark and promotion gates.

This module is deliberately not a model adapter.  A caller may provide a
synthetic/local score and verification observation, but no provider, network,
or scorer callback is accepted here.  Every promotion gate consequently fails
closed when the designated independent scorer is unavailable.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from typing import Callable, Literal

from gateway.candidate_profile_requests import CandidateLifecycleSnapshot, CandidateProfileRequests
from gateway.capability_registry import CapabilityRegistry, CapabilitySignature
from hermes_cli import kanban_db


_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_IDENTITY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,95}$")
_MAX_RECEIPT_LIFETIME_SECONDS = 86_400


def _canonical_json(value: object) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def _hash(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _require_hash(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not _HASH_RE.fullmatch(value):
        raise ValueError(f"{field} must be a SHA-256 hex digest")
    return value


def _require_identity(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not _IDENTITY_RE.fullmatch(value):
        raise ValueError(f"{field} must be a bounded canonical identity")
    return value


@dataclass(frozen=True, slots=True)
class FrozenBenchmarkCaseSet:
    """A task-specific case identity set frozen before any score is recorded."""

    case_ids: tuple[str, ...]
    non_production: bool = True

    def __post_init__(self) -> None:
        if not self.non_production:
            raise ValueError("benchmark cases must be explicitly non-production")
        if not isinstance(self.case_ids, tuple) or not self.case_ids or len(self.case_ids) > 32:
            raise ValueError("case_ids must be a non-empty bounded tuple")
        canonical = tuple(sorted(set(self.case_ids)))
        if any(not isinstance(case, str) or not _HASH_RE.fullmatch(case) for case in canonical):
            raise ValueError("benchmark case identities must be opaque SHA-256 digests")
        object.__setattr__(self, "case_ids", canonical)

    @property
    def case_set_hash(self) -> str:
        return _hash({"case_ids": self.case_ids, "non_production": self.non_production})


@dataclass(frozen=True, slots=True)
class CandidateProposal:
    """Opaque proposal identity and the declared, immutable candidate scope."""

    candidate_id: str
    proposal_author: str
    sol_reviewer: str
    proposal_hash: str
    signature_hash: str
    permissions_hash: str

    def __post_init__(self) -> None:
        _require_identity(self.candidate_id, field="candidate_id")
        _require_identity(self.proposal_author, field="proposal_author")
        _require_identity(self.sol_reviewer, field="sol_reviewer")
        for field in ("proposal_hash", "signature_hash", "permissions_hash"):
            _require_hash(getattr(self, field), field=field)

    @property
    def input_hash(self) -> str:
        return _hash(
            {
                "candidate_id": self.candidate_id,
                "permissions_hash": self.permissions_hash,
                "proposal_hash": self.proposal_hash,
                "signature_hash": self.signature_hash,
                "sol_reviewer": self.sol_reviewer,
            }
        )


BenchmarkStatus = Literal["passed", "failed", "unavailable", "rejected"]


@dataclass(frozen=True, slots=True)
class BenchmarkReceipt:
    """A bounded benchmark record that can be revalidated without a model call."""

    candidate_id: str
    proposal_input_hash: str
    case_set_hash: str
    scorer_model: str
    scores: tuple[int, ...]
    pass_threshold: int
    status: BenchmarkStatus
    issued_at: int
    expires_at: int
    result_hash: str

    def __post_init__(self) -> None:
        _require_identity(self.candidate_id, field="candidate_id")
        _require_identity(self.scorer_model, field="scorer_model")
        for field in ("proposal_input_hash", "case_set_hash", "result_hash"):
            _require_hash(getattr(self, field), field=field)
        if not isinstance(self.scores, tuple) or any(
            isinstance(score, bool) or not isinstance(score, int) or score < 0 or score > 100
            for score in self.scores
        ):
            raise ValueError("scores must be bounded integer percentages")
        if isinstance(self.pass_threshold, bool) or not isinstance(self.pass_threshold, int) or not 0 <= self.pass_threshold <= 100:
            raise ValueError("pass_threshold must be an integer percentage")
        if self.status not in {"passed", "failed", "unavailable", "rejected"}:
            raise ValueError("benchmark status is invalid")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in (self.issued_at, self.expires_at)):
            raise ValueError("receipt times must be integer timestamps")
        if not self.issued_at <= self.expires_at <= self.issued_at + _MAX_RECEIPT_LIFETIME_SECONDS:
            raise ValueError("receipt expiry is outside the bounded lifetime")
        if self.result_hash != self.expected_result_hash:
            raise ValueError("result_hash does not bind benchmark receipt contents")

    @property
    def expected_result_hash(self) -> str:
        return _hash(
            {
                "candidate_id": self.candidate_id,
                "case_set_hash": self.case_set_hash,
                "expires_at": self.expires_at,
                "issued_at": self.issued_at,
                "pass_threshold": self.pass_threshold,
                "proposal_input_hash": self.proposal_input_hash,
                "scorer_model": self.scorer_model,
                "scores": self.scores,
                "status": self.status,
            }
        )


VerificationStatus = Literal["verified", "failed", "unavailable", "rejected"]


@dataclass(frozen=True, slots=True)
class VerificationReceipt:
    """Independent result from one disposable, one-task local sandbox."""

    candidate_id: str
    benchmark_result_hash: str
    verifier_identity: str
    sandbox_id: str
    sandbox_disposable: bool
    sandbox_task_count: int
    status: VerificationStatus
    issued_at: int
    expires_at: int
    result_hash: str

    def __post_init__(self) -> None:
        for field in ("candidate_id", "verifier_identity", "sandbox_id"):
            _require_identity(getattr(self, field), field=field)
        _require_hash(self.benchmark_result_hash, field="benchmark_result_hash")
        if not isinstance(self.sandbox_disposable, bool) or self.sandbox_task_count != 1:
            raise ValueError("verification requires one disposable sandbox task")
        if self.status not in {"verified", "failed", "unavailable", "rejected"}:
            raise ValueError("verification status is invalid")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in (self.issued_at, self.expires_at)):
            raise ValueError("receipt times must be integer timestamps")
        if not self.issued_at <= self.expires_at <= self.issued_at + _MAX_RECEIPT_LIFETIME_SECONDS:
            raise ValueError("receipt expiry is outside the bounded lifetime")
        if self.result_hash != self.expected_result_hash:
            raise ValueError("result_hash does not bind verification receipt contents")

    @property
    def expected_result_hash(self) -> str:
        return _hash(
            {
                "benchmark_result_hash": self.benchmark_result_hash,
                "candidate_id": self.candidate_id,
                "expires_at": self.expires_at,
                "issued_at": self.issued_at,
                "sandbox_disposable": self.sandbox_disposable,
                "sandbox_id": self.sandbox_id,
                "sandbox_task_count": self.sandbox_task_count,
                "status": self.status,
                "verifier_identity": self.verifier_identity,
            }
        )


CanaryStatus = Literal["passed", "rejected"]


@dataclass(frozen=True, slots=True)
class CanaryReceipt:
    """A synthetic local-only no-send canary bound to a staged candidate.

    This is intentionally an observation of durable local records.  It never
    constructs a worker, dispatches a task, invokes a provider, or changes a
    registry entry.  A separate active approval is still required afterwards.
    """

    candidate_id: str
    promotion_proof_hash: str
    verification_result_hash: str
    mode: Literal["local-no-send"]
    status: CanaryStatus
    issued_at: int
    result_hash: str

    def __post_init__(self) -> None:
        _require_identity(self.candidate_id, field="candidate_id")
        for field in ("promotion_proof_hash", "verification_result_hash", "result_hash"):
            _require_hash(getattr(self, field), field=field)
        if self.mode != "local-no-send" or self.status not in {"passed", "rejected"}:
            raise ValueError("canary must be a local no-send receipt with a valid status")
        if isinstance(self.issued_at, bool) or not isinstance(self.issued_at, int):
            raise ValueError("canary issued_at must be an integer timestamp")
        if self.result_hash != self.expected_result_hash:
            raise ValueError("result_hash does not bind canary receipt contents")

    @property
    def expected_result_hash(self) -> str:
        return _hash(
            {
                "candidate_id": self.candidate_id,
                "issued_at": self.issued_at,
                "mode": self.mode,
                "promotion_proof_hash": self.promotion_proof_hash,
                "status": self.status,
                "verification_result_hash": self.verification_result_hash,
            }
        )


@dataclass(frozen=True, slots=True)
class OperatorApproval:
    """A separately recorded local operator decision; absence is not approval."""

    candidate_id: str
    approval_id: str
    operator_identity: str
    verification_result_hash: str
    target_state: Literal["staged", "active"]
    approved: bool
    issued_at: int

    def __post_init__(self) -> None:
        for field in ("candidate_id", "approval_id", "operator_identity"):
            _require_identity(getattr(self, field), field=field)
        _require_hash(self.verification_result_hash, field="verification_result_hash")
        if self.target_state not in {"staged", "active"}:
            raise ValueError("operator approval target must be staged or active")
        if not isinstance(self.approved, bool) or isinstance(self.issued_at, bool) or not isinstance(self.issued_at, int):
            raise ValueError("operator approval fields are malformed")

    @property
    def approval_hash(self) -> str:
        return _hash(
            {
                "approval_id": self.approval_id,
                "approved": self.approved,
                "candidate_id": self.candidate_id,
                "issued_at": self.issued_at,
                "operator_identity": self.operator_identity,
                "target_state": self.target_state,
                "verification_result_hash": self.verification_result_hash,
            }
        )


@dataclass(frozen=True, slots=True)
class PromotionResult:
    status: Literal["benchmarked", "sandbox", "verified", "staged", "active", "rejected"]
    reason: str
    snapshot: CandidateLifecycleSnapshot | None = None


class CandidatePromotionGate:
    """Validate only durable local evidence, then append a legal transition."""

    def __init__(
        self,
        requests: CandidateProfileRequests,
        registry: CapabilityRegistry,
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._requests = requests
        self._registry = registry
        self._clock = clock

    def benchmark(
        self,
        proposal: CandidateProposal,
        cases: FrozenBenchmarkCaseSet,
        *,
        scorer_model: str,
        scores: tuple[int, ...] = (),
        pass_threshold: int = 80,
        scorer_available: bool = False,
        receipt_lifetime_seconds: int = 3_600,
    ) -> tuple[PromotionResult, BenchmarkReceipt]:
        """Record only a local receipt; unavailable scorers cause no call or transition."""
        now = int(self._clock())
        status: BenchmarkStatus = "unavailable"
        if not self._proposal_matches_candidate(proposal):
            status = "rejected"
        elif not self._independent_scorer(proposal, scorer_model):
            status = "rejected"
        elif not scorer_available:
            status = "unavailable"
        elif not scores or len(scores) != len(cases.case_ids):
            status = "rejected"
        elif min(scores) >= pass_threshold:
            status = "passed"
        else:
            status = "failed"
        receipt = self._benchmark_receipt(
            proposal, cases, scorer_model, scores, pass_threshold, status, now, receipt_lifetime_seconds
        )
        if status != "passed":
            return PromotionResult("rejected", f"benchmark {status}"), receipt
        self._store_benchmark(receipt, proposal)
        snapshot = self._append(proposal.candidate_id, "candidate", "benchmarked", "benchmark_passed", receipt.result_hash)
        return self._transition_result("benchmarked", snapshot, "benchmark transition rejected"), receipt

    def open_disposable_sandbox(
        self, proposal: CandidateProposal, benchmark: BenchmarkReceipt, *, sandbox_id: str,
        sandbox_disposable: bool = True, sandbox_task_count: int = 1,
    ) -> PromotionResult:
        if not self.valid_benchmark(benchmark, proposal, None):
            return PromotionResult("rejected", "benchmark receipt is invalid")
        if not sandbox_disposable or sandbox_task_count != 1:
            return PromotionResult("rejected", "sandbox must be disposable and contain exactly one task")
        try:
            _require_identity(sandbox_id, field="sandbox_id")
            with self._connection() as conn:
                with kanban_db.write_txn(conn):
                    conn.execute(
                        "INSERT INTO specialist_sandbox_runs (sandbox_id, candidate_id, benchmark_result_hash, disposable, task_count, created_at) VALUES (?, ?, ?, 1, 1, ?)",
                        (sandbox_id, proposal.candidate_id, benchmark.result_hash, int(self._clock())),
                    )
        except Exception as exc:
            return PromotionResult("rejected", f"sandbox record unavailable: {type(exc).__name__}")
        return PromotionResult("benchmarked", "durable disposable sandbox record created")

    def verify(
        self,
        proposal: CandidateProposal,
        benchmark: BenchmarkReceipt,
        *,
        verifier_identity: str,
        sandbox_id: str,
        verifier_available: bool = False,
        receipt_lifetime_seconds: int = 3_600,
    ) -> tuple[PromotionResult, VerificationReceipt]:
        now = int(self._clock())
        valid_benchmark = self.valid_benchmark(benchmark, proposal, None)
        independent = self._independent_verifier(proposal, benchmark, verifier_identity)
        status: VerificationStatus = "verified"
        if not valid_benchmark or not independent or not self._has_disposable_sandbox(proposal.candidate_id, benchmark.result_hash, sandbox_id):
            status = "rejected"
        elif not verifier_available:
            status = "unavailable"
        receipt = self._verification_receipt(
            proposal.candidate_id, benchmark.result_hash, verifier_identity, sandbox_id,
            True, 1, status, now, receipt_lifetime_seconds,
        )
        if status != "verified":
            return PromotionResult("rejected", f"verification {status}"), receipt
        self._store_verification(receipt)
        snapshot = self._append(proposal.candidate_id, "benchmarked", "verified", "sandbox_verified", receipt.result_hash)
        return self._transition_result("verified", snapshot, "verification transition rejected"), receipt

    def stage(
        self,
        proposal: CandidateProposal,
        benchmark: BenchmarkReceipt,
        verification: VerificationReceipt,
        approval: OperatorApproval | None,
    ) -> PromotionResult:
        if not self.valid_benchmark(benchmark, proposal, None) or not self.valid_verification(verification, proposal, benchmark, None):
            return PromotionResult("rejected", "benchmark or verification receipt is invalid")
        if not self._valid_approval(approval, proposal, verification, "staged"):
            return PromotionResult("rejected", "separate operator approval is required before staging")
        if self._store_proof(proposal, "staged", None, benchmark, verification, approval) is None:
            return PromotionResult("rejected", "durable staged-promotion proof is unavailable")
        snapshot = self._append(proposal.candidate_id, "verified", "staged", "operator_approved", approval.approval_hash)
        return self._transition_result("staged", snapshot, "staging transition rejected")

    def run_local_no_send_canary(
        self, proposal: CandidateProposal, verification: VerificationReceipt
    ) -> tuple[PromotionResult, CanaryReceipt | None]:
        """Persist one synthetic local-only canary for an already staged profile.

        The implementation only reads and writes the local Kanban ledger.  In
        particular it does not call a model/adaptor, create a profile, or emit
        a task/message.  Repeating the same durable input safely reuses the
        same immutable receipt.
        """
        now = int(self._clock())
        current = self._requests.lifecycle_snapshot(proposal.candidate_id)
        proof = self._staged_proof(proposal.candidate_id, verification.result_hash)
        benchmark = self._benchmark_for(verification)
        if (
            current is None
            or current.lifecycle_status != "staged"
            or benchmark is None
            or not self.valid_verification(verification, proposal, benchmark, None)
            or proof is None
        ):
            return PromotionResult("rejected", "staged durable evidence is required before a no-send canary"), None
        provisional = {
            "candidate_id": proposal.candidate_id,
            "promotion_proof_hash": proof,
            "verification_result_hash": verification.result_hash,
            "mode": "local-no-send",
            "status": "passed",
            "issued_at": now,
        }
        receipt = CanaryReceipt(**provisional, result_hash=_hash(provisional))
        try:
            with self._connection() as conn:
                with kanban_db.write_txn(conn):
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO specialist_canary_receipts (
                            result_hash, candidate_id, promotion_proof_hash,
                            verification_result_hash, mode, status, issued_at, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            receipt.result_hash,
                            receipt.candidate_id,
                            receipt.promotion_proof_hash,
                            receipt.verification_result_hash,
                            receipt.mode,
                            receipt.status,
                            receipt.issued_at,
                            now,
                        ),
                    )
        except Exception as exc:
            return PromotionResult("rejected", f"canary receipt unavailable: {type(exc).__name__}"), None
        return PromotionResult("staged", "durable local no-send canary recorded", current), receipt

    def activate(
        self,
        proposal: CandidateProposal,
        signature: CapabilitySignature,
        profile_id: str,
        benchmark: BenchmarkReceipt,
        verification: VerificationReceipt,
        approval: OperatorApproval | None,
        *,
        expires_at: int | None = None,
    ) -> PromotionResult:
        if signature.signature_hash != proposal.signature_hash or signature.permissions_hash != proposal.permissions_hash:
            return PromotionResult("rejected", "activation signature or permissions differ from the benchmarked candidate")
        if not self.valid_benchmark(benchmark, proposal, None) or not self.valid_verification(verification, proposal, benchmark, None):
            return PromotionResult("rejected", "benchmark or verification receipt is invalid")
        if not self._valid_approval(approval, proposal, verification, "active"):
            return PromotionResult("rejected", "separate operator approval is required before activation")
        current = self._requests.lifecycle_snapshot(proposal.candidate_id)
        if current is None or current.lifecycle_status != "staged":
            return PromotionResult("rejected", "candidate must be staged before activation")
        if not self._has_local_no_send_canary(proposal.candidate_id, verification.result_hash):
            return PromotionResult("rejected", "durable local no-send canary is required before activation")
        proof_hash = self._store_proof(proposal, "active", profile_id, benchmark, verification, approval)
        if proof_hash is None:
            return PromotionResult("rejected", "durable authorized promotion proof is unavailable")
        try:
            self._registry.add_active_from_durable_promotion(
                profile_id=profile_id,
                signature=signature,
                candidate_id=proposal.candidate_id,
                promotion_proof_hash=proof_hash,
                expires_at=expires_at,
                now=int(self._clock()),
            )
        except Exception as exc:
            return PromotionResult("rejected", f"capability activation unavailable: {type(exc).__name__}")
        snapshot = self._append(proposal.candidate_id, "staged", "active", "profile_activated", approval.approval_hash)
        return self._transition_result("active", snapshot, "activation transition rejected")

    def valid_benchmark(
        self,
        receipt: BenchmarkReceipt,
        proposal: CandidateProposal,
        cases: FrozenBenchmarkCaseSet | None,
    ) -> bool:
        now = int(self._clock())
        return (
            receipt.status == "passed"
            and receipt.result_hash == receipt.expected_result_hash
            and receipt.issued_at <= now < receipt.expires_at
            and receipt.candidate_id == proposal.candidate_id
            and receipt.proposal_input_hash == proposal.input_hash
            and (cases is None or receipt.case_set_hash == cases.case_set_hash)
            and self._independent_scorer(proposal, receipt.scorer_model)
            and bool(receipt.scores)
            and min(receipt.scores) >= receipt.pass_threshold
            and self._proposal_matches_candidate(proposal)
            and self._stored_benchmark_matches(receipt, proposal)
        )

    def valid_verification(
        self,
        receipt: VerificationReceipt,
        proposal: CandidateProposal,
        benchmark: BenchmarkReceipt,
        cases: FrozenBenchmarkCaseSet | None,
    ) -> bool:
        now = int(self._clock())
        return (
            self.valid_benchmark(benchmark, proposal, cases)
            and receipt.status == "verified"
            and receipt.result_hash == receipt.expected_result_hash
            and receipt.issued_at <= now < receipt.expires_at
            and receipt.candidate_id == proposal.candidate_id
            and receipt.benchmark_result_hash == benchmark.result_hash
            and receipt.sandbox_disposable
            and receipt.sandbox_task_count == 1
            and self._has_disposable_sandbox(receipt.candidate_id, receipt.benchmark_result_hash, receipt.sandbox_id)
            and self._independent_verifier(proposal, benchmark, receipt.verifier_identity)
            and self._stored_verification_matches(receipt)
        )

    def _proposal_matches_candidate(self, proposal: CandidateProposal) -> bool:
        snapshot = self._requests.lifecycle_snapshot(proposal.candidate_id)
        return bool(
            snapshot
            and snapshot.lifecycle_status
            in {"candidate", "benchmarked", "verified", "staged"}
            and snapshot.signature_hash == proposal.signature_hash
            and snapshot.permissions_hash == proposal.permissions_hash
        )

    @staticmethod
    def _independent_scorer(proposal: CandidateProposal, scorer: str) -> bool:
        try:
            scorer = _require_identity(scorer, field="scorer_model")
        except ValueError:
            return False
        return scorer not in {proposal.proposal_author, proposal.sol_reviewer}

    @staticmethod
    def _independent_verifier(proposal: CandidateProposal, benchmark: BenchmarkReceipt, verifier: str) -> bool:
        try:
            verifier = _require_identity(verifier, field="verifier_identity")
        except ValueError:
            return False
        return verifier not in {proposal.proposal_author, proposal.sol_reviewer, benchmark.scorer_model}

    def _append(self, candidate_id: str, expected: str, next_status: str, reason: str, receipt_hash: str) -> CandidateLifecycleSnapshot | None:
        return self._requests.append_lifecycle_transition(
            candidate_id,
            expected_status=expected,
            next_status=next_status,
            reason_code=reason,
            receipt_hash=receipt_hash,
        )

    @staticmethod
    def _transition_result(status: Literal["benchmarked", "sandbox", "verified", "staged", "active"], snapshot: CandidateLifecycleSnapshot | None, reason: str) -> PromotionResult:
        if snapshot is None or snapshot.lifecycle_status != status:
            return PromotionResult("rejected", reason, snapshot)
        return PromotionResult(status, "append-only lifecycle transition recorded", snapshot)

    def record_operator_approval(
        self, approval: OperatorApproval, *, authenticated_operator_identity: str | None = None
    ) -> bool:
        """Persist an approval only when the dashboard auth boundary attests it.

        A caller-supplied identity is never sufficient.  The authenticated
        dashboard route supplies the verified, allowlisted subject and binds it
        to the approval's durable identity before this method can append a row.
        """
        if authenticated_operator_identity != approval.operator_identity:
            return False
        try:
            _require_identity(authenticated_operator_identity, field="authenticated_operator_identity")
        except ValueError:
            return False
        with self._connection() as conn:
            with kanban_db.write_txn(conn):
                conn.execute(
                    "INSERT INTO specialist_operator_approvals (approval_hash, approval_id, candidate_id, target_state, operator_identity, verification_result_hash, approved, issued_at, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        approval.approval_hash,
                        approval.approval_id,
                        approval.candidate_id,
                        approval.target_state,
                        approval.operator_identity,
                        approval.verification_result_hash,
                        int(approval.approved),
                        approval.issued_at,
                        int(self._clock()),
                    ),
                )
        return True

    def _valid_approval(self, approval: OperatorApproval | None, proposal: CandidateProposal, verification: VerificationReceipt, target: str) -> bool:
        if approval is None or not approval.approved or approval.target_state != target:
            return False
        if approval.candidate_id != proposal.candidate_id:
            return False
        if approval.verification_result_hash != verification.result_hash:
            return False
        with self._connection() as conn:
            row = conn.execute(
                "SELECT approval_hash, operator_identity, approved FROM specialist_operator_approvals WHERE approval_hash = ? AND candidate_id = ? AND target_state = ?",
                (approval.approval_hash, proposal.candidate_id, target),
            ).fetchone()
        return bool(
            row
            and row["operator_identity"] == approval.operator_identity
            and bool(row["approved"])
        )

    def _connection(self):
        return kanban_db.connect_closing(self._requests._db_path, board=self._requests._board)

    def _store_benchmark(self, receipt: BenchmarkReceipt, proposal: CandidateProposal) -> None:
        with self._connection() as conn:
            with kanban_db.write_txn(conn):
                conn.execute(
                    "INSERT INTO specialist_benchmark_receipts (result_hash, candidate_id, proposal_input_hash, proposal_author, sol_reviewer, signature_hash, permissions_hash, case_set_hash, scorer_model, scores_json, pass_threshold, status, issued_at, expires_at, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (receipt.result_hash, receipt.candidate_id, receipt.proposal_input_hash, proposal.proposal_author, proposal.sol_reviewer, proposal.signature_hash, proposal.permissions_hash, receipt.case_set_hash, receipt.scorer_model, _canonical_json(receipt.scores), receipt.pass_threshold, receipt.status, receipt.issued_at, receipt.expires_at, int(self._clock())),
                )

    def _store_verification(self, receipt: VerificationReceipt) -> None:
        with self._connection() as conn:
            with kanban_db.write_txn(conn):
                conn.execute(
                    "INSERT INTO specialist_verification_receipts (result_hash, candidate_id, benchmark_result_hash, verifier_identity, sandbox_id, status, issued_at, expires_at, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (receipt.result_hash, receipt.candidate_id, receipt.benchmark_result_hash, receipt.verifier_identity, receipt.sandbox_id, receipt.status, receipt.issued_at, receipt.expires_at, int(self._clock())),
                )

    def _stored_benchmark_matches(self, receipt: BenchmarkReceipt, proposal: CandidateProposal) -> bool:
        try:
            with self._connection() as conn:
                row = conn.execute("SELECT * FROM specialist_benchmark_receipts WHERE result_hash = ?", (receipt.result_hash,)).fetchone()
            return bool(row and tuple(json.loads(row["scores_json"])) == receipt.scores and all(row[field] == value for field, value in (("candidate_id", receipt.candidate_id), ("proposal_input_hash", receipt.proposal_input_hash), ("proposal_author", proposal.proposal_author), ("sol_reviewer", proposal.sol_reviewer), ("signature_hash", proposal.signature_hash), ("permissions_hash", proposal.permissions_hash), ("case_set_hash", receipt.case_set_hash), ("scorer_model", receipt.scorer_model), ("pass_threshold", receipt.pass_threshold), ("status", receipt.status), ("issued_at", receipt.issued_at), ("expires_at", receipt.expires_at))))
        except Exception:
            return False

    def _stored_verification_matches(self, receipt: VerificationReceipt) -> bool:
        try:
            with self._connection() as conn:
                row = conn.execute("SELECT * FROM specialist_verification_receipts WHERE result_hash = ?", (receipt.result_hash,)).fetchone()
            return bool(row and all(row[field] == value for field, value in (("candidate_id", receipt.candidate_id), ("benchmark_result_hash", receipt.benchmark_result_hash), ("verifier_identity", receipt.verifier_identity), ("sandbox_id", receipt.sandbox_id), ("status", receipt.status), ("issued_at", receipt.issued_at), ("expires_at", receipt.expires_at))))
        except Exception:
            return False

    def _stored_approval_matches(self, approval: OperatorApproval) -> bool:
        try:
            with self._connection() as conn:
                row = conn.execute("SELECT approval_hash FROM specialist_operator_approvals WHERE approval_hash = ? AND candidate_id = ? AND target_state = ? AND verification_result_hash = ? AND approved = 1", (approval.approval_hash, approval.candidate_id, approval.target_state, approval.verification_result_hash)).fetchone()
            return row is not None
        except Exception:
            return False

    def _has_disposable_sandbox(self, candidate_id: str, benchmark_hash: str, sandbox_id: str) -> bool:
        try:
            with self._connection() as conn:
                row = conn.execute("SELECT 1 FROM specialist_sandbox_runs WHERE sandbox_id = ? AND candidate_id = ? AND benchmark_result_hash = ? AND disposable = 1 AND task_count = 1", (sandbox_id, candidate_id, benchmark_hash)).fetchone()
            return row is not None
        except Exception:
            return False

    def _staged_proof(self, candidate_id: str, verification_hash: str) -> str | None:
        try:
            with self._connection() as conn:
                row = conn.execute(
                    """
                    SELECT proof_hash FROM specialist_promotion_proofs
                    WHERE candidate_id = ? AND target_state = 'staged'
                      AND verification_result_hash = ?
                    ORDER BY id DESC LIMIT 1
                    """,
                    (candidate_id, verification_hash),
                ).fetchone()
            return row["proof_hash"] if row else None
        except Exception:
            return None

    def _benchmark_for(self, verification: VerificationReceipt) -> BenchmarkReceipt | None:
        """Reconstruct a stored benchmark for local canary validation only."""
        try:
            with self._connection() as conn:
                row = conn.execute(
                    "SELECT * FROM specialist_benchmark_receipts WHERE result_hash = ?",
                    (verification.benchmark_result_hash,),
                ).fetchone()
            if row is None:
                return None
            scores = tuple(json.loads(row["scores_json"]))
            return BenchmarkReceipt(
                candidate_id=row["candidate_id"],
                proposal_input_hash=row["proposal_input_hash"],
                case_set_hash=row["case_set_hash"],
                scorer_model=row["scorer_model"],
                scores=scores,
                pass_threshold=row["pass_threshold"],
                status=row["status"],
                issued_at=row["issued_at"],
                expires_at=row["expires_at"],
                result_hash=row["result_hash"],
            )
        except Exception:
            return None

    def _has_local_no_send_canary(self, candidate_id: str, verification_hash: str) -> bool:
        try:
            with self._connection() as conn:
                row = conn.execute(
                    """
                    SELECT 1 FROM specialist_canary_receipts
                    WHERE candidate_id = ? AND verification_result_hash = ?
                      AND mode = 'local-no-send' AND status = 'passed'
                    """,
                    (candidate_id, verification_hash),
                ).fetchone()
            return row is not None
        except Exception:
            return False

    def _store_proof(self, proposal: CandidateProposal, target: str, profile_id: str | None, benchmark: BenchmarkReceipt, verification: VerificationReceipt, approval: OperatorApproval) -> str | None:
        proof_hash = _hash({"candidate_id": proposal.candidate_id, "target": target, "profile_id": profile_id, "signature_hash": proposal.signature_hash, "permissions_hash": proposal.permissions_hash, "benchmark": benchmark.result_hash, "verification": verification.result_hash, "approval": approval.approval_hash})
        try:
            with self._connection() as conn:
                with kanban_db.write_txn(conn):
                    conn.execute("INSERT INTO specialist_promotion_proofs (proof_hash, candidate_id, target_state, profile_id, signature_hash, permissions_hash, benchmark_result_hash, verification_result_hash, approval_hash, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (proof_hash, proposal.candidate_id, target, profile_id, proposal.signature_hash, proposal.permissions_hash, benchmark.result_hash, verification.result_hash, approval.approval_hash, int(self._clock())))
            return proof_hash
        except Exception:
            return None

    @staticmethod
    def _benchmark_receipt(
        proposal: CandidateProposal,
        cases: FrozenBenchmarkCaseSet,
        scorer: str,
        scores: tuple[int, ...],
        threshold: int,
        status: BenchmarkStatus,
        issued_at: int,
        lifetime: int,
    ) -> BenchmarkReceipt:
        expires_at = issued_at + max(1, min(lifetime, _MAX_RECEIPT_LIFETIME_SECONDS))
        provisional = {
            "candidate_id": proposal.candidate_id, "proposal_input_hash": proposal.input_hash,
            "case_set_hash": cases.case_set_hash, "scorer_model": scorer, "scores": scores,
            "pass_threshold": threshold, "status": status, "issued_at": issued_at, "expires_at": expires_at,
        }
        return BenchmarkReceipt(**provisional, result_hash=_hash(provisional))

    @staticmethod
    def _verification_receipt(
        candidate_id: str, benchmark_hash: str, verifier: str, sandbox_id: str,
        disposable: bool, task_count: int, status: VerificationStatus, issued_at: int, lifetime: int,
    ) -> VerificationReceipt:
        expires_at = issued_at + max(1, min(lifetime, _MAX_RECEIPT_LIFETIME_SECONDS))
        provisional = {
            "candidate_id": candidate_id, "benchmark_result_hash": benchmark_hash,
            "verifier_identity": verifier, "sandbox_id": sandbox_id,
            "sandbox_disposable": disposable, "sandbox_task_count": task_count,
            "status": status, "issued_at": issued_at, "expires_at": expires_at,
        }
        return VerificationReceipt(**provisional, result_hash=_hash(provisional))
