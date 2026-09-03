"""Local, fail-closed storage for specialist capability declarations.

This module deliberately does not route, dispatch, or invoke model providers.
It only persists locally verified declarations and resolves a supplied scope.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from hermes_cli import kanban_db


_MAX_EXPIRY_TIMESTAMP = 253402300799  # 9999-12-31T23:59:59Z
_RECEIPT_HASH_RE = re.compile(r"^[0-9a-f]{64}$")

# Fixed baseline profiles are deployed configuration, not candidate output.
# Keep their permitted diagnostic-only scope in this module so the registry can
# validate registration without importing the router (which imports this
# module).  Every generated profile must use durable-promotion evidence.
_FIXED_BASELINE_SCOPES = {
    "task-orchestrator": ("repository-evidence", ("audit", "inspect", "read", "review", "validate"), "repository-evidence:read"),
    "burndown-patch-steward": ("repository-evidence", ("audit", "inspect", "read", "review", "validate"), "repository-evidence:read"),
    "acceptance-gate-verifier": ("repository-evidence", ("audit", "inspect", "read", "review", "validate"), "repository-evidence:read"),
    "paper-safety-guardian": ("financial-analysis", ("audit", "inspect", "read", "review", "validate"), "financial-analysis:read"),
    "market-data-authority-auditor": ("market-data", ("audit", "inspect", "read", "review", "validate"), "market-data:read"),
    "route-execution-boundary-auditor": ("repository-evidence", ("audit", "inspect", "read", "review", "validate"), "repository-evidence:read"),
    "dependency-tooling-health-sentinel": ("repository-evidence", ("audit", "inspect", "read", "review", "validate"), "repository-evidence:read"),
    "copilot-learning-steward": ("repository-evidence", ("audit", "inspect", "read", "review", "validate"), "repository-evidence:read"),
    "mission-control-ux-auditor": ("repository-evidence", ("audit", "inspect", "read", "review", "validate"), "repository-evidence:read"),
    "research-scout": ("research", ("audit", "read", "research", "review"), "research:read"),
    "performance-sentinel": ("repository-evidence", ("audit", "inspect", "read", "review", "validate"), "repository-evidence:read"),
}


def _canonical_tokens(values: tuple[str, ...], *, field: str) -> tuple[str, ...]:
    if isinstance(values, str):
        raise TypeError(f"{field} must be a tuple of strings")
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError(f"{field} must contain only non-empty strings")
    return tuple(sorted(set(values)))


def _hash_payload(payload: object) -> str:
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class CapabilitySignature:
    """Canonical scope that a specialist profile is allowed to advise on."""

    domain: str
    actions: tuple[str, ...]
    evidence_class: str
    requested_permissions: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.domain, str) or not self.domain.strip():
            raise ValueError("domain must be a non-empty string")
        if not isinstance(self.evidence_class, str) or not self.evidence_class.strip():
            raise ValueError("evidence_class must be a non-empty string")
        object.__setattr__(self, "actions", _canonical_tokens(self.actions, field="actions"))
        object.__setattr__(
            self,
            "requested_permissions",
            _canonical_tokens(self.requested_permissions, field="requested_permissions"),
        )

    @property
    def signature_hash(self) -> str:
        return _hash_payload(
            {
                "actions": self.actions,
                "domain": self.domain,
                "evidence_class": self.evidence_class,
            }
        )

    @property
    def permissions_hash(self) -> str:
        return _hash_payload({"requested_permissions": self.requested_permissions})


@dataclass(frozen=True, slots=True)
class RegistryResolution:
    """A fail-closed registry result; only ``active_match`` has a profile."""

    status: Literal["active_match", "no_match", "ambiguous", "unavailable"]
    profile: str | None
    reason: str


def _expires_at_timestamp(expires_at: datetime | int | float | None) -> int | None:
    if expires_at is None:
        return None
    if isinstance(expires_at, datetime):
        if expires_at.tzinfo is None:
            raise ValueError("expires_at datetime must be timezone-aware")
        return int(expires_at.timestamp())
    if isinstance(expires_at, bool) or not isinstance(expires_at, (int, float)):
        raise TypeError("expires_at must be a timestamp or timezone-aware datetime")
    return int(expires_at)


def _is_unexpired_stored_timestamp(expires_at: object, *, now: int) -> bool:
    """Return false for malformed SQLite expiry metadata rather than coercing it."""
    if expires_at is None:
        return True
    if isinstance(expires_at, bool) or not isinstance(expires_at, int):
        return False
    return now < expires_at <= _MAX_EXPIRY_TIMESTAMP


def _authenticated_operator_approval_authority_available() -> bool:
    """Whether a separately verified operator-approval integration is wired.

    Phase 0 has no such integration.  Keep this explicit predicate so a future
    authenticated authority must change a narrow, reviewable boundary rather
    than silently trusting the descriptive ``operator_identity`` field.
    """
    return False


class CapabilityRegistry:
    """Persist and resolve capability records in the local Kanban SQLite DB."""

    def __init__(self, *, db_path: Path | None = None, board: str | None = None) -> None:
        self._db_path = db_path
        self._board = board

    def register_fixed_baseline(
        self,
        *,
        profile_id: str,
        signature: CapabilitySignature,
        expires_at: datetime | int | float | None = None,
    ) -> None:
        """Register one closed, configured baseline specialist declaration.

        This is intentionally not a generic profile-registration API.  It
        accepts only a profile and exact scope compiled into the fixed routing
        baseline; generated profiles can only reach active state through
        :meth:`add_active_from_durable_promotion`.
        """
        fixed_scope = _FIXED_BASELINE_SCOPES.get(profile_id)
        if fixed_scope is None:
            raise ValueError("profile_id is not a fixed baseline specialist")
        if not isinstance(signature, CapabilitySignature):
            raise TypeError("signature must be a CapabilitySignature")
        domain, allowed_actions, permission = fixed_scope
        if (
            signature.domain != domain
            or signature.evidence_class != "diagnostic-only"
            or not signature.actions
            or not set(signature.actions) <= set(allowed_actions)
            or not signature.requested_permissions
            or not set(signature.requested_permissions) <= {permission}
        ):
            raise ValueError("fixed baseline profile scope does not match its closed declaration")
        self._append_active(
            profile_id=profile_id,
            signature=signature,
            model_receipt_hash="",
            verification_receipt_hash="",
            expires_at=expires_at,
        )

    def add_active(self, **_: object) -> None:
        """Reject the former public arbitrary-profile activation API."""
        raise ValueError(
            "direct arbitrary active profile registration is disabled; use fixed baseline or durable promotion"
        )

    def _append_active(
        self,
        *,
        profile_id: str,
        signature: CapabilitySignature,
        model_receipt_hash: str,
        verification_receipt_hash: str,
        expires_at: datetime | int | float | None,
    ) -> None:
        """Internal persistence primitive reached only by closed gate paths."""

        expires_at_timestamp = _expires_at_timestamp(expires_at)
        created_at = int(time.time())
        with kanban_db.connect_closing(self._db_path, board=self._board) as conn:
            with kanban_db.write_txn(conn):
                conn.execute(
                    """
                    INSERT INTO capability_profiles (
                        profile_id, signature_hash, permissions_hash,
                        model_receipt_hash, verification_receipt_hash,
                        domain, actions_json, evidence_class, requested_permissions_json,
                        expires_at, status, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?)
                    """,
                    (
                        profile_id,
                        signature.signature_hash,
                        signature.permissions_hash,
                        model_receipt_hash,
                        verification_receipt_hash,
                        signature.domain,
                        json.dumps(signature.actions, separators=(",", ":")),
                        signature.evidence_class,
                        json.dumps(signature.requested_permissions, separators=(",", ":")),
                        expires_at_timestamp,
                        created_at,
                    ),
                )

    def add_active_from_promotion(
        self,
        *,
        profile_id: str,
        signature: CapabilitySignature,
        benchmark_receipt_hash: str,
        verification_receipt_hash: str,
        expires_at: datetime | int | float | None = None,
    ) -> None:
        """Disabled unsafe compatibility entry point.

        Hash-shaped strings alone are not authorization.  The only promotion
        entry point is :meth:`add_active_from_durable_promotion`, which reads
        and validates immutable board-local proof, receipt, sandbox, and
        approval rows before delegating to the fixed-profile append path.
        """
        del profile_id, signature, benchmark_receipt_hash, verification_receipt_hash, expires_at
        raise ValueError("hash-only promotion is disabled; use durable promotion proof")

    def add_active_from_durable_promotion(
        self,
        *,
        profile_id: str,
        signature: CapabilitySignature,
        candidate_id: str,
        promotion_proof_hash: str,
        expires_at: datetime | int | float | None = None,
        now: int | None = None,
    ) -> None:
        """Derive an active profile from a stored active-promotion proof only."""
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError("candidate_id must be a non-empty string")
        if not isinstance(promotion_proof_hash, str) or not _RECEIPT_HASH_RE.fullmatch(promotion_proof_hash):
            raise ValueError("promotion_proof_hash must be a SHA-256 hex digest")
        # No authenticated, operator-controlled approval authority is wired in
        # this integration.  Refuse even a durable-looking legacy proof rather
        # than treating a historical identity string as authorization.
        if not _authenticated_operator_approval_authority_available():
            raise ValueError("authenticated operator approval authority is unavailable")
        # The durable validation below is retained as the future gate once an
        # authenticated approval authority is separately integrated.
        with kanban_db.connect_closing(self._db_path, board=self._board) as conn:
            proof = conn.execute(
                """
                SELECT benchmark_result_hash, verification_result_hash, approval_hash
                FROM specialist_promotion_proofs
                WHERE proof_hash = ? AND candidate_id = ? AND target_state = 'active'
                  AND profile_id = ? AND signature_hash = ? AND permissions_hash = ?
                """,
                (promotion_proof_hash, candidate_id, profile_id, signature.signature_hash, signature.permissions_hash),
            ).fetchone()
            if proof is None:
                raise ValueError("no durable authorized active-promotion proof")
            benchmark = conn.execute(
                """
                SELECT proposal_author, sol_reviewer, scorer_model, status, expires_at
                FROM specialist_benchmark_receipts
                WHERE result_hash = ? AND candidate_id = ?
                """,
                (proof["benchmark_result_hash"], candidate_id),
            ).fetchone()
            verification = conn.execute(
                """
                SELECT verifier_identity, sandbox_id, status, expires_at
                FROM specialist_verification_receipts
                WHERE result_hash = ? AND candidate_id = ? AND benchmark_result_hash = ?
                """,
                (proof["verification_result_hash"], candidate_id, proof["benchmark_result_hash"]),
            ).fetchone()
            now = int(time.time()) if now is None else now
            if isinstance(now, bool) or not isinstance(now, int):
                raise ValueError("now must be an integer timestamp")
            if (
                benchmark is None
                or verification is None
                or benchmark["status"] != "passed"
                or verification["status"] != "verified"
                or not _is_unexpired_stored_timestamp(benchmark["expires_at"], now=now)
                or not _is_unexpired_stored_timestamp(verification["expires_at"], now=now)
                or benchmark["scorer_model"] in {benchmark["proposal_author"], benchmark["sol_reviewer"]}
                or verification["verifier_identity"]
                in {benchmark["proposal_author"], benchmark["sol_reviewer"], benchmark["scorer_model"]}
            ):
                raise ValueError("durable benchmark or independent verification is invalid")
            sandbox = conn.execute(
                """
                SELECT 1 FROM specialist_sandbox_runs
                WHERE sandbox_id = ? AND candidate_id = ? AND benchmark_result_hash = ?
                  AND disposable = 1 AND task_count = 1
                """,
                (verification["sandbox_id"], candidate_id, proof["benchmark_result_hash"]),
            ).fetchone()
            if sandbox is None:
                raise ValueError("durable disposable sandbox record is missing")
            approval = conn.execute(
                """
                SELECT 1 FROM specialist_operator_approvals
                WHERE approval_hash = ? AND candidate_id = ? AND target_state = 'active'
                  AND verification_result_hash = ? AND approved = 1
                """,
                (proof["approval_hash"], candidate_id, proof["verification_result_hash"]),
            ).fetchone()
            if approval is None:
                raise ValueError("durable active approval is missing")
            canary = conn.execute(
                """
                SELECT 1 FROM specialist_canary_receipts
                WHERE candidate_id = ? AND verification_result_hash = ?
                  AND mode = 'local-no-send' AND status = 'passed'
                """,
                (candidate_id, proof["verification_result_hash"]),
            ).fetchone()
            if canary is None:
                raise ValueError("durable local no-send canary is missing")
        self._append_active(
            profile_id=profile_id,
            signature=signature,
            model_receipt_hash=proof["benchmark_result_hash"],
            verification_receipt_hash=proof["verification_result_hash"],
            expires_at=expires_at,
        )

    def revoke(
        self,
        *,
        profile_id: str,
        signature: CapabilitySignature,
        reason_code: str,
        now: int | None = None,
    ) -> str:
        """Append a local rollback receipt that immediately removes a profile.

        Revocation is deliberately a separate immutable record rather than an
        update to ``capability_profiles``.  It performs no dispatch, adapter,
        model, or external action; callers must explicitly issue it.
        """
        if not isinstance(profile_id, str) or not profile_id.strip():
            raise ValueError("profile_id must be a non-empty string")
        if not isinstance(signature, CapabilitySignature):
            raise TypeError("signature must be a CapabilitySignature")
        if not isinstance(reason_code, str) or not re.fullmatch(r"[a-z0-9_]{1,64}", reason_code):
            raise ValueError("reason_code must be a bounded canonical code")
        created_at = int(time.time()) if now is None else now
        if isinstance(created_at, bool) or not isinstance(created_at, int):
            raise ValueError("now must be an integer timestamp")
        revocation_hash = _hash_payload(
            {
                "profile_id": profile_id,
                "reason_code": reason_code,
                "signature_hash": signature.signature_hash,
            }
        )
        with kanban_db.connect_closing(self._db_path, board=self._board) as conn:
            with kanban_db.write_txn(conn):
                conn.execute(
                    """
                    INSERT OR IGNORE INTO specialist_profile_revocations (
                        revocation_hash, profile_id, signature_hash, reason_code, created_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (revocation_hash, profile_id, signature.signature_hash, reason_code, created_at),
                )
        return revocation_hash

    def resolve(self, signature: CapabilitySignature) -> RegistryResolution:
        """Resolve exactly one active, unexpired, non-expanding local profile."""
        if not isinstance(signature, CapabilitySignature):
            raise TypeError("signature must be a CapabilitySignature")

        try:
            now = int(time.time())
            with kanban_db.connect_closing(self._db_path, board=self._board) as conn:
                rows = conn.execute(
                    """
                    SELECT profile_id, signature_hash, permissions_hash, domain,
                           actions_json, evidence_class, requested_permissions_json, expires_at
                    FROM capability_profiles AS profiles
                    WHERE status = 'active'
                      AND signature_hash = ?
                      AND evidence_class = ?
                      AND NOT EXISTS (
                          SELECT 1 FROM specialist_profile_revocations AS revocations
                          WHERE revocations.profile_id = profiles.profile_id
                            AND revocations.signature_hash = profiles.signature_hash
                      )
                    ORDER BY profile_id, id
                    """,
                    (signature.signature_hash, signature.evidence_class),
                ).fetchall()
        except Exception as exc:
            return RegistryResolution(
                status="unavailable",
                profile=None,
                reason=f"local capability registry unavailable: {type(exc).__name__}",
            )

        matches: list[str] = []
        requested_permissions = set(signature.requested_permissions)
        for row in rows:
            if not _is_unexpired_stored_timestamp(row["expires_at"], now=now):
                continue
            try:
                stored_signature = CapabilitySignature(
                    domain=row["domain"],
                    actions=tuple(json.loads(row["actions_json"])),
                    evidence_class=row["evidence_class"],
                    requested_permissions=tuple(json.loads(row["requested_permissions_json"])),
                )
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if (
                stored_signature.signature_hash != row["signature_hash"]
                or stored_signature.permissions_hash != row["permissions_hash"]
                or stored_signature.domain != signature.domain
                or stored_signature.actions != signature.actions
                or not set(stored_signature.requested_permissions) <= requested_permissions
            ):
                continue
            matches.append(row["profile_id"])

        if len(matches) == 1:
            return RegistryResolution(
                status="active_match",
                profile=matches[0],
                reason="exact active capability profile matched locally",
            )
        if len(matches) > 1:
            return RegistryResolution(
                status="ambiguous",
                profile=None,
                reason="multiple active capability profiles matched the requested scope",
            )
        return RegistryResolution(
            status="no_match",
            profile=None,
            reason="no active unexpired profile matched the requested scope",
        )
