"""Local, inert candidate requests for missing specialist capabilities.

This ledger deliberately has no provider, model, profile-creation, or dispatch
dependency. A row is a bounded request for later human-governed review, never a
profile that can receive work.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

from gateway.capability_registry import CapabilityRegistry, CapabilitySignature, RegistryResolution
from hermes_cli import kanban_db


CandidateRequestStatus = Literal["candidate", "duplicate", "cooldown", "rejected"]
_LIFECYCLE_NONTERMINAL = frozenset({"candidate", "benchmarked", "verified", "staged", "active"})
_LIFECYCLE_TERMINAL = frozenset({"rejected", "expired", "revoked"})
_ALLOWED_LIFECYCLE_TRANSITIONS = {
    "candidate": "benchmarked",
    "benchmarked": "verified",
    "verified": "staged",
    "staged": "active",
}
_DEFAULT_COOLDOWN_SECONDS = 3_600
_MAX_SOURCE_KEY_CHARS = 512
_MAX_EVIDENCE_REFERENCES = 16
_OPAQUE_REFERENCE_RE = re.compile(r"^[0-9a-f]{64}$")
_REASON_CODE_RE = re.compile(r"^[a-z0-9_]{1,64}$")
_LOCAL_READ_ONLY_SCOPES: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    "financial-analysis": (
        frozenset({"audit", "inspect", "read", "review", "validate"}),
        frozenset({"financial-analysis:read"}),
    ),
    "market-data": (
        frozenset({"audit", "inspect", "read", "review", "validate"}),
        frozenset({"market-data:read"}),
    ),
    "repository-evidence": (
        frozenset({"audit", "inspect", "read", "review", "validate"}),
        frozenset({"repository-evidence:read"}),
    ),
    "research": (
        frozenset({"audit", "read", "research", "review"}),
        frozenset({"research:read"}),
    ),
}
_POLICY = {
    "external_egress": False,
    "local_read_only_scopes": sorted(_LOCAL_READ_ONLY_SCOPES),
    "profile_creation": False,
    "requested_permissions": "explicit-local-read-only",
    "sandbox_tasks": 1,
}


def _canonical_json(value: object) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def _hash(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


DEFAULT_POLICY_DIGEST = _hash(_POLICY)


@dataclass(frozen=True, slots=True)
class OpaqueEvidenceReference:
    """A pre-hashed opaque evidence identity suitable for durable storage."""

    digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.digest, str) or not _OPAQUE_REFERENCE_RE.fullmatch(self.digest):
            raise ValueError("opaque evidence references must be SHA-256 hex digests")


@dataclass(frozen=True, slots=True)
class SanitizedTaskEnvelope:
    """Only bounded opaque evidence digests may enter the candidate ledger."""

    evidence_refs: tuple[OpaqueEvidenceReference, ...] = ()


@dataclass(frozen=True, slots=True)
class CandidateProfileRequest:
    """Result of opening an inert candidate request, never a dispatch target."""

    request_id: str
    status: CandidateRequestStatus
    profile_id: None
    reason: str


@dataclass(frozen=True, slots=True)
class CandidateLifecycleSnapshot:
    """The immutable scope and newest append-only state for one candidate."""

    candidate_id: str
    request_hash: str
    signature_hash: str
    permissions_hash: str
    policy_digest: str
    lifecycle_status: str


def _capability_rejection_code(signature: CapabilitySignature) -> str | None:
    scope = _LOCAL_READ_ONLY_SCOPES.get(signature.domain)
    if scope is None or signature.evidence_class != "diagnostic-only":
        return "unapproved_local_scope"
    allowed_actions, allowed_permissions = scope
    if not signature.actions or not set(signature.actions) <= allowed_actions:
        return "non_read_only_action"
    if not signature.requested_permissions or not set(signature.requested_permissions) <= allowed_permissions:
        return "unapproved_permission"
    return None


def _result_reason(reason_code: str) -> str:
    return {
        "invalid_evidence_refs": "candidate requests store only sanitized evidence references",
        "invalid_policy_digest": "candidate policy digest does not match the fixed local policy",
        "non_read_only_action": "candidate requests require read-only actions",
        "unapproved_local_scope": "candidate requests require an approved local read-only scope",
        "unapproved_permission": "candidate requests require approved local read-only permissions",
        "unresolved_scope": "candidate request requires a no-match or ambiguous local resolution",
    }.get(reason_code, "candidate request rejected by fixed local policy")


def _sanitize_evidence_refs(envelope: SanitizedTaskEnvelope | None) -> tuple[str, ...] | None:
    if envelope is None:
        return ()
    if isinstance(envelope, SanitizedTaskEnvelope):
        refs = envelope.evidence_refs
    else:
        return None
    if not isinstance(refs, tuple) or len(refs) > _MAX_EVIDENCE_REFERENCES:
        return None
    normalized: list[str] = []
    for reference in refs:
        if not isinstance(reference, OpaqueEvidenceReference):
            return None
        normalized.append(reference.digest)
    return tuple(sorted(set(normalized)))


class CandidateProfileRequests:
    """Open idempotent local candidate requests under a fixed least-privilege policy."""

    def __init__(
        self,
        *,
        db_path: Path | None = None,
        board: str | None = None,
        cooldown_seconds: int = _DEFAULT_COOLDOWN_SECONDS,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if isinstance(cooldown_seconds, bool) or not isinstance(cooldown_seconds, int) or cooldown_seconds <= 0:
            raise ValueError("cooldown_seconds must be a positive integer")
        self._db_path = db_path
        self._board = board
        self._cooldown_seconds = cooldown_seconds
        self._clock = clock

    def open_or_reuse(
        self,
        signature: CapabilitySignature,
        *,
        source_key: str,
        resolution: RegistryResolution | None = None,
        envelope: SanitizedTaskEnvelope | None = None,
        policy_digest: str = DEFAULT_POLICY_DIGEST,
    ) -> CandidateProfileRequest:
        """Create or reuse a request only after a concrete local no-match lookup.

        ``resolution`` is retained temporarily for call-site compatibility but
        is deliberately ignored. It is untrusted caller input and cannot
        authorize candidate persistence; this method always resolves the
        supplied signature against its own local Kanban-backed registry.
        """
        if not isinstance(signature, CapabilitySignature):
            raise TypeError("signature must be a CapabilitySignature")
        if not isinstance(source_key, str) or not source_key.strip() or len(source_key) > _MAX_SOURCE_KEY_CHARS:
            raise ValueError("source_key must be a bounded non-empty string")
        local_resolution = CapabilityRegistry(
            db_path=self._db_path, board=self._board
        ).resolve(signature)
        if local_resolution.status not in {"no_match", "ambiguous"}:
            return CandidateProfileRequest(
                request_id="",
                status="rejected",
                profile_id=None,
                reason="candidate request requires a local no-match or ambiguous resolution",
            )
        evidence_refs = _sanitize_evidence_refs(envelope)
        reason_code = _capability_rejection_code(signature)
        stored_policy_digest = DEFAULT_POLICY_DIGEST
        if not isinstance(policy_digest, str) or policy_digest != DEFAULT_POLICY_DIGEST:
            stored_policy_digest = _hash({"untrusted_policy_digest": policy_digest})
            reason_code = "invalid_policy_digest"
        if evidence_refs is None:
            reason_code = "invalid_evidence_refs"

        request_hash = _hash(
            {
                "policy_digest": stored_policy_digest,
                "signature": {
                    "actions": signature.actions,
                    "domain": signature.domain,
                    "evidence_class": signature.evidence_class,
                    "requested_permissions": signature.requested_permissions,
                },
                "source_key": source_key,
            }
        )
        return self._open_or_reuse_latest(
            request_hash=request_hash,
            signature=signature,
            source_key=source_key,
            policy_digest=stored_policy_digest,
            evidence_ref_hashes=evidence_refs or (),
            reason_code=reason_code,
        )

    def _open_or_reuse_latest(
        self,
        *,
        request_hash: str,
        signature: CapabilitySignature,
        source_key: str,
        policy_digest: str,
        evidence_ref_hashes: tuple[str, ...],
        reason_code: str | None,
    ) -> CandidateProfileRequest:
        now = int(self._clock())
        with kanban_db.connect_closing(self._db_path, board=self._board) as conn:
            with kanban_db.write_txn(conn):
                row = conn.execute(
                    """
                    SELECT request_id, lifecycle_status, cooldown_until
                    FROM candidate_profile_requests
                    WHERE request_hash = ?
                    ORDER BY id DESC LIMIT 1
                    """,
                    (request_hash,),
                ).fetchone()
                if row is not None and row["lifecycle_status"] in _LIFECYCLE_TERMINAL:
                    if isinstance(row["cooldown_until"], int) and now < row["cooldown_until"]:
                        return CandidateProfileRequest(
                            request_id=row["request_id"], status="cooldown", profile_id=None,
                            reason="repeated terminal candidate request is in bounded cooldown",
                        )
                elif row is not None and row["lifecycle_status"] in _LIFECYCLE_NONTERMINAL:
                    return CandidateProfileRequest(
                        request_id=row["request_id"], status="duplicate", profile_id=None,
                        reason="reused existing nonterminal inert candidate request",
                    )
                lifecycle_status = "rejected" if reason_code else "candidate"
                cooldown_until = now + self._cooldown_seconds if reason_code else None
                stored_reason_code = reason_code or "candidate_opened"
                request_id = self._insert(
                    conn, request_hash=request_hash, signature=signature, source_key=source_key,
                    policy_digest=policy_digest, evidence_ref_hashes=evidence_ref_hashes,
                    lifecycle_status=lifecycle_status, reason_code=stored_reason_code,
                    cooldown_until=cooldown_until, now=now,
                )
        if reason_code:
            return CandidateProfileRequest(
                request_id=request_id, status="rejected", profile_id=None,
                reason=_result_reason(reason_code),
            )
        return CandidateProfileRequest(
            request_id=request_id, status="candidate", profile_id=None,
            reason="local no-match queued for bounded inert candidate review",
        )

    def lifecycle_snapshot(self, candidate_id: str) -> CandidateLifecycleSnapshot | None:
        """Read a candidate's latest state without treating a transition as mutable.

        Lifecycle rows share the original request hash and are appended with a
        derived request id.  The original candidate id remains the stable
        receipt identity throughout promotion.
        """
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError("candidate_id must be a non-empty string")
        with kanban_db.connect_closing(self._db_path, board=self._board) as conn:
            original = conn.execute(
                """
                SELECT request_id, request_hash, signature_hash, permissions_hash, policy_digest
                FROM candidate_profile_requests WHERE request_id = ?
                """,
                (candidate_id,),
            ).fetchone()
            if original is None:
                return None
            latest = conn.execute(
                """
                SELECT lifecycle_status FROM candidate_profile_requests
                WHERE request_hash = ? ORDER BY id DESC LIMIT 1
                """,
                (original["request_hash"],),
            ).fetchone()
        if latest is None:
            return None
        return CandidateLifecycleSnapshot(
            candidate_id=original["request_id"],
            request_hash=original["request_hash"],
            signature_hash=original["signature_hash"],
            permissions_hash=original["permissions_hash"],
            policy_digest=original["policy_digest"],
            lifecycle_status=latest["lifecycle_status"],
        )

    def append_lifecycle_transition(
        self,
        candidate_id: str,
        *,
        expected_status: str,
        next_status: str,
        reason_code: str,
        receipt_hash: str,
    ) -> CandidateLifecycleSnapshot | None:
        """Append one monotonic lifecycle observation, never update a candidate.

        ``receipt_hash`` is intentionally opaque and only contributes to a
        deterministic derived row id; it is not persisted as raw advisory or
        benchmark content in the candidate ledger.
        """
        if _ALLOWED_LIFECYCLE_TRANSITIONS.get(expected_status) != next_status:
            raise ValueError("candidate lifecycle transition is not permitted")
        if not _REASON_CODE_RE.fullmatch(reason_code):
            raise ValueError("reason_code must be a bounded canonical code")
        if not _OPAQUE_REFERENCE_RE.fullmatch(receipt_hash):
            raise ValueError("receipt_hash must be a SHA-256 hex digest")
        with kanban_db.connect_closing(self._db_path, board=self._board) as conn:
            with kanban_db.write_txn(conn):
                original = conn.execute(
                    """
                    SELECT request_id, request_hash, signature_hash, permissions_hash, source_key_hash,
                           policy_digest, evidence_ref_hashes_json
                    FROM candidate_profile_requests WHERE request_id = ?
                    """,
                    (candidate_id,),
                ).fetchone()
                if original is None:
                    return None
                latest = conn.execute(
                    """
                    SELECT lifecycle_status FROM candidate_profile_requests
                    WHERE request_hash = ? ORDER BY id DESC LIMIT 1
                    """,
                    (original["request_hash"],),
                ).fetchone()
                if latest is None or latest["lifecycle_status"] != expected_status:
                    return None
                transition_id = (
                    f"cpr_{original['request_hash'][:24]}_"
                    f"{_hash((candidate_id, expected_status, next_status, receipt_hash))[:8]}"
                )
                conn.execute(
                    """
                    INSERT INTO candidate_profile_requests (
                        request_id, request_hash, signature_hash, permissions_hash, source_key_hash,
                        policy_digest, evidence_ref_hashes_json, lifecycle_status, reason_code,
                        cooldown_until, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?)
                    """,
                    (
                        transition_id,
                        original["request_hash"],
                        original["signature_hash"],
                        original["permissions_hash"],
                        original["source_key_hash"],
                        original["policy_digest"],
                        original["evidence_ref_hashes_json"],
                        next_status,
                        reason_code,
                        int(self._clock()),
                    ),
                )
        return self.lifecycle_snapshot(candidate_id)

    @staticmethod
    def _insert(
        conn: object,
        *,
        request_hash: str,
        signature: CapabilitySignature,
        source_key: str,
        policy_digest: str,
        evidence_ref_hashes: tuple[str, ...],
        lifecycle_status: str,
        reason_code: str,
        cooldown_until: int | None,
        now: int,
    ) -> str:
        if not _OPAQUE_REFERENCE_RE.fullmatch(request_hash):
            raise ValueError("request_hash must be a SHA-256 hex digest")
        if any(not _OPAQUE_REFERENCE_RE.fullmatch(value) for value in evidence_ref_hashes):
            raise ValueError("evidence_ref_hashes must contain only SHA-256 hex digests")
        if lifecycle_status not in _LIFECYCLE_NONTERMINAL | _LIFECYCLE_TERMINAL:
            raise ValueError("lifecycle_status must be a declared candidate lifecycle state")
        if not _OPAQUE_REFERENCE_RE.fullmatch(policy_digest):
            policy_digest = _hash({"untrusted_policy_digest": policy_digest})
        if not _REASON_CODE_RE.fullmatch(reason_code):
            reason_code = "internal_rejection"
        request_id = f"cpr_{request_hash[:24]}_{_hash((request_hash, lifecycle_status, now))[:8]}"
        conn.execute(
            """
            INSERT INTO candidate_profile_requests (
                request_id, request_hash, signature_hash, permissions_hash, source_key_hash,
                policy_digest, evidence_ref_hashes_json, lifecycle_status, reason_code,
                cooldown_until, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                request_id, request_hash, signature.signature_hash, signature.permissions_hash,
                _hash(source_key), policy_digest, _canonical_json(evidence_ref_hashes),
                lifecycle_status, reason_code, cooldown_until, now,
            ),
        )
        return request_id
