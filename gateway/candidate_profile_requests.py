"""Bounded, inert requests for missing specialist capabilities.

Rows contain canonical hashes and opaque evidence references only. Creating a
candidate never creates a profile, dispatches work, or grants authority.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from gateway.capability_registry import CapabilityRegistry, CapabilitySignature, RegistryResolution
from hermes_cli import kanban_db


CandidateRequestStatus = Literal["candidate", "duplicate", "cooldown", "rejected"]
_DEFAULT_COOLDOWN_SECONDS = 3_600
_MAX_SOURCE_KEY_CHARS = 512
_MAX_EVIDENCE_REFERENCES = 16
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_ALLOWED_LOCAL_SCOPES: dict[str, tuple[frozenset[str], frozenset[str]]] = {
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
    "local_read_only_scopes": sorted(_ALLOWED_LOCAL_SCOPES),
    "profile_creation": False,
    "requested_permissions": "explicit-local-read-only",
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS candidate_profile_requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT NOT NULL UNIQUE,
    request_hash TEXT NOT NULL,
    signature_hash TEXT NOT NULL,
    permissions_hash TEXT NOT NULL,
    source_key_hash TEXT NOT NULL,
    policy_digest TEXT NOT NULL,
    evidence_ref_hashes_json TEXT NOT NULL,
    lifecycle_status TEXT NOT NULL,
    reason_code TEXT NOT NULL,
    cooldown_until INTEGER,
    created_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_candidate_request_identity
ON candidate_profile_requests(request_hash, id);
CREATE TRIGGER IF NOT EXISTS candidate_profile_requests_no_update
BEFORE UPDATE ON candidate_profile_requests BEGIN
    SELECT RAISE(ABORT, 'candidate_profile_requests is append-only');
END;
CREATE TRIGGER IF NOT EXISTS candidate_profile_requests_no_delete
BEFORE DELETE ON candidate_profile_requests BEGIN
    SELECT RAISE(ABORT, 'candidate_profile_requests is append-only');
END;
"""


def _canonical_json(value: object) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def _hash(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


DEFAULT_POLICY_DIGEST = _hash(_POLICY)


@dataclass(frozen=True, slots=True)
class OpaqueEvidenceReference:
    """A pre-hashed evidence identity safe for durable local storage."""

    digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.digest, str) or not _HASH_RE.fullmatch(self.digest):
            raise ValueError("opaque evidence references must be SHA-256 hex digests")


@dataclass(frozen=True, slots=True)
class SanitizedTaskEnvelope:
    """The only task data accepted by the candidate ledger."""

    evidence_refs: tuple[OpaqueEvidenceReference, ...] = ()


@dataclass(frozen=True, slots=True)
class CandidateProfileRequest:
    """Result of opening an inert candidate request."""

    request_id: str
    status: CandidateRequestStatus
    profile_id: None
    reason: str


def _capability_rejection(signature: CapabilitySignature) -> str | None:
    scope = _ALLOWED_LOCAL_SCOPES.get(signature.domain)
    if scope is None or signature.evidence_class != "diagnostic-only":
        return "unapproved_local_scope"
    actions, permissions = scope
    if not signature.actions or not set(signature.actions) <= actions:
        return "non_read_only_action"
    if not signature.requested_permissions or not set(signature.requested_permissions) <= permissions:
        return "unapproved_permission"
    return None


def _reason(reason_code: str) -> str:
    return {
        "invalid_evidence_refs": "candidate requests store only sanitized evidence references",
        "invalid_policy_digest": "candidate policy digest does not match the fixed local policy",
        "non_read_only_action": "candidate requests require read-only actions",
        "unapproved_local_scope": "candidate requests require an approved local read-only scope",
        "unapproved_permission": "candidate requests require approved local read-only permissions",
    }.get(reason_code, "candidate request rejected by fixed local policy")


def _sanitize_evidence(envelope: SanitizedTaskEnvelope | None) -> tuple[str, ...] | None:
    if envelope is None:
        return ()
    if not isinstance(envelope, SanitizedTaskEnvelope):
        return None
    refs = envelope.evidence_refs
    if not isinstance(refs, tuple) or len(refs) > _MAX_EVIDENCE_REFERENCES:
        return None
    if any(not isinstance(reference, OpaqueEvidenceReference) for reference in refs):
        return None
    return tuple(sorted({reference.digest for reference in refs}))


class CandidateProfileRequests:
    """Open idempotent candidate rows under a fixed least-privilege policy."""

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

    @contextmanager
    def _connection(self) -> Iterator[object]:
        with kanban_db.connect_closing(self._db_path, board=self._board) as conn:
            conn.executescript(_SCHEMA)
            yield conn

    def open_or_reuse(
        self,
        signature: CapabilitySignature,
        *,
        source_key: str,
        resolution: RegistryResolution | None = None,
        envelope: SanitizedTaskEnvelope | None = None,
        policy_digest: str = DEFAULT_POLICY_DIGEST,
    ) -> CandidateProfileRequest:
        """Create or reuse a row only after this board's own registry lookup."""
        del resolution
        if not isinstance(signature, CapabilitySignature):
            raise TypeError("signature must be a CapabilitySignature")
        if not isinstance(source_key, str) or not source_key.strip() or len(source_key) > _MAX_SOURCE_KEY_CHARS:
            raise ValueError("source_key must be a bounded non-empty string")

        local_resolution = CapabilityRegistry(db_path=self._db_path, board=self._board).resolve(signature)
        if local_resolution.status != "no_match":
            return CandidateProfileRequest(
                request_id="",
                status="rejected",
                profile_id=None,
                reason=f"candidate request requires a local no-match; resolution was {local_resolution.status}",
            )

        evidence_refs = _sanitize_evidence(envelope)
        reason_code = _capability_rejection(signature)
        stored_policy_digest = DEFAULT_POLICY_DIGEST
        if policy_digest != DEFAULT_POLICY_DIGEST:
            stored_policy_digest = _hash({"untrusted_policy_digest": policy_digest})
            reason_code = "invalid_policy_digest"
        if evidence_refs is None:
            evidence_refs = ()
            reason_code = "invalid_evidence_refs"

        source_key_hash = _hash(source_key)
        request_hash = _hash(
            {
                "evidence_refs": evidence_refs,
                "permissions_hash": signature.permissions_hash,
                "policy_digest": stored_policy_digest,
                "signature_hash": signature.signature_hash,
                "source_key_hash": source_key_hash,
            }
        )
        now = int(self._clock())

        with self._connection() as conn:
            with kanban_db.write_txn(conn):
                latest = conn.execute(
                    """
                    SELECT request_id, lifecycle_status, cooldown_until
                    FROM candidate_profile_requests
                    WHERE request_hash = ? ORDER BY id DESC LIMIT 1
                    """,
                    (request_hash,),
                ).fetchone()
                if latest is not None:
                    if latest["lifecycle_status"] == "candidate":
                        return CandidateProfileRequest(
                            latest["request_id"], "duplicate", None, "matching candidate already exists"
                        )
                    cooldown_until = latest["cooldown_until"]
                    if isinstance(cooldown_until, int) and now < cooldown_until:
                        return CandidateProfileRequest(
                            latest["request_id"], "cooldown", None, "matching rejection is cooling down"
                        )

                lifecycle_status = "rejected" if reason_code else "candidate"
                cooldown_until = now + self._cooldown_seconds if reason_code else None
                request_id = f"cpr_{request_hash[:24]}_{_hash((lifecycle_status, now))[:8]}"
                conn.execute(
                    """
                    INSERT INTO candidate_profile_requests (
                        request_id, request_hash, signature_hash, permissions_hash,
                        source_key_hash, policy_digest, evidence_ref_hashes_json,
                        lifecycle_status, reason_code, cooldown_until, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        request_id,
                        request_hash,
                        signature.signature_hash,
                        signature.permissions_hash,
                        source_key_hash,
                        stored_policy_digest,
                        _canonical_json(evidence_refs),
                        lifecycle_status,
                        reason_code or "local_no_match",
                        cooldown_until,
                        now,
                    ),
                )

        if reason_code:
            return CandidateProfileRequest(request_id, "rejected", None, _reason(reason_code))
        return CandidateProfileRequest(
            request_id,
            "candidate",
            None,
            "local no-match queued for bounded inert candidate review",
        )
