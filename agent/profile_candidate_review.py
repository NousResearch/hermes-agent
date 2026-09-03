"""Inert, provider-neutral receipts for specialist-profile advisory proposals.

Phase 0 deliberately authorizes no specialist advisory adapter.  This module
therefore has no provider clients, subprocesses, network calls, registry
writes, or candidate-lifecycle writes.  It only canonicalizes a bounded local
request and returns an auditable ``advisory_unavailable`` receipt until a later
operator-approved integration supplies a separately reviewed adapter.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Literal

from gateway.candidate_profile_requests import (
    DEFAULT_POLICY_DIGEST,
    OpaqueEvidenceReference,
    SanitizedTaskEnvelope,
)
from gateway.capability_registry import CapabilitySignature


_MAX_ID_CHARS = 96
_MAX_TEXT_CHARS = 4_096
_MAX_ROLE_NAME_CHARS = 96
_MAX_BENCHMARK_CASES = 16
_MAX_LIMITATIONS = 16
_MAX_PERMISSIONS = 16
_MAX_TIMEOUT_SECONDS = 600
_MAX_MODEL_CALLS = 1
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CANDIDATE_ID_RE = re.compile(r"^cpr_[0-9a-f]{24}_[0-9a-f]{8}$")
_ROLE_NAME_RE = re.compile(r"^[a-z][a-z0-9-]{0,95}$")
_SECRET_LIKE_RE = re.compile(
    r"(?i)(?:\b(?:api[_-]?key|access[_-]?token|refresh[_-]?token|secret|password)\s*[:=]"
    r"|\bauthorization\s*:\s*bearer\b|\bbearer\s+[a-z0-9._-]{8,}|-----BEGIN(?: [A-Z]+)? PRIVATE KEY-----)"
)


def _canonical_json(value: object) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def _hash(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _bounded_text(value: object, *, field: str, max_chars: int) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > max_chars:
        raise ValueError(f"{field} must be a bounded non-empty string")
    return value


def _reject_secret_like(*values: str) -> None:
    if any(_SECRET_LIKE_RE.search(value) for value in values):
        raise ValueError("receipt-only advisory data cannot contain secret-like material")


def _canonical_strings(value: object, *, field: str, max_items: int) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, tuple) or len(value) > max_items:
        raise ValueError(f"{field} must be a bounded tuple of strings")
    normalized = tuple(sorted(set(value)))
    if any(not isinstance(item, str) or not item.strip() or len(item) > _MAX_TEXT_CHARS for item in normalized):
        raise ValueError(f"{field} must contain bounded non-empty strings")
    return normalized


def _validated_evidence_refs(envelope: SanitizedTaskEnvelope) -> tuple[str, ...]:
    if not isinstance(envelope, SanitizedTaskEnvelope) or not isinstance(envelope.evidence_refs, tuple):
        raise TypeError("envelope must contain only a sanitized evidence-reference tuple")
    refs = envelope.evidence_refs
    if len(refs) > _MAX_BENCHMARK_CASES:
        raise ValueError("envelope has too many evidence references")
    if any(not isinstance(reference, OpaqueEvidenceReference) for reference in refs):
        raise ValueError("envelope must contain opaque evidence references")
    return tuple(sorted({reference.digest for reference in refs}))


@dataclass(frozen=True, slots=True)
class IntegrationCapability:
    """A sanitized Phase-0 compatibility row, never dispatch authority."""

    agent: str
    supported_operation: str
    authentication_owner: str
    human_interaction_required: bool
    egress_class: str
    allowed: bool
    evidence: str
    failure_fallback: str

    def __post_init__(self) -> None:
        for field in (
            "agent",
            "supported_operation",
            "authentication_owner",
            "egress_class",
            "evidence",
            "failure_fallback",
        ):
            _bounded_text(getattr(self, field), field=field, max_chars=_MAX_TEXT_CHARS)
        _reject_secret_like(
            self.agent,
            self.supported_operation,
            self.authentication_owner,
            self.egress_class,
            self.evidence,
            self.failure_fallback,
        )
        if not isinstance(self.human_interaction_required, bool) or not isinstance(self.allowed, bool):
            raise TypeError("capability flags must be booleans")
        if self.failure_fallback != "advisory_unavailable":
            raise ValueError("advisory capability fallback must be advisory_unavailable")


@dataclass(frozen=True, slots=True)
class AdvisoryLimits:
    """Finite budget supplied with a receipt-only advisory request."""

    max_input_bytes: int
    max_model_calls: int
    timeout_seconds: int

    def __post_init__(self) -> None:
        for field, maximum in (
            ("max_input_bytes", _MAX_TEXT_CHARS * 2),
            ("max_model_calls", _MAX_MODEL_CALLS),
            ("timeout_seconds", _MAX_TIMEOUT_SECONDS),
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0 or value > maximum:
                raise ValueError(f"{field} must be a positive bounded integer")


@dataclass(frozen=True, slots=True)
class AdvisoryRequest:
    """Only the bounded, opaque material a future advisory adapter could receive."""

    candidate_request_id: str
    signature: CapabilitySignature
    envelope: SanitizedTaskEnvelope
    limits: AdvisoryLimits
    policy_digest: str = DEFAULT_POLICY_DIGEST

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_request_id, str) or not _CANDIDATE_ID_RE.fullmatch(self.candidate_request_id):
            raise ValueError("candidate_request_id must be a canonical opaque candidate id")
        if not isinstance(self.signature, CapabilitySignature):
            raise TypeError("signature must be a CapabilitySignature")
        _reject_secret_like(*self.signature.requested_permissions)
        if not isinstance(self.limits, AdvisoryLimits):
            raise TypeError("limits must be AdvisoryLimits")
        if not isinstance(self.policy_digest, str) or self.policy_digest != DEFAULT_POLICY_DIGEST:
            raise ValueError("policy_digest must match the fixed local candidate policy")
        if self.encoded_size > self.limits.max_input_bytes:
            raise ValueError("bounded advisory request exceeds max_input_bytes")

    @property
    def canonical_payload(self) -> dict[str, object]:
        return {
            "candidate_request_id": self.candidate_request_id,
            "evidence_ref_hashes": _validated_evidence_refs(self.envelope),
            "limits": {
                "max_input_bytes": self.limits.max_input_bytes,
                "max_model_calls": self.limits.max_model_calls,
                "timeout_seconds": self.limits.timeout_seconds,
            },
            "permissions_hash": self.signature.permissions_hash,
            "policy_digest": self.policy_digest,
            "signature_hash": self.signature.signature_hash,
        }

    @property
    def encoded_size(self) -> int:
        return len(_canonical_json(self.canonical_payload).encode("utf-8"))

    @property
    def input_hash(self) -> str:
        return _hash(self.canonical_payload)


@dataclass(frozen=True, slots=True)
class AdvisoryProfileProposal:
    """Untrusted advisory material; validation never promotes or persists it."""

    role_name: str
    soul_markdown: str
    benchmark_cases: tuple[str, ...]
    claimed_permissions: tuple[str, ...]
    limitations: tuple[str, ...]
    source_receipt_hash: str

    def __post_init__(self) -> None:
        if not _ROLE_NAME_RE.fullmatch(self.role_name):
            raise ValueError("role_name must be a bounded lowercase role label")
        _bounded_text(self.soul_markdown, field="soul_markdown", max_chars=_MAX_TEXT_CHARS)
        object.__setattr__(self, "benchmark_cases", _canonical_strings(
            self.benchmark_cases, field="benchmark_cases", max_items=_MAX_BENCHMARK_CASES
        ))
        object.__setattr__(self, "claimed_permissions", _canonical_strings(
            self.claimed_permissions, field="claimed_permissions", max_items=_MAX_PERMISSIONS
        ))
        object.__setattr__(self, "limitations", _canonical_strings(
            self.limitations, field="limitations", max_items=_MAX_LIMITATIONS
        ))
        if not isinstance(self.source_receipt_hash, str) or not _SHA256_RE.fullmatch(self.source_receipt_hash):
            raise ValueError("source_receipt_hash must be a SHA-256 digest")


ReceiptStatus = Literal[
    "advisory_received",
    "advisory_unavailable",
    "advisory_timeout",
    "advisory_malformed",
]


@dataclass(frozen=True, slots=True)
class ReviewReceipt:
    """A local observation bound to exactly one candidate packet and budget."""

    provider_label: str
    candidate_request_id: str
    input_hash: str
    policy_digest: str
    max_input_bytes: int
    max_model_calls: int
    timeout_seconds: int
    elapsed_seconds: int
    model_calls: int
    status: ReceiptStatus
    reason: str

    def __post_init__(self) -> None:
        _bounded_text(self.provider_label, field="provider_label", max_chars=64)
        if not isinstance(self.candidate_request_id, str) or not _CANDIDATE_ID_RE.fullmatch(self.candidate_request_id):
            raise ValueError("candidate_request_id must be a canonical opaque candidate id")
        if not isinstance(self.input_hash, str) or not _SHA256_RE.fullmatch(self.input_hash):
            raise ValueError("input_hash must be a SHA-256 digest")
        if not isinstance(self.policy_digest, str) or not _SHA256_RE.fullmatch(self.policy_digest):
            raise ValueError("policy_digest must be a SHA-256 digest")
        for field in ("max_input_bytes", "max_model_calls", "timeout_seconds", "elapsed_seconds", "model_calls"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field} must be a non-negative integer")
        if self.status not in {
            "advisory_received",
            "advisory_unavailable",
            "advisory_timeout",
            "advisory_malformed",
        }:
            raise ValueError("receipt status is not an advisory-only status")
        _bounded_text(self.reason, field="reason", max_chars=512)
        _reject_secret_like(self.provider_label, self.reason)

    @property
    def source_receipt_hash(self) -> str:
        return _hash({
            "candidate_request_id": self.candidate_request_id,
            "elapsed_seconds": self.elapsed_seconds,
            "input_hash": self.input_hash,
            "max_input_bytes": self.max_input_bytes,
            "max_model_calls": self.max_model_calls,
            "model_calls": self.model_calls,
            "policy_digest": self.policy_digest,
            "provider_label": self.provider_label,
            "reason": self.reason,
            "status": self.status,
            "timeout_seconds": self.timeout_seconds,
        })


ReviewStatus = Literal["accepted", "advisory_unavailable", "rejected"]


@dataclass(frozen=True, slots=True)
class AdvisoryReviewResult:
    """The receipt-only outcome.  ``proposal`` is always ``None`` in Phase 0."""

    status: ReviewStatus
    receipt: ReviewReceipt
    proposal: AdvisoryProfileProposal | None = None
    reason: str = ""


@dataclass(frozen=True, slots=True)
class ProposalValidationResult:
    status: Literal["accepted", "rejected"]
    reason: str


def _unavailable_receipt(capability: IntegrationCapability, request: AdvisoryRequest, *, reason: str) -> ReviewReceipt:
    return ReviewReceipt(
        provider_label=capability.agent,
        candidate_request_id=request.candidate_request_id,
        input_hash=request.input_hash,
        policy_digest=request.policy_digest,
        max_input_bytes=request.limits.max_input_bytes,
        max_model_calls=request.limits.max_model_calls,
        timeout_seconds=request.limits.timeout_seconds,
        elapsed_seconds=0,
        model_calls=0,
        status="advisory_unavailable",
        reason=reason,
    )


def review_with(capability: IntegrationCapability, request: AdvisoryRequest) -> AdvisoryReviewResult:
    """Return an inert receipt without looking up or calling any provider.

    An allowed compatibility row is intentionally insufficient: no external
    adapter is wired by Phase 0, and this function must never substitute a
    provider or transform the originating task into a failure.
    """
    if not isinstance(capability, IntegrationCapability):
        raise TypeError("capability must be an IntegrationCapability")
    if not isinstance(request, AdvisoryRequest):
        raise TypeError("request must be an AdvisoryRequest")
    reason = (
        "integration capability is not currently allowed"
        if not capability.allowed
        else "no approved advisory adapter is wired"
    )
    receipt = _unavailable_receipt(capability, request, reason=reason)
    return AdvisoryReviewResult(status="advisory_unavailable", receipt=receipt, reason=reason)


def validate_receipt(
    receipt: ReviewReceipt,
    request: AdvisoryRequest,
    capability: IntegrationCapability,
) -> AdvisoryReviewResult:
    """Validate a supplied local observation without accepting a provider output."""
    if not isinstance(receipt, ReviewReceipt) or not isinstance(request, AdvisoryRequest):
        raise TypeError("receipt and request must use the advisory receipt types")
    if not isinstance(capability, IntegrationCapability):
        raise TypeError("capability must be an IntegrationCapability")
    if receipt.provider_label != capability.agent:
        return AdvisoryReviewResult("rejected", receipt, reason="receipt provider does not match capability")
    if receipt.candidate_request_id != request.candidate_request_id or receipt.input_hash != request.input_hash:
        return AdvisoryReviewResult("rejected", receipt, reason="receipt candidate or input binding does not match request")
    if receipt.policy_digest != request.policy_digest:
        return AdvisoryReviewResult("rejected", receipt, reason="receipt policy digest does not match request")
    if (
        receipt.max_input_bytes != request.limits.max_input_bytes
        or receipt.max_model_calls != request.limits.max_model_calls
        or receipt.timeout_seconds != request.limits.timeout_seconds
    ):
        return AdvisoryReviewResult("rejected", receipt, reason="receipt budget does not match request")
    if receipt.model_calls > receipt.max_model_calls:
        return AdvisoryReviewResult("rejected", receipt, reason="receipt model calls exceed requested budget")
    if receipt.elapsed_seconds > receipt.timeout_seconds:
        return AdvisoryReviewResult("rejected", receipt, reason="receipt timeout exceeds requested budget")
    if receipt.status == "advisory_timeout":
        return AdvisoryReviewResult("rejected", receipt, reason="advisory timed out")
    if receipt.status == "advisory_malformed":
        return AdvisoryReviewResult("rejected", receipt, reason="advisory receipt is malformed")
    if receipt.status == "advisory_received":
        if not capability.allowed:
            return AdvisoryReviewResult("rejected", receipt, reason="integration capability is not currently allowed")
        # A Phase-0 compatibility row is descriptive, not an adapter grant.
        # No implementation is allowed to inject an advisory result merely by
        # setting ``allowed=True``; a later task must add a separately approved
        # concrete adapter authority before this branch can ever accept data.
        return AdvisoryReviewResult("rejected", receipt, reason="no advisory adapter is wired in Phase 0")
    return AdvisoryReviewResult("advisory_unavailable", receipt, reason=receipt.reason)


def validate_proposal(
    proposal: AdvisoryProfileProposal,
    request: AdvisoryRequest,
    receipt: ReviewReceipt | None,
    capability: IntegrationCapability,
) -> ProposalValidationResult:
    """Reject unsafe advisory text or permission expansion without persisting it."""
    if not isinstance(proposal, AdvisoryProfileProposal) or not isinstance(request, AdvisoryRequest):
        raise TypeError("proposal and request must use the advisory types")
    if receipt is None:
        return ProposalValidationResult("rejected", "proposal requires a bound advisory receipt")
    if not isinstance(receipt, ReviewReceipt) or not isinstance(capability, IntegrationCapability):
        return ProposalValidationResult("rejected", "proposal receipt or capability is malformed")
    receipt_result = validate_receipt(receipt, request, capability)
    if proposal.source_receipt_hash != receipt.source_receipt_hash:
        return ProposalValidationResult("rejected", "proposal receipt binding does not match")
    text = "\n".join(
        (
            proposal.role_name,
            proposal.soul_markdown,
            *proposal.benchmark_cases,
            *proposal.claimed_permissions,
            *proposal.limitations,
        )
    )
    if _SECRET_LIKE_RE.search(text):
        return ProposalValidationResult("rejected", "proposal contains secret-like material")
    if not set(proposal.claimed_permissions) <= set(request.signature.requested_permissions):
        return ProposalValidationResult("rejected", "proposal expands requested permissions")
    if receipt_result.status != "accepted":
        return ProposalValidationResult("rejected", f"proposal receipt rejected: {receipt_result.reason}")
    return ProposalValidationResult("accepted", "proposal is bounded advisory input only")
