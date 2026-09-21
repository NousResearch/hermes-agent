"""Contracts for the inert specialist-profile advisory boundary."""

from __future__ import annotations

import hashlib

import pytest

from agent.profile_candidate_review import (
    AdvisoryLimits,
    AdvisoryProfileProposal,
    AdvisoryRequest,
    IntegrationCapability,
    ReviewReceipt,
    review_with,
    validate_proposal,
    validate_receipt,
)
from gateway.candidate_profile_requests import (
    DEFAULT_POLICY_DIGEST,
    OpaqueEvidenceReference,
    SanitizedTaskEnvelope,
)
from gateway.capability_registry import CapabilitySignature


SIGNATURE = CapabilitySignature(
    domain="market-data",
    actions=("audit", "read"),
    evidence_class="diagnostic-only",
    requested_permissions=("market-data:read",),
)


def _reference(label: str) -> OpaqueEvidenceReference:
    return OpaqueEvidenceReference(digest=hashlib.sha256(label.encode()).hexdigest())


def _capability(*, agent: str = "claude", allowed: bool = False) -> IntegrationCapability:
    return IntegrationCapability(
        agent=agent,
        supported_operation="bounded local receipt",
        authentication_owner="operator",
        human_interaction_required=True,
        egress_class="external-provider",
        allowed=allowed,
        evidence="phase-0 receipt",
        failure_fallback="advisory_unavailable",
    )


def _request() -> AdvisoryRequest:
    return AdvisoryRequest(
        candidate_request_id="cpr_1234567890abcdef12345678_12345678",
        signature=SIGNATURE,
        envelope=SanitizedTaskEnvelope(evidence_refs=(_reference("case-1"),)),
        limits=AdvisoryLimits(max_input_bytes=2_048, max_model_calls=1, timeout_seconds=5),
        policy_digest=DEFAULT_POLICY_DIGEST,
    )


def _received_receipt(
    request: AdvisoryRequest,
    *,
    policy_digest: str | None = None,
    max_input_bytes: int | None = None,
    max_model_calls: int | None = None,
    timeout_seconds: int | None = None,
    elapsed_seconds: int = 0,
    model_calls: int = 0,
    status: str = "advisory_received",
) -> ReviewReceipt:
    return ReviewReceipt(
        provider_label="claude",
        candidate_request_id=request.candidate_request_id,
        input_hash=request.input_hash,
        policy_digest=policy_digest or request.policy_digest,
        max_input_bytes=max_input_bytes if max_input_bytes is not None else request.limits.max_input_bytes,
        max_model_calls=max_model_calls if max_model_calls is not None else request.limits.max_model_calls,
        timeout_seconds=timeout_seconds if timeout_seconds is not None else request.limits.timeout_seconds,
        elapsed_seconds=elapsed_seconds,
        model_calls=model_calls,
        status=status,
        reason="synthetic local receipt for contract validation",
    )


def _proposal(receipt: ReviewReceipt, *, claimed_permissions: tuple[str, ...] = ("market-data:read",),
              soul_markdown: str = "Read the bounded evidence and report uncertainty.") -> AdvisoryProfileProposal:
    return AdvisoryProfileProposal(
        role_name="market-data-reviewer",
        soul_markdown=soul_markdown,
        benchmark_cases=("opaque-case",),
        claimed_permissions=claimed_permissions,
        limitations=("diagnostic-only",),
        source_receipt_hash=receipt.source_receipt_hash,
    )


def test_unavailable_claude_adapter_emits_bound_receipt_and_never_substitutes_a_model():
    result = review_with(_capability(), _request())

    assert result.status == "advisory_unavailable"
    assert result.proposal is None
    assert result.receipt.provider_label == "claude"
    assert result.receipt.candidate_request_id == _request().candidate_request_id
    assert result.receipt.input_hash == _request().input_hash
    assert result.receipt.policy_digest == DEFAULT_POLICY_DIGEST
    assert result.receipt.model_calls == 0
    assert result.receipt.source_receipt_hash


def test_allowed_capability_without_a_wired_adapter_remains_unavailable_not_a_provider_call():
    result = review_with(_capability(agent="codex", allowed=True), _request())

    assert result.status == "advisory_unavailable"
    assert result.receipt.provider_label == "codex"
    assert result.receipt.model_calls == 0
    assert result.receipt.reason == "no approved advisory adapter is wired"


def test_timeout_receipt_is_rejected_when_it_exceeds_bound_budget():
    request = _request()
    result = validate_receipt(
        ReviewReceipt(
            provider_label="claude",
            candidate_request_id=request.candidate_request_id,
            input_hash=request.input_hash,
            policy_digest=request.policy_digest,
            max_input_bytes=request.limits.max_input_bytes,
            max_model_calls=request.limits.max_model_calls,
            timeout_seconds=request.limits.timeout_seconds,
            elapsed_seconds=6,
            model_calls=0,
            status="advisory_timeout",
            reason="local timeout observation",
        ),
        request,
        _capability(),
    )

    assert result.status == "rejected"
    assert result.reason == "receipt timeout exceeds requested budget"


def test_malformed_receipt_cannot_be_bound_to_another_candidate_or_input():
    request = _request()
    result = validate_receipt(
        ReviewReceipt(
            provider_label="claude",
            candidate_request_id="cpr_aaaaaaaaaaaaaaaaaaaaaaaa_bbbbbbbb",
            input_hash="0" * 64,
            policy_digest=request.policy_digest,
            max_input_bytes=request.limits.max_input_bytes,
            max_model_calls=0,
            timeout_seconds=request.limits.timeout_seconds,
            elapsed_seconds=0,
            model_calls=0,
            status="advisory_unavailable",
            reason="malformed synthetic receipt",
        ),
        request,
        _capability(),
    )

    assert result.status == "rejected"
    assert result.reason == "receipt candidate or input binding does not match request"


def test_proposal_rejects_secret_bearing_content():
    request = _request()
    receipt = _received_receipt(request)
    proposal = _proposal(receipt, soul_markdown="Use api_key=not-a-secret-here when reviewing.")

    result = validate_proposal(proposal, request, receipt, _capability(allowed=True))

    assert result.status == "rejected"
    assert result.reason == "proposal contains secret-like material"


def test_proposal_rejects_unapproved_permission_delta():
    request = _request()
    receipt = _received_receipt(request)
    proposal = _proposal(receipt, claimed_permissions=("market-data:read", "market-data:write"))

    result = validate_proposal(proposal, request, receipt, _capability(allowed=True))

    assert result.status == "rejected"
    assert result.reason == "proposal expands requested permissions"


def test_proposal_requires_a_bound_receipt():
    request = _request()
    receipt = _received_receipt(request)

    result = validate_proposal(_proposal(receipt), request, None, _capability(allowed=True))

    assert result.status == "rejected"
    assert result.reason == "proposal requires a bound advisory receipt"


def test_phase0_rejects_an_injected_received_receipt_even_when_compatibility_is_allowed():
    request = _request()
    receipt = _received_receipt(request)

    result = validate_proposal(_proposal(receipt), request, receipt, _capability(allowed=True))

    assert result.status == "rejected"
    assert result.reason == "proposal receipt rejected: no advisory adapter is wired in Phase 0"


@pytest.mark.parametrize(
    ("receipt_kwargs", "expected_reason"),
    (
        ({"elapsed_seconds": 6, "status": "advisory_timeout"}, "receipt timeout exceeds requested budget"),
        ({"policy_digest": "d" * 64}, "receipt policy digest does not match request"),
        ({"max_input_bytes": 1}, "receipt budget does not match request"),
        ({"model_calls": 2}, "receipt model calls exceed requested budget"),
    ),
)
def test_proposal_rejects_any_invalid_bound_receipt(receipt_kwargs, expected_reason):
    request = _request()
    receipt = _received_receipt(request, **receipt_kwargs)

    result = validate_proposal(_proposal(receipt), request, receipt, _capability(allowed=True))

    assert result.status == "rejected"
    assert result.reason == f"proposal receipt rejected: {expected_reason}"


def test_proposal_rejects_secret_like_claimed_permission_before_scope_comparison():
    request = _request()
    receipt = _received_receipt(request)

    result = validate_proposal(
        _proposal(receipt, claimed_permissions=("api_key=not-a-secret-here",)),
        request,
        receipt,
        _capability(allowed=True),
    )

    assert result.status == "rejected"
    assert result.reason == "proposal contains secret-like material"


def test_request_rejects_secret_like_requested_permission_at_input_boundary():
    secret_signature = CapabilitySignature(
        domain="market-data",
        actions=("read",),
        evidence_class="diagnostic-only",
        requested_permissions=("api_key=not-a-secret-here",),
    )

    with pytest.raises(ValueError, match="secret-like"):
        AdvisoryRequest(
            candidate_request_id="cpr_1234567890abcdef12345678_12345678",
            signature=secret_signature,
            envelope=SanitizedTaskEnvelope(evidence_refs=(_reference("secret-permission"),)),
            limits=AdvisoryLimits(max_input_bytes=2_048, max_model_calls=1, timeout_seconds=5),
        )
