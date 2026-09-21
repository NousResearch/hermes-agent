"""Fail-closed routing tests for active specialist capability resolution."""

from __future__ import annotations

import pytest

from gateway.capability_registry import CapabilityRegistry, CapabilitySignature
from gateway.specialist_routing import (
    RouteKind,
    SPECIALIST_PROFILES,
    SpecialistRouteDecision,
    capability_signature_for_profile,
    parse_specialist_response,
    resolve_route,
)


MARKET_DATA = CapabilitySignature(
    domain="market-data",
    actions=("audit", "read"),
    evidence_class="diagnostic-only",
    requested_permissions=("market-data:read",),
)
CANDIDATE_PROFILE_JSON = (
    '{"kind":"specialist","profile":"generated-market-data-candidate",'
    '"confidence":0.95,"reason":"new capability appears useful",'
    '"title":"Generated candidate"}'
)


@pytest.fixture
def registry(tmp_path):
    return CapabilityRegistry(db_path=tmp_path / "capabilities.db")


def _fixed_profile_decision() -> SpecialistRouteDecision:
    return SpecialistRouteDecision(
        kind=RouteKind.SPECIALIST,
        profile="burndown-patch-steward",
        confidence=0.95,
        reason="bounded patch work",
        title="Patch confirmed failures",
        audit_reason="specialist",
    )


def test_classifier_cannot_dispatch_a_candidate_profile(registry):
    decision = parse_specialist_response(CANDIDATE_PROFILE_JSON)

    assert decision.dispatches is False
    assert decision.audit_reason == "unknown_profile"


def test_every_fixed_classifier_profile_has_a_closed_local_capability_signature():
    signatures = {
        profile: capability_signature_for_profile(profile) for profile in SPECIALIST_PROFILES
    }

    assert all(signature is not None for signature in signatures.values())
    assert capability_signature_for_profile("generated-market-data-candidate") is None


def test_active_registry_match_precedes_fixed_profile_classifier(registry):
    registry.register_fixed_baseline(profile_id="market-data-authority-auditor", signature=MARKET_DATA)

    decision = resolve_route(MARKET_DATA, registry, fallback=_fixed_profile_decision())

    assert decision.dispatches is True
    assert decision.profile == "market-data-authority-auditor"
    assert decision.audit_reason == "active_registry_match"


def test_no_active_registry_match_uses_existing_fixed_profile_fallback(registry):
    decision = resolve_route(MARKET_DATA, registry, fallback=_fixed_profile_decision())

    assert decision == _fixed_profile_decision()


def test_candidate_decision_never_bypasses_fixed_or_active_profiles(registry):
    candidate = SpecialistRouteDecision(
        kind=RouteKind.SPECIALIST,
        profile="generated-market-data-candidate",
        confidence=0.95,
        reason="model suggestion",
        title="Generated candidate",
        audit_reason="specialist",
    )

    decision = resolve_route(MARKET_DATA, registry, fallback=candidate)

    assert decision.dispatches is False
    assert decision.audit_reason == "inactive_profile"


def test_registry_exception_falls_back_to_normal_chat_with_auditable_reason():
    class BrokenRegistry:
        def resolve(self, signature):
            raise RuntimeError("database is unavailable")

    decision = resolve_route(MARKET_DATA, BrokenRegistry(), fallback=_fixed_profile_decision())

    assert decision.kind is RouteKind.GENERAL
    assert decision.dispatches is False
    assert decision.audit_reason == "registry_unavailable"
