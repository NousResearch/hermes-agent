"""End-to-end contracts for provider-free specialist discovery."""

from __future__ import annotations

import pytest


def _registry_api():
    from gateway.capability_registry import CapabilityRegistry, CapabilitySignature

    return CapabilityRegistry, CapabilitySignature


def _discovery_api():
    try:
        from gateway.specialist_discovery import SpecialistDiscovery
    except ImportError as exc:  # RED: the split starts without the orchestrator.
        pytest.fail(f"specialist discovery orchestration is unavailable: {exc}")
    return SpecialistDiscovery


def _repository_review():
    _, CapabilitySignature = _registry_api()
    return CapabilitySignature(
        domain="repository-evidence",
        actions=("read", "review"),
        evidence_class="diagnostic-only",
        requested_permissions=("repository-evidence:read",),
    )


def test_configured_match_returns_profile_without_candidate_side_effect(tmp_path):
    CapabilityRegistry, _ = _registry_api()
    SpecialistDiscovery = _discovery_api()
    db_path = tmp_path / "discovery.db"
    signature = _repository_review()
    registry = CapabilityRegistry(
        db_path=db_path,
        configured_profiles={"repository-reviewer": signature},
    )
    registry.register_configured_profile("repository-reviewer")
    discovery = SpecialistDiscovery(db_path=db_path)

    decision = discovery.resolve_or_request(signature, source_key="gateway:request-1")

    assert decision.status == "active_match"
    assert decision.profile == "repository-reviewer"
    assert decision.candidate_request_id is None


def test_missing_scope_returns_inert_candidate_identity(tmp_path):
    SpecialistDiscovery = _discovery_api()
    discovery = SpecialistDiscovery(db_path=tmp_path / "discovery.db")

    decision = discovery.resolve_or_request(
        _repository_review(),
        source_key="gateway:request-1",
    )

    assert decision.status == "candidate"
    assert decision.profile is None
    assert decision.candidate_request_id.startswith("cpr_")


def test_ambiguous_match_is_preserved_without_candidate_side_effect(tmp_path):
    CapabilityRegistry, CapabilitySignature = _registry_api()
    SpecialistDiscovery = _discovery_api()
    db_path = tmp_path / "discovery.db"
    signature = CapabilitySignature(
        domain="repository-evidence",
        actions=("read",),
        evidence_class="diagnostic-only",
        requested_permissions=("repository-evidence:read",),
    )
    registry = CapabilityRegistry(
        db_path=db_path,
        configured_profiles={"one": signature, "two": signature},
    )
    registry.register_configured_profile("one")
    registry.register_configured_profile("two")

    decision = SpecialistDiscovery(db_path=db_path).resolve_or_request(
        signature,
        source_key="gateway:ambiguous",
    )

    assert decision.status == "ambiguous"
    assert decision.profile is None
    assert decision.candidate_request_id is None


def test_disallowed_scope_fails_closed_without_profile(tmp_path):
    _, CapabilitySignature = _registry_api()
    SpecialistDiscovery = _discovery_api()
    discovery = SpecialistDiscovery(db_path=tmp_path / "discovery.db")
    write_scope = CapabilitySignature(
        domain="repository-evidence",
        actions=("write",),
        evidence_class="diagnostic-only",
        requested_permissions=("repository-evidence:write",),
    )

    decision = discovery.resolve_or_request(
        write_scope,
        source_key="gateway:request-2",
    )

    assert decision.status == "rejected"
    assert decision.profile is None
