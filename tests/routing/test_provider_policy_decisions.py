"""Decision tests for multi-provider policy kernel — RED PHASE.

These tests exercise decide_provider_route() beyond the PR1 stub.
They MUST fail initially; GREEN phase wires real decision logic.
"""

from __future__ import annotations

import pytest

from agent.provider_policy import (
    DomainPolicy,
    ProviderPolicy,
    RoutingDecision,
    decide_provider_route,
    normalize_provider_policy,
)


def _base_config(**overrides):
    cfg = {
        "model": {"provider": "copilot", "default": "gpt-5.4"},
        "compression": {},
        "auxiliary": {},
        "delegation": {},
    }
    cfg.update(overrides)
    return cfg


# ---------------------------------------------------------------------------
# primary domain — real decision, not stub
# ---------------------------------------------------------------------------
def test_primary_decision_returns_primary_provider_and_model():
    policy = normalize_provider_policy(_base_config())
    decision = decide_provider_route("primary", policy)

    assert decision.selected_provider == "copilot"
    assert decision.selected_model == "gpt-5.4"
    # RED: stub currently returns "pr1_stub"
    assert decision.decision_source == "primary"
    assert decision.cross_provider is False


def test_primary_decision_exposes_smart_routing_flag_in_metadata():
    cfg = _base_config(smart_model_routing={"enabled": True})
    policy = normalize_provider_policy(cfg)
    decision = decide_provider_route("primary", policy)

    # RED: stub metadata does not propagate primary domain metadata
    assert decision.metadata.get("smart_model_routing_enabled") is True


# ---------------------------------------------------------------------------
# auxiliary domain — explicit provider must flag cross_provider
# ---------------------------------------------------------------------------
def test_auxiliary_explicit_provider_marks_cross_provider_true():
    cfg = _base_config(auxiliary={
        "vision": {"provider": "openrouter", "model": "gemini-2.5-flash"},
    })
    policy = normalize_provider_policy(cfg)
    decision = decide_provider_route(
        "auxiliary", policy, context={"task": "vision"}
    )

    assert decision.selected_provider == "openrouter"
    assert decision.selected_model == "gemini-2.5-flash"
    assert decision.cross_provider is True
    assert "openrouter" in decision.reason.lower() or "auxiliary" in decision.reason.lower()


def test_auxiliary_without_task_context_falls_back_to_primary():
    cfg = _base_config(auxiliary={
        "vision": {"provider": "openrouter", "model": "gemini-2.5-flash"},
    })
    policy = normalize_provider_policy(cfg)
    # No context → no task hint → should route to primary, not blindly pick one aux task
    decision = decide_provider_route("auxiliary", policy)
    assert decision.selected_provider == "copilot"
    assert decision.selected_model == "gpt-5.4"


# ---------------------------------------------------------------------------
# delegation domain — explicit > inherit_primary
# ---------------------------------------------------------------------------
def test_delegation_decision_uses_explicit_delegation_provider():
    cfg = _base_config(delegation={
        "provider": "copilot",
        "model": "gpt-5-mini",
    })
    policy = normalize_provider_policy(cfg)
    decision = decide_provider_route("delegation", policy)

    assert decision.selected_provider == "copilot"
    assert decision.selected_model == "gpt-5-mini"
    assert decision.decision_source in {"delegation_explicit", "delegation"}


def test_delegation_inherits_primary_when_unset():
    policy = normalize_provider_policy(_base_config())
    decision = decide_provider_route("delegation", policy)

    assert decision.selected_provider == "copilot"
    assert decision.selected_model == "gpt-5.4"
    assert decision.decision_source in {"delegation_inherit", "delegation"}


# ---------------------------------------------------------------------------
# fallback domain — ordered chain
# ---------------------------------------------------------------------------
def test_fallback_decision_picks_first_candidate():
    cfg = _base_config(fallback_providers=[
        {"provider": "anthropic", "model": "claude-sonnet-4.6"},
        {"provider": "openrouter", "model": "gpt-5.4"},
    ])
    policy = normalize_provider_policy(cfg)
    decision = decide_provider_route("fallback", policy)

    assert decision.selected_provider == "anthropic"
    assert decision.selected_model == "claude-sonnet-4.6"
    assert decision.cross_provider is True


def test_fallback_decision_when_disabled_returns_primary():
    policy = normalize_provider_policy(_base_config())
    decision = decide_provider_route("fallback", policy)

    # No fallbacks configured → decision should still be deterministic;
    # selected = primary with a reason signalling disabled chain
    assert decision.selected_provider == "copilot"
    assert decision.selected_model == "gpt-5.4"
    assert "disabled" in decision.reason.lower() or decision.decision_source.endswith("disabled")


# ---------------------------------------------------------------------------
# unknown domain — defensive behavior
# ---------------------------------------------------------------------------
def test_unknown_domain_returns_decision_with_unknown_source():
    policy = normalize_provider_policy(_base_config())
    decision = decide_provider_route("does-not-exist", policy)

    assert isinstance(decision, RoutingDecision)
    assert decision.decision_source in {"unknown_domain", "unknown"}
