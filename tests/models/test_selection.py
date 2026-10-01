from __future__ import annotations

from dataclasses import fields

import pytest

from models import (
    CapabilityRequirements,
    ModelRef,
    SelectionCandidate,
    SelectionConstraints,
    SelectionPolicy,
    SelectionReason,
    SelectionRequest,
    build_selection_candidate,
    select_model,
)
from models.metadata import ModelMetadata
from providers.routing import InvocationRequest, InvocationRoute


def _metadata(ref: ModelRef, **kwargs) -> ModelMetadata:
    return ModelMetadata(ref=ref, **kwargs)


def _candidate(
    provider: str,
    model: str,
    *,
    catalogued: bool = True,
    stable_order: int = 0,
    api_mode: str = "chat_completions",
    runtime_kind: str = "http",
    **metadata,
) -> SelectionCandidate:
    ref = ModelRef(provider, model)
    route = InvocationRoute(
        provider=ref.provider,
        model=ref.model,
        base_url="https://example.invalid/v1",
        api_mode=api_mode,
        runtime_kind=runtime_kind,
        is_routing_aggregator=False,
        source="test",
    )
    return SelectionCandidate(
        ref=ref,
        metadata=_metadata(ref, **metadata),
        route=route,
        catalogued=catalogued,
        stable_order=stable_order,
        source="test",
    )


def test_constraints_filter_canonical_facts_and_preserve_unknown():
    unknown = _candidate("openai", "unknown", supports_tools=None, supports_vision=None)
    text = _candidate("openai", "text", supports_tools=True, supports_vision=False)
    vision = _candidate(
        "anthropic", "vision", supports_tools=True, supports_vision=True,
        context_window=200_000,
    )

    result = select_model(SelectionRequest(
        candidates=(unknown, text, vision),
        constraints=SelectionConstraints(
            allowed_providers=("anthropic",),
            capabilities=CapabilityRequirements(
                tools=True, vision=True, minimum_context_window=128_000,
            ),
        ),
    ))

    assert result.selected == vision
    assert result.eligible == (vision,)
    rejected = {item.ref: item.reasons for item in result.rejected}
    assert "provider_not_allowed" in rejected[unknown.ref]
    assert "provider_not_allowed" in rejected[text.ref]


def test_unknown_capability_does_not_satisfy_required_true():
    unknown = _candidate("openai", "unknown", supports_vision=None)
    result = select_model(SelectionRequest(
        candidates=(unknown,),
        constraints=SelectionConstraints(
            capabilities=CapabilityRequirements(vision=True),
        ),
    ))
    assert result.selected is None
    assert result.reason is SelectionReason.NO_ELIGIBLE_CANDIDATE
    assert result.rejected[0].reasons == ("requires_vision=True",)


def test_policy_ranking_is_deterministic_across_input_order():
    first = _candidate("openai", "alpha", stable_order=20)
    second = _candidate("anthropic", "beta", stable_order=10)
    policy = SelectionPolicy(provider_order=("openai", "anthropic"))

    left = select_model(SelectionRequest(candidates=(second, first), policy=policy))
    right = select_model(SelectionRequest(candidates=(first, second), policy=policy))

    assert left.selected == right.selected == first
    assert tuple(item.ref for item in left.eligible) == tuple(item.ref for item in right.eligible)


def test_preferred_identity_beats_provider_order_and_reports_reason():
    preferred = _candidate("anthropic", "beta")
    other = _candidate("openai", "alpha")
    result = select_model(SelectionRequest(
        candidates=(other, preferred),
        purpose="auxiliary_fast",
        policy=SelectionPolicy(
            name="fast",
            preferred=(preferred.ref,),
            provider_order=("openai", "anthropic"),
        ),
    ))
    assert result.selected == preferred
    assert result.reason is SelectionReason.PREFERRED_MATCH
    assert result.purpose == "auxiliary_fast"
    assert result.policy == "fast"


def test_explicit_target_bypasses_ranking_but_not_constraints():
    explicit = _candidate("anthropic", "vision", supports_vision=True)
    other = _candidate("openai", "alpha", supports_vision=True)
    policy = SelectionPolicy(preferred=(other.ref,))

    chosen = select_model(SelectionRequest(
        candidates=(other, explicit),
        explicit=explicit.ref,
        policy=policy,
    ))
    assert chosen.selected == explicit
    assert chosen.eligible == (explicit,)
    assert chosen.reason is SelectionReason.EXPLICIT_MATCH

    rejected = select_model(SelectionRequest(
        candidates=(other, explicit),
        explicit=explicit.ref,
        constraints=SelectionConstraints(allowed_providers=("openai",)),
        policy=policy,
    ))
    assert rejected.selected is None
    assert rejected.reason is SelectionReason.EXPLICIT_UNAVAILABLE
    assert any(item.ref == explicit.ref for item in rejected.rejected)


def test_explicit_target_never_silently_substitutes():
    available = _candidate("openai", "alpha")
    missing = ModelRef("anthropic", "missing")
    result = select_model(SelectionRequest(
        candidates=(available,),
        explicit=missing,
        policy=SelectionPolicy(preferred=(available.ref,)),
    ))
    assert result.selected is None
    assert result.eligible == ()
    assert result.rejected[-1].ref == missing
    assert result.rejected[-1].reasons == ("not_in_universe",)


def test_candidate_builder_uses_canonical_route_owner():
    ref = ModelRef("actual", "gpt-5.4")
    candidate = build_selection_candidate(
        ref,
        _metadata(ref, supports_tools=True),
        InvocationRequest(
            provider="actual",
            model="ignored",
            base_url="https://api.actual.inc/v1",
            configured_api_mode="codex_responses",
        ),
    )
    assert candidate.route.provider == "actual"
    assert candidate.route.model == "gpt-5.4"
    assert candidate.route.api_mode == "chat_completions"
    assert candidate.route.source == "provider_mandate"


def test_candidate_and_request_reject_ambiguous_identity():
    ref = ModelRef("openai", "alpha")
    with pytest.raises(ValueError, match="metadata identity"):
        SelectionCandidate(
            ref=ref,
            metadata=_metadata(ModelRef("openai", "other")),
            route=InvocationRoute(
                "openai", "alpha", "", "chat_completions", "http", False, "test"
            ),
        )

    candidate = _candidate("openai", "alpha")
    with pytest.raises(ValueError, match="unique canonical"):
        SelectionRequest(candidates=(candidate, candidate))


def test_selection_result_exposes_no_secret_or_runtime_object_fields():
    result = select_model(SelectionRequest(candidates=(_candidate("openai", "alpha"),)))
    result_fields = {item.name for item in fields(result)}
    forbidden = {"api_key", "token", "credentials", "credential_pool", "client", "config"}
    assert result_fields.isdisjoint(forbidden)

def test_identity_only_candidate_supports_explicit_selection():
    candidate = SelectionCandidate(ref=ModelRef("openai", "gpt-test"))
    result = select_model(SelectionRequest(
        candidates=(candidate,),
        explicit=candidate.ref,
    ))
    assert result.selected == candidate
    assert result.reason is SelectionReason.EXPLICIT_MATCH


def test_missing_candidate_facts_do_not_satisfy_fact_constraints():
    candidate = SelectionCandidate(ref=ModelRef("openai", "gpt-test"))
    result = select_model(SelectionRequest(
        candidates=(candidate,),
        constraints=SelectionConstraints(
            capabilities=CapabilityRequirements(tools=True),
            allowed_api_modes=("chat_completions",),
        ),
    ))
    assert result.selected is None
    assert result.rejected[0].reasons == (
        "api_mode_not_allowed",
        "requires_tools=True",
    )
