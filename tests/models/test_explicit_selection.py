from __future__ import annotations

import pytest

from models import ModelRef
from models.selection import (
    ExplicitAlias,
    ExplicitDetectionFacts,
    ExplicitProviderFacts,
    ExplicitSelectionError,
    select_explicit_model,
)


def _facts(
    provider: str,
    *,
    alias_models: tuple[str, ...] = (),
    catalog_models: tuple[str, ...] = (),
    model_aliases: tuple[tuple[str, str], ...] = (),
    identity_aliases: tuple[str, ...] = (),
    aggregator: bool = False,
    normalization_ids: tuple[str, ...] = (),
) -> ExplicitProviderFacts:
    return ExplicitProviderFacts(
        provider=provider,
        alias_models=alias_models,
        catalog_models=catalog_models,
        model_aliases=model_aliases,
        identity_aliases=identity_aliases,
        aggregator=aggregator,
        normalization_ids=normalization_ids,
    )


def test_provider_qualified_input_is_interpreted_in_selection_domain():
    result = select_explicit_model(
        "anthropic:claude-sonnet-5",
        "openai",
        provider_facts=(_facts("anthropic"), _facts("openai")),
        known_provider_ids=("anthropic", "openai"),
    )
    assert result.selected.ref == ModelRef("anthropic", "claude-sonnet-5")
    assert result.selected.source == "explicit"


def test_direct_alias_routes_provider_and_preserves_alias_provenance():
    result = select_explicit_model(
        "fast",
        "openai",
        provider_facts=(_facts("openai"), _facts("anthropic")),
        direct_aliases=(ExplicitAlias("fast", ModelRef("anthropic", "claude-fast")),),
    )
    assert result.selected.ref == ModelRef("anthropic", "claude-fast")
    assert result.matched_alias == "fast"


def test_explicit_provider_keeps_provider_when_direct_alias_belongs_elsewhere():
    result = select_explicit_model(
        "fast",
        "openai",
        explicit_provider="openai",
        provider_facts=(_facts("openai"), _facts("anthropic")),
        direct_aliases=(ExplicitAlias("fast", ModelRef("anthropic", "claude-fast")),),
    )
    assert result.selected.ref == ModelRef("openai", "claude-fast")
    assert result.matched_alias == ""


def test_semantic_alias_falls_back_only_to_eligible_provider():
    result = select_explicit_model(
        "sonnet",
        "openai",
        provider_facts=(
            _facts("openai", alias_models=("gpt-5.4",)),
            _facts("anthropic", alias_models=("claude-sonnet-5",)),
        ),
        fallback_providers=("anthropic",),
    )
    assert result.selected.ref == ModelRef("anthropic", "claude-sonnet-5")
    assert result.selected.source == "alias_fallback"
    assert result.matched_alias == "sonnet"


def test_semantic_alias_without_candidate_fails_in_selection_domain():
    with pytest.raises(ExplicitSelectionError, match="alias_unavailable"):
        select_explicit_model(
            "sonnet",
            "openai",
            provider_facts=(_facts("openai", alias_models=("gpt-5.4",)),),
        )


def test_aggregator_catalog_match_beats_configured_provider_detection():
    result = select_explicit_model(
        "claude-sonnet-5",
        "openrouter",
        provider_facts=(
            _facts(
                "openrouter",
                catalog_models=("anthropic/claude-sonnet-5",),
                aggregator=True,
            ),
            _facts("anthropic"),
        ),
        configured_matches=(ModelRef("anthropic", "claude-sonnet-5"),),
    )
    assert result.selected.ref == ModelRef("openrouter", "anthropic/claude-sonnet-5")
    assert result.selected.source == "catalog"


def test_configured_provider_prefers_current_equivalent_identity():
    result = select_explicit_model(
        "model-x",
        "custom:relay",
        provider_facts=(
            _facts(
                "relay",
                identity_aliases=("custom:relay",),
            ),
        ),
        configured_matches=(
            ModelRef("relay", "model-x"),
            ModelRef("other", "model-x"),
        ),
    )
    assert result.selected.ref == ModelRef("custom:relay", "model-x")


def test_multiple_configured_providers_are_ambiguous():
    with pytest.raises(ExplicitSelectionError) as exc:
        select_explicit_model(
            "model-x",
            "openai",
            provider_facts=(_facts("openai"),),
            configured_matches=(
                ModelRef("relay-a", "model-x"),
                ModelRef("relay-b", "model-x"),
            ),
        )
    assert exc.value.code == "ambiguous_configured"
    assert exc.value.providers == ("relay-a", "relay-b")


def test_block_provider_fallback_keeps_unresolved_model_on_current_provider():
    result = select_explicit_model(
        "model-x",
        "nous",
        provider_facts=(_facts("nous"), _facts("relay")),
        configured_matches=(ModelRef("relay", "model-x"),),
        detection=ExplicitDetectionFacts(openrouter_candidate=ModelRef("openai", "model-x")),
        block_provider_fallback=True,
    )
    assert result.selected.ref == ModelRef("nous", "model-x")
    assert result.selected.source == "current"
