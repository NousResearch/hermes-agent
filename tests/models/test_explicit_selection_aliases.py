from __future__ import annotations

from dataclasses import fields

import pytest

from models import AmbiguousModelAliasError, ModelRef
from models.selection import ExplicitAlias, ExplicitProviderFacts, select_explicit_model


def _facts(
    provider: str,
    *,
    alias_models: tuple[str, ...] = (),
    catalog_models: tuple[str, ...] = (),
    aggregator: bool = False,
) -> ExplicitProviderFacts:
    return ExplicitProviderFacts(
        provider=provider,
        alias_models=alias_models,
        catalog_models=catalog_models,
        aggregator=aggregator,
    )


def test_direct_alias_beats_catalog_interpretation():
    result = select_explicit_model(
        "glm",
        "openrouter",
        provider_facts=(
            _facts(
                "openrouter",
                catalog_models=("vendor/glm",),
                aggregator=True,
            ),
            _facts("custom"),
        ),
        direct_aliases=(ExplicitAlias("glm", ModelRef("custom", "glm-4.7")),),
    )
    assert result.selected.ref == ModelRef("custom", "glm-4.7")
    assert result.matched_alias == "glm"


def test_direct_alias_input_is_trimmed():
    result = select_explicit_model(
        "  myalias  ",
        "openrouter",
        provider_facts=(_facts("openrouter"), _facts("custom")),
        direct_aliases=(ExplicitAlias("myalias", ModelRef("custom", "my-model")),),
    )
    assert result.selected.ref == ModelRef("custom", "my-model")


def test_semantic_alias_ambiguity_preserves_candidate_ordering():
    facts = _facts(
        "anthropic",
        alias_models=(
            "claude-opus-4-1",
            "claude-opus-4-7",
            "claude-opus-4-8",
            "claude-opus-4-20250514",
        ),
    )
    with pytest.raises(AmbiguousModelAliasError) as exc:
        select_explicit_model("opus", "anthropic", provider_facts=(facts,))
    assert exc.value.candidates[0] == "claude-opus-4-8"
    assert set(exc.value.candidates) == {
        "claude-opus-4-1",
        "claude-opus-4-7",
        "claude-opus-4-8",
        "claude-opus-4-20250514",
    }


def test_unsynced_new_alias_candidate_sorts_first():
    facts = _facts(
        "anthropic",
        alias_models=(
            "claude-opus-4-7",
            "claude-opus-4-8",
            "claude-opus-4-20250514",
            "claude-opus-4-9",
        ),
    )
    with pytest.raises(AmbiguousModelAliasError) as exc:
        select_explicit_model("opus", "anthropic", provider_facts=(facts,))
    assert exc.value.candidates[0] == "claude-opus-4-9"


def test_single_semantic_alias_match_resolves():
    result = select_explicit_model(
        "opus",
        "anthropic",
        provider_facts=(
            _facts(
                "anthropic",
                alias_models=("claude-opus-4-8", "claude-sonnet-4-6"),
            ),
        ),
    )
    assert result.selected.ref == ModelRef("anthropic", "claude-opus-4-8")
    assert result.matched_alias == "opus"


def test_fallback_provider_eligibility_is_caller_supplied():
    facts = (
        _facts("openai", alias_models=("gpt-5.4",)),
        _facts("anthropic", alias_models=("claude-sonnet-5",)),
    )
    with pytest.raises(ValueError, match="alias_unavailable"):
        select_explicit_model("sonnet", "openai", provider_facts=facts)

    result = select_explicit_model(
        "sonnet",
        "openai",
        provider_facts=facts,
        fallback_providers=("anthropic",),
    )
    assert result.selected.ref == ModelRef("anthropic", "claude-sonnet-5")


def test_provider_facts_carry_no_credential_state():
    names = {field.name for field in fields(ExplicitProviderFacts)}
    assert names.isdisjoint(
        {"authenticated", "api_key", "token", "credentials", "credential_provider"}
    )
