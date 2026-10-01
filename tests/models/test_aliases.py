from __future__ import annotations

import pytest

from models import (
    AmbiguousModelAliasError,
    MODEL_ALIASES,
    ModelAliasPattern,
    resolve_declared_model_id,
    resolve_model_alias,
)


def test_builtin_model_aliases_live_in_model_domain():
    assert MODEL_ALIASES["sonnet"] == ModelAliasPattern("anthropic", "claude-sonnet")
    assert MODEL_ALIASES["gpt"] == ModelAliasPattern("openai", "gpt")
    assert MODEL_ALIASES["glm"] == ModelAliasPattern("z-ai", "glm")


def test_provider_declared_alias_wins_without_catalogue_membership():
    assert resolve_model_alias(
        "fable",
        "proc-provider",
        ["claude-opus-5[1m]"],
        provider_aliases={"fable": "claude-fable-5-1[1m]"},
    ) == "claude-fable-5-1[1m]"


def test_declared_model_resolves_exact_id_case_insensitively():
    assert resolve_declared_model_id(
        "CLAUDE-OPUS-5[1M]",
        "proc-provider",
        ["claude-opus-5[1m]"],
    ) == "claude-opus-5[1m]"


def test_declared_model_resolves_unique_extended_id():
    assert resolve_declared_model_id(
        "claude-opus-5",
        "proc-provider",
        ["claude-opus-5[1m]", "claude-sonnet-5[1m]"],
    ) == "claude-opus-5[1m]"


def test_declared_model_prefix_ambiguity_is_explicit():
    with pytest.raises(AmbiguousModelAliasError) as exc:
        resolve_declared_model_id(
            "claude-opus",
            "proc-provider",
            ["claude-opus-5[1m]", "claude-opus-4-8[1m]"],
        )
    assert exc.value.candidates == (
        "claude-opus-5[1m]",
        "claude-opus-4-8[1m]",
    )


def test_family_alias_ambiguity_is_sorted_only_for_display():
    with pytest.raises(AmbiguousModelAliasError) as exc:
        resolve_model_alias(
            "opus",
            "anthropic",
            [
                "claude-opus-4-20250514",
                "claude-opus-4-8",
                "claude-opus-4-7",
            ],
        )
    assert exc.value.candidates == (
        "claude-opus-4-8",
        "claude-opus-4-7",
        "claude-opus-4-20250514",
    )
