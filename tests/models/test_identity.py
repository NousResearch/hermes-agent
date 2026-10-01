from __future__ import annotations

import pytest

from models import (
    AmbiguousModelAliasError,
    ModelAliasPattern,
    ModelRef,
    format_model_ref,
    normalize_model_id,
    normalize_model_ref,
    parse_configured_provider_ref,
    parse_model_ref,
    resolve_model_alias,
)


def test_model_ref_canonicalizes_provider_alias_and_trims_model():
    assert ModelRef("claude", "  claude-sonnet-4.6  ") == ModelRef(
        "anthropic", "claude-sonnet-4.6"
    )


def test_model_ref_preserves_named_custom_provider_identity():
    ref = ModelRef("custom:Relay", "model-x")
    assert ref == ModelRef("custom:relay", "model-x")


def test_parse_model_ref_accepts_known_provider_alias():
    ref = parse_model_ref(
        "claude:claude-sonnet-4.6",
        "openrouter",
        known_provider_ids={"anthropic", "openrouter"},
    )
    assert ref == ModelRef("anthropic", "claude-sonnet-4.6")


def test_parse_model_ref_does_not_treat_unknown_colon_as_provider():
    ref = parse_model_ref(
        "qwen3:8b",
        "custom",
        known_provider_ids={"custom", "openrouter"},
    )
    assert ref == ModelRef("custom", "qwen3:8b")


def test_parse_model_ref_uses_longest_named_custom_provider():
    ref = parse_model_ref(
        "custom:relay:west:qwen3:8b",
        "openrouter",
        known_provider_ids={"custom", "openrouter"},
        named_custom_provider_ids={"custom:relay", "custom:relay:west"},
    )
    assert ref == ModelRef("custom:relay:west", "qwen3:8b")


def test_parse_model_ref_supports_generic_custom_provider():
    ref = parse_model_ref(
        "custom:qwen3:8b",
        "openrouter",
        known_provider_ids={"custom", "openrouter"},
    )
    assert ref == ModelRef("custom", "qwen3:8b")


def test_unqualified_slash_model_stays_on_default_provider():
    ref = parse_model_ref(
        "anthropic/claude-sonnet-4.6",
        "openrouter",
        known_provider_ids={"anthropic", "openrouter"},
    )
    assert ref == ModelRef("openrouter", "anthropic/claude-sonnet-4.6")


def test_configured_provider_ref_requires_explicit_configured_prefix():
    assert parse_configured_provider_ref(
        "relay/qwen3:8b",
        {"relay"},
    ) == ModelRef("relay", "qwen3:8b")
    assert parse_configured_provider_ref(
        "anthropic/claude-sonnet-4.6",
        {"relay"},
    ) is None


def test_model_ref_format_parse_round_trip_with_named_custom_and_model_colon():
    original = ModelRef("custom:relay", "qwen3:8b")
    encoded = format_model_ref(original)
    assert encoded == "custom:relay:qwen3:8b"
    assert parse_model_ref(
        encoded,
        "",
        known_provider_ids={"custom"},
        named_custom_provider_ids={"custom:relay"},
    ) == original


def test_empty_model_formats_as_empty_choice():
    assert format_model_ref(ModelRef("openrouter", "")) == ""


def test_normalization_seam_is_conservative_before_provider_rules_move():
    assert normalize_model_id(
        "custom:relay",
        "  Vendor/Model.Name:Tag  ",
        known_ids={"Vendor/Model.Name:Tag"},
    ) == "Vendor/Model.Name:Tag"
    assert normalize_model_ref(
        ModelRef("custom:relay", " Vendor/Model.Name:Tag ")
    ) == ModelRef("custom:relay", "Vendor/Model.Name:Tag")


def test_model_alias_resolves_against_direct_provider_family():
    aliases = {"sonnet": ModelAliasPattern("anthropic", "claude-sonnet")}
    assert resolve_model_alias(
        "SONNET",
        "anthropic",
        ["claude-haiku-4.5", "claude-sonnet-4.6"],
        aliases,
    ) == "claude-sonnet-4.6"


def test_model_alias_resolves_aggregator_vendor_family():
    aliases = {"sonnet": ModelAliasPattern("anthropic", "claude-sonnet")}
    assert resolve_model_alias(
        "sonnet",
        "openrouter",
        ["openai/gpt-5.4", "anthropic/claude-sonnet-4.6"],
        aliases,
    ) == "anthropic/claude-sonnet-4.6"


def test_model_alias_never_silently_selects_ambiguous_version():
    aliases = {"sonnet": ModelAliasPattern("anthropic", "claude-sonnet")}
    with pytest.raises(AmbiguousModelAliasError) as exc:
        resolve_model_alias(
            "sonnet",
            "anthropic",
            ["claude-sonnet-4.5", "claude-sonnet-4.6"],
            aliases,
        )
    assert exc.value.candidates == ("claude-sonnet-4.6", "claude-sonnet-4.5")


def test_unknown_model_alias_returns_none():
    assert resolve_model_alias(
        "future",
        "anthropic",
        ["claude-sonnet-4.6"],
        {"sonnet": ModelAliasPattern("anthropic", "claude-sonnet")},
    ) is None
