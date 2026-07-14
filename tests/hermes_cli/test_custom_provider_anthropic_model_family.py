"""Tests for per-model Anthropic family hints on custom providers."""

from unittest.mock import patch

import pytest

from hermes_cli.config import get_custom_provider_anthropic_model_family


def _providers():
    return [
        {
            "name": "opaque-anthropic",
            "base_url": "https://router.example/anthropic/",
            "models": {
                "ep-opus": {
                    "anthropic_model_family": "  claude-opus-4-8  ",
                },
                "ep-fable": {
                    "anthropic_model_family": "claude-fable-5",
                },
            },
        }
    ]


def test_returns_stripped_family_for_exact_url_and_model():
    assert get_custom_provider_anthropic_model_family(
        "ep-opus",
        "https://router.example/anthropic",
        _providers(),
    ) == "claude-opus-4-8"


def test_trailing_slash_is_ignored_and_models_resolve_independently():
    providers = _providers()
    assert get_custom_provider_anthropic_model_family(
        "ep-opus",
        "https://router.example/anthropic/",
        providers,
    ) == "claude-opus-4-8"
    assert get_custom_provider_anthropic_model_family(
        "ep-fable",
        "https://router.example/anthropic",
        providers,
    ) == "claude-fable-5"


@pytest.mark.parametrize(
    ("model", "base_url"),
    [
        ("missing", "https://router.example/anthropic"),
        ("ep-opus", "https://other.example/anthropic"),
        ("", "https://router.example/anthropic"),
        ("ep-opus", ""),
    ],
)
def test_missing_or_mismatched_route_returns_none(model, base_url):
    assert get_custom_provider_anthropic_model_family(
        model,
        base_url,
        _providers(),
    ) is None


@pytest.mark.parametrize("invalid", [None, "", "   ", 7, [], {}])
def test_invalid_family_values_return_none(invalid):
    providers = [
        {
            "base_url": "https://router.example/anthropic",
            "models": {"ep": {"anthropic_model_family": invalid}},
        }
    ]
    assert get_custom_provider_anthropic_model_family(
        "ep",
        "https://router.example/anthropic",
        providers,
    ) is None


def test_malformed_entries_are_skipped_without_hiding_a_later_match():
    providers = [
        "not-a-dict",
        None,
        {"base_url": "https://router.example/anthropic", "models": []},
        {
            "base_url": "https://router.example/anthropic",
            "models": {"ep": "not-a-dict"},
        },
        {
            "base_url": "https://router.example/anthropic",
            "models": {
                "ep": {"anthropic_model_family": "claude-opus-4-6"},
            },
        },
    ]
    assert get_custom_provider_anthropic_model_family(
        "ep",
        "https://router.example/anthropic",
        providers,
    ) == "claude-opus-4-6"


def test_explicit_provider_list_does_not_reload_configuration():
    with patch("hermes_cli.config.get_compatible_custom_providers") as loader:
        assert get_custom_provider_anthropic_model_family(
            "ep-fable",
            "https://router.example/anthropic",
            _providers(),
        ) == "claude-fable-5"
    loader.assert_not_called()
