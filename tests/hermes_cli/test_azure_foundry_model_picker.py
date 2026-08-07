"""Regression coverage for Azure Foundry deployment aliases in model pickers."""

from __future__ import annotations

import hermes_cli.config as config_mod
import hermes_cli.model_switch as model_switch
import hermes_cli.models as models


def test_azure_foundry_aliases_keep_primary_first_and_deduplicate_fallbacks(monkeypatch):
    monkeypatch.setattr(
        models,
        "_get_model_config_dict",
        lambda: {
            "provider": "azure-foundry",
            "default": "gpt-5.6-sol",
            "base_url": "https://example.openai.azure.com/openai/v1",
        },
    )
    monkeypatch.setattr(
        config_mod,
        "load_config",
        lambda: {
            "fallback_providers": [
                {"provider": "azure-foundry", "model": "GPT-5.6-SOL"},
                {"provider": "azure-foundry", "model": "gpt-5.5"},
            ]
        },
    )

    assert models.provider_model_ids("azure-foundry") == ["gpt-5.6-sol", "gpt-5.5"]


def test_azure_foundry_picker_surfaces_fallback_alias_when_another_provider_is_primary(monkeypatch):
    monkeypatch.setattr(
        models,
        "_get_model_config_dict",
        lambda: {"provider": "xai-oauth", "default": "grok-4.5"},
    )
    monkeypatch.setattr(
        config_mod,
        "load_config",
        lambda: {
            "fallback_providers": [
                {
                    "provider": "azure-foundry",
                    "model": "gpt-5.6-sol",
                    "base_url": "https://example.openai.azure.com/openai/v1",
                }
            ]
        },
    )

    assert models.provider_model_ids("azure-foundry") == ["gpt-5.6-sol"]


def test_azure_foundry_picker_supports_legacy_fallback_model(monkeypatch):
    monkeypatch.setattr(
        models,
        "_get_model_config_dict",
        lambda: {"provider": "xai-oauth", "default": "grok-4.5"},
    )
    monkeypatch.setattr(
        config_mod,
        "load_config",
        lambda: {
            "fallback_model": {
                "provider": "azure-foundry",
                "model": "gpt-5.6-sol",
            }
        },
    )

    assert models.provider_model_ids("azure-foundry") == ["gpt-5.6-sol"]


def test_azure_foundry_aliases_bypass_stale_raw_model_cache(monkeypatch):
    monkeypatch.setattr(
        models,
        "_get_model_config_dict",
        lambda: {"provider": "azure-foundry", "default": "gpt-5.6-sol"},
    )
    monkeypatch.setattr(config_mod, "load_config", lambda: {})
    monkeypatch.setattr(
        models,
        "_load_provider_models_cache",
        lambda: {
            "azure-foundry": {
                "fp": models._credential_fingerprint("azure-foundry"),
                "at": 9_999_999_999,
                "models": ["text-embedding-3-large", "gpt-image-1"],
            }
        },
    )
    saved = []
    monkeypatch.setattr(models, "_save_provider_models_cache", saved.append)

    assert models.cached_provider_model_ids("azure-foundry") == ["gpt-5.6-sol"]
    assert saved and "azure-foundry" not in saved[-1]


def test_interactive_picker_retains_azure_row_with_configured_alias(monkeypatch):
    monkeypatch.setattr(
        model_switch,
        "list_authenticated_providers",
        lambda **_kwargs: [
            {
                "slug": "azure-foundry",
                "name": "Azure Foundry",
                "models": ["gpt-5.6-sol"],
                "total_models": 1,
            }
        ],
    )

    rows = model_switch.list_picker_providers()

    assert [row["slug"] for row in rows] == ["azure-foundry"]
    assert rows[0]["models"] == ["gpt-5.6-sol"]
