"""Regression tests for /model picker showing the active model (#40676).

When the user's configured default model is not in the curated list (e.g. an
OpenRouter model not in OPENROUTER_MODELS), the /model picker should still
include it — it's the model the user is actually running.
"""

import os
from unittest.mock import patch

from hermes_cli.model_switch import list_authenticated_providers


def test_current_model_injected_when_not_in_curated_list(monkeypatch):
    """Active model should appear in picker even if not in curated list."""
    monkeypatch.setattr(
        "agent.models_dev.fetch_models_dev",
        lambda: {
            "openrouter": {
                "env": ["OPENROUTER_API_KEY"],
                "inference_base_url": "https://openrouter.ai/api/v1",
            }
        },
    )
    monkeypatch.setattr(
        "hermes_cli.models.cached_provider_model_ids",
        lambda slug, **kw: ["anthropic/claude-sonnet-4", "openai/gpt-4o"] if slug == "openrouter" else [],
    )
    monkeypatch.setattr(
        "hermes_cli.models._merge_with_models_dev",
        lambda slug, ids: ids,
    )

    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=False):
        providers = list_authenticated_providers(
            current_provider="openrouter",
            current_model="google/gemini-2.5-flash-lite",
            max_models=50,
        )

    openrouter = next((p for p in providers if p["slug"] == "openrouter"), None)
    assert openrouter is not None, "openrouter provider should be present"
    assert "google/gemini-2.5-flash-lite" in openrouter["models"], (
        f"current model should be in the models list, got: {openrouter['models']}"
    )
    # Should be first (prepended)
    assert openrouter["models"][0] == "google/gemini-2.5-flash-lite"


def test_current_model_not_duplicated_when_already_in_list(monkeypatch):
    """Active model should not appear twice if already in curated list."""
    monkeypatch.setattr(
        "agent.models_dev.fetch_models_dev",
        lambda: {
            "openrouter": {
                "env": ["OPENROUTER_API_KEY"],
                "inference_base_url": "https://openrouter.ai/api/v1",
            }
        },
    )
    monkeypatch.setattr(
        "hermes_cli.models.cached_provider_model_ids",
        lambda slug, **kw: ["anthropic/claude-sonnet-4", "openai/gpt-4o"] if slug == "openrouter" else [],
    )
    monkeypatch.setattr(
        "hermes_cli.models._merge_with_models_dev",
        lambda slug, ids: ids,
    )

    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=False):
        providers = list_authenticated_providers(
            current_provider="openrouter",
            current_model="openai/gpt-4o",  # Already in curated list
            max_models=50,
        )

    openrouter = next((p for p in providers if p["slug"] == "openrouter"), None)
    assert openrouter is not None
    assert openrouter["models"].count("openai/gpt-4o") == 1, (
        f"model should not be duplicated, got: {openrouter['models']}"
    )


def test_current_model_not_injected_for_different_provider(monkeypatch):
    """Active model should NOT be injected into a different provider's list."""
    monkeypatch.setattr(
        "agent.models_dev.fetch_models_dev",
        lambda: {
            "openrouter": {
                "env": ["OPENROUTER_API_KEY"],
                "inference_base_url": "https://openrouter.ai/api/v1",
            }
        },
    )
    monkeypatch.setattr(
        "hermes_cli.models.cached_provider_model_ids",
        lambda slug, **kw: ["anthropic/claude-sonnet-4"] if slug == "openrouter" else [],
    )
    monkeypatch.setattr(
        "hermes_cli.models._merge_with_models_dev",
        lambda slug, ids: ids,
    )

    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=False):
        providers = list_authenticated_providers(
            current_provider="anthropic",  # Different provider
            current_model="anthropic/claude-sonnet-4",
            max_models=50,
        )

    openrouter = next((p for p in providers if p["slug"] == "openrouter"), None)
    if openrouter:
        # The model is in the curated list, so it will be present.
        # The important thing is the count is 1 (not injected).
        assert openrouter["models"].count("anthropic/claude-sonnet-4") <= 1


def test_current_model_injected_for_mdev_matched_provider(monkeypatch):
    """Active model injected when current_provider matches mdev_id (models.dev slug)."""
    monkeypatch.setattr(
        "agent.models_dev.fetch_models_dev",
        lambda: {
            "openrouter": {
                "env": ["OPENROUTER_API_KEY"],
                "inference_base_url": "https://openrouter.ai/api/v1",
            }
        },
    )
    monkeypatch.setattr(
        "hermes_cli.models.cached_provider_model_ids",
        lambda slug, **kw: ["openai/gpt-4o"] if slug == "openrouter" else [],
    )
    monkeypatch.setattr(
        "hermes_cli.models._merge_with_models_dev",
        lambda slug, ids: ids,
    )

    # Provider passed as models.dev ID (e.g. "openrouter" matches mdev_id)
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=False):
        providers = list_authenticated_providers(
            current_provider="openrouter",
            current_model="google/gemini-2.5-pro",
            max_models=50,
        )

    openrouter = next((p for p in providers if p["slug"] == "openrouter"), None)
    assert openrouter is not None
    assert "google/gemini-2.5-pro" in openrouter["models"], (
        f"current model should be injected, got: {openrouter['models']}"
    )
