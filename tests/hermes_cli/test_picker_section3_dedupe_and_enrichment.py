"""Regression tests for /model picker Section 3 dedupe and live count enrichment.

Covers two related bugs in ``list_authenticated_providers``:

1. Section 3 (user-defined providers from config.yaml ``providers:``) appended
   rows without checking ``seen_slugs``, producing duplicate entries when a
   provider name overlapped with a built-in (issue #7524).

2. User-defined providers like ``openai-direct`` that supply only a
   ``base_url`` + ``key_env`` (no explicit ``models:`` list) showed
   ``total_models=0`` in the picker even though the live ``/v1/models``
   catalog returned a full list. Counts are now enriched from
   ``provider_model_ids`` (and a direct ``/v1/models`` probe for
   user-defined providers) so the CLI picker matches what the TUI
   gateway already showed after selection.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from hermes_cli.model_switch import list_authenticated_providers


# --- Section 3 dedupe -------------------------------------------------------


@patch.dict(os.environ, {"OPENROUTER_API_KEY": "fake-or"}, clear=False)
def test_user_provider_does_not_duplicate_built_in(monkeypatch):
    """A user-defined provider sharing a built-in slug must not appear twice."""
    # `openrouter` is a built-in provider; redefining it in config.yaml
    # must NOT add a second row to the picker.
    user_providers = {
        "openrouter": {
            "name": "OpenRouter (custom)",
            "api": "https://openrouter.ai/api/v1",
            "default_model": "anthropic/claude-opus-4-6",
        }
    }

    providers = list_authenticated_providers(
        current_provider="",
        user_providers=user_providers,
        custom_providers=[],
    )

    matches = [p for p in providers if p["slug"].lower() == "openrouter"]
    assert len(matches) == 1, (
        f"openrouter should appear once; got {len(matches)} entries: {matches}"
    )


def test_user_provider_unique_slug_still_appears(monkeypatch):
    """A user-defined provider with a unique slug must still be listed."""
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr("hermes_cli.providers.HERMES_OVERLAYS", {})

    user_providers = {
        "my-private-llm": {
            "name": "My Private LLM",
            "api": "http://internal.example.com/v1",
            "default_model": "internal-model-1",
        }
    }

    providers = list_authenticated_providers(
        current_provider="",
        user_providers=user_providers,
        custom_providers=[],
    )

    assert any(p["slug"] == "my-private-llm" for p in providers), (
        "User-defined providers with unique slugs should still be listed"
    )


# --- Live count enrichment --------------------------------------------------


def test_user_provider_count_enriched_from_live_probe(monkeypatch):
    """openai-direct (no explicit models in config) should show the live count."""
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr("hermes_cli.providers.HERMES_OVERLAYS", {})

    user_providers = {
        "openai-direct": {
            "name": "OpenAI Direct",
            "base_url": "https://api.openai.com/v1",
            "key_env": "OPENAI_API_KEY",
            # no `models:` and no `default_model` — live probe is the
            # only source of catalog data
        }
    }

    fake_catalog = [f"gpt-fake-{i}" for i in range(122)]

    def fake_fetch(api_key, base_url, timeout=5.0):
        assert base_url == "https://api.openai.com/v1"
        return fake_catalog

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr("hermes_cli.models.fetch_api_models", fake_fetch)
    # Also stub provider_model_ids since openai-direct has its own branch
    # there which would call the real network.
    monkeypatch.setattr(
        "hermes_cli.models.provider_model_ids",
        lambda slug, **kw: fake_catalog if slug == "openai-direct" else [],
    )

    providers = list_authenticated_providers(
        current_provider="",
        user_providers=user_providers,
        custom_providers=[],
        max_models=8,
    )

    od = next((p for p in providers if p["slug"] == "openai-direct"), None)
    assert od is not None, "openai-direct should appear in picker"
    assert od["total_models"] == 122, (
        f"openai-direct should show live count 122, got {od['total_models']}"
    )
    # The visible models list is capped to max_models
    assert len(od["models"]) == 8


def test_user_provider_explicit_models_not_overwritten(monkeypatch):
    """When a user provider declares explicit models, live probe must not overwrite them."""
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr("hermes_cli.providers.HERMES_OVERLAYS", {})

    user_providers = {
        "local-ollama": {
            "name": "Local Ollama",
            "api": "http://localhost:11434/v1",
            "default_model": "model-a",
            "models": ["model-a", "model-b", "model-c", "model-d"],
        }
    }

    # Even if the live probe returns something different, the explicit
    # config must win.
    monkeypatch.setattr(
        "hermes_cli.models.provider_model_ids",
        lambda slug, **kw: ["surprise-model-1", "surprise-model-2"],
    )

    providers = list_authenticated_providers(
        current_provider="",
        user_providers=user_providers,
        custom_providers=[],
        max_models=50,
    )

    lo = next((p for p in providers if p["slug"] == "local-ollama"), None)
    assert lo is not None
    assert lo["total_models"] == 4
    assert "surprise-model-1" not in lo["models"]


def test_live_probe_failure_falls_back_to_config(monkeypatch):
    """If the live probe raises, the picker must still return config-only data."""
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr("hermes_cli.providers.HERMES_OVERLAYS", {})

    user_providers = {
        "openai-direct": {
            "name": "OpenAI Direct",
            "base_url": "https://api.openai.com/v1",
            "key_env": "OPENAI_API_KEY",
        }
    }

    def boom(*args, **kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr("hermes_cli.models.provider_model_ids", boom)
    monkeypatch.setattr("hermes_cli.models.fetch_api_models", boom)

    providers = list_authenticated_providers(
        current_provider="",
        user_providers=user_providers,
        custom_providers=[],
    )

    od = next((p for p in providers if p["slug"] == "openai-direct"), None)
    assert od is not None, "openai-direct should still appear when live probe fails"
    assert od["total_models"] == 0  # config has no models, probe failed
