"""Regression tests for two OpenAI/OpenRouter model-picker bugs.

Bug 1 — OpenAI picker dumped the raw ``/v1/models`` catalog
    ``provider_model_ids("openai")`` hit ``api.openai.com/v1/models`` and
    returned the full 120+ entry catalog (embeddings, whisper, tts, dall-e,
    moderation, gpt-3.5, …). The picker now filters the live official-host
    catalog to current chat families and preserves custom endpoint discovery.
    Custom OpenAI-compatible endpoints (proxies, gateways) keep the live list
    verbatim so discovery still works.

Bug 2 — OpenRouter appeared authenticated whenever OPENAI_API_KEY was set
    OpenRouter's HermesOverlay carried ``extra_env_vars=("OPENAI_API_KEY",)``.
    ``list_authenticated_providers`` reads ``extra_env_vars`` to decide whether
    a provider has credentials, so any OpenAI user saw a phantom OpenRouter
    row. The overlay entry is removed; runtime credential resolution still
    falls back to OPENAI_API_KEY for explicitly-selected OpenRouter (handled
    in runtime_provider.py, independent of the overlay).
"""

import os
from unittest.mock import patch

import pytest

from hermes_cli import models as M
from hermes_cli.providers import HERMES_OVERLAYS


# --- Bug 2: overlay no longer lists OPENAI_API_KEY --------------------------

def test_openrouter_overlay_does_not_list_openai_api_key():
    overlay = HERMES_OVERLAYS["openrouter"]
    assert "OPENAI_API_KEY" not in overlay.extra_env_vars


# --- Bug 1: default OpenAI endpoint filters to curated agentic models -------

def test_default_openai_endpoint_filters_to_curated(monkeypatch):
    """The large /v1/models dump is reduced to current chat models."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    curated = M._PROVIDER_MODELS["openai-api"]
    # Live catalog: every curated model PLUS a pile of non-agentic junk.
    live = list(curated) + [
        "text-embedding-3-large", "whisper-1", "tts-1", "dall-e-3",
        "gpt-3.5-turbo", "davinci-002", "omni-moderation-latest",
    ]
    with patch.object(M, "fetch_api_models", return_value=live):
        result = M.provider_model_ids("openai-api", force_refresh=True)

    from hermes_cli.models_catalog_static import openai_chat_models

    # Only current chat models survive, in stable version order, with no junk.
    assert result == openai_chat_models(curated)
    for m in result:
        assert m in curated


def test_default_openai_endpoint_intersects_account_access(monkeypatch):
    """Curated models the account can't access are dropped (intersection)."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    curated = M._PROVIDER_MODELS["openai-api"]
    # Account only serves the first two curated models.
    live = list(curated[:2]) + ["text-embedding-3-large", "whisper-1"]
    with patch.object(M, "fetch_api_models", return_value=live):
        result = M.provider_model_ids("openai-api", force_refresh=True)

    assert result == list(curated[:2])


def test_unknown_new_gpt_family_is_offered(monkeypatch):
    """A future GPT family appears as soon as the account-scoped listing serves it."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    unknown = "gpt-99-nebula"
    assert unknown not in M._PROVIDER_MODELS["openai-api"]
    with patch.object(M, "fetch_api_models", return_value=["gpt-5.5", unknown]):
        result = M.provider_model_ids("openai-api", force_refresh=True)

    assert result == [unknown, "gpt-5.5"]


@pytest.mark.parametrize("stale", ["gpt-4o", "gpt-4.1", "gpt-5.4", "gpt-5.3-codex", "gpt-5-mini"])
def test_models_below_the_floor_are_not_offered(monkeypatch, stale):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    with patch.object(M, "fetch_api_models", return_value=[stale, "gpt-5.5"]):
        result = M.provider_model_ids("openai-api", force_refresh=True)

    assert result == ["gpt-5.5"]


def test_offline_picker_uses_floor_without_changing_routing_catalog(monkeypatch):
    """Picker policy does not remove older explicit model names from provider routing."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    assert "gpt-5.4" in M._PROVIDER_MODELS["openai-api"]
    assert "gpt-5.4" not in M.provider_model_ids("openai-api", force_refresh=True)


def test_legacy_openai_picker_uses_current_api_fallback(monkeypatch):
    """The legacy runtime id shares the direct API picker without sharing its routing catalog."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    result = M.provider_model_ids("openai", force_refresh=True)

    assert "gpt-5.5" in result
    assert "gpt-5.4" not in result
    assert M._PROVIDER_MODELS["openai"] != M._PROVIDER_MODELS["openai-api"]


def test_successful_below_floor_catalog_is_authoritatively_empty(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    with patch.object(M, "fetch_api_models", return_value=["gpt-4o"]):
        result = M.provider_model_ids("openai-api", force_refresh=True)

    assert result == []


def test_successful_empty_catalog_is_authoritative(monkeypatch):
    """An answered empty listing is distinct from a failed probe (`None`)."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    with patch.object(M, "fetch_api_models", return_value=[]):
        result = M.provider_model_ids("openai-api", force_refresh=True)

    assert result == []


def test_non_chat_families_and_dated_snapshots_are_filtered():
    from hermes_cli.models_catalog_static import openai_chat_models

    noise = [
        "gpt-audio-1.5", "gpt-image-2.5-flare", "gpt-realtime-2.1", "gpt-live-1",
        "gpt-transcribe", "gpt-5.5-2026-04-23", "text-embedding-3-large", "o3-pro",
    ]
    assert openai_chat_models(noise) == []
    assert openai_chat_models(["gpt-5.5-2026-04-23", "gpt-5.5"]) == ["gpt-5.5"]


def test_astra_is_offered_only_by_successful_account_discovery(monkeypatch):
    """A gated preview may enrich the picker only when this API key lists it."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    with patch.object(M, "fetch_api_models", return_value=["gpt-5.6-sol", "gpt-6-astra"]):
        discovered = M.provider_model_ids("openai-api", force_refresh=True)
    with patch.object(M, "fetch_api_models", return_value=["gpt-5.6-sol"]):
        not_entitled = M.provider_model_ids("openai-api", force_refresh=True)
    with patch.object(M, "fetch_api_models", side_effect=RuntimeError("discovery unavailable")):
        discovery_failed = M.provider_model_ids("openai-api", force_refresh=True)

    assert "gpt-6-astra" in discovered
    assert "gpt-6-astra" not in not_entitled
    assert "gpt-6-astra" not in discovery_failed
