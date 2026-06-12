"""Regression tests for two OpenAI/OpenRouter model-picker bugs.

Bug 1 — OpenAI picker dumped the raw ``/v1/models`` catalog
    ``provider_model_ids("openai")`` hit ``api.openai.com/v1/models`` and
    returned the full 120+ entry catalog (embeddings, whisper, tts, dall-e,
    moderation, gpt-3.5, …). The ``hermes model`` CLI shows only the curated
    agentic list. The picker now intersects the live default-endpoint catalog
    with the curated list (preserving curated order) so both surfaces match.
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

from unittest.mock import patch


from hermes_cli import models as M
from hermes_cli.providers import HERMES_OVERLAYS


# --- Bug 2: overlay no longer lists OPENAI_API_KEY --------------------------

def test_openrouter_overlay_does_not_list_openai_api_key():
    overlay = HERMES_OVERLAYS["openrouter"]
    assert "OPENAI_API_KEY" not in overlay.extra_env_vars


# --- Bug 1: default OpenAI endpoint filters to curated agentic models -------

def test_default_openai_endpoint_filters_to_curated(monkeypatch):
    """The 126-model /v1/models dump is intersected with the curated list."""
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

    # Only curated models survive, in curated order, no junk.
    assert result == list(curated)
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


def test_default_openai_endpoint_falls_back_when_no_curated_access(monkeypatch):
    """If the account serves none of the curated models, fall back to curated."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    curated = M._PROVIDER_MODELS["openai-api"]
    live = ["text-embedding-3-large", "whisper-1", "tts-1"]  # all junk
    with patch.object(M, "fetch_api_models", return_value=live):
        result = M.provider_model_ids("openai-api", force_refresh=True)

    # No curated overlap -> serve the curated defaults so the picker isn't empty.
    assert result == list(curated)


def test_custom_openai_compatible_endpoint_keeps_live_list(monkeypatch):
    """Custom OPENAI_BASE_URL endpoints keep the live catalog verbatim."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://my-proxy.example.com/v1")

    live = ["custom-model-a", "custom-model-b", "some-embedding-model"]
    with patch.object(M, "fetch_api_models", return_value=live):
        result = M.provider_model_ids("openai-api", force_refresh=True)

    assert result == live
