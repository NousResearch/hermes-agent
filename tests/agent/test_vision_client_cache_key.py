"""Tests for vision client cache key including model (#14085).

Verifies that _client_cache_key() and _get_cached_client() distinguish
between requests to the same provider with different models, so that
e.g. two OpenRouter vision calls with different models do not collide.
"""

from unittest.mock import patch, MagicMock

import pytest


def _stub_resolve_provider_client(provider, model, async_mode, **kw):
    """Return a unique mock client for each call."""
    client = MagicMock(name=f"client-{provider}-{model}")
    client.api_key = "test"
    client.base_url = kw.get("explicit_base_url", "http://localhost:8081/v1")
    return client, model or "default-model"


@pytest.fixture(autouse=True)
def _clean_client_cache():
    """Clear the client cache before and after each test."""
    import agent.auxiliary_client as ac
    ac._client_cache.clear()
    yield
    ac._client_cache.clear()


class TestCacheKeyIncludesModel:
    """_client_cache_key must include the model so same-provider
    requests with different models get separate cache entries."""

    def test_different_models_produce_different_keys(self):
        from agent.auxiliary_client import _client_cache_key

        key_a = _client_cache_key("openrouter", async_mode=False,
                                  model="google/gemini-3-flash-preview")
        key_b = _client_cache_key("openrouter", async_mode=False,
                                  model="openai/gpt-4o")

        assert key_a != key_b, (
            "Same provider with different models must produce different cache keys"
        )

    def test_same_model_produces_same_key(self):
        from agent.auxiliary_client import _client_cache_key

        key_a = _client_cache_key("openrouter", async_mode=False,
                                  model="google/gemini-3-flash-preview")
        key_b = _client_cache_key("openrouter", async_mode=False,
                                  model="google/gemini-3-flash-preview")

        assert key_a == key_b, (
            "Same provider and model must produce the same cache key"
        )

    def test_none_model_matches_empty_string(self):
        from agent.auxiliary_client import _client_cache_key

        key_none = _client_cache_key("openrouter", async_mode=False, model=None)
        key_empty = _client_cache_key("openrouter", async_mode=False, model="")

        assert key_none == key_empty, (
            "None and empty-string model should produce the same key"
        )


class TestGetCachedClientModelIsolation:
    """_get_cached_client must return different clients for different
    models on the same provider."""

    def test_different_models_return_different_clients(self):
        from agent.auxiliary_client import _get_cached_client

        with patch("agent.auxiliary_client.resolve_provider_client",
                   side_effect=_stub_resolve_provider_client):
            client_a, model_a = _get_cached_client(
                "openrouter", "google/gemini-3-flash-preview", async_mode=False)
            client_b, model_b = _get_cached_client(
                "openrouter", "openai/gpt-4o", async_mode=False)

        assert client_a is not client_b, (
            "Same provider with different models must NOT return the same "
            "cached client — this is the bug from #14085"
        )
        assert model_a == "google/gemini-3-flash-preview"
        assert model_b == "openai/gpt-4o"

    def test_same_model_returns_cached_client(self):
        from agent.auxiliary_client import _get_cached_client

        with patch("agent.auxiliary_client.resolve_provider_client",
                   side_effect=_stub_resolve_provider_client):
            client_a, _ = _get_cached_client(
                "openrouter", "google/gemini-3-flash-preview", async_mode=False)
            client_b, _ = _get_cached_client(
                "openrouter", "google/gemini-3-flash-preview", async_mode=False)

        assert client_a is client_b, (
            "Same provider and model should return the same cached client"
        )

    def test_resolve_called_once_per_model(self):
        """resolve_provider_client should be called once per unique
        (provider, model), not on every invocation."""
        from agent.auxiliary_client import _get_cached_client

        with patch("agent.auxiliary_client.resolve_provider_client",
                   side_effect=_stub_resolve_provider_client) as mock_resolve:
            _get_cached_client("openrouter", "model-a", async_mode=False)
            _get_cached_client("openrouter", "model-a", async_mode=False)
            _get_cached_client("openrouter", "model-b", async_mode=False)

        assert mock_resolve.call_count == 2, (
            "Expected 2 calls to resolve_provider_client (one per unique model), "
            f"got {mock_resolve.call_count}"
        )
