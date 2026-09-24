"""Regression coverage for provider-scoped auxiliary cache eviction (#113022)."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import agent.auxiliary_client as aux
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


def _cache_entry() -> tuple[MagicMock, str, None]:
    return (MagicMock(), "test-model", None)


def test_evict_cached_clients_matches_provider_after_profile_home_key(monkeypatch):
    """OAuth rotation must evict both sync and async entries for its provider."""
    anthropic_sync = _cache_entry()
    anthropic_async = _cache_entry()
    other_provider = _cache_entry()
    sync_key = aux._client_cache_key("anthropic", async_mode=False)
    async_key = aux._client_cache_key("anthropic", async_mode=True)
    other_key = aux._client_cache_key("openai-codex", async_mode=False)
    monkeypatch.setattr(aux, "_client_cache", {
        sync_key: anthropic_sync,
        async_key: anthropic_async,
        other_key: other_provider,
    })

    aux._evict_cached_clients("anthropic")

    assert sync_key not in aux._client_cache
    assert async_key not in aux._client_cache
    assert other_key in aux._client_cache
    other_provider[0].close.assert_not_called()


def test_evict_cached_clients_is_scoped_to_the_calling_profile(monkeypatch, tmp_path):
    """A rotation in profile A must not drop profile B's client for the same provider."""
    key_a = aux._client_cache_key("anthropic", async_mode=False)
    token = set_hermes_home_override(tmp_path / "profiles" / "b")
    try:
        key_b = aux._client_cache_key("anthropic", async_mode=False)
    finally:
        reset_hermes_home_override(token)
    assert key_a[0] != key_b[0]
    monkeypatch.setattr(aux, "_client_cache", {key_a: _cache_entry(), key_b: _cache_entry()})

    aux._evict_cached_clients("anthropic")

    assert key_a not in aux._client_cache
    assert key_b in aux._client_cache


def test_evict_cached_clients_does_not_close_possibly_in_flight_client(monkeypatch):
    """Eviction pops the entry; a concurrent caller mid-request must not get a closed client."""
    entry = _cache_entry()
    key = aux._client_cache_key("anthropic", async_mode=False)
    monkeypatch.setattr(aux, "_client_cache", {key: entry})

    aux._evict_cached_clients("anthropic")

    assert key not in aux._client_cache
    entry[0].close.assert_not_called()


def test_pool_rotation_evicts_client_built_with_revoked_credential(monkeypatch):
    """A 401 pool rotation drops the old provider client before retry/fallback."""
    stale_entry = _cache_entry()
    stale_key = aux._client_cache_key("anthropic", async_mode=False)
    monkeypatch.setattr(aux, "_client_cache", {stale_key: stale_entry})

    pool = MagicMock()
    pool.has_credentials.return_value = True
    pool.try_refresh_current.return_value = None
    pool.mark_exhausted_and_rotate.return_value = SimpleNamespace(id="fresh-oauth-entry")
    auth_error = Exception("revoked OAuth token")
    auth_error.status_code = 401

    with patch("agent.auxiliary_client.load_pool", return_value=pool):
        assert aux._recover_provider_pool("anthropic", auth_error, failed_api_key="revoked-token") is True

    assert stale_key not in aux._client_cache
    pool.mark_exhausted_and_rotate.assert_called_once()
