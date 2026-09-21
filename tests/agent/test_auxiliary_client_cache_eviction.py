"""Regression coverage for provider-scoped auxiliary cache eviction (#113022)."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

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
    pool.try_refresh_matching.return_value = None
    pool.mark_exhausted_and_rotate.return_value = SimpleNamespace(id="fresh-oauth-entry")
    auth_error = Exception("revoked OAuth token")
    auth_error.status_code = 401

    with patch("agent.auxiliary_client.load_pool", return_value=pool):
        assert aux._recover_provider_pool("anthropic", auth_error, failed_api_key="revoked-token") is True

    assert stale_key not in aux._client_cache
    pool.mark_exhausted_and_rotate.assert_called_once()


def test_pool_recovery_without_failed_key_does_not_exhaust_current_credential():
    """An unattributed stale-client error must not park the pool's healthy current entry."""
    pool = MagicMock()
    pool.has_credentials.return_value = True
    rate_error = Exception("rate limited")
    rate_error.status_code = 429

    with patch("agent.auxiliary_client.load_pool", return_value=pool):
        assert aux._recover_provider_pool("zai", rate_error) is False

    pool.mark_exhausted_and_rotate.assert_not_called()


def test_retry_failure_is_attributed_to_rebuilt_client_not_pool_current():
    """A stale A failure followed by a B retry failure attributes each request to its own key."""
    stale_client = SimpleNamespace(api_key="stale-key-a", base_url="https://api.z.ai/v1")
    retry_client = SimpleNamespace(api_key="healthy-key-b")
    first_error = Exception("payment required")
    first_error.status_code = 402
    rate_error = Exception("rate limited")
    rate_error.status_code = 429
    route = aux._LadderRoute(
        client=stale_client, task="approval", tag="", async_mode=False,
        base_info="https://api.z.ai/v1", resolved_provider="zai", resolved_model="glm",
        resolved_base_url=None, resolved_api_key=None, resolved_api_mode="chat_completions",
        final_model="glm", main_runtime=None, route_info=None,
    )

    with (
        patch("agent.auxiliary_client._recover_provider_pool", return_value=True) as recover,
        patch("agent.auxiliary_client._prepare_same_provider_retry", return_value=(retry_client, {})),
        patch("agent.auxiliary_client._relay_sync_completion", side_effect=rate_error),
    ):
        ladder = aux._ladder_credential_rungs(first_error, route, {}, False)
        assert next(ladder).kind == "retry_same_provider"
        with pytest.raises(Exception) as caught:
            aux._retry_same_provider_sync(
                resolved_provider="zai", resolved_api_mode="chat_completions", task="approval",
            )
        with pytest.raises(StopIteration):
            ladder.throw(caught.value)

    assert getattr(caught.value, aux._FAILED_API_KEY_ATTR) == "healthy-key-b"
    assert [call.kwargs["failed_api_key"] for call in recover.call_args_list] == [
        "stale-key-a", "healthy-key-b",
    ]
