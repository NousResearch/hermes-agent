"""Regression test for #21013: auxiliary client cache not cleared on .env hot-reload.

Two layers:
- ``shutdown_cached_clients`` clears ``_client_cache`` so subsequent calls create
  fresh clients with updated API keys.
- ``_reload_runtime_env_preserving_config_authority`` drops the cache only when the
  per-turn .env reload actually changed the environment (rotated keys), keeping the
  cache warm otherwise.
"""

from unittest.mock import MagicMock

from agent.auxiliary_client import (
    _client_cache,
    _client_cache_lock,
    shutdown_cached_clients,
    _store_cached_client,
    _client_cache_key,
)

# Placeholder secrets for cache-key discrimination only — never sent anywhere.
_OLD_KEY = "old-test-key-placeholder"
_NEW_KEY = "new-test-key-placeholder"


def _populate_cache(api_key: str = _OLD_KEY) -> tuple:
    """Store one mock client under the given api_key and return its cache key."""
    key = _client_cache_key(
        "test_provider",
        async_mode=False,
        base_url="https://example.com",
        api_key=api_key,
    )
    _store_cached_client(key, MagicMock(), "test-model")
    return key


class TestShutdownCachedClients:
    """shutdown_cached_clients should clear _client_cache."""

    def setup_method(self):
        """Ensure cache starts empty."""
        with _client_cache_lock:
            _client_cache.clear()

    def test_clears_cache(self):
        """Cache is empty after shutdown_cached_clients()."""
        _populate_cache()

        with _client_cache_lock:
            assert len(_client_cache) == 1

        shutdown_cached_clients()

        with _client_cache_lock:
            assert len(_client_cache) == 0

    def test_closes_sync_clients(self):
        """Sync clients should have .close() called during shutdown."""
        key = _client_cache_key(
            "test_provider",
            async_mode=False,
            base_url="https://example.com",
            api_key=_OLD_KEY,
        )
        mock_client = MagicMock()
        _store_cached_client(key, mock_client, "test-model")

        shutdown_cached_clients()

        mock_client.close.assert_called_once()

    def test_cache_stays_empty_after_shutdown(self):
        """After shutdown, new entries can be stored but old ones are gone."""
        key = _populate_cache(_OLD_KEY)

        shutdown_cached_clients()

        with _client_cache_lock:
            assert key not in _client_cache

        # New entry with new key works
        new_key = _client_cache_key(
            "test_provider",
            async_mode=False,
            base_url="https://example.com",
            api_key=_NEW_KEY,
        )
        new_client = MagicMock()
        _store_cached_client(new_key, new_client, "new-model")

        with _client_cache_lock:
            assert new_key in _client_cache
            assert _client_cache[new_key][0] is new_client


class TestCacheKeyIncludesApiKey:
    """Cache key should include api_key to distinguish rotated keys."""

    def test_different_keys_produce_different_cache_keys(self):
        key_old = _client_cache_key(
            "custom",
            async_mode=False,
            base_url="https://api.example.com",
            api_key=_OLD_KEY,
        )
        key_new = _client_cache_key(
            "custom",
            async_mode=False,
            base_url="https://api.example.com",
            api_key=_NEW_KEY,
        )
        assert key_old != key_new

    def test_empty_api_key_consistent(self):
        """When api_key is None/empty, cache key should be stable."""
        key_none = _client_cache_key(
            "custom",
            async_mode=False,
            base_url="https://api.example.com",
            api_key=None,
        )
        key_empty = _client_cache_key(
            "custom",
            async_mode=False,
            base_url="https://api.example.com",
            api_key="",
        )
        # Both should map to empty string in the key
        assert key_none == key_empty


class TestReloadInvalidatesCacheOnEnvChange:
    """The per-turn .env reload drops cached clients only when the env changed (#21013)."""

    def setup_method(self):
        with _client_cache_lock:
            _client_cache.clear()

    def teardown_method(self):
        with _client_cache_lock:
            _client_cache.clear()

    def _reload(self, monkeypatch, *, multiplex: bool, env_change: bool) -> None:
        import gateway.run as gateway_run

        monkeypatch.setattr(
            "agent.secret_scope.is_multiplex_active", lambda: multiplex)
        monkeypatch.setattr(
            gateway_run, "_bridge_max_turns_from_config", lambda *a, **k: None)

        def fake_load_dotenv(**kwargs):
            if env_change:
                monkeypatch.setenv("HERMES_TEST_ROTATED_KEY", _NEW_KEY)

        monkeypatch.setattr(gateway_run, "load_hermes_dotenv", fake_load_dotenv)
        gateway_run._reload_runtime_env_preserving_config_authority()

    def test_env_change_clears_cache(self, monkeypatch):
        """A reload that rotated keys must drop cached auxiliary clients."""
        _populate_cache()
        with _client_cache_lock:
            assert len(_client_cache) == 1

        self._reload(monkeypatch, multiplex=False, env_change=True)

        with _client_cache_lock:
            assert len(_client_cache) == 0

    def test_env_unchanged_keeps_cache(self, monkeypatch):
        """A no-op reload keeps the cache warm so per-turn turns don't rebuild clients."""
        key = _populate_cache()

        self._reload(monkeypatch, multiplex=False, env_change=False)

        with _client_cache_lock:
            assert key in _client_cache

    def test_multiplex_active_keeps_cache(self, monkeypatch):
        """Multiplex never reloads .env globally, so the cache must stay untouched."""
        key = _populate_cache()

        self._reload(monkeypatch, multiplex=True, env_change=True)

        with _client_cache_lock:
            assert key in _client_cache
