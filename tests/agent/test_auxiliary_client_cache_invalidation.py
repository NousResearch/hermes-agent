"""Regression test for #21013: auxiliary client cache not cleared on .env hot-reload.

Verifies that shutdown_cached_clients() clears the _client_cache so that
subsequent calls create fresh clients with updated API keys.
"""

import threading
from unittest.mock import MagicMock

import pytest

from agent.auxiliary_client import (
    _client_cache,
    _client_cache_lock,
    shutdown_cached_clients,
    _store_cached_client,
    _client_cache_key,
)


class TestShutdownCachedClients:
    """shutdown_cached_clients should clear _client_cache."""

    def setup_method(self):
        """Ensure cache starts empty."""
        with _client_cache_lock:
            _client_cache.clear()

    def test_clears_cache(self):
        """Cache is empty after shutdown_cached_clients()."""
        # Populate cache with a fake entry
        key = _client_cache_key(
            "test_provider",
            async_mode=False,
            base_url="https://example.com",
            api_key="old-key",
        )
        mock_client = MagicMock()
        mock_client.close = MagicMock()
        _store_cached_client(key, mock_client, "test-model")

        with _client_cache_lock:
            assert len(_client_cache) == 1

        # Act
        shutdown_cached_clients()

        # Assert
        with _client_cache_lock:
            assert len(_client_cache) == 0

    def test_closes_sync_clients(self):
        """Sync clients should have .close() called during shutdown."""
        key = _client_cache_key(
            "test_provider",
            async_mode=False,
            base_url="https://example.com",
            api_key="old-key",
        )
        mock_client = MagicMock()
        _store_cached_client(key, mock_client, "test-model")

        shutdown_cached_clients()

        mock_client.close.assert_called_once()

    def test_cache_stays_empty_after_shutdown(self):
        """After shutdown, new entries can be stored but old ones are gone."""
        key = _client_cache_key(
            "test_provider",
            async_mode=False,
            base_url="https://example.com",
            api_key="old-key",
        )
        _store_cached_client(key, MagicMock(), "old-model")

        shutdown_cached_clients()

        with _client_cache_lock:
            assert key not in _client_cache

        # New entry with new key works
        new_key = _client_cache_key(
            "test_provider",
            async_mode=False,
            base_url="https://example.com",
            api_key="new-key",
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
            api_key="old-key-123",
        )
        key_new = _client_cache_key(
            "custom",
            async_mode=False,
            base_url="https://api.example.com",
            api_key="new-key-456",
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
