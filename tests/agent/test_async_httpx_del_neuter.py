"""Tests for the AsyncHttpxClientWrapper.__del__ neuter fix.

The OpenAI SDK's ``AsyncHttpxClientWrapper.__del__`` schedules
``aclose()`` via ``asyncio.get_running_loop().create_task()``.  When GC
fires during CLI idle time, prompt_toolkit's event loop picks up the task
and crashes with "Event loop is closed" because the underlying TCP
transport is bound to a dead worker loop.

The three-layer defence:
1. ``neuter_async_httpx_del()`` replaces ``__del__`` with a no-op.
2. A custom asyncio exception handler silences residual errors.
3. ``cleanup_stale_async_clients()`` evicts stale cache entries.
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Layer 1: neuter_async_httpx_del
# ---------------------------------------------------------------------------

class TestNeuterAsyncHttpxDel:
    """Verify neuter_async_httpx_del replaces __del__ on the SDK class."""

    def test_del_becomes_noop(self):
        """After neuter, __del__ should do nothing (no RuntimeError)."""
        from agent.auxiliary_client import neuter_async_httpx_del

        try:
            from openai._base_client import AsyncHttpxClientWrapper
        except ImportError:
            pytest.skip("openai SDK not installed")

        # Save original so we can restore
        original_del = AsyncHttpxClientWrapper.__del__
        try:
            neuter_async_httpx_del()
            # The patched __del__ should be a no-op lambda
            assert AsyncHttpxClientWrapper.__del__ is not original_del
            # Calling it should not raise, even without a running loop
            wrapper = MagicMock(spec=AsyncHttpxClientWrapper)
            AsyncHttpxClientWrapper.__del__(wrapper)  # Should be silent
        finally:
            # Restore original to avoid leaking into other tests
            AsyncHttpxClientWrapper.__del__ = original_del




# ---------------------------------------------------------------------------
# Layer 3: cleanup_stale_async_clients
# ---------------------------------------------------------------------------

class TestCleanupStaleAsyncClients:
    """Verify stale cache entries are evicted and force-closed."""

    def test_removes_stale_entries(self):
        """Entries with a closed loop should be evicted."""
        from agent.auxiliary_client import (
            _client_cache,
            _client_cache_key,
            _client_cache_lock,
            cleanup_stale_async_clients,
        )

        # Create a loop, close it, make a cache entry
        loop = asyncio.new_event_loop()
        loop.close()

        mock_client = MagicMock()
        # Give it _client attribute for _force_close_async_httpx
        mock_client._client = MagicMock()
        mock_client._client.is_closed = False

        key = _client_cache_key("test_stale", async_mode=True)
        with _client_cache_lock:
            _client_cache[key] = (mock_client, "test-model", loop)

        try:
            cleanup_stale_async_clients()
            mock_client.close.assert_called_once()
            with _client_cache_lock:
                assert key not in _client_cache, "Stale entry should be removed"
        finally:
            # Clean up in case test fails
            with _client_cache_lock:
                _client_cache.pop(key, None)

    def test_awaits_async_close_for_closed_loop(self):
        from agent.auxiliary_client import (
            _client_cache,
            _client_cache_key,
            _client_cache_lock,
            cleanup_stale_async_clients,
        )

        class AsyncClient:
            def __init__(self):
                self._client = MagicMock()
                self._client.is_closed = False
                self.closed = False

            async def close(self):
                self.closed = True

        loop = asyncio.new_event_loop()
        loop.close()
        client = AsyncClient()
        key = _client_cache_key("test_async_close", async_mode=True)
        with _client_cache_lock:
            _client_cache[key] = (client, "test-model", loop)

        try:
            cleanup_stale_async_clients()
            assert client.closed
        finally:
            with _client_cache_lock:
                _client_cache.pop(key, None)


    def test_shutdown_closes_outside_cache_lock(self):
        from agent.auxiliary_client import (
            _client_cache,
            _client_cache_key,
            _client_cache_lock,
            shutdown_cached_clients,
        )

        lock_observations = []

        class Client:
            _client = None

            def close(self):
                acquired = _client_cache_lock.acquire(blocking=False)
                lock_observations.append(acquired)
                if acquired:
                    _client_cache_lock.release()

        key = _client_cache_key("test_shutdown_lock", async_mode=False)
        with _client_cache_lock:
            previous = dict(_client_cache)
            _client_cache.clear()
            _client_cache[key] = (Client(), "test-model", None)

        try:
            shutdown_cached_clients()
        finally:
            with _client_cache_lock:
                _client_cache.clear()
                _client_cache.update(previous)

        assert lock_observations == [True]

    def test_shutdown_does_not_await_live_foreign_loop_client(self):
        from agent.auxiliary_client import (
            _client_cache,
            _client_cache_key,
            _client_cache_lock,
            shutdown_cached_clients,
        )

        owner_loop = asyncio.new_event_loop()

        class Client:
            def __init__(self):
                self.awaited = False

            async def close(self):
                self.awaited = True

        client = Client()
        key = _client_cache_key("test_shutdown_foreign_loop", async_mode=True)
        with _client_cache_lock:
            previous = dict(_client_cache)
            _client_cache.clear()
            _client_cache[key] = (client, "test-model", owner_loop)

        try:
            shutdown_cached_clients()
            assert client.awaited is False
        finally:
            owner_loop.close()
            with _client_cache_lock:
                _client_cache.clear()
                _client_cache.update(previous)

    def test_keeps_live_entries(self):
        """Entries with an open loop should be preserved."""
        from agent.auxiliary_client import (
            _client_cache,
            _client_cache_key,
            _client_cache_lock,
            cleanup_stale_async_clients,
        )

        loop = asyncio.new_event_loop()  # NOT closed

        mock_client = MagicMock()
        key = _client_cache_key("test_live", async_mode=True)
        with _client_cache_lock:
            _client_cache[key] = (mock_client, "test-model", loop)

        try:
            cleanup_stale_async_clients()
            with _client_cache_lock:
                assert key in _client_cache, "Live entry should be preserved"
        finally:
            loop.close()
            with _client_cache_lock:
                _client_cache.pop(key, None)

    def test_keeps_entries_without_loop(self):
        """Sync entries (cached_loop=None) should be preserved."""
        from agent.auxiliary_client import (
            _client_cache,
            _client_cache_key,
            _client_cache_lock,
            cleanup_stale_async_clients,
        )

        mock_client = MagicMock()
        key = _client_cache_key("test_sync", async_mode=False)
        with _client_cache_lock:
            _client_cache[key] = (mock_client, "test-model", None)

        try:
            cleanup_stale_async_clients()
            with _client_cache_lock:
                assert key in _client_cache, "Sync entry should be preserved"
        finally:
            with _client_cache_lock:
                _client_cache.pop(key, None)


# ---------------------------------------------------------------------------
# Cache bounded growth (#10200)
# ---------------------------------------------------------------------------

class TestClientCacheBoundedGrowth:
    """Verify the cache stays bounded when loops change (fix for #10200).

    Previously, loop_id was part of the cache key, so every new event loop
    created a new entry for the same provider config.  Now loop identity is
    validated at hit time and stale entries are replaced in-place.
    """

    def test_same_key_replaces_stale_loop_entry(self):
        """When the loop changes, the old entry should be replaced, not duplicated."""
        from agent.auxiliary_client import (
            _client_cache,
            _client_cache_key,
            _client_cache_lock,
            _get_cached_client,
        )

        key = _client_cache_key(
            "test_replace",
            async_mode=True,
            task="",
        )

        # Simulate a stale entry from a closed loop
        old_loop = asyncio.new_event_loop()
        old_loop.close()
        old_client = MagicMock()
        old_client._client = MagicMock()
        old_client._client.is_closed = False

        with _client_cache_lock:
            _client_cache[key] = (old_client, "old-model", old_loop)

        try:
            # Now call _get_cached_client — should detect stale loop and evict
            with patch("agent.auxiliary_client.resolve_provider_client") as mock_resolve:
                mock_resolve.return_value = (MagicMock(), "new-model")
                client, model = _get_cached_client(
                    "test_replace", async_mode=True,
                )
            # The old entry should have been replaced
            with _client_cache_lock:
                assert key in _client_cache, "Key should still exist (replaced)"
                entry = _client_cache[key]
                assert entry[1] == "new-model", "Should have the new model"
        finally:
            with _client_cache_lock:
                _client_cache.pop(key, None)


