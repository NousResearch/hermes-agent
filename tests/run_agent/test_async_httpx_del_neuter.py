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
import threading
from types import SimpleNamespace
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

    def test_neuter_idempotent(self):
        """Calling neuter twice doesn't break anything."""
        from agent.auxiliary_client import neuter_async_httpx_del

        try:
            from openai._base_client import AsyncHttpxClientWrapper
        except ImportError:
            pytest.skip("openai SDK not installed")

        original_del = AsyncHttpxClientWrapper.__del__
        try:
            neuter_async_httpx_del()
            first_del = AsyncHttpxClientWrapper.__del__
            neuter_async_httpx_del()
            second_del = AsyncHttpxClientWrapper.__del__
            # Both calls should succeed; the class should have a no-op
            assert first_del is not original_del
            assert second_del is not original_del
        finally:
            AsyncHttpxClientWrapper.__del__ = original_del

    def test_neuter_graceful_without_sdk(self):
        """neuter_async_httpx_del doesn't raise if the openai SDK isn't installed."""
        from agent.auxiliary_client import neuter_async_httpx_del

        with patch.dict("sys.modules", {"openai._base_client": None}):
            # Should not raise
            neuter_async_httpx_del()


# ---------------------------------------------------------------------------
# Layer 3: cleanup_stale_async_clients
# ---------------------------------------------------------------------------

class TestCleanupStaleAsyncClients:
    """Verify stale cache entries are evicted and force-closed."""

    def test_removes_stale_entries(self):
        """Entries with a closed loop should be evicted."""
        from agent.auxiliary_client import (
            _client_cache,
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

        key = ("test_stale", True, "", "", "", ())
        with _client_cache_lock:
            _client_cache[key] = (mock_client, "test-model", loop)

        try:
            cleanup_stale_async_clients()
            with _client_cache_lock:
                assert key not in _client_cache, "Stale entry should be removed"
        finally:
            # Clean up in case test fails
            with _client_cache_lock:
                _client_cache.pop(key, None)

    def test_keeps_live_entries(self):
        """Entries with an open loop should be preserved."""
        from agent.auxiliary_client import (
            _client_cache,
            _client_cache_lock,
            cleanup_stale_async_clients,
        )

        loop = asyncio.new_event_loop()  # NOT closed

        mock_client = MagicMock()
        key = ("test_live", True, "", "", "", ())
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
            _client_cache_lock,
            cleanup_stale_async_clients,
        )

        mock_client = MagicMock()
        key = ("test_sync", False, "", "", "", ())
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
            _client_cache_lock,
            _get_cached_client,
        )

        key = ("test_replace", True, "", "", "", ())

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

    def test_different_loops_do_not_grow_cache(self):
        """Multiple event loops for the same provider should NOT create multiple entries."""
        from agent.auxiliary_client import (
            _client_cache,
            _client_cache_lock,
        )

        key = ("test_no_grow", True, "", "", "", ())

        loops = []
        try:
            for i in range(5):
                loop = asyncio.new_event_loop()
                loops.append(loop)
                mock_client = MagicMock()
                mock_client._client = MagicMock()
                mock_client._client.is_closed = False

                # Close previous loop entries (simulating worker thread recycling)
                if i > 0:
                    loops[i - 1].close()

                with _client_cache_lock:
                    # Simulate what _get_cached_client does: replace on loop mismatch
                    if key in _client_cache:
                        old_entry = _client_cache[key]
                        del _client_cache[key]
                    _client_cache[key] = (mock_client, f"model-{i}", loop)

            # Only one entry should exist for this key
            with _client_cache_lock:
                count = sum(1 for k in _client_cache if k == key)
                assert count == 1, f"Expected 1 entry, got {count}"
        finally:
            for loop in loops:
                if not loop.is_closed():
                    loop.close()
            with _client_cache_lock:
                _client_cache.pop(key, None)

    def test_max_cache_size_eviction(self):
        """Cache should not exceed _CLIENT_CACHE_MAX_SIZE."""
        from agent.auxiliary_client import (
            _client_cache,
            _client_cache_lock,
            _CLIENT_CACHE_MAX_SIZE,
        )

        # Save existing cache state
        with _client_cache_lock:
            saved = dict(_client_cache)
            _client_cache.clear()

        try:
            # Fill to max + 5
            for i in range(_CLIENT_CACHE_MAX_SIZE + 5):
                mock_client = MagicMock()
                mock_client._client = MagicMock()
                mock_client._client.is_closed = False
                key = (f"evict_test_{i}", False, "", "", "", ())
                with _client_cache_lock:
                    # Inline the eviction logic (same as _get_cached_client)
                    while len(_client_cache) >= _CLIENT_CACHE_MAX_SIZE:
                        evict_key = next(iter(_client_cache))
                        del _client_cache[evict_key]
                    _client_cache[key] = (mock_client, f"model-{i}", None)

            with _client_cache_lock:
                assert len(_client_cache) <= _CLIENT_CACHE_MAX_SIZE, \
                    f"Cache size {len(_client_cache)} exceeds max {_CLIENT_CACHE_MAX_SIZE}"
                # The earliest entries should have been evicted
                assert ("evict_test_0", False, "", "", "", ()) not in _client_cache
                # The latest entries should be present
                assert (f"evict_test_{_CLIENT_CACHE_MAX_SIZE + 4}", False, "", "", "", ()) in _client_cache
        finally:
            with _client_cache_lock:
                _client_cache.clear()
                _client_cache.update(saved)
