"""Tests for _store_cached_client async-close guard in agent.auxiliary_client."""
import pytest


def test_store_cached_client_skips_async_close():
    """Cache eviction must not call async close() synchronously.

    When the evicted client's close() is a coroutine function, calling it
    from the synchronous _store_cached_client path creates an unawaited
    coroutine warning.  The guard must skip async close() and let
    _force_close_async_httpx handle the transport teardown.
    """
    import warnings
    from agent.auxiliary_client import _store_cached_client, _client_cache

    class AsyncCloseClient:
        async def close(self):
            return None

    class ReplacementClient:
        pass

    cache_key = ("test_async_close", "model")
    old_client = AsyncCloseClient()
    _client_cache[cache_key] = (old_client, "model", None)

    new_client = ReplacementClient()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _store_cached_client(cache_key, new_client, "model")

    runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    coroutine_warnings = [w for w in runtime_warnings if "coroutine" in str(w.message).lower()]
    assert coroutine_warnings == [], (
        f"Unawaited coroutine warnings: {coroutine_warnings}"
    )
    assert _client_cache[cache_key][0] is new_client


def test_store_cached_client_calls_sync_close():
    """Sync close() must still be called during cache eviction."""
    from agent.auxiliary_client import _store_cached_client, _client_cache

    close_called = []

    class SyncCloseClient:
        def close(self):
            close_called.append(True)

    class ReplacementClient:
        pass

    cache_key = ("test_sync_close", "model")
    old_client = SyncCloseClient()
    _client_cache[cache_key] = (old_client, "model", None)

    new_client = ReplacementClient()
    _store_cached_client(cache_key, new_client, "model")

    assert close_called, "Sync close() should have been called"
    assert _client_cache[cache_key][0] is new_client
