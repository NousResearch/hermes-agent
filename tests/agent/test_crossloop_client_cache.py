"""Tests for cross-loop client cache isolation fix (#2681).

Verifies that _get_cached_client() returns different AsyncOpenAI clients
when called from different event loops, preventing the httpx deadlock
that occurs when a cached async client bound to loop A is reused on loop B.

This test file is self-contained and does not import the full tool chain,
so it can run without optional dependencies like firecrawl.
"""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch, MagicMock
from types import SimpleNamespace

import pytest


# ---------------------------------------------------------------------------
# Minimal stubs so we can import _get_cached_client without the full tree
# ---------------------------------------------------------------------------

def _stub_resolve_provider_client(provider, model, async_mode, **kw):
    """Return a unique mock client each time, simulating AsyncOpenAI creation."""
    client = MagicMock(name=f"client-{provider}-async={async_mode}")
    client.api_key = "test"
    client.base_url = kw.get("explicit_base_url", "http://localhost:8081/v1")
    return client, model or "test-model"


@pytest.fixture(autouse=True)
def _clean_client_cache():
    """Clear the client cache before each test."""
    import importlib
    # We need to patch before importing
    with patch.dict("sys.modules", {}):
        pass
    # Import and clear
    import agent.auxiliary_client as ac
    ac._client_cache.clear()
    yield
    ac._client_cache.clear()


class TestCrossLoopCacheIsolation:
    """Verify async clients are cached per-event-loop, not globally."""

    def test_same_loop_reuses_client(self):
        """Within a single event loop, the same client should be returned."""
        from agent.auxiliary_client import _get_cached_client, _client_cache

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        with patch("agent.auxiliary_client.resolve_provider_client",
                    side_effect=_stub_resolve_provider_client):
            client1, _ = _get_cached_client("custom", "m1", async_mode=True,
                                             base_url="http://localhost:8081/v1")
            client2, _ = _get_cached_client("custom", "m1", async_mode=True,
                                             base_url="http://localhost:8081/v1")

        assert client1 is client2, (
            "Same loop should return the same cached client"
        )
        loop.close()

    def test_different_loops_get_different_clients(self):
        """Different event loops must get separate client instances."""
        from agent.auxiliary_client import _get_cached_client

        results = {}

        def _get_client_on_new_loop(name):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            with patch("agent.auxiliary_client.resolve_provider_client",
                        side_effect=_stub_resolve_provider_client):
                client, _ = _get_cached_client("custom", "m1", async_mode=True,
                                                 base_url="http://localhost:8081/v1")
            results[name] = (id(client), id(loop))
            # Don't close loop — simulates real usage where loops persist

        t1 = threading.Thread(target=_get_client_on_new_loop, args=("a",))
        t2 = threading.Thread(target=_get_client_on_new_loop, args=("b",))
        t1.start(); t1.join()
        t2.start(); t2.join()

        client_id_a, loop_id_a = results["a"]
        client_id_b, loop_id_b = results["b"]

        assert loop_id_a != loop_id_b, "Test setup error: same loop on both threads"
        assert client_id_a != client_id_b, (
            "Different event loops got the SAME cached client — this causes "
            "httpx cross-loop deadlocks in gateway mode (#2681)"
        )

    def test_sync_clients_not_affected(self):
        """Sync clients (async_mode=False) should still be cached globally,
        since httpx.Client (sync) doesn't bind to an event loop."""
        from agent.auxiliary_client import _get_cached_client

        results = {}

        def _get_sync_client(name):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            with patch("agent.auxiliary_client.resolve_provider_client",
                        side_effect=_stub_resolve_provider_client):
                client, _ = _get_cached_client("custom", "m1", async_mode=False,
                                                 base_url="http://localhost:8081/v1")
            results[name] = id(client)

        t1 = threading.Thread(target=_get_sync_client, args=("a",))
        t2 = threading.Thread(target=_get_sync_client, args=("b",))
        t1.start(); t1.join()
        t2.start(); t2.join()

        assert results["a"] == results["b"], (
            "Sync clients should be shared across threads (no loop binding)"
        )

    def test_gateway_simulation_no_deadlock(self):
        """Simulate gateway mode: _run_async spawns a thread with asyncio.run(),
        which creates a new loop. The cached client must be created on THAT loop,
        not reused from a different one."""
        from agent.auxiliary_client import _get_cached_client

        # Simulate: first call on "gateway loop"
        gateway_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(gateway_loop)

        with patch("agent.auxiliary_client.resolve_provider_client",
                    side_effect=_stub_resolve_provider_client):
            gateway_client, _ = _get_cached_client("custom", "m1", async_mode=True,
                                                     base_url="http://localhost:8081/v1")

        # Simulate: _run_async spawns a thread with asyncio.run()
        worker_client_id = [None]
        def _worker():
            async def _inner():
                with patch("agent.auxiliary_client.resolve_provider_client",
                            side_effect=_stub_resolve_provider_client):
                    client, _ = _get_cached_client("custom", "m1", async_mode=True,
                                                     base_url="http://localhost:8081/v1")
                worker_client_id[0] = id(client)
            asyncio.run(_inner())

        t = threading.Thread(target=_worker)
        t.start()
        t.join()

        assert worker_client_id[0] != id(gateway_client), (
            "Worker thread (asyncio.run) got the gateway's cached client — "
            "this is the exact cross-loop scenario that causes httpx deadlocks. "
            "The cache key must include the event loop identity (#2681)"
        )
        gateway_loop.close()

    def test_closed_loop_client_discarded(self):
        """A cached client whose loop has closed should be replaced."""
        from agent.auxiliary_client import _get_cached_client

        loop1 = asyncio.new_event_loop()
        asyncio.set_event_loop(loop1)

        with patch("agent.auxiliary_client.resolve_provider_client",
                    side_effect=_stub_resolve_provider_client):
            client1, _ = _get_cached_client("custom", "m1", async_mode=True,
                                             base_url="http://localhost:8081/v1")

        loop1.close()

        # New loop on same thread
        loop2 = asyncio.new_event_loop()
        asyncio.set_event_loop(loop2)

        with patch("agent.auxiliary_client.resolve_provider_client",
                    side_effect=_stub_resolve_provider_client):
            client2, _ = _get_cached_client("custom", "m1", async_mode=True,
                                             base_url="http://localhost:8081/v1")

        assert client1 is not client2, (
            "Client from closed loop should not be reused"
        )
        loop2.close()
