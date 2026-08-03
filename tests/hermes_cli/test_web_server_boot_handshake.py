"""
Integration tests for the desktop boot handshake fix (PR #50231 / issue #50209).

Simulates a slow hermes_cli.gateway import (15-30 s on a fresh Windows install
with Defender scanning every new .pyc) by patching the two helpers that touch
the blocking import and measuring event-loop freedom + response latency.

Three scenarios are covered:

1. _lifespan fire-and-forget: patched _warm_gateway_module sleeps N seconds in
   a thread; TestClient startup must complete in << N seconds (event loop not
   blocked, HERMES_DASHBOARD_READY would fire immediately).

2. get_status run_in_executor: patched _resolve_restart_drain_timeout sleeps N
   seconds in a thread; a concurrent fast endpoint (/api/version) must respond
   during the wait, proving the event loop stayed free.

3. No orphan accumulation: three concurrent /api/status requests all receive a
   200 response — no socket timeouts, no connection resets.
"""

from __future__ import annotations

import asyncio
import time
import threading
from unittest.mock import patch

import pytest

import hermes_cli.web_server as web_server_mod

SLOW_SECONDS = 3  # represents the Defender worst-case (scaled down for CI speed)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_slow_warm(seconds: float):
    """Return a _warm_gateway_module replacement that sleeps in the caller thread."""
    def _slow():
        time.sleep(seconds)
    return _slow


def _make_slow_drain(seconds: float):
    """Return a _resolve_restart_drain_timeout replacement that sleeps in thread."""
    def _slow():
        time.sleep(seconds)
        return 180.0
    return _slow


# ---------------------------------------------------------------------------
# Test 1 — _lifespan fire-and-forget does not block the event loop
# ---------------------------------------------------------------------------

def test_lifespan_warmup_is_nonblocking():
    """
    _warm_gateway_module runs in an executor (fire-and-forget).
    Even if it sleeps for SLOW_SECONDS, TestClient startup must complete
    in well under that time — proving the event loop was never blocked and
    HERMES_DASHBOARD_READY would have fired without delay.
    """
    from fastapi.testclient import TestClient

    with patch.object(web_server_mod, "_warm_gateway_module", _make_slow_warm(SLOW_SECONDS)):
        t0 = time.perf_counter()
        with TestClient(web_server_mod.app, raise_server_exceptions=False) as _client:
            startup_ms = (time.perf_counter() - t0) * 1000

    # Startup must complete in under half of SLOW_SECONDS (generous margin).
    # If the import were synchronous, startup would block for >= SLOW_SECONDS.
    threshold_ms = (SLOW_SECONDS * 1000) / 2
    assert startup_ms < threshold_ms, (
        f"_lifespan blocked the event loop: startup took {startup_ms:.0f} ms "
        f"but slow import is {SLOW_SECONDS * 1000:.0f} ms — "
        f"fire-and-forget is not working."
    )


# ---------------------------------------------------------------------------
# Test 2 — get_status run_in_executor keeps event loop free for other requests
# ---------------------------------------------------------------------------

def test_get_status_does_not_block_event_loop():
    """
    /api/status calls _resolve_restart_drain_timeout via run_in_executor.
    While that slow call is running in a thread, a concurrent fast request
    (/api/version) must still get a response — proving the event loop stayed
    free during the import.
    """
    import httpx
    from anyio import from_thread, to_thread

    results: dict[str, float] = {}
    errors: list[str] = []

    async def _run():
        transport = httpx.ASGITransport(app=web_server_mod.app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            # Fire both requests concurrently
            async with asyncio.TaskGroup() as tg:
                async def _status():
                    t = time.perf_counter()
                    r = await client.get("/api/status", timeout=SLOW_SECONDS + 5)
                    results["status_ms"] = (time.perf_counter() - t) * 1000
                    results["status_code"] = r.status_code

                async def _version():
                    # Small delay so /api/status starts first
                    await asyncio.sleep(0.1)
                    t = time.perf_counter()
                    r = await client.get("/api/version", timeout=5)
                    results["version_ms"] = (time.perf_counter() - t) * 1000
                    results["version_code"] = r.status_code

                tg.create_task(_status())
                tg.create_task(_version())

    with patch.object(
        web_server_mod, "_resolve_restart_drain_timeout", _make_slow_drain(SLOW_SECONDS)
    ):
        asyncio.run(_run())

    # /api/version must have responded well before /api/status finished
    assert "version_ms" in results, "Fast endpoint never responded"
    assert "status_ms" in results, "/api/status never responded"

    version_ms = results["version_ms"]
    status_ms = results["status_ms"]

    # /api/version should respond in < SLOW_SECONDS (event loop free)
    assert version_ms < SLOW_SECONDS * 1000, (
        f"/api/version took {version_ms:.0f} ms — event loop was blocked by "
        f"/api/status (which waited {status_ms:.0f} ms for the slow import)."
    )

    # /api/status itself eventually returns 200
    assert results.get("status_code") == 200, (
        f"/api/status returned {results.get('status_code')} instead of 200"
    )


# ---------------------------------------------------------------------------
# Test 3 — no orphan accumulation: concurrent probes all receive 200
# ---------------------------------------------------------------------------

def test_concurrent_status_probes_all_respond():
    """
    Three concurrent /api/status requests must all receive HTTP 200.
    If the event loop were blocked, later requests would pile up and
    the desktop shell would eventually reset the connection (WinError 10054).
    """
    import httpx

    PROBES = 3
    responses: list[int] = []

    async def _run():
        transport = httpx.ASGITransport(app=web_server_mod.app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            tasks = [
                client.get("/api/status", timeout=SLOW_SECONDS + 5)
                for _ in range(PROBES)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for r in results:
                if isinstance(r, Exception):
                    responses.append(-1)
                else:
                    responses.append(r.status_code)

    with patch.object(
        web_server_mod, "_resolve_restart_drain_timeout", _make_slow_drain(SLOW_SECONDS)
    ):
        asyncio.run(_run())

    failed = [c for c in responses if c != 200]
    assert not failed, (
        f"{len(failed)}/{PROBES} probes failed (codes: {responses}). "
        f"This would cause WinError 10054 and orphan accumulation on desktop."
    )
