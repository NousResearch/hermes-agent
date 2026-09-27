"""
Integration tests for the desktop boot handshake fix (PR #50231 / issue #50209).

Simulates a slow hermes_cli.gateway import (15-30 s on a fresh Windows install
with Defender scanning every new .pyc) by patching the two helpers that touch
the blocking import and measuring response latency.

Covered: the gateway warmup import completes before the lifespan yields
(#73083/#73291), backend startup is not blocked by hosted-room recovery, shutdown joins the
state.db reconcile worker, and /api/status runs its slow drain-timeout resolution off
the event loop so a concurrent fast endpoint (/api/version) still responds.
"""

from __future__ import annotations

import asyncio
import time
import threading
from unittest.mock import patch

import hermes_cli.web_server as web_server_mod
import hermes_cli.web_server_lifecycle as _web_server_lifecycle

SLOW_SECONDS = 1  # represents the Defender worst-case (scaled down for CI speed)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_slow_drain(seconds: float):
    """Return a _resolve_restart_drain_timeout replacement that sleeps in thread."""
    def _slow():
        time.sleep(seconds)
        return 180.0
    return _slow


def test_lifespan_warmup_is_synchronous(monkeypatch):
    """_warm_gateway_module must finish on the event-loop thread before the
    lifespan yields (#73083/#73291). On Windows + Python 3.11 the heavy import
    holds the GIL, so moving it to run_in_executor / a background task froze
    the loop after the socket opened and the Desktop's ready-probe timed out.
    Running it before the yield means no request is served until it is done."""
    from fastapi.testclient import TestClient

    warm: dict[str, object] = {}

    def _record_warm():
        try:
            asyncio.get_running_loop()
            warm["on_loop_thread"] = True
        except RuntimeError:
            warm["on_loop_thread"] = False
        warm["done"] = True

    monkeypatch.setattr(web_server_mod, "_warm_gateway_module", _record_warm)
    with TestClient(web_server_mod.app, raise_server_exceptions=False):
        assert warm.get("done") is True, "startup completed before the gateway warmup ran"
        assert warm.get("on_loop_thread") is True, (
            "gateway warmup was moved off the lifespan (executor/background) — "
            "the socket now accepts probes while the GIL-holding import runs"
        )


def test_hosted_room_recovery_cannot_block_or_abort_backend_startup(monkeypatch):
    from fastapi.testclient import TestClient
    from tui_gateway import methods_groups

    started = threading.Event()
    release = threading.Event()

    def blocked_failure():
        started.set()
        release.wait(timeout=2.0)
        raise RuntimeError("state.db is locked")

    monkeypatch.setattr(web_server_mod, "_warm_gateway_module", lambda: None)
    monkeypatch.setattr(methods_groups, "stop_hosted_room_service", lambda **_kwargs: True)
    # Warm lifespan imports first so the bound below measures the recovery hook, not cold imports.
    monkeypatch.setattr(methods_groups, "start_hosted_room_service", lambda *a, **k: None)
    with TestClient(web_server_mod.app, raise_server_exceptions=False):
        pass
    monkeypatch.setattr(methods_groups, "start_hosted_room_service", blocked_failure)

    before = time.perf_counter()
    with TestClient(web_server_mod.app, raise_server_exceptions=False):
        assert started.wait(timeout=1.0)
        assert time.perf_counter() - before < 1.0
        release.set()

def test_lifespan_shutdown_joins_statedb_reconcile_worker(monkeypatch):
    """The eager state.db reconcile runs off the startup path but never outlives
    the lifespan: shutdown joins it, so its sqlite connection is only ever closed
    by the thread stepping it (a daemon copy left running had its connection
    closed cross-thread by teardown and segfaulted the interpreter)."""
    from fastapi.testclient import TestClient

    started = threading.Event()
    finished = threading.Event()

    def slow_reconcile():
        started.set()
        time.sleep(SLOW_SECONDS)
        finished.set()

    monkeypatch.setattr(web_server_mod, "_warm_gateway_module", lambda: None)
    monkeypatch.setattr(web_server_mod, "_eager_reconcile_own_session_db", slow_reconcile)

    before = time.perf_counter()
    with TestClient(web_server_mod.app, raise_server_exceptions=False):
        assert started.wait(timeout=1.0)
        # Off the startup path: the socket is up long before the worker is done.
        assert time.perf_counter() - before < SLOW_SECONDS * 0.8

    assert finished.is_set(), "lifespan shutdown returned before the reconcile worker finished"
    assert not any(t.name == "statedb-eager-reconcile" for t in threading.enumerate())

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
        _web_server_lifecycle, "_resolve_restart_drain_timeout", _make_slow_drain(SLOW_SECONDS)
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
