"""Tests for the Chronos cron-fire webhook (POST /api/cron/fire) — Phase 4E.2.

The webhook authenticates a NAS-minted JWT via the pluggable fire-verifier
(NOT API_SERVER_KEY), then runs the job via the resolved provider's fire_due in
the background, returning 202. These tests monkeypatch the verifier and
resolve_cron_scheduler — the verifier itself is tested with real crypto in
test_chronos_verify.py.
"""

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter, cors_middleware

_MOD = "gateway.platforms.api_server"


def _make_adapter() -> APIServerAdapter:
    return APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "sk-secret"}))


def _create_app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application(middlewares=[cors_middleware])
    app["api_server_adapter"] = adapter
    app.router.add_post("/api/cron/fire", adapter._handle_cron_fire)
    return app


@pytest.fixture
def adapter():
    return _make_adapter()


class _SpyProvider:
    """Records fire_due calls; stands in for the resolved provider."""

    def __init__(self):
        self.fired = []

    def fire_due(self, job_id, *, adapters=None, loop=None):
        self.fired.append(job_id)
        return True


@pytest.mark.asyncio
async def test_valid_fire_reservation_blocks_drain_before_body_and_task(adapter, monkeypatch):
    runner = SimpleNamespace(_draining=False, _external_drain_active=False)
    body_started = asyncio.Event()
    release_body = asyncio.Event()
    fired = threading.Event()
    release_fire = threading.Event()

    class BlockingProvider:
        def fire_due(self, job_id, *, adapters=None, loop=None):
            fired.set()
            release_fire.wait(timeout=2)
            return True

    original_json = web.Request.json

    async def delayed_json(request):
        body_started.set()
        await release_body.wait()
        return await original_json(request)

    monkeypatch.setattr("cron.scheduler_provider.resolve_cron_scheduler", BlockingProvider)
    monkeypatch.setattr(
        "plugins.cron_providers.chronos.verify.get_fire_verifier",
        lambda: (lambda **kw: {"purpose": "cron_fire"}),
    )
    app = _create_app(adapter)
    with patch("gateway.run._gateway_runner_ref", lambda: runner), patch.object(
        web.Request, "json", delayed_json
    ):
        async with TestClient(TestServer(app)) as cli:
            request_task = asyncio.create_task(
                cli.post(
                    "/api/cron/fire",
                    headers={"Authorization": "Bearer good"},
                    json={"job_id": "abc123"},
                )
            )
            await body_started.wait()
            assert adapter.active_agent_work_count() == 1

            release_body.set()
            response = await request_task
            assert response.status == 202
            await asyncio.to_thread(fired.wait, 2)
            assert adapter.active_agent_work_count() == 1
            release_fire.set()
            for _ in range(50):
                if adapter.active_agent_work_count() == 0:
                    break
                await asyncio.sleep(0.01)

    assert adapter.active_agent_work_count() == 0


@pytest.mark.asyncio
async def test_missing_job_id_400(adapter, monkeypatch):
    """Valid token but no job_id → 400, no fire."""
    spy = _SpyProvider()
    monkeypatch.setattr("cron.scheduler_provider.resolve_cron_scheduler", lambda: spy)
    monkeypatch.setattr(
        "plugins.cron_providers.chronos.verify.get_fire_verifier",
        lambda: (lambda **kw: {"purpose": "cron_fire"}),
    )

    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post("/api/cron/fire",
                              headers={"Authorization": "Bearer good"},
                              json={})
        assert resp.status == 400
    assert spy.fired == []


@pytest.mark.asyncio
async def test_fire_does_not_require_api_server_key(adapter, monkeypatch):
    """The fire endpoint must NOT gate on API_SERVER_KEY — auth is the NAS-JWT.
    A request with NO API key header but a valid fire token still succeeds."""
    spy = _SpyProvider()
    monkeypatch.setattr("cron.scheduler_provider.resolve_cron_scheduler", lambda: spy)
    monkeypatch.setattr(
        "plugins.cron_providers.chronos.verify.get_fire_verifier",
        lambda: (lambda **kw: {"purpose": "cron_fire"}),
    )

    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        # Bearer is the FIRE token, not the API_SERVER_KEY "sk-secret".
        resp = await cli.post("/api/cron/fire",
                              headers={"Authorization": "Bearer nas-jwt"},
                              json={"job_id": "j9"})
        assert resp.status == 202
    for _ in range(50):
        if spy.fired:
            break
        await asyncio.sleep(0.01)
    assert spy.fired == ["j9"]


@pytest.mark.asyncio
async def test_sync_verifier_runs_off_the_event_loop(adapter, monkeypatch):
    """The verifier resolves the signing key from a JWKS URL — a synchronous
    HTTP GET on a cache miss. It must run via asyncio.to_thread, NOT inline on
    the event loop, or a slow/rate-limited portal stalls every other adapter
    sharing the loop. Proof: the sync verifier executes on a worker thread, not
    the loop thread.
    """
    loop_thread_id = threading.get_ident()
    seen = {}

    def blocking_verifier(**kw):
        seen["thread_id"] = threading.get_ident()
        return {"purpose": "cron_fire"}

    spy = _SpyProvider()
    monkeypatch.setattr("cron.scheduler_provider.resolve_cron_scheduler", lambda: spy)
    monkeypatch.setattr(
        "plugins.cron_providers.chronos.verify.get_fire_verifier",
        lambda: blocking_verifier,
    )

    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post("/api/cron/fire",
                              headers={"Authorization": "Bearer good"},
                              json={"job_id": "off-loop"})
        assert resp.status == 202

    # If the verifier had run inline on the loop, its thread id would equal the
    # loop thread's; to_thread puts it on a distinct worker thread.
    assert seen["thread_id"] != loop_thread_id


@pytest.mark.asyncio
async def test_crashing_verifier_fails_closed_401(adapter, monkeypatch):
    """A verifier that raises must be treated as a rejection (401), never admit
    the fire, and never surface as a 500 — this is the only inbound that can
    trigger remote job execution, so it fails closed.
    """
    spy = _SpyProvider()
    monkeypatch.setattr("cron.scheduler_provider.resolve_cron_scheduler", lambda: spy)

    def exploding_verifier(**kw):
        raise RuntimeError("JWKS endpoint unreachable")

    monkeypatch.setattr(
        "plugins.cron_providers.chronos.verify.get_fire_verifier",
        lambda: exploding_verifier,
    )

    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post("/api/cron/fire",
                              headers={"Authorization": "Bearer boom"},
                              json={"job_id": "abc123"})
        assert resp.status == 401

    await asyncio.sleep(0.05)
    assert spy.fired == []


@pytest.mark.asyncio
async def test_async_verifier_is_awaited(adapter, monkeypatch):
    """A coroutine verifier (a future async escape-hatch) is awaited directly
    rather than dispatched to a thread — a valid async verify still fires.
    """
    spy = _SpyProvider()
    monkeypatch.setattr("cron.scheduler_provider.resolve_cron_scheduler", lambda: spy)

    async def async_verifier(**kw):
        return {"purpose": "cron_fire", "aud": "agent:x"}

    monkeypatch.setattr(
        "plugins.cron_providers.chronos.verify.get_fire_verifier",
        lambda: async_verifier,
    )

    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post("/api/cron/fire",
                              headers={"Authorization": "Bearer good"},
                              json={"job_id": "async-ok"})
        assert resp.status == 202

    for _ in range(50):
        if spy.fired:
            break
        await asyncio.sleep(0.01)
    assert spy.fired == ["async-ok"]
