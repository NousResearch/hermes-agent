"""A Chronos fire that lands while the gateway is still starting waits for its adapters (cold boot race).

A backend that stops the guest on sleep (Azure) wakes the agent FOR the fire, so the fire reaches
``POST /api/cron/fire`` in the first second of a cold boot. The api_server adapter listens before the
gateway has published any adapter into ``runner.adapters`` (``_publish_primary_adapter`` registers an
adapter only after its ``connect()`` returns), so the handler's snapshot was an empty dict, which
``or None`` turned into ``None``. The job then ran with no adapters and a relay-fronted platform failed
with "has no live gateway transport", losing the result. The handler must wait for the gateway to finish
starting (``runner._running``) and then hand over the LIVE adapters.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms import api_server_fire_startup
from gateway.platforms.api_server import APIServerAdapter, cors_middleware


@pytest.fixture
def adapter():
    return APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "sk-secret"}))


class _SpyProvider:
    def __init__(self):
        self.claimed = []
        self.fired = []

    def claim_fire(self, job_id):
        self.claimed.append(job_id)
        return {"id": job_id, "execution_id": f"exec-{job_id}"}

    def fire_claimed(self, job, *, adapters=None, loop=None):
        self.fired.append((job["id"], adapters))
        return True


@pytest.fixture
def provider(monkeypatch):
    spy = _SpyProvider()
    monkeypatch.setattr("cron.scheduler_provider.resolve_cron_scheduler", lambda: spy)
    monkeypatch.setattr(
        "plugins.cron_providers.chronos.verify.get_fire_verifier",
        lambda: (lambda **kw: {"purpose": "cron_fire"}),
    )
    return spy


async def _post_fire(adapter, runner):
    app = web.Application(middlewares=[cors_middleware])
    app["api_server_adapter"] = adapter
    app.router.add_post("/api/cron/fire", adapter._handle_cron_fire)
    with patch("gateway.run._gateway_runner_ref", lambda: runner):
        async with TestClient(TestServer(app)) as cli:
            return await cli.post(
                "/api/cron/fire", headers={"Authorization": "Bearer good"}, json={"job_id": "job-1"})


async def _wait_for(predicate, timeout=2.0):
    for _ in range(int(timeout / 0.01)):
        if predicate():
            return True
        await asyncio.sleep(0.01)
    return False


@pytest.mark.asyncio
async def test_fire_during_startup_waits_and_receives_the_live_adapters(adapter, provider):
    """The cold-boot shape: api_server already accepting, adapters not yet published."""
    relay = object()
    runner = SimpleNamespace(_draining=False, _external_drain_active=False, _running=False, adapters={})

    async def _finish_startup():
        await asyncio.sleep(0.3)
        runner.adapters["relay"] = relay
        runner._running = True

    finisher = asyncio.ensure_future(_finish_startup())
    resp = await _post_fire(adapter, runner)
    await finisher

    assert resp.status == 202
    assert await _wait_for(lambda: provider.fired)
    _job_id, adapters = provider.fired[0]
    assert adapters is runner.adapters and adapters.get("relay") is relay


@pytest.mark.asyncio
async def test_fire_is_retryable_and_never_claimed_when_startup_does_not_finish(adapter, provider, monkeypatch):
    monkeypatch.setattr(api_server_fire_startup, "FIRE_STARTUP_WAIT_SECONDS", 0.2)
    runner = SimpleNamespace(_draining=False, _external_drain_active=False, _running=False, adapters={})

    resp = await _post_fire(adapter, runner)

    assert resp.status == 503
    assert provider.claimed == [] and provider.fired == []  # nothing durably claimed for a fire we refused


@pytest.mark.asyncio
async def test_started_gateway_is_not_delayed(adapter, provider):
    runner = SimpleNamespace(_draining=False, _external_drain_active=False, _running=True, adapters={"relay": object()})

    loop = asyncio.get_running_loop()
    started = loop.time()
    resp = await _post_fire(adapter, runner)

    assert resp.status == 202
    assert loop.time() - started < 1.0
