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
            resp = await cli.post(
                "/api/cron/fire", headers={"Authorization": "Bearer good"}, json={"job_id": "job-1"})
            await resp.read()  # keep the body readable after the client closes
            return resp


async def _wait_for(predicate, timeout=2.0):
    for _ in range(int(timeout / 0.01)):
        if predicate():
            return True
        await asyncio.sleep(0.01)
    return False


class _StartingRunner:
    """A gateway that has not finished starting. ``entered`` is set the first time the gate reads
    ``_running``, so a test changes the runner only once the handler is actually waiting (no timers)."""

    def __init__(self):
        self._draining = False
        self._external_drain_active = False
        self.adapters = {}
        self.started = False
        self.entered = asyncio.Event()

    @property
    def _running(self):
        self.entered.set()
        return self.started


async def _once_waiting(runner, change):
    await asyncio.wait_for(runner.entered.wait(), timeout=10.0)
    change()


@pytest.mark.asyncio
async def test_fire_during_startup_waits_and_receives_the_live_adapters(adapter, provider):
    """The cold-boot shape: api_server already accepting, adapters not yet published."""
    relay = object()
    runner = _StartingRunner()

    def _finish_startup():
        runner.adapters["relay"] = relay
        runner.started = True

    finisher = asyncio.ensure_future(_once_waiting(runner, _finish_startup))
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

    # NAS classifies this refusal by the exact body string and honours Retry-After.
    assert resp.status == 503 and resp.headers["Retry-After"] == "60"
    assert (await resp.json())["error"] == "gateway unreachable; retry"
    assert provider.claimed == [] and provider.fired == []  # nothing durably claimed for a fire we refused


@pytest.mark.asyncio
async def test_gateway_that_starts_draining_mid_boot_is_refused_without_the_full_wait(
        adapter, provider, monkeypatch):
    monkeypatch.setattr(api_server_fire_startup, "FIRE_STARTUP_WAIT_SECONDS", 30.0)
    runner = _StartingRunner()
    stopper = asyncio.ensure_future(_once_waiting(runner, lambda: setattr(runner, "_draining", True)))
    resp = await asyncio.wait_for(_post_fire(adapter, runner), timeout=10.0)
    await stopper

    assert resp.status == 503 and resp.headers["Retry-After"] == "60"
    assert provider.claimed == [] and provider.fired == []


@pytest.mark.asyncio
async def test_gateway_that_finishes_starting_into_a_drain_is_refused(adapter, provider, monkeypatch):
    """A restart drain keeps ``_running`` True, so readiness alone must not admit a fire that waited."""
    monkeypatch.setattr(api_server_fire_startup, "FIRE_STARTUP_WAIT_SECONDS", 30.0)
    runner = _StartingRunner()

    def _start_into_drain():
        runner.started = True
        runner._draining = True

    flipper = asyncio.ensure_future(_once_waiting(runner, _start_into_drain))
    resp = await asyncio.wait_for(_post_fire(adapter, runner), timeout=10.0)
    await flipper

    assert resp.status == 503
    assert provider.claimed == [] and provider.fired == []


@pytest.mark.asyncio
async def test_started_gateway_is_accepted_immediately(adapter, provider):
    runner = SimpleNamespace(_draining=False, _external_drain_active=False, _running=True, adapters={"relay": object()})

    resp = await _post_fire(adapter, runner)

    assert resp.status == 202


@pytest.mark.asyncio
async def test_slow_token_verify_spends_the_startup_budget(adapter, provider, monkeypatch):
    """The budget runs from handler entry: a slow JWKS fetch plus a full wait would outlast the dashboard
    forwarder's timeout, NAS would retry while this handler still ran the job, and it would run twice."""
    monkeypatch.setattr(api_server_fire_startup, "FIRE_STARTUP_WAIT_SECONDS", 0.2)

    async def _slow_verifier(**_kw):  # outlasts the whole budget; host load only makes it slower
        await asyncio.sleep(0.5)
        return {"purpose": "cron_fire"}

    monkeypatch.setattr("plugins.cron_providers.chronos.verify.get_fire_verifier", lambda: _slow_verifier)
    runner = _StartingRunner()
    # Startup completes as soon as the gate first looks: a budget measured from the gate would then
    # admit the fire on its next poll; one measured from entry is already spent and refuses at once.
    finisher = asyncio.ensure_future(_once_waiting(runner, lambda: setattr(runner, "started", True)))
    resp = await _post_fire(adapter, runner)
    await finisher

    assert resp.status == 503
    assert provider.claimed == [] and provider.fired == []
