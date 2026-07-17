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


class _Reservation:
    def __init__(self, provider):
        self.provider = provider

    def release(self):
        return None


def _reserve(provider):
    return lambda: _Reservation(provider)


@pytest.mark.asyncio
async def test_valid_token_accepts_and_fires(adapter, monkeypatch):
    """Valid NAS-JWT + {job_id} → 202 and fire_due invoked with that id."""
    spy = _SpyProvider()
    monkeypatch.setattr("cron.scheduler_runtime.reserve_active_scheduler_provider", _reserve(spy))
    # verifier returns claims (valid token)
    monkeypatch.setattr(
        "plugins.cron_providers.chronos.verify.get_fire_verifier",
        lambda: (lambda **kw: {"purpose": "cron_fire", "aud": "agent:x"}),
    )

    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post("/api/cron/fire",
                              headers={"Authorization": "Bearer good"},
                              json={"job_id": "abc123"})
        assert resp.status == 202
        data = await resp.json()
        assert data["job_id"] == "abc123"

    # fire runs in a background thread/task — give it a beat to land.
    for _ in range(50):
        if spy.fired:
            break
        await asyncio.sleep(0.01)
    assert spy.fired == ["abc123"]


@pytest.mark.asyncio
async def test_invalid_token_401_and_no_fire(adapter, monkeypatch):
    """Bad/forged token → 401, fire_due NOT invoked."""
    spy = _SpyProvider()
    monkeypatch.setattr("cron.scheduler_runtime.reserve_active_scheduler_provider", _reserve(spy))
    monkeypatch.setattr(
        "plugins.cron_providers.chronos.verify.get_fire_verifier",
        lambda: (lambda **kw: None),  # verification fails
    )

    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post("/api/cron/fire",
                              headers={"Authorization": "Bearer forged"},
                              json={"job_id": "abc123"})
        assert resp.status == 401

    await asyncio.sleep(0.05)
    assert spy.fired == []


@pytest.mark.asyncio
async def test_missing_token_401(adapter, monkeypatch):
    """No Authorization header → verifier gets empty token → 401."""
    spy = _SpyProvider()
    monkeypatch.setattr("cron.scheduler_runtime.reserve_active_scheduler_provider", _reserve(spy))
    # Real verifier: empty token returns None.
    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post("/api/cron/fire", json={"job_id": "abc123"})
        assert resp.status == 401
    assert spy.fired == []


@pytest.mark.asyncio
async def test_valid_token_refuses_during_gateway_drain(adapter, monkeypatch):
    spy = _SpyProvider()
    runner = SimpleNamespace(_draining=False, _external_drain_active=True)
    monkeypatch.setattr("cron.scheduler_runtime.reserve_active_scheduler_provider", _reserve(spy))
    monkeypatch.setattr(
        "plugins.cron_providers.chronos.verify.get_fire_verifier",
        lambda: (lambda **kw: {"purpose": "cron_fire"}),
    )

    app = _create_app(adapter)
    with patch("gateway.run._gateway_runner_ref", lambda: runner):
        async with TestClient(TestServer(app)) as cli:
            response = await cli.post(
                "/api/cron/fire",
                headers={"Authorization": "Bearer good"},
                json={"job_id": "abc123"},
            )
            payload = await response.json()

    assert response.status == 503
    assert payload["error"]["code"] == "gateway_draining"
    assert spy.fired == []


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

    monkeypatch.setattr(
        "cron.scheduler_runtime.reserve_active_scheduler_provider",
        _reserve(BlockingProvider()),
    )
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
async def test_fire_task_cancellation_holds_reservation_until_worker_exits(
    adapter, monkeypatch
):
    fired = threading.Event()
    release_fire = threading.Event()
    reservation_released = threading.Event()

    class BlockingProvider:
        def fire_due(self, job_id, *, adapters=None, loop=None):
            fired.set()
            release_fire.wait(timeout=2)
            return True

    class TrackingReservation:
        provider = BlockingProvider()

        def release(self):
            reservation_released.set()

    monkeypatch.setattr(
        "cron.scheduler_runtime.reserve_active_scheduler_provider",
        lambda: TrackingReservation(),
    )
    monkeypatch.setattr(
        "plugins.cron_providers.chronos.verify.get_fire_verifier",
        lambda: (lambda **kw: {"purpose": "cron_fire"}),
    )

    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        response = await cli.post(
            "/api/cron/fire",
            headers={"Authorization": "Bearer good"},
            json={"job_id": "abc123"},
        )
        assert response.status == 202
        assert await asyncio.to_thread(fired.wait, 2)
        task = next(iter(adapter._background_tasks))

        task.cancel()
        await asyncio.sleep(0.05)
        assert not task.done()
        assert not reservation_released.is_set()
        assert adapter.active_agent_work_count() == 1

        release_fire.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        for _ in range(50):
            if adapter.active_agent_work_count() == 0:
                break
            await asyncio.sleep(0.01)

    assert reservation_released.is_set()
    assert adapter.active_agent_work_count() == 0


@pytest.mark.asyncio
async def test_fire_inner_task_cancellation_does_not_spin_or_release_early(
    adapter, monkeypatch
):
    fired = threading.Event()
    release_fire = threading.Event()
    reservation_released = threading.Event()

    class BlockingProvider:
        def fire_due(self, job_id, *, adapters=None, loop=None):
            fired.set()
            release_fire.wait(timeout=2)
            return True

    class TrackingReservation:
        provider = BlockingProvider()

        def release(self):
            reservation_released.set()

    monkeypatch.setattr(
        "cron.scheduler_runtime.reserve_active_scheduler_provider",
        lambda: TrackingReservation(),
    )
    monkeypatch.setattr(
        "plugins.cron_providers.chronos.verify.get_fire_verifier",
        lambda: (lambda **kw: {"purpose": "cron_fire"}),
    )

    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        response = await cli.post(
            "/api/cron/fire",
            headers={"Authorization": "Bearer good"},
            json={"job_id": "abc123"},
        )
        assert response.status == 202
        assert await asyncio.to_thread(fired.wait, 2)
        outer = next(
            task
            for task in asyncio.all_tasks()
            if task.get_name() == "cron-fire-supervisor"
        )
        inner = next(
            task
            for task in asyncio.all_tasks()
            if task.get_name() == "cron-fire-worker"
        )

        outer.cancel()
        inner.cancel()
        # This yield would hang if the supervisor busy-spun on the cancelled
        # inner task.
        await asyncio.wait_for(asyncio.sleep(0.05), timeout=1)
        assert outer.done()
        assert not reservation_released.is_set()
        assert adapter.active_agent_work_count() == 1

        release_fire.set()
        assert await asyncio.to_thread(reservation_released.wait, 2)
        for _ in range(50):
            if adapter.active_agent_work_count() == 0:
                break
            await asyncio.sleep(0.01)

    assert adapter.active_agent_work_count() == 0


@pytest.mark.asyncio
async def test_fire_provider_exception_is_consumed_and_redacted(
    adapter, monkeypatch, caplog
):
    raw_secret = "provider-error-secret-must-not-leak"
    reservation_released = threading.Event()

    class FailingProvider:
        def fire_due(self, job_id, *, adapters=None, loop=None):
            raise RuntimeError(raw_secret)

    class TrackingReservation:
        provider = FailingProvider()

        def release(self):
            reservation_released.set()

    monkeypatch.setattr(
        "cron.scheduler_runtime.reserve_active_scheduler_provider",
        lambda: TrackingReservation(),
    )
    monkeypatch.setattr(
        "plugins.cron_providers.chronos.verify.get_fire_verifier",
        lambda: (lambda **kw: {"purpose": "cron_fire"}),
    )

    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        response = await cli.post(
            "/api/cron/fire",
            headers={"Authorization": "Bearer good"},
            json={"job_id": "abc123"},
        )
        assert response.status == 202
        assert await asyncio.to_thread(reservation_released.wait, 2)
        for _ in range(50):
            if not adapter._background_tasks:
                break
            await asyncio.sleep(0.01)

    assert raw_secret not in caplog.text
    assert "cron fire: provider worker failed" in caplog.text
    assert adapter.active_agent_work_count() == 0


@pytest.mark.asyncio
async def test_missing_job_id_400(adapter, monkeypatch):
    """Valid token but no job_id → 400, no fire."""
    spy = _SpyProvider()
    monkeypatch.setattr("cron.scheduler_runtime.reserve_active_scheduler_provider", _reserve(spy))
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
async def test_valid_fire_fails_closed_without_active_leased_provider(
    adapter, monkeypatch
):
    monkeypatch.setattr(
        "cron.scheduler_runtime.reserve_active_scheduler_provider", lambda: None
    )
    monkeypatch.setattr(
        "plugins.cron_providers.chronos.verify.get_fire_verifier",
        lambda: (lambda **kw: {"purpose": "cron_fire"}),
    )
    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        response = await cli.post(
            "/api/cron/fire",
            headers={"Authorization": "Bearer good"},
            json={"job_id": "abc123"},
        )
        payload = await response.json()
    assert response.status == 503
    assert payload == {"error": "scheduler ownership unavailable"}


@pytest.mark.asyncio
async def test_fire_does_not_require_api_server_key(adapter, monkeypatch):
    """The fire endpoint must NOT gate on API_SERVER_KEY — auth is the NAS-JWT.
    A request with NO API key header but a valid fire token still succeeds."""
    spy = _SpyProvider()
    monkeypatch.setattr("cron.scheduler_runtime.reserve_active_scheduler_provider", _reserve(spy))
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
