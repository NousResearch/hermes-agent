"""Webex webhook callbacks share the multiplexer's listener without sharing secrets."""

import asyncio
import hashlib
import hmac
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import Platform, PlatformConfig
from gateway.platforms.shared_ingress import dispatch_profile_ingress
from plugins.platforms.webex.adapter import WebexAdapter


@pytest.fixture
def profile_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    profile = home / "profiles" / "coder"
    profile.mkdir(parents=True)
    (profile / ".env").write_text("WEBEX_WEBHOOK_SECRET=secret-coder\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr("hermes_constants._default_hermes_root_memo", None)
    monkeypatch.setattr("agent.secret_scope._MULTIPLEX_ACTIVE", True)
    return profile


def _adapter(monkeypatch):
    adapter = WebexAdapter(PlatformConfig(enabled=True, token="coder-token", extra={
        "connection_mode": "webhook",
        "public_url": "https://public.example/p/coder",
        "secret": "secret-coder",
    }))
    adapter._shared_listener_profile = "coder"
    adapter.set_owner_profile("coder")
    monkeypatch.setattr(adapter, "_load_bot_identity", AsyncMock())
    monkeypatch.setattr(adapter, "_ensure_managed_webhooks", AsyncMock())
    monkeypatch.setattr(adapter, "_acquire_platform_lock", MagicMock(return_value=True))
    monkeypatch.setattr(adapter, "_release_platform_lock", MagicMock())
    monkeypatch.setattr(adapter, "_poll_loop", AsyncMock(side_effect=asyncio.Event().wait))
    return adapter


def _signature(body, secret):
    return hmac.new(secret.encode(), body, hashlib.sha1).hexdigest()


@pytest.mark.asyncio
async def test_secondary_webhook_uses_shared_listener_and_own_secret(profile_home, monkeypatch):
    adapter = _adapter(monkeypatch)
    runner = SimpleNamespace(
        adapters={}, _profile_adapters={"coder": {Platform("webex"): adapter}})
    adapter.gateway_runner = runner
    seen_homes = []
    verify_signature = adapter._verify_signature

    def verify(headers, body):
        from hermes_constants import get_hermes_home
        seen_homes.append(get_hermes_home())
        return verify_signature(headers, body)

    monkeypatch.setattr(adapter, "_verify_signature", verify)

    async def forward(request):
        return await dispatch_profile_ingress(
            runner, request.match_info["profile"], request.match_info["tail"], request)

    app = web.Application()
    app.router.add_route("*", "/p/{profile}/{tail:.*}", forward)
    body = b'{"resource":"messages","event":"created","data":{"id":"probe"}}'
    async with TestClient(TestServer(app)) as client:
        # The existing listener owns this port; the secondary must not bind it again.
        adapter._port = client.server.port
        runner.adapters[Platform.API_SERVER] = SimpleNamespace(
            _host="127.0.0.1", _port=client.server.port)
        try:
            assert await adapter.connect()
            assert adapter._runner is None
            assert adapter._target_url == "https://public.example/p/coder/webex/webhook"
            ok = await client.post("/p/coder/webex/webhook", data=body, headers={
                "X-Spark-Signature": _signature(body, "secret-coder")})
            assert ok.status == 200
            assert adapter._message_queue.get_nowait()["data"]["id"] == "probe"
            wrong = await client.post("/p/coder/webex/webhook", data=body, headers={
                "X-Spark-Signature": _signature(body, "default-secret")})
            assert wrong.status == 403
            assert adapter._message_queue.empty()
            assert (await client.post("/p/unknown/webex/webhook", data=body)).status == 404
            assert (await client.post("/webex/webhook", data=body)).status == 404
            assert seen_homes == [profile_home, profile_home]
        finally:
            await adapter.disconnect()
        assert (await client.post("/p/coder/webex/webhook", data=body)).status == 404


@pytest.mark.asyncio
async def test_shared_webhook_startup_failure_removes_published_app(profile_home, monkeypatch):
    adapter = _adapter(monkeypatch)
    adapter.gateway_runner = SimpleNamespace(adapters={})
    monkeypatch.setattr(adapter, "_ensure_managed_webhooks", AsyncMock(
        side_effect=RuntimeError("registration failed")))

    assert await adapter.connect() is False
    assert getattr(adapter, "_shared_ingress_app", None) is None
    assert adapter._session is None
    adapter._release_platform_lock.assert_called_once()
