"""Tests for POST /api/delivery/send — loopback cron delivery (#86249)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import Platform, PlatformConfig
from gateway.platforms.api_server import APIServerAdapter, cors_middleware


def _make_adapter() -> APIServerAdapter:
    return APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "sk-secret-sixteen"}))


def _create_app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application(middlewares=[cors_middleware])
    app["api_server_adapter"] = adapter
    app.router.add_post("/api/delivery/send", adapter._handle_delivery_send)
    return app


@pytest.mark.asyncio
async def test_delivery_send_uses_live_relay_transport():
    adapter = _make_adapter()
    send_result = SimpleNamespace(success=True, message_id="m-1", error=None)
    transport = MagicMock()
    transport.send = AsyncMock(return_value=send_result)

    runner = SimpleNamespace(adapters={Platform.RELAY: MagicMock()})
    adapter.gateway_runner = runner
    app = _create_app(adapter)

    with patch(
        "gateway.delivery.resolve_delivery_transport", return_value=transport,
    ), patch(
        "gateway.config.load_gateway_config", return_value=MagicMock(),
    ):
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/api/delivery/send",
                json={
                    "platform": "discord",
                    "chat_id": "1517373704248758474",
                    "content": "Nightly report.",
                },
                headers={"Authorization": "Bearer sk-secret-sixteen"},
            )
            body = await resp.json()

    assert resp.status == 200
    assert body == {"success": True, "message_id": "m-1"}
    transport.send.assert_awaited_once()


@pytest.mark.asyncio
async def test_delivery_send_rejects_bad_auth():
    adapter = _make_adapter()
    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as client:
        resp = await client.post(
            "/api/delivery/send",
            json={"platform": "discord", "chat_id": "1", "content": "x"},
            headers={"Authorization": "Bearer wrong"},
        )
    assert resp.status == 401


@pytest.mark.asyncio
async def test_delivery_send_503_without_adapters():
    adapter = _make_adapter()
    adapter.gateway_runner = SimpleNamespace(adapters={})
    app = _create_app(adapter)
    with patch("gateway.run._gateway_runner_ref", lambda: None):
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/api/delivery/send",
                json={"platform": "discord", "chat_id": "1", "content": "x"},
                headers={"Authorization": "Bearer sk-secret-sixteen"},
            )
            body = await resp.json()
    assert resp.status == 503
    assert "adapters" in body["error"].lower() or "live" in body["error"].lower()
