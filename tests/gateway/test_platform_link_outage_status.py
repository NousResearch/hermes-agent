"""Mattermost and Signal report their link to the platform: connected while it is up, not connected
during an outage, connected again once the adapter's own reconnect succeeds.

``gateway_state.json`` (read by ``hermes gateway status``, the dashboard and the Desktop health pill)
and ``adapter.is_connected`` both follow the link. Each adapter runs its real ``connect()`` and receive
loop against a local stand-in server on 127.0.0.1; the outage is the stand-in going away and coming
back on the same port.
"""

from __future__ import annotations

import asyncio

import pytest
from aiohttp import web

from gateway.config import PlatformConfig
from gateway.status import publish_runtime_status, read_runtime_status
from tests.waiting import async_wait_until


def _platform_state(platform: str) -> str:
    return ((read_runtime_status() or {}).get("platforms", {}).get(platform) or {}).get("state", "")


class _StandIn:
    """An aiohttp server that can be stopped and restarted on the same port."""

    def __init__(self) -> None:
        self.port = 0
        self.open_streams = 0
        self._down = asyncio.Event()
        self._runner: web.AppRunner | None = None

    def routes(self, app: web.Application) -> None:
        raise NotImplementedError

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    async def start(self) -> None:
        self._down = asyncio.Event()
        app = web.Application()
        self.routes(app)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", self.port)
        await site.start()
        self.port = self._runner.addresses[0][1]

    async def stop(self) -> None:
        if not self._down.is_set():
            self._down.set()
            await self._runner.cleanup()


class _MattermostStandIn(_StandIn):
    def routes(self, app: web.Application) -> None:
        app.router.add_get("/api/v4/users/me", self._me)
        app.router.add_get("/api/v4/websocket", self._websocket)

    async def _me(self, request: web.Request) -> web.Response:
        return web.json_response({"id": "bot-user-id", "username": "hermes"})

    async def _websocket(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.receive_json()  # authentication_challenge
        self.open_streams += 1
        await self._down.wait()
        self.open_streams -= 1
        await ws.close()
        return ws


class _SignalStandIn(_StandIn):
    def routes(self, app: web.Application) -> None:
        app.router.add_get("/api/v1/check", self._check)
        app.router.add_get("/api/v1/events", self._events)

    async def _check(self, request: web.Request) -> web.Response:
        return web.Response(status=200)

    async def _events(self, request: web.Request) -> web.StreamResponse:
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        await resp.write(b":keepalive\n\n")
        self.open_streams += 1
        await self._down.wait()
        self.open_streams -= 1
        return resp


async def _assert_outage_reported(adapter, server: _StandIn, platform: str) -> None:
    publish_runtime_status(platform=platform, platform_state="connected")  # the runner's stamp after connect()
    await async_wait_until(lambda: server.open_streams == 1, timeout=10.0, what=f"{platform} receive stream open")
    await async_wait_until(lambda: _platform_state(platform) == "connected", timeout=5.0,
                           what=f"{platform} reported connected")
    assert adapter.is_connected is True

    await server.stop()
    await async_wait_until(lambda: not adapter.is_connected, timeout=10.0,
                           what=f"{platform} with no live platform link reported as not connected")
    await async_wait_until(lambda: _platform_state(platform) == "disconnected", timeout=5.0,
                           what=f"{platform} status written as disconnected")

    await server.start()
    await async_wait_until(lambda: adapter.is_connected and server.open_streams == 1, timeout=10.0,
                           what=f"{platform} reconnected")
    await async_wait_until(lambda: _platform_state(platform) == "connected", timeout=5.0,
                           what=f"{platform} status written as connected again")


@pytest.mark.asyncio
async def test_mattermost_websocket_outage_is_reported_until_the_link_returns(monkeypatch):
    from plugins.platforms.mattermost import adapter as mattermost

    monkeypatch.setattr(mattermost, "_RECONNECT_BASE_DELAY", 0.01)
    monkeypatch.setattr(mattermost, "_RECONNECT_MAX_DELAY", 0.05)
    server = _MattermostStandIn()
    await server.start()
    adapter = mattermost.MattermostAdapter(
        PlatformConfig(enabled=True, token="test-token", extra={"url": server.url}))
    try:
        assert await adapter.connect() is True
        await _assert_outage_reported(adapter, server, "mattermost")
    finally:
        await adapter.disconnect()
        await server.stop()


@pytest.mark.asyncio
async def test_signal_event_stream_outage_is_reported_until_the_link_returns(monkeypatch):
    from plugins.platforms.signal import adapter as signal

    monkeypatch.setattr(signal, "SSE_RETRY_DELAY_INITIAL", 0.01)
    monkeypatch.setattr(signal, "SSE_RETRY_DELAY_MAX", 0.05)
    server = _SignalStandIn()
    await server.start()
    adapter = signal.SignalAdapter(
        PlatformConfig(enabled=True, extra={"http_url": server.url, "account": "+15550000001"}))
    try:
        assert await adapter.connect() is True
        await _assert_outage_reported(adapter, server, "signal")
    finally:
        await adapter.disconnect()
        await server.stop()
