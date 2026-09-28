"""Auth gate for `hermes proxy` (#126757).

The proxy attaches the operator's real upstream credential to every forwarded
request. Before this gate it accepted any bearer (or none), never checked
Origin (browser pages could spend the credential via CORS-simple cross-site
POSTs with no preflight), and happily bound a non-loopback host with no auth at
all — an open public credential proxy.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

aiohttp = pytest.importorskip("aiohttp")
from aiohttp import web  # noqa: E402

from hermes_cli.proxy.adapters.base import UpstreamAdapter, UpstreamCredential
from hermes_cli.proxy.server import (  # noqa: E402
    PROXY_TOKEN_ENV, _is_loopback_host, create_app, run_server,
)


class FakeAdapter(UpstreamAdapter):
    """Test adapter returning a fixed credential; no disk access."""

    def __init__(self, base_url: str):
        self._base_url = base_url
        self.calls = 0

    @property
    def name(self): return "fake"

    @property
    def display_name(self): return "Fake Provider"

    @property
    def allowed_paths(self): return frozenset(["/chat/completions"])

    def is_authenticated(self): return True

    def get_credential(self):
        self.calls += 1
        return UpstreamCredential(
            bearer="real-upstream-bearer", base_url=self._base_url,
            expires_at="2099-01-01T00:00:00Z",
        )

    def get_retry_credential(self, *, failed_credential, status_code):
        return None


async def _start_runner(app: "web.Application"):
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, host="127.0.0.1", port=0)
    await site.start()
    port = list(site._server.sockets)[0].getsockname()[1]  # type: ignore[union-attr]
    return runner, f"http://127.0.0.1:{port}"


def _build_capturing_upstream(captured: Dict[str, Any]) -> "web.Application":
    async def echo(request):
        captured["calls"] += 1
        return web.json_response({"reached": True})
    app = web.Application()
    app.router.add_route("*", "/v1/chat/completions", echo)
    return app


@pytest.mark.asyncio
async def test_no_token_accepts_anything_on_loopback():
    """Default loopback posture is unchanged: any bearer works (back-compat)."""
    captured: Dict[str, Any] = {"calls": 0}
    up_runner, up_base = await _start_runner(_build_capturing_upstream(captured))
    adapter = FakeAdapter(f"{up_base}/v1")
    px_runner, px_base = await _start_runner(create_app(adapter))
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{px_base}/v1/chat/completions",
                json={"model": "m"},
                headers={"Authorization": "Bearer anything-at-all"},
            ) as resp:
                assert resp.status == 200
                assert (await resp.json())["reached"] is True
    finally:
        await px_runner.cleanup()
        await up_runner.cleanup()


@pytest.mark.asyncio
async def test_configured_token_rejects_wrong_and_missing_bearer():
    captured: Dict[str, Any] = {"calls": 0}
    up_runner, up_base = await _start_runner(_build_capturing_upstream(captured))
    adapter = FakeAdapter(f"{up_base}/v1")
    px_runner, px_base = await _start_runner(create_app(adapter, proxy_token="sekrit"))
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{px_base}/v1/chat/completions", json={"model": "m"}) as resp:
                assert resp.status == 401  # no bearer at all
            async with session.post(
                f"{px_base}/v1/chat/completions", json={"model": "m"},
                headers={"Authorization": "Bearer wrong"},
            ) as resp:
                assert resp.status == 401
            async with session.post(
                f"{px_base}/v1/chat/completions", json={"model": "m"},
                headers={"Authorization": "Bearer sekrit"},
            ) as resp:
                assert resp.status == 200  # correct token still forwards
    finally:
        await px_runner.cleanup()
        await up_runner.cleanup()
    assert captured["calls"] == 1  # only the authorized request reached upstream


@pytest.mark.asyncio
async def test_browser_origin_request_is_refused():
    """A page-originated request (Origin header) must never spend the credential,
    even with the right bearer: CORS-simple POSTs skip preflight entirely."""
    captured: Dict[str, Any] = {"calls": 0}
    up_runner, up_base = await _start_runner(_build_capturing_upstream(captured))
    adapter = FakeAdapter(f"{up_base}/v1")
    px_runner, px_base = await _start_runner(create_app(adapter, proxy_token="sekrit"))
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{px_base}/v1/chat/completions",
                json={"model": "m"},
                headers={"Authorization": "Bearer sekrit", "Origin": "https://evil.example"},
            ) as resp:
                assert resp.status == 403
    finally:
        await px_runner.cleanup()
        await up_runner.cleanup()
    assert captured["calls"] == 0


@pytest.mark.asyncio
async def test_off_loopback_bind_without_token_refuses_to_start():
    adapter = FakeAdapter("https://upstream.example/v1")
    with pytest.raises(ValueError, match="without an auth token"):
        await run_server(adapter, host="0.0.0.0", port=0, proxy_token="")


def test_loopback_host_detection():
    assert _is_loopback_host("127.0.0.1") is True
    assert _is_loopback_host("::1") is True
    assert _is_loopback_host("localhost") is True
    assert _is_loopback_host("0.0.0.0") is False
    assert _is_loopback_host("192.168.1.5") is False


@pytest.mark.asyncio
async def test_off_loopback_bind_with_token_starts():
    adapter = FakeAdapter("https://upstream.example/v1")
    shutdown = asyncio.Event()
    task = asyncio.ensure_future(
        run_server(adapter, host="127.0.0.1", port=0, shutdown_event=shutdown, proxy_token="t")
    )
    await asyncio.sleep(0.2)
    shutdown.set()
    await asyncio.wait_for(task, timeout=5)
