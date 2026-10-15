"""Webex transport failures retain the host retry and ambiguity contracts."""

import asyncio
import re

import aiohttp
from aiohttp import web
from aiohttp.test_utils import TestServer
import pytest

from gateway.config import PlatformConfig
from plugins.platforms.webex.adapter import WebexAdapter


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [503, 429])
async def test_initial_transient_failure_retries_complete_reply(tmp_path, monkeypatch, status):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    requests = []
    delivered = []

    async def post_message(request):
        payload = await request.json()
        requests.append(payload)
        if len(requests) == 1:
            return web.json_response({"message": "try again"}, status=status, headers={"Retry-After": "0"})
        delivered.append(payload)
        return web.json_response({"id": f"message-{len(requests)}"})

    app = web.Application()
    app.router.add_post("/messages", post_message)
    async with TestServer(app) as server, aiohttp.ClientSession() as session:
        monkeypatch.setattr("plugins.platforms.webex.adapter.API_BASE", str(server.make_url("/")).rstrip("/"))
        adapter = WebexAdapter(PlatformConfig(token="token"))
        adapter._session = session
        result = await adapter._send_with_retry("room", "x" * 8000, metadata={"thread_id": "thread"}, base_delay=0)

    assert result.success
    assert len(requests) == 3
    assert "".join(re.sub(r" \(\d+/\d+\)$", "", payload["markdown"]) for payload in delivered) == "x" * 8000
    assert all(payload["parentId"] == "thread" for payload in delivered)


@pytest.mark.asyncio
async def test_initial_auth_refusal_does_not_send_formatting_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    requests = []

    async def post_message(request):
        requests.append(await request.json())
        if len(requests) == 1:
            return web.json_response({"message": "invalid token"}, status=401)
        return web.json_response({"id": "unexpected-fallback"})

    app = web.Application()
    app.router.add_post("/messages", post_message)
    async with TestServer(app) as server, aiohttp.ClientSession() as session:
        monkeypatch.setattr("plugins.platforms.webex.adapter.API_BASE", str(server.make_url("/")).rstrip("/"))
        adapter = WebexAdapter(PlatformConfig(token="token"))
        adapter._session = session
        result = await adapter._send_with_retry("room", "x" * 8000, base_delay=0)

    assert result.success is False
    assert result.error_kind == "forbidden"
    assert len(requests) == 1


@pytest.mark.asyncio
async def test_initial_timeout_does_not_resend_an_ambiguous_message(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    release = asyncio.Event()
    requests = []

    async def post_message(request):
        requests.append(await request.json())
        if len(requests) == 1:
            await release.wait()
        return web.json_response({"id": "possibly-delivered"})

    app = web.Application()
    app.router.add_post("/messages", post_message)
    async with TestServer(app) as server, aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=0.05)) as session:
        monkeypatch.setattr("plugins.platforms.webex.adapter.API_BASE", str(server.make_url("/")).rstrip("/"))
        adapter = WebexAdapter(PlatformConfig(token="token"))
        adapter._session = session
        try:
            result = await adapter._send_with_retry("room", "x" * 8000, base_delay=0)
        finally:
            release.set()

    assert result.success is False
    assert "timed out" in result.error.lower()
    assert len(requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["send", "edit", "file", "image_url"])
async def test_outbound_writes_keep_rate_limit_cooldown(tmp_path, monkeypatch, operation):
    requests = []

    async def refuse(request):
        await request.read()
        requests.append(request.method)
        return web.json_response({"message": "rate limited"}, status=429, headers={"Retry-After": "7"})

    app = web.Application()
    app.router.add_post("/messages", refuse)
    app.router.add_put("/messages/existing", refuse)
    async with TestServer(app) as server, aiohttp.ClientSession() as session:
        monkeypatch.setattr("plugins.platforms.webex.adapter.API_BASE", str(server.make_url("/")).rstrip("/"))
        adapter = WebexAdapter(PlatformConfig(token="token"))
        adapter._session = session
        if operation == "edit":
            result = await adapter.edit_message("room", "existing", "Reply")
        elif operation == "file":
            path = tmp_path / "report.txt"
            path.write_bytes(b"file contents")
            result = await adapter.send_document("room", str(path), caption="Report")
        elif operation == "image_url":
            result = await adapter.send_image("room", "https://example.com/image.png")
        else:
            result = await adapter.send("room", "Reply")

    assert result.success is False
    assert result.error_kind == "rate_limited"
    assert result.retryable is True
    assert result.retry_after == 7
    assert requests == (["PUT"] if operation == "edit" else ["POST"])
