"""Real PTB request state must survive abandoned initialization (#138649)."""

import asyncio
import json

import httpx
import pytest
import pytest_asyncio
from telegram.ext import Application

from gateway.config import PlatformConfig
from plugins.platforms.telegram import adapter as telegram


@pytest_asyncio.fixture
async def bot_api():
    calls = []
    handlers = set()

    async def respond(reader, writer):
        task = asyncio.current_task()
        handlers.add(task)
        try:
            headers = await reader.readuntil(b"\r\n\r\n")
            length = next(int(line.split(b":", 1)[1]) for line in headers.split(b"\r\n")
                          if line.lower().startswith(b"content-length:"))
            await reader.readexactly(length)
            calls.append(headers.split(b"\r\n", 1)[0])
            body = json.dumps({"ok": True, "result": {
                "id": 123, "is_bot": True, "first_name": "Local", "username": "local_bot",
            }}).encode()
            writer.write(b"HTTP/1.1 200 OK\r\nConnection: close\r\nContent-Type: application/json\r\n"
                         + f"Content-Length: {len(body)}\r\n\r\n".encode() + body)
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()
            handlers.discard(task)

    server = await asyncio.start_server(respond, "127.0.0.1", 0)
    try:
        yield f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}", calls
    finally:
        server.close()
        await server.wait_closed()
        await asyncio.gather(*handlers, return_exceptions=True)


def prepare(monkeypatch, url, mode="direct"):
    adapter = telegram.TelegramAdapter(PlatformConfig(enabled=True, token="123:LOCAL_TEST"))
    monkeypatch.setenv("HERMES_TELEGRAM_DISABLE_FALLBACK_IPS", "1" if mode == "direct" else "0")
    monkeypatch.setattr(adapter, "_fallback_ips", lambda: ["149.154.166.110"])
    monkeypatch.setattr(telegram, "resolve_proxy_url", lambda *a, **kw: url if mode == "proxy" else None)
    monkeypatch.setattr(adapter, "_wire_plugin_handlers", lambda app: None)
    monkeypatch.setattr(adapter, "_register_handlers", lambda app: None)
    monkeypatch.setattr(telegram, "env_float", lambda key, default: 2.0 if key == "HERMES_TELEGRAM_INIT_TIMEOUT" else default)
    monkeypatch.setattr(telegram, "_DRAIN_TIMEOUT", 2.0)
    sleep = asyncio.sleep

    async def backoff(_delay):
        await sleep(0)

    monkeypatch.setattr(telegram.asyncio, "sleep", backoff)
    return adapter, Application.builder().token(adapter.config.token).base_url(url + "/bot")


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["direct", "proxy", "fallback"])
async def test_retry_reaches_socket_while_abandoned_transport_and_cleanup_are_still_waiting(monkeypatch, bot_api, mode, caplog):
    url, calls = bot_api
    adapter, builder = prepare(monkeypatch, url, mode)
    requests = await adapter._build_ptb_requests()
    client = requests[0]._client
    inner = client._transport_for_url(httpx.URL(url))
    release = asyncio.Event()
    blocked = set()

    class WedgedTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            blocked.add(asyncio.current_task())
            # Fault injection at the network boundary: a cancellation-resistant wait BEFORE a socket.
            while not release.is_set():
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    pass
            return await inner.handle_async_request(request)

        async def aclose(self):
            # The same wait also prevents old-generation cleanup from completing.
            await release.wait()
            await inner.aclose()

    transport = WedgedTransport()
    if client._transport is inner:
        client._transport = transport
        requests[0]._client_kwargs["transport"] = transport
    for pattern, mounted in client._mounts.items():
        if mounted is inner:
            client._mounts[pattern] = transport
    original_shutdown = type(requests[0]).shutdown

    async def shutdown(request):
        if request is requests[0]:
            await release.wait()
        await original_shutdown(request)

    monkeypatch.setattr(type(requests[0]), "shutdown", shutdown)
    adapter._app = builder.request(requests[0]).get_updates_request(requests[1]).build()
    adapter._bot = adapter._app.bot
    old_app = adapter._app
    try:
        await asyncio.wait_for(adapter._initialize_app_with_retries(builder), 12)
        assert calls and all(b"/getMe " in call for call in calls)
        assert blocked and all(not task.done() for task in blocked)
        assert not release.is_set()
        assert not set(old_app.bot._request) & set(adapter._bot._request)
        diagnostics = [record.getMessage() for record in caplog.records
                       if "initialization await chain:" in record.getMessage()]
        assert diagnostics
        assert any("handle_async_request" in message and "wait" in message for message in diagnostics)
        assert all("LOCAL_TEST" not in message and "http://" not in message for message in diagnostics)
        # A late old-app shutdown must never close the successful successor's clients.
        release.set()
        await asyncio.gather(*blocked, return_exceptions=True)
        await telegram._shutdown_abandoned_app(old_app)
        assert (await adapter._bot.get_me()).id == adapter._bot.id
    finally:
        release.set()
        await asyncio.gather(*blocked, return_exceptions=True)
        await telegram._shutdown_abandoned_app(old_app)
        await telegram._shutdown_abandoned_app(adapter._app)


@pytest.mark.asyncio
async def test_network_error_rebuild_owns_both_requests_after_late_old_app_shutdown(monkeypatch, bot_api):
    url, calls = bot_api
    adapter, builder = prepare(monkeypatch, url)
    requests = await adapter._build_ptb_requests()
    adapter._app = builder.request(requests[0]).get_updates_request(requests[1]).build()
    adapter._bot = adapter._app.bot
    old_app = adapter._app

    async def fail_initialize():
        raise OSError("transient network failure")

    original_initialize = type(old_app).initialize

    async def initialize(app):
        if app is old_app:
            return await fail_initialize()
        return await original_initialize(app)

    monkeypatch.setattr(type(old_app), "initialize", initialize)
    try:
        await adapter._initialize_app_with_retries(builder)
        assert calls
        assert not set(old_app.bot._request) & set(adapter._bot._request)
        await telegram._shutdown_abandoned_app(old_app)
        assert (await adapter._bot.get_me()).id == adapter._bot.id
    finally:
        await telegram._shutdown_abandoned_app(old_app)
        await telegram._shutdown_abandoned_app(adapter._app)
