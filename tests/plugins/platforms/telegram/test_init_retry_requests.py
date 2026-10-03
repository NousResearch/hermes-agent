"""Telegram initialization retries must not inherit an abandoned app's requests."""

import asyncio

import httpx
import pytest

telegram = pytest.importorskip("telegram")
if not getattr(telegram, "__file__", None):
    pytest.skip("requires python-telegram-bot", allow_module_level=True)

from telegram.ext import Application  # noqa: E402
from telegram.request import HTTPXRequest  # noqa: E402

from gateway.config import PlatformConfig  # noqa: E402
from plugins.platforms.telegram.adapter import TelegramAdapter  # noqa: E402


@pytest.mark.asyncio
async def test_timeout_retry_uses_open_requests_while_abandoned_requests_stay_closed(monkeypatch):
    entered, release = asyncio.Event(), asyncio.Event()
    first_app = None
    api_calls = []

    def respond(request):
        api_calls.append(request.url.path)
        return httpx.Response(200, json={
            "ok": True,
            "result": {"id": 123, "is_bot": True, "first_name": "Test", "username": "test_bot"},
        })

    monkeypatch.setattr(
        HTTPXRequest, "_build_client",
        lambda self: httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    )
    monkeypatch.setenv("HERMES_TELEGRAM_DISABLE_FALLBACK_IPS", "1")
    monkeypatch.setenv("HERMES_TELEGRAM_INIT_TIMEOUT", "0.05")
    monkeypatch.setenv("HERMES_TELEGRAM_HTTP_POOL_SIZE", "17")
    monkeypatch.setenv("HERMES_TELEGRAM_HTTP_CONNECT_TIMEOUT", "0.7")

    class SlowFirstApplication(Application):
        async def initialize(self):
            if self is first_app:
                for request in self.bot._request:
                    await request.initialize()
                entered.set()
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    await release.wait()
                return
            await super().initialize()

    adapter = TelegramAdapter(PlatformConfig(
        enabled=True, token="123:AAA", extra={"proxy_url": "http://localhost:1234"},
    ))
    builder = Application.builder().token("123:AAA").application_class(SlowFirstApplication)
    general, updates = await adapter._build_ptb_requests()
    builder.request(general).get_updates_request(updates)
    adapter._app = first_app = builder.build()
    adapter._bot = first_app.bot
    monkeypatch.setattr(adapter, "_wire_plugin_handlers", lambda app: None)
    monkeypatch.setattr(adapter, "_register_handlers", lambda app: None)

    try:
        ladder = asyncio.create_task(adapter._initialize_app_with_retries(builder))
        await asyncio.wait_for(entered.wait(), timeout=5)
        await asyncio.wait_for(ladder, timeout=10)

        successor = adapter._app
        assert successor is not first_app
        assert all(old is not new for old, new in zip(first_app.bot._request, successor.bot._request))
        for old, new in zip(first_app.bot._request, successor.bot._request):
            assert new._client_kwargs["proxy"] == old._client_kwargs["proxy"] == "http://localhost:1234"
            assert new._client_kwargs["timeout"].connect == old._client_kwargs["timeout"].connect == 0.7
            assert new._client_kwargs["limits"].max_connections == 17
        assert all(request._client.is_closed for request in first_app.bot._request)
        assert all(not request._client.is_closed for request in successor.bot._request)
        assert (await successor.bot.get_me()).id == 123
        assert api_calls
        for request in first_app.bot._request:
            await request.initialize()
        assert all(request._client.is_closed for request in first_app.bot._request)
    finally:
        release.set()
        await asyncio.sleep(0)
        if adapter._app is not None:
            await adapter._app.shutdown()
        for request in first_app.bot._request:
            await request.shutdown()
