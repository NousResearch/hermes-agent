"""A transient Bot API failure on the first webhook registration must not leave the webhook unset.

Real PTB ``Application`` + ``Updater.start_webhook`` against the local Bot API stand-in, whose first
``setWebhook`` answers 502: startup must retry, succeed and leave the webhook registered (#141397).
"""

import socket

import pytest

pytest.importorskip("telegram.ext")
pytest.importorskip("tornado")

from telegram.ext import Application

from hermes_agent.gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter
from tests.fakes.platforms.telegram_standin import TelegramStandin

pytestmark = pytest.mark.platforms("posix")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.asyncio
async def test_first_set_webhook_refused_then_startup_registers_it(monkeypatch):
    standin = TelegramStandin().start()
    registered = []
    standin._m_setWebhook = lambda p: registered.append(p["url"]) or True  # the stand-in has no setWebhook
    standin.fail("setWebhook", {"ok": False, "error_code": 502, "description": "Bad Gateway"}, status=502)
    monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", "s3cret")
    monkeypatch.setenv("TELEGRAM_WEBHOOK_PORT", str(_free_port()))
    monkeypatch.setenv("TELEGRAM_WEBHOOK_HOST", "127.0.0.1")
    app = Application.builder().token(standin.token).base_url(standin.api_base).base_file_url(standin.file_base).build()
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token=standin.token))
    adapter._app = app
    try:
        await app.initialize()
        await adapter._start_webhook_mode("https://example.test/telegram", is_reconnect=True)
        assert app.updater.running
        assert [c.faulted for c in standin.calls_of("setWebhook")] == [True, False]
        assert registered == ["https://example.test/telegram"]
    finally:
        if app.updater.running:
            await app.updater.stop()
        await app.shutdown()
        standin.stop()
