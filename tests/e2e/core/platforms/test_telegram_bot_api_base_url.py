"""A custom ``extra.base_url`` without the Bot API ``/bot`` path must still produce a valid URL.

PTB appends the token straight to ``base_url``; a bare server URL turned the token's secret half
into a port (``InvalidURL("Invalid port: 'AA…'")``, #81787).
"""
import httpx
import pytest

pytest.importorskip("telegram")

from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter
from plugins.platforms.telegram.standalone import _telegram_bot
from plugins.platforms.telegram.telegram_network import bot_api_urls

TOKEN = "123456789:AAGURFJSzXoINq_Fj_Srvuf7mpZ4XRXO6rQ"


@pytest.mark.parametrize("base_url, expected", [
    ("http://127.0.0.1:8081", "http://127.0.0.1:8081/bot"),
    ("http://127.0.0.1:8081/", "http://127.0.0.1:8081/bot"),
    ("http://127.0.0.1:8081/bot", "http://127.0.0.1:8081/bot"),  # documented form, unchanged
    ("https://proxy.example/tg/bot{token}", "https://proxy.example/tg/bot{token}"),
])
def test_standalone_bot_resolves_a_valid_api_url(base_url, expected):
    bot = _telegram_bot(TOKEN, {"base_url": base_url})
    httpx.URL(f"{bot.base_url}/getMe")  # raises InvalidURL on the bug
    want = expected.replace("{token}", TOKEN) if "{token}" in expected else expected + TOKEN
    assert bot.base_url == want


@pytest.mark.asyncio
async def test_gateway_connect_builds_application_against_the_normalized_url(monkeypatch):
    """Drive the real ``connect()`` builder path; stop right after the Application is built."""
    built = {}
    from telegram.ext import ApplicationBuilder

    def capture_build(self):
        built["base_url"], built["base_file_url"] = self._base_url, self._base_file_url
        raise RuntimeError("stop after build")

    monkeypatch.setattr(ApplicationBuilder, "build", capture_build)
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token=TOKEN, extra={"base_url": "http://127.0.0.1:8081"}))
    monkeypatch.setattr(adapter, "_acquire_platform_lock", lambda *a, **k: True)
    await adapter.connect()
    assert built == {"base_url": "http://127.0.0.1:8081/bot", "base_file_url": "http://127.0.0.1:8081/file/bot"}


@pytest.mark.parametrize("extra, expected", [
    ({"base_url": "http://127.0.0.1:8081"},
     {"base_url": "http://127.0.0.1:8081/bot", "base_file_url": "http://127.0.0.1:8081/file/bot"}),
    # A reverse proxy's own route is configured, not missing: kept exactly as written (files too).
    ({"base_url": "https://proxy.example/telegram/"},
     {"base_url": "https://proxy.example/telegram/", "base_file_url": "https://proxy.example/telegram/"}),
    ({"base_url": "https://proxy.example/telegram/", "base_file_url": "https://proxy.example/files/"},
     {"base_url": "https://proxy.example/telegram/", "base_file_url": "https://proxy.example/files/"}),
    ({"base_url": "http://127.0.0.1:8081/bot", "base_file_url": "http://127.0.0.1:8081"},
     {"base_url": "http://127.0.0.1:8081/bot", "base_file_url": "http://127.0.0.1:8081/file/bot"}),
])
def test_only_bare_server_urls_get_the_bot_api_path(extra, expected):
    assert bot_api_urls(extra) == expected
