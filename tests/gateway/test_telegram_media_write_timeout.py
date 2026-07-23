"""Behavior tests for Telegram's configurable media upload timeout."""

import asyncio
import sys
from unittest.mock import MagicMock

from gateway.config import PlatformConfig


def _ensure_telegram_mock():
    if "telegram" in sys.modules and hasattr(sys.modules["telegram"], "__file__"):
        return
    telegram_mod = MagicMock()
    telegram_mod.ext.ContextTypes.DEFAULT_TYPE = type(None)
    telegram_mod.constants.ParseMode.MARKDOWN_V2 = "MarkdownV2"
    telegram_mod.constants.ChatType.GROUP = "group"
    telegram_mod.constants.ChatType.SUPERGROUP = "supergroup"
    telegram_mod.constants.ChatType.CHANNEL = "channel"
    telegram_mod.constants.ChatType.PRIVATE = "private"
    telegram_modules = (
        "telegram",
        "telegram.ext",
        "telegram.constants",
        "telegram.request",
    )
    for name in telegram_modules:
        sys.modules.setdefault(name, telegram_mod)


_ensure_telegram_mock()

from plugins.platforms.telegram import adapter as tg_adapter  # noqa: E402
from plugins.platforms.telegram.adapter import TelegramAdapter  # noqa: E402


class _StopConnect(Exception):
    """Stop connect after request construction and before network access."""


class _RecordingHTTPXRequest:
    """Record each HTTPXRequest constructor call."""

    instances: list = []

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.__class__.instances.append(self)


def _build_requests(monkeypatch, *, telegram_config=None):
    _RecordingHTTPXRequest.instances = []

    async def _no_fallback_ips():
        return []

    monkeypatch.setattr(tg_adapter, "discover_fallback_ips", _no_fallback_ips)
    monkeypatch.setattr(tg_adapter, "resolve_proxy_url", lambda *args, **kwargs: None)
    monkeypatch.setattr(tg_adapter, "HTTPXRequest", _RecordingHTTPXRequest)

    extra = tg_adapter._apply_yaml_config({}, telegram_config or {}) or {}
    adapter = TelegramAdapter(
        PlatformConfig(enabled=True, token="test-token", extra=extra)
    )
    monkeypatch.setattr(
        adapter,
        "_acquire_platform_lock",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(adapter, "_fallback_ips", lambda: [])

    builder = MagicMock()
    builder.token.return_value = builder
    builder.request.return_value = builder
    builder.get_updates_request.return_value = builder
    builder.build.side_effect = _StopConnect
    application = MagicMock()
    application.builder.return_value = builder
    monkeypatch.setattr(tg_adapter, "Application", application)

    asyncio.run(adapter.connect())
    return list(_RecordingHTTPXRequest.instances)


def test_configured_media_write_timeout_is_passed_to_httpx_request(monkeypatch):
    requests = _build_requests(
        monkeypatch,
        telegram_config={"extra": {"media_write_timeout": 180.0}},
    )

    assert len(requests) == 2
    assert all(
        request.kwargs.get("media_write_timeout") == 180.0
        for request in requests
    )


def test_unset_media_write_timeout_uses_ptb_default(monkeypatch):
    requests = _build_requests(monkeypatch)

    assert len(requests) == 2
    assert all("media_write_timeout" not in request.kwargs for request in requests)
