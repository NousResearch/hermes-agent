"""Regression tests for gateway Telegram adapter HTTP timeout configuration.

Covers that ``request_kwargs`` passed to HTTPXRequest includes
``media_write_timeout`` so large media uploads (send_video, send_document)
respect HERMES_TELEGRAM_HTTP_MEDIA_WRITE_TIMEOUT instead of silently
falling back to python-telegram-bot's 20s default. See issue #21757.
"""

import sys
from unittest.mock import MagicMock

import pytest

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

    telegram_mod.error.NetworkError = type("NetworkError", (OSError,), {})
    telegram_mod.error.TimedOut = type("TimedOut", (OSError,), {})
    telegram_mod.error.BadRequest = type("BadRequest", (Exception,), {})

    for name in ("telegram", "telegram.ext", "telegram.constants", "telegram.request"):
        sys.modules.setdefault(name, telegram_mod)
    sys.modules.setdefault("telegram.error", telegram_mod.error)


_ensure_telegram_mock()

from gateway.platforms.telegram import TelegramAdapter  # noqa: E402


@pytest.mark.asyncio
async def test_connect_passes_media_write_timeout_to_httpxrequest(monkeypatch):
    """connect() must include media_write_timeout in request_kwargs.

    Without this key, python-telegram-bot's HTTPXRequest defaults to 20.0s
    for media uploads, ignoring HERMES_TELEGRAM_HTTP_MEDIA_WRITE_TIMEOUT.
    """
    captured: list[dict] = []

    def _capture_httpx(**kwargs):
        captured.append(kwargs)
        return MagicMock()

    async def _noop():
        return []

    monkeypatch.setenv("HERMES_TELEGRAM_DISABLE_FALLBACK_IPS", "1")
    monkeypatch.setattr("gateway.platforms.telegram.discover_fallback_ips", _noop)
    monkeypatch.setattr("gateway.platforms.telegram.HTTPXRequest", _capture_httpx)
    monkeypatch.setattr(
        "gateway.status.acquire_scoped_lock",
        lambda scope, identity, metadata=None: (True, None),
    )

    builder_mock = MagicMock()
    builder_mock.request.return_value = builder_mock
    builder_mock.get_updates_request.return_value = builder_mock
    built = MagicMock()
    built.bot = MagicMock()
    builder_mock.build.return_value = built
    app_mock = MagicMock()
    app_mock.builder.return_value = builder_mock
    monkeypatch.setattr("gateway.platforms.telegram.Application", app_mock)

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    await adapter.connect()

    assert captured, "HTTPXRequest was never called"
    kwargs = captured[0]
    assert "media_write_timeout" in kwargs, (
        "media_write_timeout missing from request_kwargs — large media uploads "
        "will time out at the 20s default even when "
        "HERMES_TELEGRAM_HTTP_MEDIA_WRITE_TIMEOUT is set"
    )
    assert kwargs["media_write_timeout"] == pytest.approx(300.0)


@pytest.mark.asyncio
async def test_connect_media_write_timeout_respects_env(monkeypatch):
    """HERMES_TELEGRAM_HTTP_MEDIA_WRITE_TIMEOUT env var is forwarded to HTTPXRequest."""
    captured: list[dict] = []

    def _capture_httpx(**kwargs):
        captured.append(kwargs)
        return MagicMock()

    async def _noop():
        return []

    monkeypatch.setenv("HERMES_TELEGRAM_DISABLE_FALLBACK_IPS", "1")
    monkeypatch.setenv("HERMES_TELEGRAM_HTTP_MEDIA_WRITE_TIMEOUT", "600")
    monkeypatch.setattr("gateway.platforms.telegram.discover_fallback_ips", _noop)
    monkeypatch.setattr("gateway.platforms.telegram.HTTPXRequest", _capture_httpx)
    monkeypatch.setattr(
        "gateway.status.acquire_scoped_lock",
        lambda scope, identity, metadata=None: (True, None),
    )

    builder_mock = MagicMock()
    builder_mock.request.return_value = builder_mock
    builder_mock.get_updates_request.return_value = builder_mock
    built = MagicMock()
    built.bot = MagicMock()
    builder_mock.build.return_value = built
    app_mock = MagicMock()
    app_mock.builder.return_value = builder_mock
    monkeypatch.setattr("gateway.platforms.telegram.Application", app_mock)

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    await adapter.connect()

    assert captured
    assert captured[0]["media_write_timeout"] == pytest.approx(600.0)
