"""
Tests for HTTP timeout wiring in gateway/platforms/telegram.py.

Verifies that the request_kwargs passed to PTB's ``HTTPXRequest`` honor
the ``HERMES_TELEGRAM_HTTP_*`` environment variables — most importantly
``HERMES_TELEGRAM_HTTP_MEDIA_WRITE_TIMEOUT``, which controls how long the
adapter will wait for multipart media uploads (``send_video``,
``send_document``, etc.) before giving up.

PTB 22.7's ``HTTPXRequest`` defaults ``media_write_timeout`` to 20s,
which is too short for typical video uploads on real-world uplinks.
The adapter overrides it to 300s by default and lets the env var bump
it higher.
"""

import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig


def _ensure_telegram_mock():
    """Install mock telegram modules so TelegramAdapter can be imported."""
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


@pytest.fixture
def _mock_connect_dependencies(monkeypatch):
    """Stub out network-discovery and lock acquisition so connect() proceeds
    far enough to construct HTTPXRequest, and then bails on the application
    builder (we don't care about anything past that)."""

    async def _no_fallback_ips():
        return []

    monkeypatch.setattr(
        "gateway.platforms.telegram.discover_fallback_ips",
        _no_fallback_ips,
    )
    monkeypatch.setattr(
        "gateway.status.acquire_scoped_lock",
        lambda scope, identity, metadata=None: (True, None),
    )
    monkeypatch.setattr(
        "gateway.status.release_scoped_lock",
        lambda scope, identity: None,
    )

    builder = MagicMock()
    builder.token.return_value = builder
    builder.request.return_value = builder
    builder.get_updates_request.return_value = builder
    builder.base_url.return_value = builder
    builder.base_file_url.return_value = builder
    builder.build.side_effect = RuntimeError("stop after HTTPXRequest construction")

    monkeypatch.setattr(
        "gateway.platforms.telegram.Application",
        SimpleNamespace(builder=MagicMock(return_value=builder)),
    )


def _capture_httpx_calls(monkeypatch):
    """Replace HTTPXRequest with a recorder; returns the call list."""
    calls = []

    def _recorder(**kwargs):
        calls.append(kwargs)
        return MagicMock()

    monkeypatch.setattr("gateway.platforms.telegram.HTTPXRequest", _recorder)
    return calls


@pytest.mark.asyncio
async def test_media_write_timeout_default_is_300s(
    monkeypatch, _mock_connect_dependencies
):
    """Without env override, media_write_timeout should default to 300s.

    PTB's library default is 20s, which fails for real-world media uploads.
    """
    monkeypatch.delenv("HERMES_TELEGRAM_HTTP_MEDIA_WRITE_TIMEOUT", raising=False)
    calls = _capture_httpx_calls(monkeypatch)

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    try:
        await adapter.connect()
    except RuntimeError:
        pass  # builder.build raises by design — we already captured kwargs

    assert calls, "HTTPXRequest was never called"
    assert calls[0]["media_write_timeout"] == 300.0
    # The other timeouts should still be set for parity / regression coverage
    assert calls[0]["write_timeout"] == 20.0
    assert calls[0]["read_timeout"] == 20.0
    assert calls[0]["connect_timeout"] == 10.0
    assert calls[0]["pool_timeout"] == 8.0


@pytest.mark.asyncio
async def test_media_write_timeout_env_override(
    monkeypatch, _mock_connect_dependencies
):
    """Setting HERMES_TELEGRAM_HTTP_MEDIA_WRITE_TIMEOUT must propagate."""
    monkeypatch.setenv("HERMES_TELEGRAM_HTTP_MEDIA_WRITE_TIMEOUT", "600")
    calls = _capture_httpx_calls(monkeypatch)

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    try:
        await adapter.connect()
    except RuntimeError:
        pass

    assert calls, "HTTPXRequest was never called"
    assert calls[0]["media_write_timeout"] == 600.0


@pytest.mark.asyncio
async def test_media_write_timeout_env_invalid_falls_back_to_default(
    monkeypatch, _mock_connect_dependencies
):
    """A non-float env value must not crash; it falls back to the default."""
    monkeypatch.setenv("HERMES_TELEGRAM_HTTP_MEDIA_WRITE_TIMEOUT", "not-a-number")
    calls = _capture_httpx_calls(monkeypatch)

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    try:
        await adapter.connect()
    except RuntimeError:
        pass

    assert calls, "HTTPXRequest was never called"
    assert calls[0]["media_write_timeout"] == 300.0
