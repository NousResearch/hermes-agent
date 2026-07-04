import sys
from unittest.mock import AsyncMock, MagicMock

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
    for name in ("telegram", "telegram.ext", "telegram.constants", "telegram.request"):
        sys.modules.setdefault(name, telegram_mod)
    sys.modules.setdefault("telegram.error", telegram_mod.error)


_ensure_telegram_mock()

from plugins.platforms.telegram import adapter as tg_adapter  # noqa: E402
from plugins.platforms.telegram.adapter import TelegramAdapter  # noqa: E402


@pytest.mark.asyncio
async def test_connect_retries_when_initialize_wall_deadline_expires(monkeypatch):
    """A wedged initialize() attempt must not trap startup on attempt 1/8."""
    fake_app = MagicMock()
    fake_app.bot = MagicMock()
    fake_app.initialize = AsyncMock(return_value=None)
    fake_app.start = AsyncMock()
    fake_app.add_handler = MagicMock()

    chainable = MagicMock()
    chainable.token.return_value = chainable
    chainable.request.return_value = chainable
    chainable.get_updates_request.return_value = chainable
    chainable.build.return_value = fake_app

    builder_root = MagicMock()
    builder_root.builder.return_value = chainable
    monkeypatch.setattr(tg_adapter, "Application", builder_root)
    monkeypatch.setattr(tg_adapter, "HTTPXRequest", MagicMock)
    monkeypatch.setattr(tg_adapter, "discover_fallback_ips", AsyncMock(return_value=[]))
    monkeypatch.setattr(tg_adapter, "resolve_proxy_url", lambda *a, **k: None)
    monkeypatch.setattr(tg_adapter.asyncio, "sleep", AsyncMock())

    deadline_calls = 0

    async def _fake_deadline(awaitable, timeout):
        nonlocal deadline_calls
        deadline_calls += 1
        if deadline_calls == 1:
            awaitable.close()
            raise tg_adapter.asyncio.TimeoutError()
        return await awaitable

    monkeypatch.setattr(tg_adapter, "_await_with_thread_deadline", _fake_deadline)

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    monkeypatch.setattr(adapter, "_acquire_platform_lock", lambda *a, **k: True)
    monkeypatch.setattr(adapter, "_fallback_ips", lambda: [])
    monkeypatch.setattr(adapter, "_delete_webhook_best_effort", AsyncMock())
    monkeypatch.setattr(adapter, "_start_polling_resilient", AsyncMock(return_value=True))
    monkeypatch.setattr(adapter, "_polling_heartbeat_loop", AsyncMock(return_value=None))
    monkeypatch.setattr(adapter, "_start_post_connect_housekeeping", MagicMock())

    assert await adapter.connect() is True

    assert fake_app.initialize.call_count == 2
    assert fake_app.initialize.await_count == 1
    assert deadline_calls == 2
    tg_adapter.asyncio.sleep.assert_awaited_once_with(1)
    fake_app.start.assert_awaited_once()
