"""Tests for opt-in Telegram per-message latency/typing diagnostics (#102761).

Disabled by default; ``HERMES_TELEGRAM_LATENCY_DIAGNOSTICS=1`` turns on
INFO-level logs for (a) message age at adapter accept time and (b)
sendChatAction(typing) duration/outcome. Must not change any Telegram
processing behavior — only add logging.
"""

import logging
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig, Platform
from gateway.platforms.base import MessageType
from plugins.platforms.telegram.adapter import TelegramAdapter

_LOGGER_NAME = "plugins.platforms.telegram.adapter"


def _make_adapter():
    config = PlatformConfig(enabled=True, token="fake-token")
    adapter = object.__new__(TelegramAdapter)
    adapter.config = config
    adapter._config = config
    adapter._platform = Platform.TELEGRAM
    adapter.platform = Platform.TELEGRAM
    adapter._connected = True
    adapter._dm_topics = {}
    adapter._dm_topics_config = []
    adapter._reply_to_mode = "first"
    adapter._fallback_ips = []
    return adapter


def _make_message(date):
    import plugins.platforms.telegram.adapter as telegram_mod

    return SimpleNamespace(
        text="hi",
        caption=None,
        chat=SimpleNamespace(
            id=-100123,
            type=telegram_mod.ChatType.SUPERGROUP,
            is_forum=False,
            title="Group",
        ),
        from_user=SimpleNamespace(id=456, full_name="Alice"),
        message_thread_id=None,
        is_topic_message=False,
        reply_to_message=None,
        message_id=999,
        date=date,
    )


def test_diagnostics_disabled_by_default(monkeypatch, caplog):
    monkeypatch.delenv("HERMES_TELEGRAM_LATENCY_DIAGNOSTICS", raising=False)
    adapter = _make_adapter()
    message = _make_message(datetime.now(timezone.utc) - timedelta(seconds=5))

    with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
        adapter._build_message_event(message, msg_type=MessageType.TEXT, update_id=7)

    assert "message age" not in caplog.text


def test_message_age_logged_when_enabled(monkeypatch, caplog):
    monkeypatch.setenv("HERMES_TELEGRAM_LATENCY_DIAGNOSTICS", "1")
    adapter = _make_adapter()
    message = _make_message(datetime.now(timezone.utc) - timedelta(seconds=5))

    with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
        adapter._build_message_event(message, msg_type=MessageType.TEXT, update_id=7)

    matching = [r for r in caplog.records if "message age" in r.message]
    assert len(matching) == 1
    assert "update_id=7" in matching[0].message
    assert "message_id=999" in matching[0].message
    # age should be ~5s, not clamped, and not the raw message text.
    assert "hi" not in matching[0].message


def test_naive_timestamp_treated_as_utc(monkeypatch, caplog):
    monkeypatch.setenv("HERMES_TELEGRAM_LATENCY_DIAGNOSTICS", "1")
    adapter = _make_adapter()
    naive_now = (datetime.now(timezone.utc) - timedelta(seconds=2)).replace(tzinfo=None)
    message = _make_message(naive_now)

    with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
        adapter._build_message_event(message, msg_type=MessageType.TEXT, update_id=1)

    matching = [r for r in caplog.records if "message age" in r.message]
    assert len(matching) == 1


def test_future_timestamp_clamped_to_zero(monkeypatch, caplog):
    """Negative clock skew (platform timestamp in the future) must clamp to 0s."""
    monkeypatch.setenv("HERMES_TELEGRAM_LATENCY_DIAGNOSTICS", "1")
    adapter = _make_adapter()
    message = _make_message(datetime.now(timezone.utc) + timedelta(seconds=30))

    with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
        adapter._build_message_event(message, msg_type=MessageType.TEXT, update_id=2)

    matching = [r for r in caplog.records if "message age" in r.message]
    assert len(matching) == 1
    assert "0.000s" in matching[0].message


@pytest.mark.asyncio
async def test_typing_diagnostics_disabled_by_default(monkeypatch, caplog):
    monkeypatch.delenv("HERMES_TELEGRAM_LATENCY_DIAGNOSTICS", raising=False)
    adapter = _make_adapter()
    adapter._bot = AsyncMock()
    adapter._telegram_typing_cooldown_until = {}
    adapter._telegram_typing_cooldown_seconds = 30.0

    with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
        await adapter.send_typing("123")

    assert "sendChatAction" not in caplog.text


@pytest.mark.asyncio
async def test_typing_success_logs_duration_when_enabled(monkeypatch, caplog):
    monkeypatch.setenv("HERMES_TELEGRAM_LATENCY_DIAGNOSTICS", "1")
    adapter = _make_adapter()
    adapter._bot = AsyncMock()
    adapter._telegram_typing_cooldown_until = {}
    adapter._telegram_typing_cooldown_seconds = 30.0

    with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
        await adapter.send_typing("123")

    matching = [r for r in caplog.records if "sendChatAction" in r.message]
    assert len(matching) == 1
    assert "succeeded" in matching[0].message
    assert "chat_id=123" in matching[0].message


_SECRET_TOKEN = "123456789:AAFakeSecretTelegramBotTokenABCDEFGHIJ"


@pytest.mark.asyncio
async def test_typing_failure_logs_sanitized_error_when_enabled(monkeypatch, caplog):
    monkeypatch.setenv("HERMES_TELEGRAM_LATENCY_DIAGNOSTICS", "1")
    adapter = _make_adapter()
    adapter._bot = AsyncMock(
        send_chat_action=AsyncMock(
            side_effect=RuntimeError(
                f"https://api.telegram.org/bot{_SECRET_TOKEN}/sendChatAction failed"
            )
        )
    )
    adapter._telegram_typing_cooldown_until = {}
    adapter._telegram_typing_cooldown_seconds = 30.0

    with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
        await adapter.send_typing("123")

    matching = [r for r in caplog.records if "sendChatAction" in r.message]
    assert len(matching) == 1
    assert "failed" in matching[0].message
    assert _SECRET_TOKEN not in matching[0].message
