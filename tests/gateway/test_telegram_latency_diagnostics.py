"""Opt-in Telegram ingress and typing latency diagnostics."""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent
from gateway.session import SessionSource
from plugins.platforms.telegram import adapter as adapter_module
from plugins.platforms.telegram.adapter import TelegramAdapter


def _make_adapter() -> TelegramAdapter:
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter._bot = AsyncMock()
    return adapter


def _event(*, timestamp: datetime) -> MessageEvent:
    return MessageEvent(
        text="private message text",
        timestamp=timestamp,
        platform_update_id=77,
        message_id="42",
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="123456789"),
        metadata={},
    )


def test_message_age_ms_handles_naive_utc_and_clamps_clock_skew() -> None:
    now = datetime(2026, 9, 4, 8, 0, 0, tzinfo=timezone.utc)

    assert (
        adapter_module._telegram_message_age_ms(
            now.replace(tzinfo=None) - timedelta(seconds=1.25), now=now
        )
        == 1250
    )
    assert (
        adapter_module._telegram_message_age_ms(now + timedelta(seconds=10), now=now)
        == 0
    )
    assert adapter_module._telegram_message_age_ms(None, now=now) is None


def test_chat_ref_is_stable_and_does_not_expose_raw_chat_id() -> None:
    first = adapter_module._telegram_chat_ref("123456789")

    assert first == adapter_module._telegram_chat_ref("123456789")
    assert first != adapter_module._telegram_chat_ref("987654321")
    assert "123456789" not in first


def test_inbound_diagnostics_are_disabled_by_default(monkeypatch, caplog) -> None:
    monkeypatch.delenv("HERMES_TELEGRAM_LATENCY_DIAGNOSTICS", raising=False)
    adapter = _make_adapter()

    with caplog.at_level(logging.INFO):
        adapter._log_inbound_latency(
            _event(timestamp=datetime.now(timezone.utc)), stage="adapter_received"
        )

    assert "TelegramLatency" not in caplog.text


def test_inbound_diagnostics_log_correlatable_sanitized_age(
    monkeypatch, caplog
) -> None:
    monkeypatch.setenv("HERMES_TELEGRAM_LATENCY_DIAGNOSTICS", "1")
    adapter = _make_adapter()
    event = _event(timestamp=datetime.now(timezone.utc) - timedelta(seconds=2))

    with caplog.at_level(logging.INFO):
        adapter._log_inbound_latency(event, stage="adapter_received")

    assert "TelegramLatency" in caplog.text
    assert "stage=adapter_received" in caplog.text
    assert "update_id=77" in caplog.text
    assert "message_id=42" in caplog.text
    assert "platform_age_ms=" in caplog.text
    assert "chat_ref=" in caplog.text
    assert "123456789" not in caplog.text
    assert event.text not in caplog.text
    assert event.metadata == {}
    assert hasattr(event, "_telegram_latency_received_monotonic")


@pytest.mark.asyncio
async def test_gateway_dispatch_diagnostic_preserves_base_behavior(
    monkeypatch, caplog
) -> None:
    monkeypatch.setenv("HERMES_TELEGRAM_LATENCY_DIAGNOSTICS", "true")
    adapter = _make_adapter()
    event = _event(timestamp=datetime.now(timezone.utc) - timedelta(seconds=1))

    adapter._log_inbound_latency(event, stage="adapter_received")
    base_handle = AsyncMock()
    with (
        caplog.at_level(logging.INFO),
        patch.object(BasePlatformAdapter, "handle_message", base_handle),
    ):
        await adapter.handle_message(event)

    assert "stage=gateway_dispatch" in caplog.text
    assert "adapter_queue_ms=" in caplog.text
    base_handle.assert_awaited_once_with(event)


@pytest.mark.asyncio
async def test_typing_success_logs_duration_when_enabled(monkeypatch, caplog) -> None:
    monkeypatch.setenv("HERMES_TELEGRAM_LATENCY_DIAGNOSTICS", "yes")
    adapter = _make_adapter()

    with caplog.at_level(logging.INFO):
        await adapter.send_typing("123456789")

    assert "TelegramLatency" in caplog.text
    assert "operation=send_chat_action" in caplog.text
    assert "outcome=success" in caplog.text
    assert "duration_ms=" in caplog.text
    assert "123456789" not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("bot_connected", "cooldown", "reason"),
    (
        (False, False, "not_connected"),
        (True, True, "cooldown"),
    ),
)
async def test_typing_skip_reason_is_visible_when_enabled(
    monkeypatch, caplog, bot_connected, cooldown, reason
) -> None:
    monkeypatch.setenv("HERMES_TELEGRAM_LATENCY_DIAGNOSTICS", "1")
    adapter = _make_adapter()
    if not bot_connected:
        adapter._bot = None
    monkeypatch.setattr(adapter, "_typing_in_cooldown", lambda _chat_id: cooldown)

    with caplog.at_level(logging.INFO):
        await adapter.send_typing("123456789")

    assert "outcome=skipped" in caplog.text
    assert f"reason={reason}" in caplog.text
    assert "123456789" not in caplog.text


@pytest.mark.asyncio
async def test_typing_failure_logs_redacted_error_when_enabled(
    monkeypatch, caplog
) -> None:
    monkeypatch.setenv("HERMES_TELEGRAM_LATENCY_DIAGNOSTICS", "on")
    adapter = _make_adapter()
    secret = "123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghi"
    adapter._bot.send_chat_action = AsyncMock(
        side_effect=RuntimeError(
            f"request failed https://api.telegram.org/bot{secret}/sendChatAction"
        )
    )

    with caplog.at_level(logging.INFO):
        await adapter.send_typing("987654321")

    assert "outcome=failure" in caplog.text
    assert secret not in caplog.text
    assert "987654321" not in caplog.text


@pytest.mark.asyncio
async def test_typing_fallback_failure_logs_the_final_error(
    monkeypatch, caplog
) -> None:
    monkeypatch.setenv("HERMES_TELEGRAM_LATENCY_DIAGNOSTICS", "1")
    adapter = _make_adapter()
    adapter._bot.send_chat_action = AsyncMock(
        side_effect=(
            ValueError("primary thread error"),
            RuntimeError("fallback transport error"),
        )
    )

    with caplog.at_level(logging.INFO):
        await adapter.send_typing(
            "987654321",
            metadata={"thread_id": "99", "telegram_dm_topic_reply_fallback": True},
        )

    assert "outcome=failure" in caplog.text
    assert "fallback transport error" in caplog.text
    assert "primary thread error" not in caplog.text


@pytest.mark.asyncio
async def test_typing_fallback_cancellation_is_logged_and_propagated(
    monkeypatch, caplog
) -> None:
    monkeypatch.setenv("HERMES_TELEGRAM_LATENCY_DIAGNOSTICS", "1")
    adapter = _make_adapter()
    adapter._bot.send_chat_action = AsyncMock(
        side_effect=(
            ValueError("primary thread error"),
            asyncio.CancelledError(),
        )
    )

    with caplog.at_level(logging.INFO), pytest.raises(asyncio.CancelledError):
        await adapter.send_typing(
            "987654321",
            metadata={"thread_id": "99", "telegram_dm_topic_reply_fallback": True},
        )

    assert "outcome=cancelled" in caplog.text
    assert "duration_ms=" in caplog.text
    assert "987654321" not in caplog.text


@pytest.mark.asyncio
async def test_typing_cancellation_is_logged_and_propagated(
    monkeypatch, caplog
) -> None:
    monkeypatch.setenv("HERMES_TELEGRAM_LATENCY_DIAGNOSTICS", "1")
    adapter = _make_adapter()
    adapter._bot.send_chat_action = AsyncMock(side_effect=asyncio.CancelledError())

    with caplog.at_level(logging.INFO), pytest.raises(asyncio.CancelledError):
        await adapter.send_typing("123456789")

    assert "outcome=cancelled" in caplog.text
