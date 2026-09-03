"""Cron standalone delivery retries only proven pre-connect DNS failures."""

from __future__ import annotations

import socket
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from cron.scheduler import _deliver_result
from gateway.config import Platform
from plugins.platforms.telegram import adapter as _telegram_adapter  # noqa: F401


def _gateway_config():
    platform_config = SimpleNamespace(enabled=True, token="test-token", extra={})
    return SimpleNamespace(platforms={Platform.TELEGRAM: platform_config})


def _job():
    return {
        "id": "dns-retry",
        "name": "dns-retry",
        "deliver": "origin",
        "origin": {"platform": "telegram", "chat_id": "123"},
    }


def test_cron_delivery_retries_gaierror_before_connect():
    bot = SimpleNamespace(send_message=AsyncMock())
    bot.send_message.side_effect = [
        socket.gaierror(socket.EAI_AGAIN, "Temporary failure in name resolution"),
        socket.gaierror(socket.EAI_AGAIN, "Temporary failure in name resolution"),
        SimpleNamespace(message_id=42),
    ]

    with (
        patch("gateway.config.load_gateway_config", return_value=_gateway_config()),
        patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}),
        patch("telegram.Bot", return_value=bot),
        patch("tools.send_message_tool.asyncio.sleep", new=AsyncMock()),
    ):
        result = _deliver_result(_job(), "daily report")

    assert result is None
    assert bot.send_message.await_count == 3


def test_cron_delivery_does_not_retry_ambiguous_connection_error():
    bot = SimpleNamespace(
        send_message=AsyncMock(side_effect=ConnectionError("connection reset"))
    )

    with (
        patch("gateway.config.load_gateway_config", return_value=_gateway_config()),
        patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}),
        patch("telegram.Bot", return_value=bot),
        patch("tools.send_message_tool.asyncio.sleep", new=AsyncMock()) as sleep,
    ):
        result = _deliver_result(_job(), "daily report")

    assert "Telegram send failed" in result
    assert bot.send_message.await_count == 1
    sleep.assert_not_awaited()
