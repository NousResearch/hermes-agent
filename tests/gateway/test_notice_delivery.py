from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import SendResult
from gateway.run import GatewayRunner
from gateway.session import SessionSource


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.SLACK,
        chat_id="C123",
        chat_type="channel",
        user_id="U123",
        thread_id="111.222",
    )


def _make_runner(extra=None):
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={
            Platform.SLACK: PlatformConfig(enabled=True, token="***", extra=extra or {})
        }
    )
    adapter = MagicMock()
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="public-1"))
    adapter.send_private_notice = AsyncMock(return_value=SendResult(success=True, message_id="private-1"))
    runner.adapters = {Platform.SLACK: adapter}
    return runner, adapter


@pytest.mark.asyncio
async def test_deliver_platform_notice_uses_private_delivery_when_configured():
    runner, adapter = _make_runner(extra={"notice_delivery": "private"})

    await runner._deliver_platform_notice(_make_source(), "hello")

    adapter.send_private_notice.assert_awaited_once_with(
        "C123",
        "U123",
        "hello",
        # user_id rides along since the R3-5 per-turn identity stamp (Slack
        # sources carry their author in thread metadata for the connector's
        # chat.startStream recipient fields). Additive and harmless here.
        metadata={"thread_id": "111.222", "user_id": "U123", "_interim_send": True},
    )
    adapter.send.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("private_fallback", [False, True])
async def test_operational_notice_is_not_a_final_reply(private_fallback):
    runner, adapter = _make_runner(
        extra={"notice_delivery": "private" if private_fallback else "public"})
    adapter.send_private_notice.return_value = SendResult(success=False, error="unavailable")
    source = _make_source()
    finals = []

    async def receive(chat_id, content, metadata=None):
        if not (metadata or {}).get("_interim_send"):
            finals.append(content)
        return SendResult(success=True, message_id="notice")

    adapter.send.side_effect = receive
    await runner._deliver_platform_notice(source, "Use /sethome to set your home channel.")
    adapter.send.assert_awaited_once_with(
        "C123", "Use /sethome to set your home channel.",
        metadata={"thread_id": "111.222", "user_id": "U123", "_interim_send": True},
    )
    assert finals == []
    await adapter.send("C123", "Timer set.", metadata={"thread_id": "111.222"})
    assert finals == ["Timer set."]
