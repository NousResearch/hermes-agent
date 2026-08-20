from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig, SessionResetPolicy
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


def _make_email_source() -> SessionSource:
    return SessionSource(
        platform=Platform.EMAIL,
        chat_id="person@example.com",
        chat_type="dm",
        user_id="person@example.com",
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
        metadata={"thread_id": "111.222"},
    )
    adapter.send.assert_not_awaited()


def test_standalone_reset_and_setup_notices_are_suppressed_for_email():
    """Email maps standalone notices to separate messages, unlike chat apps."""
    assert GatewayRunner._should_send_standalone_notice(_make_email_source()) is False
    assert GatewayRunner._should_send_standalone_notice(_make_source()) is True


def _make_auto_reset_runner():
    runner = object.__new__(GatewayRunner)
    adapter = MagicMock()
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="reset-1"))
    runner._adapter_for_source = MagicMock(return_value=adapter)
    runner._reset_notice_session_info = MagicMock(return_value=None)
    runner._thread_metadata_for_source = MagicMock(return_value={"thread_id": "111.222"})
    return runner, adapter


@pytest.mark.asyncio
async def test_auto_reset_notice_does_not_send_a_standalone_email():
    runner, adapter = _make_auto_reset_runner()

    await runner._deliver_auto_reset_notice(
        _make_email_source(),
        "idle",
        SessionResetPolicy(notify=True, idle_minutes=60),
        had_activity=True,
    )

    adapter.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_auto_reset_notice_still_sends_for_chat_platforms():
    runner, adapter = _make_auto_reset_runner()

    await runner._deliver_auto_reset_notice(
        _make_source(),
        "idle",
        SessionResetPolicy(notify=True, idle_minutes=60),
        had_activity=True,
    )

    adapter.send.assert_awaited_once()
