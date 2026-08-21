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
        metadata={"thread_id": "111.222", "user_id": "U123"},
    )
    adapter.send.assert_not_awaited()


@pytest.mark.parametrize("reset_reason", ["suspended", "resume_pending_expired"])
def test_reset_notice_platform_exclusion_is_absolute(reset_reason):
    policy = SessionResetPolicy(
        notify=True,
        notify_exclude_platforms=("email",),
    )

    assert not GatewayRunner._should_send_auto_reset_notice(
        "email", reset_reason, policy, had_activity=True
    )


def test_reset_notice_still_sends_on_included_chat_platform():
    policy = SessionResetPolicy(
        notify=True,
        notify_exclude_platforms=("email",),
    )

    assert GatewayRunner._should_send_auto_reset_notice(
        "telegram", "idle", policy, had_activity=True
    )


