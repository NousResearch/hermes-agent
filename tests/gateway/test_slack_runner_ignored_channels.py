import pytest
from unittest.mock import AsyncMock

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType, SendResult
from gateway.run import (
    GatewayRunner,
    _is_slack_ignored_channel,
    _slack_ignored_channels_from_gateway_config,
)
from gateway.session import SessionSource


def _config_with_slack_extra(extra=None):
    return GatewayConfig(
        platforms={
            Platform.SLACK: PlatformConfig(enabled=True, extra=extra or {}),
        }
    )


def test_slack_ignored_channels_from_config_list():
    config = _config_with_slack_extra({"ignored_channels": ["C_PRD", " C_OTHER ", ""]})

    assert _slack_ignored_channels_from_gateway_config(config) == {"C_PRD", "C_OTHER"}


def test_slack_ignored_channel_matches_thread_scoped_chat_id():
    config = _config_with_slack_extra({"ignored_channels": "C_PRD"})

    assert _is_slack_ignored_channel(config, "C_PRD")
    assert _is_slack_ignored_channel(config, "C_PRD:1782283787.899249")


def test_slack_ignored_channel_supports_wildcard():
    config = _config_with_slack_extra({"ignored_channels": "*"})

    assert _is_slack_ignored_channel(config, "C_ANY")


@pytest.mark.asyncio
async def test_runner_drops_slack_ignored_channel_before_auth_hooks_and_sessions(monkeypatch):
    runner = object.__new__(GatewayRunner)
    runner.config = _config_with_slack_extra({"ignored_channels": "C_PRD"})
    runner._startup_restore_in_progress = False

    # If the guard regresses, _handle_message will proceed into hooks/auth/session
    # setup and one of these sentinels will fail the test.
    runner.session_store = object()
    monkeypatch.setattr(
        "hermes_cli.plugins.invoke_hook",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("hook should not run")),
    )
    runner._is_user_authorized = lambda source: (_ for _ in ()).throw(AssertionError("auth should not run"))

    event = MessageEvent(
        text="<@U_BOT> review this PRD",
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.SLACK,
            user_id="U_USER",
            user_name="shubham",
            chat_id="C_PRD",
            chat_type="group",
        ),
    )

    assert await runner._handle_message(event) is None


@pytest.mark.asyncio
async def test_runner_drops_thread_scoped_slack_ignored_channel():
    runner = object.__new__(GatewayRunner)
    runner.config = _config_with_slack_extra({"ignored_channels": "C_PRD"})
    runner._startup_restore_in_progress = False
    runner._is_user_authorized = lambda source: (_ for _ in ()).throw(AssertionError("auth should not run"))
    runner.session_store = None

    event = MessageEvent(
        text="<@U_BOT> review this PRD",
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.SLACK,
            user_id="U_USER",
            user_name="shubham",
            chat_id="C_PRD:1782283787.899249",
            chat_type="group",
            thread_id="1782283787.899249",
        ),
    )

    assert await runner._handle_message(event) is None


@pytest.mark.asyncio
async def test_platform_notice_suppressed_for_slack_ignored_channel():
    runner = object.__new__(GatewayRunner)
    runner.config = _config_with_slack_extra({"ignored_channels": "C_PRD"})
    adapter = type("Adapter", (), {})()
    adapter.send = AsyncMock(return_value=SendResult(success=True))
    adapter.send_private_notice = AsyncMock(return_value=SendResult(success=True))
    runner.adapters = {Platform.SLACK: adapter}
    runner._thread_metadata_for_source = lambda source: {"thread_id": source.thread_id}

    source = SessionSource(
        platform=Platform.SLACK,
        user_id="U_USER",
        chat_id="C_PRD",
        chat_type="group",
        thread_id="1782283787.899249",
    )

    await runner._deliver_platform_notice(source, "No home channel is set for Slack")

    adapter.send.assert_not_called()
    adapter.send_private_notice.assert_not_called()
