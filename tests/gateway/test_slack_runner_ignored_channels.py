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


def test_slack_ignored_channels_env_bridge_fallback(monkeypatch):
    """SLACK_IGNORED_CHANNELS (set by the plugin's YAML→env bridge) is
    honored when PlatformConfig.extra carries no ignored_channels (#46925)."""
    monkeypatch.setenv("SLACK_IGNORED_CHANNELS", "C_ENV1, C_ENV2")
    config = _config_with_slack_extra({})

    assert _slack_ignored_channels_from_gateway_config(config) == {"C_ENV1", "C_ENV2"}
    assert _is_slack_ignored_channel(config, "C_ENV1")
    assert not _is_slack_ignored_channel(config, "C_OTHER")


def test_slack_ignored_channels_extra_wins_over_env(monkeypatch):
    """Explicit PlatformConfig.extra config takes precedence over the env
    bridge fallback."""
    monkeypatch.setenv("SLACK_IGNORED_CHANNELS", "C_ENV")
    config = _config_with_slack_extra({"ignored_channels": ["C_CFG"]})

    assert _slack_ignored_channels_from_gateway_config(config) == {"C_CFG"}
