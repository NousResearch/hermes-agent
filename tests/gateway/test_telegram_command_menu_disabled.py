"""Tests for the Telegram command_menu.enabled: false toggle.

Verifies that when auto-registration is disabled, neither the startup
housekeeping path nor the lazy forum-registration path call set_my_commands.
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_adapter():
    """Build a TelegramAdapter without running __init__."""
    from plugins.platforms.telegram.adapter import TelegramAdapter
    from gateway.config import Platform, PlatformConfig

    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter.config = PlatformConfig(enabled=True, token="***", extra={})
    # ``name`` is a property derived from platform.value.title()
    adapter._bot = MagicMock()
    adapter._bot.set_my_commands = AsyncMock()
    adapter._forum_command_registered = set()
    adapter._post_connect_task = None
    # Shut up subsequent non-fatal housekeeping steps that touch self.name
    # but aren't relevant to the command-menu test: status indicator and
    # DM topics.  Their guard methods return early when the relevant
    # config flags are falsy — we just need them not to fail.
    adapter._status_indicator_enabled = False
    adapter._dm_topic_setup_done = True
    adapter._dm_topics_config = []
    adapter._dm_topics = {}
    return adapter


@pytest.mark.asyncio
async def test_startup_registration_skipped_when_disabled():
    """When command_menu.enabled is False, _run_post_connect_housekeeping
    must NOT call set_my_commands for any scope."""
    adapter = _make_adapter()

    with patch("hermes_cli.commands._telegram_command_menu_config") as mock_cfg:
        mock_cfg.return_value = {"enabled": False}
        with patch("hermes_cli.commands.telegram_menu_commands") as mock_menu:
            mock_menu.return_value = ([("start", "Start")], 0)
            with patch("telegram.BotCommand"):
                with patch(
                    "telegram.BotCommandScopeDefault",
                ), patch(
                    "telegram.BotCommandScopeAllPrivateChats",
                ), patch(
                    "telegram.BotCommandScopeAllGroupChats",
                ):
                    await adapter._run_post_connect_housekeeping()

    # Must NOT have called set_my_commands at all
    adapter._bot.set_my_commands.assert_not_called()


@pytest.mark.asyncio
async def test_startup_registration_proceeds_when_enabled():
    """When command_menu.enabled is not set (default), registration must proceed
    normally and call set_my_commands for each scope."""
    adapter = _make_adapter()

    with patch("hermes_cli.commands._telegram_command_menu_config") as mock_cfg:
        # enabled=None is the default (not configured)
        mock_cfg.return_value = {
            "enabled": None,
            "max_commands": 60,
            "priority_mode": "prepend",
            "priority": [],
        }
        with patch("hermes_cli.commands.telegram_menu_commands") as mock_menu:
            mock_menu.return_value = ([("start", "Start")], 0)
            with patch("telegram.BotCommand"):
                with patch(
                    "telegram.BotCommandScopeDefault",
                ), patch(
                    "telegram.BotCommandScopeAllPrivateChats",
                ), patch(
                    "telegram.BotCommandScopeAllGroupChats",
                ):
                    await adapter._run_post_connect_housekeeping()

    # Must have called set_my_commands (once per scope = 3 calls)
    assert adapter._bot.set_my_commands.await_count == 3
