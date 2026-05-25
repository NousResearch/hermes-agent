"""Tests for lazy forum command registration in TelegramAdapter."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform, PlatformConfig


def _make_test_adapter(extra=None):
    """Build a TelegramAdapter without running __init__."""
    from gateway.platforms.telegram import TelegramAdapter

    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter.config = PlatformConfig(enabled=True, token="***", extra=extra or {})
    # ``name`` is a property derived from platform.value.title()
    adapter._bot = MagicMock()
    adapter._bot.set_my_commands = AsyncMock()
    adapter._forum_command_registered = set()
    adapter._forum_lock = asyncio.Lock()
    return adapter


def _forum_message(chat_id=-100, is_forum=True):
    return SimpleNamespace(
        chat=SimpleNamespace(id=chat_id, is_forum=is_forum),
    )


@pytest.mark.asyncio
async def test_ensure_forum_commands_skips_non_forum():
    adapter = _make_test_adapter()
    msg = _forum_message(is_forum=False)
    await adapter._ensure_forum_commands(msg)
    adapter._bot.set_my_commands.assert_not_called()


@pytest.mark.asyncio
async def test_ensure_forum_commands_skips_already_registered():
    adapter = _make_test_adapter()
    adapter._forum_command_registered.add(-100)
    msg = _forum_message(is_forum=True)
    await adapter._ensure_forum_commands(msg)
    adapter._bot.set_my_commands.assert_not_called()


@pytest.mark.asyncio
async def test_ensure_forum_commands_registers_once():
    adapter = _make_test_adapter({"slash_commands": {"mode": "all"}})
    msg = _forum_message(chat_id=-123, is_forum=True)

    with patch("hermes_cli.commands.telegram_menu_commands") as mock_menu:
        mock_menu.return_value = ([("new", "Start new session"), ("help", "Show help")], 0)
        with patch("telegram.BotCommand") as MockBotCommand:
            instances = []

            def _make_cmd(name, desc):
                cmd = MagicMock()
                cmd.name = name
                cmd.description = desc
                instances.append(cmd)
                return cmd

            MockBotCommand.side_effect = _make_cmd
            with patch("telegram.BotCommandScopeChat") as MockScope:
                # Track the chat_id passed to the BotCommandScopeChat constructor
                # so the assertions below see an int instead of a bare MagicMock.
                def _make_scope(chat_id):
                    s = MagicMock()
                    s.chat_id = chat_id
                    return s
                MockScope.side_effect = _make_scope
                await adapter._ensure_forum_commands(msg)

    assert -123 in adapter._forum_command_registered
    adapter._bot.set_my_commands.assert_awaited_once()
    args, kwargs = adapter._bot.set_my_commands.call_args
    assert len(args[0]) == 2  # two BotCommand instances
    assert kwargs["scope"] is not None
    assert isinstance(kwargs["scope"].chat_id, int)
    assert kwargs["scope"].chat_id == -123


def test_telegram_menu_entries_for_modes():
    adapter = _make_test_adapter()

    with patch("hermes_cli.commands.telegram_menu_commands") as mock_menu:
        mock_menu.return_value = ([("help", "Show help"), ("restart", "Restart")], 0)
        assert adapter._telegram_menu_entries_for_mode("none") == ([], 0)
        assert adapter._telegram_menu_entries_for_mode("minimal")[0] == [("help", "Show available commands")]
        assert adapter._telegram_menu_entries_for_mode("persona")[0] == [
            ("help", "Show available commands"),
            ("status", "Show session info"),
            ("new", "Start a new session (fresh session ID + history)"),
        ]
        assert adapter._telegram_menu_entries_for_mode("all") == (
            [("help", "Show help"), ("restart", "Restart")],
            0,
        )


@pytest.mark.asyncio
async def test_sync_bot_commands_registers_broad_and_user_overrides():
    adapter = _make_test_adapter(
        {
            "slash_commands": {
                "mode": "none",
                "users": {"123": "all", "*": "minimal", "not-an-id": "persona"},
            }
        }
    )

    with patch("hermes_cli.commands.telegram_menu_commands") as mock_menu:
        mock_menu.return_value = ([("restart", "Restart")], 0)
        with patch("telegram.BotCommand") as MockBotCommand:
            def _make_cmd(name, desc):
                cmd = MagicMock()
                cmd.name = name
                cmd.description = desc
                return cmd

            MockBotCommand.side_effect = _make_cmd
            with patch("telegram.BotCommandScopeChat") as MockScope:
                def _make_scope(chat_id):
                    s = MagicMock()
                    s.chat_id = chat_id
                    return s

                MockScope.side_effect = _make_scope
                await adapter._sync_bot_commands()

    assert adapter._bot.set_my_commands.await_count == 4
    calls = adapter._bot.set_my_commands.await_args_list
    # Wildcard/minimal mode controls the three broad scopes.
    assert all([cmd.name for cmd in call.args[0]] == ["help"] for call in calls[:3])
    # Explicit user 123 gets all-mode commands in a chat-specific scope.
    assert [cmd.name for cmd in calls[3].args[0]] == ["restart"]
    assert calls[3].kwargs["scope"].chat_id == 123


@pytest.mark.asyncio
async def test_ensure_forum_commands_uses_default_mode():
    adapter = _make_test_adapter({"slash_commands": {"mode": "minimal"}})
    msg = _forum_message(chat_id=-321, is_forum=True)

    with patch("telegram.BotCommand") as MockBotCommand:
        def _make_cmd(name, desc):
            cmd = MagicMock()
            cmd.name = name
            cmd.description = desc
            return cmd

        MockBotCommand.side_effect = _make_cmd
        with patch("telegram.BotCommandScopeChat") as MockScope:
            MockScope.side_effect = lambda chat_id: SimpleNamespace(chat_id=chat_id)
            await adapter._ensure_forum_commands(msg)

    args, kwargs = adapter._bot.set_my_commands.call_args
    assert [cmd.name for cmd in args[0]] == ["help"]
    assert kwargs["scope"].chat_id == -321


@pytest.mark.asyncio
async def test_ensure_forum_commands_handles_set_failure():
    adapter = _make_test_adapter({"slash_commands": {"mode": "all"}})
    msg = _forum_message(chat_id=-456, is_forum=True)
    adapter._bot.set_my_commands.side_effect = Exception("Telegram API error")

    with patch("hermes_cli.commands.telegram_menu_commands") as mock_menu:
        mock_menu.return_value = ([("new", "Start new session")], 0)
        # Should NOT raise despite the API error
        await adapter._ensure_forum_commands(msg)

    # On failure we don't retry for this chat, so it's added to the set
    # to avoid hammering a broken chat.
    assert -456 not in adapter._forum_command_registered


@pytest.mark.asyncio
async def test_ensure_forum_commands_race_safety():
    """Two concurrent coroutines must not double-register the same chat."""
    adapter = _make_test_adapter({"slash_commands": {"mode": "all"}})
    msg = _forum_message(chat_id=-789, is_forum=True)

    with patch("hermes_cli.commands.telegram_menu_commands") as mock_menu:
        mock_menu.return_value = ([("new", "Start new session")], 0)
        with patch("telegram.BotCommand"):
            with patch("telegram.BotCommandScopeChat"):
                coro1 = adapter._ensure_forum_commands(msg)
                coro2 = adapter._ensure_forum_commands(msg)
                await asyncio.gather(coro1, coro2)

    # The lock should make this exactly 1 call, not 2.
    assert adapter._bot.set_my_commands.await_count == 1
