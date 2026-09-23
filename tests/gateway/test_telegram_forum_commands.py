"""Tests for lazy forum command registration in TelegramAdapter."""

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform, PlatformConfig


def _make_test_adapter():
    """Build a TelegramAdapter without running __init__."""
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter.config = PlatformConfig(enabled=True, token="***", extra={})
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
async def test_ensure_forum_commands_registers_once():
    adapter = _make_test_adapter()
    msg = _forum_message(chat_id=-123, is_forum=True)

    with patch("hermes_cli.commands_platforms.telegram_menu_commands") as mock_menu:
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


@pytest.mark.asyncio
async def test_ensure_forum_commands_race_safety():
    """Two concurrent coroutines must not double-register the same chat."""
    adapter = _make_test_adapter()
    msg = _forum_message(chat_id=-789, is_forum=True)

    with patch("hermes_cli.commands_platforms.telegram_menu_commands") as mock_menu:
        mock_menu.return_value = ([("new", "Start new session")], 0)
        with patch("telegram.BotCommand"):
            with patch("telegram.BotCommandScopeChat"):
                coro1 = adapter._ensure_forum_commands(msg)
                coro2 = adapter._ensure_forum_commands(msg)
                await asyncio.gather(coro1, coro2)

    # The lock should make this exactly 1 call, not 2.
    assert adapter._bot.set_my_commands.await_count == 1


async def _menu_build_leaves_loop_free(adapter, run_site):
    """Drive ``run_site`` while the menu builder blocks; return whether the loop kept ticking."""
    scan_started = threading.Event()
    loop_ticked = threading.Event()

    def _blocking_menu(*, max_commands):
        scan_started.set()
        return ([("help", "Show help")], 0) if loop_ticked.wait(timeout=1) else ([], 0)

    with patch("hermes_cli.commands_platforms.telegram_menu_commands", _blocking_menu), \
            patch("hermes_cli.commands_platforms.telegram_menu_max_commands", lambda: 60), \
            patch("telegram.BotCommand", lambda c, d: SimpleNamespace(command=c, description=d)), \
            patch("telegram.BotCommandScopeChat", lambda chat_id: SimpleNamespace(chat_id=chat_id)):
        task = asyncio.create_task(run_site())
        await asyncio.to_thread(scan_started.wait, 1)
        # This line only runs while the builder is still inside the wait if the loop is free.
        loop_ticked.set()
        await task
    return all(
        [c.command for c in call.args[0]] == ["help"] for call in adapter._bot.set_my_commands.await_args_list
    ) and adapter._bot.set_my_commands.await_count > 0


@pytest.mark.asyncio
async def test_command_menu_skill_scan_runs_off_the_event_loop():
    """A slow skill scan (#110707) must not block the gateway loop from either menu site:
    post-connect registration and lazy forum registration on an inbound message."""
    adapter = _make_test_adapter()
    assert await _menu_build_leaves_loop_free(adapter, adapter._register_command_menu)
    adapter = _make_test_adapter()
    forum_msg = _forum_message(chat_id=-123, is_forum=True)
    assert await _menu_build_leaves_loop_free(adapter, lambda: adapter._ensure_forum_commands(forum_msg))
