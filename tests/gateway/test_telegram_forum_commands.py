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


@pytest.mark.asyncio
async def test_register_command_menu_builds_skill_menu_off_event_loop(monkeypatch):
    """A slow skill scan must not starve Telegram's reconnect event loop."""
    import telegram

    adapter = _make_test_adapter()
    scan_started = threading.Event()
    release_scan = threading.Event()
    loop_progressed = threading.Event()
    observed = {}

    def _blocking_menu(*, max_commands):
        assert max_commands == 60
        scan_started.set()
        release_scan.wait(timeout=1)
        observed["loop_progressed"] = loop_progressed.is_set()
        return [("help", "Show help"), ("skill", "Run a skill")], 0

    monkeypatch.setattr("hermes_cli.commands_platforms.telegram_menu_max_commands", lambda: 60)
    monkeypatch.setattr("hermes_cli.commands_platforms.telegram_menu_commands", _blocking_menu)
    monkeypatch.setattr(
        telegram, "BotCommand", lambda command, description: SimpleNamespace(command=command, description=description))
    # This releases a pre-fix synchronous scan so the regression test cannot hang the suite.
    safety_release = threading.Timer(0.2, release_scan.set)
    safety_release.start()
    task = asyncio.create_task(adapter._register_command_menu())
    try:
        await asyncio.to_thread(scan_started.wait)
        loop_progressed.set()
        release_scan.set()
        await task
    finally:
        release_scan.set()
        safety_release.cancel()
        if not task.done():
            task.cancel()
            await task

    assert observed["loop_progressed"] is True
    assert adapter._bot.set_my_commands.await_count == 3
    for call in adapter._bot.set_my_commands.await_args_list:
        assert [command.command for command in call.args[0]] == ["help", "skill"]
