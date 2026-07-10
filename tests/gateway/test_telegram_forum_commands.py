"""Tests for lazy forum command registration in TelegramAdapter."""

import asyncio
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
    adapter = _make_test_adapter()
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


@pytest.mark.asyncio
async def test_ensure_forum_commands_handles_set_failure():
    adapter = _make_test_adapter()
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
    adapter = _make_test_adapter()
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


# ---------------------------------------------------------------------------
# Proactive per-chat scope refresh for configured forum groups (#<issue>)
# ---------------------------------------------------------------------------


def test_configured_forum_chat_ids_list_shape():
    adapter = _make_test_adapter()
    adapter.config.extra["group_topics"] = [
        {"chat_id": "-1001", "topics": [{"thread_id": 3, "name": "A"}]},
        {"chat_id": -1002, "topics": []},
        {"no_chat_id": True},  # ignored
    ]
    assert adapter._configured_forum_chat_ids() == {-1001, -1002}


def test_configured_forum_chat_ids_dict_shape():
    adapter = _make_test_adapter()
    adapter.config.extra["group_topics"] = {
        "-1001": [{"thread_id": 3}],
        "-1002": [],
        "not-an-int": [],  # ignored (unparseable)
    }
    assert adapter._configured_forum_chat_ids() == {-1001, -1002}


def test_configured_forum_chat_ids_absent():
    adapter = _make_test_adapter()
    assert adapter._configured_forum_chat_ids() == set()


@pytest.mark.asyncio
async def test_register_configured_forum_scopes_pushes_per_chat():
    """Configured forum groups get their per-chat command scope set at connect
    time — without waiting for a message — so their menus stay in sync with the
    global scopes when the command set changes."""
    adapter = _make_test_adapter()
    adapter.config.extra["group_topics"] = [
        {"chat_id": "-1001", "topics": []},
        {"chat_id": "-1002", "topics": []},
    ]
    # Opaque sentinel list; the adapter must pass it through to set_my_commands
    # verbatim (the exact command objects are built by the caller).
    bot_commands = [object(), object()]

    # ``telegram`` is a sys.modules mock here (see conftest); track the chat_id
    # passed to BotCommandScopeChat so assertions see an int, not a bare mock.
    with patch("telegram.BotCommandScopeChat") as MockScope:
        def _make_scope(chat_id):
            s = MagicMock()
            s.chat_id = chat_id
            return s
        MockScope.side_effect = _make_scope
        await adapter._register_configured_forum_command_scopes(bot_commands)

    # One set_my_commands per configured forum chat, each carrying the full
    # command list under a chat-scoped BotCommandScopeChat.
    assert adapter._bot.set_my_commands.await_count == 2
    scoped_chat_ids = set()
    for call in adapter._bot.set_my_commands.await_args_list:
        assert call.args[0] is bot_commands
        scoped_chat_ids.add(call.kwargs["scope"].chat_id)
    assert scoped_chat_ids == {-1001, -1002}
    # Registering here suppresses redundant lazy registration later.
    assert adapter._forum_command_registered == {-1001, -1002}


@pytest.mark.asyncio
async def test_register_configured_forum_scopes_noop_without_config():
    adapter = _make_test_adapter()
    await adapter._register_configured_forum_command_scopes([])
    adapter._bot.set_my_commands.assert_not_called()
