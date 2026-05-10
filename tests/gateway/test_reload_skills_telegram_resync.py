"""Tests for `/reload-skills` refreshing the Telegram command menu."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gateway.config import Platform
from gateway.run import GatewayRunner, MessageEvent


class DefaultScope:
    pass


class ChatScope:
    def __init__(self, chat_id):
        self.chat_id = chat_id


class PrivateScope:
    pass


class GroupScope:
    pass


def _make_adapter():
    """Construct a TelegramAdapter without __init__ side effects."""
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter._bot = SimpleNamespace(set_my_commands=AsyncMock())
    adapter._forum_command_registered = set()
    adapter._forum_lock = asyncio.Lock()
    return adapter


class TestTelegramReloadSkillsResync:
    @pytest.mark.asyncio
    async def test_refresh_skill_group_registers_current_menu_for_startup_scopes(self):
        adapter = _make_adapter()

        with patch(
            "hermes_cli.commands.telegram_menu_max_commands",
            return_value=60,
        ), patch(
            "hermes_cli.commands.telegram_menu_commands",
            return_value=([("start", "Start"), ("example_skill", "Example skill")], 0),
        ), patch(
            "telegram.BotCommand",
            side_effect=lambda command, description: (command, description),
        ), patch(
            "telegram.BotCommandScopeDefault",
            new=DefaultScope,
        ), patch(
            "telegram.BotCommandScopeAllPrivateChats",
            new=PrivateScope,
        ), patch(
            "telegram.BotCommandScopeAllGroupChats",
            new=GroupScope,
        ):
            count, hidden = await adapter.refresh_skill_group()

        assert (count, hidden) == (2, 0)
        assert adapter._bot.set_my_commands.await_count == 3
        for call in adapter._bot.set_my_commands.await_args_list:
            assert call.args[0] == [
                ("start", "Start"),
                ("example_skill", "Example skill"),
            ]
        assert [type(call.kwargs["scope"]) for call in adapter._bot.set_my_commands.await_args_list] == [
            DefaultScope,
            PrivateScope,
            GroupScope,
        ]

    @pytest.mark.asyncio
    async def test_refresh_skill_group_rereads_current_menu_commands(self):
        adapter = _make_adapter()

        with patch(
            "hermes_cli.commands.telegram_menu_max_commands",
            return_value=60,
        ), patch(
            "hermes_cli.commands.telegram_menu_commands",
            side_effect=[
                ([("old_skill", "Old skill")], 0),
                ([("new_skill", "New skill")], 0),
            ],
        ), patch(
            "telegram.BotCommand",
            side_effect=lambda command, description: (command, description),
        ), patch(
            "telegram.BotCommandScopeDefault",
            new=DefaultScope,
        ), patch(
            "telegram.BotCommandScopeAllPrivateChats",
            new=PrivateScope,
        ), patch(
            "telegram.BotCommandScopeAllGroupChats",
            new=GroupScope,
        ):
            await adapter.refresh_skill_group()
            await adapter.refresh_skill_group()

        second_refresh_calls = adapter._bot.set_my_commands.await_args_list[3:6]
        assert len(second_refresh_calls) == 3
        for call in second_refresh_calls:
            assert call.args[0] == [("new_skill", "New skill")]

    @pytest.mark.asyncio
    async def test_refresh_skill_group_refreshes_cached_forum_chat_scopes(self):
        adapter = _make_adapter()
        adapter._forum_command_registered = {111, 222}

        with patch(
            "hermes_cli.commands.telegram_menu_max_commands",
            return_value=60,
        ), patch(
            "hermes_cli.commands.telegram_menu_commands",
            return_value=([("example_skill", "Example skill")], 0),
        ), patch(
            "telegram.BotCommand",
            side_effect=lambda command, description: (command, description),
        ), patch(
            "telegram.BotCommandScopeDefault",
            new=DefaultScope,
        ), patch(
            "telegram.BotCommandScopeAllPrivateChats",
            new=PrivateScope,
        ), patch(
            "telegram.BotCommandScopeAllGroupChats",
            new=GroupScope,
        ), patch(
            "telegram.BotCommandScopeChat",
            new=ChatScope,
        ):
            count, hidden = await adapter.refresh_skill_group()

        assert (count, hidden) == (1, 0)
        assert adapter._bot.set_my_commands.await_count == 5
        forum_scopes = [
            call.kwargs["scope"]
            for call in adapter._bot.set_my_commands.await_args_list
            if isinstance(call.kwargs["scope"], ChatScope)
        ]
        assert [scope.chat_id for scope in forum_scopes] == [111, 222]

    @pytest.mark.asyncio
    async def test_refresh_skill_group_uses_configurable_menu_cap(self):
        adapter = _make_adapter()

        with patch(
            "hermes_cli.commands.telegram_menu_max_commands",
            return_value=7,
        ), patch(
            "hermes_cli.commands.telegram_menu_commands",
            return_value=([("example_skill", "Example skill")], 3),
        ) as menu_commands, patch(
            "telegram.BotCommand",
            side_effect=lambda command, description: (command, description),
        ), patch(
            "telegram.BotCommandScopeDefault",
            new=DefaultScope,
        ), patch(
            "telegram.BotCommandScopeAllPrivateChats",
            new=PrivateScope,
        ), patch(
            "telegram.BotCommandScopeAllGroupChats",
            new=GroupScope,
        ):
            count, hidden = await adapter.refresh_skill_group()

        assert (count, hidden) == (1, 3)
        menu_commands.assert_called_once_with(max_commands=7)

    @pytest.mark.asyncio
    async def test_reload_skills_handler_calls_telegram_refresh(self):
        runner = object.__new__(GatewayRunner)

        telegram_adapter = SimpleNamespace(
            name="telegram",
            refresh_skill_group=AsyncMock(return_value=(42, 0)),
        )
        passive_adapter = SimpleNamespace(name="passive")
        runner.adapters = {
            "telegram": telegram_adapter,
            "passive": passive_adapter,
        }

        fake_result = {
            "added": [{"name": "example-skill", "description": "Example skill"}],
            "removed": [],
            "total": 42,
            "errors": [],
        }
        event = MessageEvent(
            text="/reload-skills",
            source=SimpleNamespace(
                platform=Platform.TELEGRAM,
                channel="chat-id",
                user_id="user-id",
                thread_id=None,
            ),
            raw_message={},
        )
        runner._session_key_for_source = lambda source: None
        runner._pending_skills_reload_notes = {}

        with patch("agent.skill_commands.reload_skills", return_value=fake_result):
            result = await runner._handle_reload_skills_command(event)

        telegram_adapter.refresh_skill_group.assert_awaited_once()
        assert "Skills Reloaded" in result
        assert "example-skill" in result
