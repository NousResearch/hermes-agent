import sys
from types import SimpleNamespace

import pytest

from hermes_cli.commands import (
    resolve_command,
    telegram_bot_commands,
    telegram_menu_commands,
)
from gateway.config import PlatformConfig
from gateway.platforms.telegram import TelegramAdapter


class _FakeBot:
    def __init__(self):
        self.command_calls = []
        self.menu_button_calls = []

    async def set_my_commands(self, commands, scope=None):
        self.command_calls.append((commands, scope))

    async def set_chat_menu_button(self, menu_button=None, **kwargs):
        self.menu_button_calls.append((menu_button, kwargs))


class _MenuButtonFailureBot(_FakeBot):
    async def set_chat_menu_button(self, menu_button=None, **kwargs):
        from telegram.error import TelegramError

        raise TelegramError("menu button unavailable")


@pytest.mark.asyncio
async def test_telegram_menu_registration_sets_commands_and_menu_button(monkeypatch):
    """Command menu registration uses production adapter code."""
    monkeypatch.setattr(
        "hermes_cli.commands.telegram_menu_commands",
        lambda max_commands=100: ([("new", "Start a new session"), ("help", "Show help")], 0),
    )

    telegram_mod = sys.modules.get("telegram")
    monkeypatch.setattr(
        telegram_mod,
        "BotCommand",
        lambda name, description: SimpleNamespace(name=name, description=description),
        raising=False,
    )
    monkeypatch.setattr(
        telegram_mod,
        "MenuButtonCommands",
        lambda: SimpleNamespace(type="commands"),
        raising=False,
    )

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    fake_bot = _FakeBot()
    adapter._bot = fake_bot

    await adapter._register_command_menu()

    assert len(fake_bot.command_calls) == 1
    commands, scope = fake_bot.command_calls[0]
    assert scope is None
    assert [cmd.name for cmd in commands] == ["new", "help"]
    assert len(fake_bot.menu_button_calls) == 1
    menu_button, kwargs = fake_bot.menu_button_calls[0]
    assert menu_button.type == "commands"
    assert kwargs == {}


@pytest.mark.asyncio
async def test_telegram_menu_button_failure_does_not_break_command_registration(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.commands.telegram_menu_commands",
        lambda max_commands=100: ([("start", "Show Telegram quick-start help")], 0),
    )

    telegram_mod = sys.modules.get("telegram")
    monkeypatch.setattr(
        telegram_mod,
        "BotCommand",
        lambda name, description: SimpleNamespace(name=name, description=description),
        raising=False,
    )

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    fake_bot = _MenuButtonFailureBot()
    adapter._bot = fake_bot

    await adapter._register_command_menu()

    assert len(fake_bot.command_calls) == 1
    commands, _scope = fake_bot.command_calls[0]
    assert [cmd.name for cmd in commands] == ["start"]


def test_start_command_is_registered_for_gateway_and_telegram_menu():
    start = resolve_command("start")

    assert start is not None
    assert start.name == "start"
    assert start.gateway_only is True
    assert any(name == "start" for name, _description in telegram_bot_commands())
    menu_commands, _hidden_count = telegram_menu_commands(max_commands=100)
    assert any(name == "start" for name, _description in menu_commands)
