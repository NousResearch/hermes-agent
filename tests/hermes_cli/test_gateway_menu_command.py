import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType, SendResult
from gateway.session import SessionSource
from gateway.slash_commands import GatewaySlashCommandsMixin
from hermes_cli.commands import (
    ACTIVE_SESSION_BYPASS_COMMANDS,
    gateway_help_lines,
    resolve_command,
    telegram_menu_commands,
)


class _FakeControlDeckAdapter:
    def __init__(self, *, success: bool = True):
        self.success = success
        self.called_with = None

    async def send_control_deck(self, chat_id, metadata=None):
        self.called_with = (chat_id, metadata)
        return SendResult(success=self.success, message_id="deck-1" if self.success else None)


class _FakeRunner(GatewaySlashCommandsMixin):
    def __init__(self, adapter=None):
        self.adapters = {Platform.TELEGRAM: adapter} if adapter else {}

    def _thread_metadata_for_source(self, source):
        return {"thread_id": source.thread_id}


def _menu_event():
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="123",
        user_id="42",
        thread_id="7",
        chat_type="dm",
    )
    return MessageEvent(text="/menu", message_type=MessageType.TEXT, source=source)


def test_menu_command_registered_with_aliases():
    cmd = resolve_command("menu")

    assert cmd is not None
    assert cmd.name == "menu"
    assert resolve_command("deck") is cmd
    assert resolve_command("control") is cmd


def test_menu_command_visible_in_gateway_help_and_telegram_priority_menu():
    help_lines = gateway_help_lines()
    telegram_menu, _hidden = telegram_menu_commands(max_commands=30)

    assert any(line.startswith("`/menu`") for line in help_lines)
    assert telegram_menu[0][0] == "menu"


def test_menu_is_active_session_bypass_command():
    assert "menu" in ACTIVE_SESSION_BYPASS_COMMANDS


@pytest.mark.asyncio
async def test_menu_handler_sends_telegram_control_deck_without_duplicate_text():
    adapter = _FakeControlDeckAdapter(success=True)
    runner = _FakeRunner(adapter)

    result = await runner._handle_menu_command(_menu_event())

    assert result == ""
    assert adapter.called_with == ("123", {"thread_id": "7"})


@pytest.mark.asyncio
async def test_menu_handler_falls_back_to_text_when_adapter_send_fails():
    adapter = _FakeControlDeckAdapter(success=False)
    runner = _FakeRunner(adapter)

    result = await runner._handle_menu_command(_menu_event())

    assert "Hermes Control Deck" in result
    assert "`/cc-codex <задача>`" in result
