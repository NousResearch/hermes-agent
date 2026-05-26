from hermes_cli.commands import GATEWAY_KNOWN_COMMANDS, SUBCOMMANDS, gateway_help_lines, resolve_command

from types import SimpleNamespace

import pytest

from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner


def test_call_command_is_gateway_visible():
    cmd = resolve_command("call")

    assert cmd is not None
    assert cmd.name == "call"
    assert "call" in GATEWAY_KNOWN_COMMANDS
    assert "/call" in "\n".join(gateway_help_lines())


def test_call_command_has_expected_subcommands():
    assert SUBCOMMANDS["/call"] == ["browser", "native", "status", "end"]


def _runner():
    runner = object.__new__(GatewayRunner)
    runner.config = SimpleNamespace(extra={})
    runner._call_manager = None
    return runner


def _event(text="/call", chat_type="dm", platform="telegram"):
    source = SimpleNamespace(
        platform=SimpleNamespace(value=platform),
        chat_id="123",
        user_id="456",
        chat_type=chat_type,
    )
    return MessageEvent(text=text, message_type=MessageType.TEXT, source=source)


@pytest.mark.asyncio
async def test_handle_call_rejects_group_chat():
    result = await _runner()._handle_call_command(_event(chat_type="group"))

    assert "private-only" in result


@pytest.mark.asyncio
async def test_handle_call_native_is_loudly_unavailable_for_telegram():
    result = await _runner()._handle_call_command(_event("/call native"))

    assert "SimpleX-native" in result
    assert "SimpleX" in result


@pytest.mark.asyncio
async def test_handle_call_status_reports_idle():
    result = await _runner()._handle_call_command(_event("/call status"))

    assert "No active call" in result
