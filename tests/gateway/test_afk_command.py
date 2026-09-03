from unittest.mock import AsyncMock, MagicMock

import pytest

from agent import afk
from hermes_cli.commands import COMMANDS, GATEWAY_KNOWN_COMMANDS, resolve_command
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.session import SessionSource


def test_afk_is_shared_and_busy_dispatchable():
    command = resolve_command("afk")
    assert command is not None
    assert command.busy_policy == "dispatch"
    assert "/afk" in COMMANDS
    assert "afk" in GATEWAY_KNOWN_COMMANDS
    assert "on" in command.args_hint and "off" in command.args_hint


def test_afk_command_renders_only_sanitized_reason(monkeypatch):
    monkeypatch.setattr(
        afk, "get_state", lambda: {"engaged_at": "now", "reason": "lunch (safe)"}
    )
    reply = afk.handle_command("status")
    assert "lunch (safe)" in reply


@pytest.mark.asyncio
async def test_gateway_afk_handler_uses_durable_shared_command(monkeypatch):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    event = MagicMock()
    event.get_command_args.return_value = "on picking up the kids"
    monkeypatch.setattr(afk, "handle_command", lambda args: f"handled:{args}")
    assert await runner._handle_afk_command(event) == "handled:on picking up the kids"


def test_cli_afk_uses_shared_command(monkeypatch):
    pytest.importorskip("prompt_toolkit")
    from cli import HermesCLI

    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id = "test"
    cli._pending_resume_sessions = None
    cli._console_print = MagicMock()
    monkeypatch.setattr(afk, "handle_command", lambda args: f"handled:{args}")
    assert cli.process_command("/afk status") is True
    assert cli._console_print.call_args.args[0] == "handled:status"


@pytest.mark.asyncio
async def test_afk_busy_dispatches_without_interrupting(monkeypatch):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._handle_afk_command = AsyncMock(return_value="afk while busy")
    event = MagicMock()
    command = resolve_command("afk")
    assert (
        await runner._dispatch_busy_slash_command(event, command, "session", object())
        == "afk while busy"
    )


def test_afk_keeps_existing_gateway_admin_gate():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={
            Platform.DISCORD: PlatformConfig(
                enabled=True,
                token="token",
                extra={"allow_admin_from": ["admin"], "user_allowed_commands": []},
            )
        }
    )
    source = SessionSource(
        platform=Platform.DISCORD, user_id="guest", chat_id="c", chat_type="dm"
    )
    denial = runner._check_slash_access(source, "afk")
    assert denial is not None and "admin-only" in denial
