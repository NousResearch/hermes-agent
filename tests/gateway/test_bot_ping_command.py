"""Tests for gateway /bot-ping command.

The /bot-ping command is a liveness check that replies with 'pong' without
triggering LLM inference.  Inspired by OpenClaw's /bot-ping for QQBot.
"""

import asyncio

import pytest

from hermes_cli.commands import (
    COMMAND_REGISTRY,
    GATEWAY_KNOWN_COMMANDS,
    resolve_command,
)


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


class TestBotPingRegistry:
    """Verify /bot-ping is properly registered in the command registry."""

    def test_bot_ping_is_registered(self):
        cmd = resolve_command("bot-ping")
        assert cmd is not None
        assert cmd.name == "bot-ping"

    def test_bot_ping_is_gateway_only(self):
        cmd = resolve_command("bot-ping")
        assert cmd.gateway_only is True
        assert cmd.cli_only is False

    def test_bot_ping_category_is_info(self):
        cmd = resolve_command("bot-ping")
        assert cmd.category == "Info"

    def test_bot_ping_description_mentions_pong(self):
        cmd = resolve_command("bot-ping")
        assert "pong" in cmd.description.lower()

    def test_bot_ping_in_gateway_known_commands(self):
        assert "bot-ping" in GATEWAY_KNOWN_COMMANDS

    def test_bot_ping_no_aliases(self):
        cmd = resolve_command("bot-ping")
        assert cmd.aliases == ()

    def test_bot_ping_no_args(self):
        cmd = resolve_command("bot-ping")
        assert cmd.args_hint == ""


# ---------------------------------------------------------------------------
# Handler tests
# ---------------------------------------------------------------------------


class TestBotPingHandler:
    """Verify the /bot-ping handler returns 'pong'."""

    def test_handler_returns_pong(self):
        from gateway.slash_commands import GatewaySlashCommandsMixin

        result = asyncio.run(
            GatewaySlashCommandsMixin._handle_bot_ping_command(None, None)  # type: ignore[arg-type]
        )
        assert result == "pong"

    def test_handler_returns_string(self):
        """Ensure handler returns a plain string, not an EphemeralReply or None."""
        from gateway.slash_commands import GatewaySlashCommandsMixin

        result = asyncio.run(
            GatewaySlashCommandsMixin._handle_bot_ping_command(None, None)  # type: ignore[arg-type]
        )
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Dispatch tests
# ---------------------------------------------------------------------------


class TestBotPingDispatch:
    """Verify /bot-ping is routed through the gateway dispatch chain."""

    def test_bot_ping_resolves_with_leading_slash(self):
        cmd = resolve_command("/bot-ping")
        assert cmd is not None
        assert cmd.name == "bot-ping"

    def test_bot_ping_does_not_resolve_as_unknown(self):
        assert resolve_command("bot-ping") is not None
        assert resolve_command("nonexistent-ping") is None
