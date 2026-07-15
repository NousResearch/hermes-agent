"""Tests for gateway /bot-ping command.

The /bot-ping command is a liveness check that replies with 'pong' without
triggering LLM inference.  Inspired by OpenClaw's /bot-ping for QQBot.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from hermes_cli.commands import (
    COMMAND_REGISTRY,
    GATEWAY_KNOWN_COMMANDS,
    resolve_command,
    should_bypass_active_session,
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
# Bypass-active-session tests
# ---------------------------------------------------------------------------


class TestBotPingActiveSessionBypass:
    """Verify /bot-ping bypasses the active-session guard."""

    def test_should_bypass_active_session(self):
        """/bot-ping must be recognized as a bypass command."""
        assert should_bypass_active_session("bot-ping") is True

    def test_should_bypass_active_session_with_slash(self):
        assert should_bypass_active_session("/bot-ping") is True

    def test_bypass_false_for_none(self):
        assert should_bypass_active_session(None) is False


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


# ---------------------------------------------------------------------------
# Active-session dispatch integration tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestBotPingActiveSessionDispatch:
    """Verify /bot-ping works during an active session:
    returns 'pong', does NOT queue, does NOT interrupt, does NOT reach
    the LLM/agent path, and still obeys slash access policy.
    """

    @staticmethod
    def _make_source():
        from gateway.config import Platform
        from gateway.session import SessionSource

        return SessionSource(
            platform=Platform.TELEGRAM,
            user_id="u1",
            chat_id="c1",
            user_name="tester",
            chat_type="dm",
        )

    @staticmethod
    def _make_event(text: str):
        from gateway.platforms.base import MessageEvent

        return MessageEvent(
            text=text,
            source=TestBotPingActiveSessionDispatch._make_source(),
            message_id="m1",
        )

    def _make_runner(self):
        """Build a bare GatewayRunner with _running_agents populated."""
        from gateway.config import GatewayConfig, Platform, PlatformConfig
        from gateway.run import GatewayRunner

        runner = object.__new__(GatewayRunner)
        runner.config = GatewayConfig(
            platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
        )
        adapter = MagicMock()
        adapter.send = AsyncMock()
        runner.adapters = {Platform.TELEGRAM: adapter}
        runner._voice_mode = {}
        runner.hooks = SimpleNamespace(
            emit=AsyncMock(), emit_collect=AsyncMock(return_value=[]), loaded_hooks=False
        )
        runner._session_model_overrides = {}
        runner._pending_model_notes = {}
        runner._background_tasks = set()
        runner._running_agents = {"agent:main:telegram:dm:c1:u1": MagicMock()}
        runner._running_agents_ts = {"agent:main:telegram:dm:c1:u1": 1000.0}
        runner._running_agent_generations = {}
        runner._pending_messages = {}
        runner._pending_approvals = {}
        runner._session_db = None
        runner._is_user_authorized = lambda _source: True
        runner._format_session_info = lambda: ""
        runner._agent_cache = {}
        runner._agent_cache_lock = None
        runner._telegram_topic_root_sessions = set()
        runner._active_session_leases = {}
        runner._command_hook_results = {}
        runner._external_drain_active = False
        runner._draining = False
        runner.session_store = MagicMock()
        return runner

    def _make_runner_with_slash_access(self):
        """Build a runner with an enabled slash-access policy that restricts
        commands to admins only, to verify bot-ping is NOT a public bypass."""
        from gateway.config import GatewayConfig, Platform, PlatformConfig
        from gateway.run import GatewayRunner

        runner = object.__new__(GatewayRunner)
        runner.config = GatewayConfig(
            platforms={
                Platform.TELEGRAM: PlatformConfig(
                    enabled=True,
                    token="***",
                    extra={
                        "allow_admin_from": ["admin1"],
                        "user_allowed_commands": [],
                    },
                )
            }
        )
        adapter = MagicMock()
        adapter.send = AsyncMock()
        runner.adapters = {Platform.TELEGRAM: adapter}
        runner._voice_mode = {}
        runner.hooks = SimpleNamespace(
            emit=AsyncMock(), emit_collect=AsyncMock(return_value=[]), loaded_hooks=False
        )
        runner._session_model_overrides = {}
        runner._pending_model_notes = {}
        runner._background_tasks = set()
        runner._running_agents = {"agent:main:telegram:dm:c1:u1": MagicMock()}
        runner._running_agents_ts = {"agent:main:telegram:dm:c1:u1": 1000.0}
        runner._running_agent_generations = {}
        runner._pending_messages = {}
        runner._pending_approvals = {}
        runner._session_db = None
        runner._is_user_authorized = lambda _source: True
        runner._format_session_info = lambda: ""
        runner._agent_cache = {}
        runner._agent_cache_lock = None
        runner._telegram_topic_root_sessions = set()
        runner._active_session_leases = {}
        runner._command_hook_results = {}
        runner._external_drain_active = False
        runner._draining = False
        runner.session_store = MagicMock()
        return runner

    async def test_active_session_returns_pong(self):
        """/bot-ping returns 'pong' while agent is running."""
        runner = self._make_runner()
        event = self._make_event("/bot-ping")
        result = await runner._handle_message(event)
        assert result == "pong"

    async def test_active_session_does_not_queue_message(self):
        """Verify bot-ping does not enqueue a pending message."""
        runner = self._make_runner()
        event = self._make_event("/bot-ping")
        await runner._handle_message(event)
        # No pending message should match our event
        for key, pending in runner._pending_messages.items():
            assert pending != event, f"bot-ping was queued for {key}"

    async def test_active_session_does_not_interrupt_agent(self):
        """Verify bot-ping does not call interrupt on the running agent."""
        runner = self._make_runner()
        event = self._make_event("/bot-ping")
        agent_key = "agent:main:telegram:dm:c1:u1"
        original_interrupt = runner._running_agents[agent_key].interrupt
        await runner._handle_message(event)
        # interrupt should NOT have been called
        original_interrupt.assert_not_called()

    async def test_active_session_does_not_trigger_llm_handler(self):
        """bot-ping is dispatched directly, not sent as user text."""
        runner = self._make_runner()
        event = self._make_event("/bot-ping")
        result = await runner._handle_message(event)
        assert result == "pong"
        assert event.text == "/bot-ping"

    async def test_active_session_slash_access_denied(self):
        """bot-ping respects slash access policy when active session exists."""
        from gateway.config import Platform
        from gateway.platforms.base import MessageEvent
        from gateway.session import SessionSource

        runner = self._make_runner_with_slash_access()
        # Non-admin user
        source = SessionSource(
            platform=Platform.TELEGRAM,
            user_id="nonadmin",
            chat_id="c1",
            user_name="nonadmin_user",
            chat_type="dm",
        )
        event = MessageEvent(text="/bot-ping", source=source, message_id="m2")
        result = await runner._handle_message(event)
        # Should be denied, not return pong
        assert result != "pong"
        assert "admin" in result.lower() or "⛔" in result or "denied" in result.lower()

    async def test_active_session_cold_path_returns_pong(self):
        """Cold path (no active agent) also returns pong."""
        runner = self._make_runner()
        # Remove the running agent entry to simulate cold path
        runner._running_agents = {}
        runner._running_agents_ts = {}
        event = self._make_event("/bot-ping")
        result = await runner._handle_message(event)
        assert result == "pong"
