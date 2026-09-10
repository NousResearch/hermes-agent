"""Tests for gateway /bot-ping command.

The /bot-ping command is a liveness check that replies with 'pong' without
triggering LLM inference.  Inspired by OpenClaw's /bot-ping for QQBot.
"""

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from hermes_cli.commands import (
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

    def test_bot_ping_busy_policy(self):
        cmd = resolve_command("bot-ping")
        assert cmd.busy_policy == "dispatch"
        assert cmd.busy_handler is None
        assert cmd.execute is None

    def test_bot_ping_in_active_session_bypass_commands(self):
        from hermes_cli.commands import ACTIVE_SESSION_BYPASS_COMMANDS

        assert "bot-ping" in ACTIVE_SESSION_BYPASS_COMMANDS


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

    def test_bot_ping_is_in_plain_command_handler_table(self):
        """The plain-command table (idle *and* busy dispatch) maps bot-ping to its handler."""
        from gateway.run import GatewayRunner

        runner = object.__new__(GatewayRunner)
        handlers = runner._gateway_plain_command_handlers()
        assert handlers.get("bot-ping") == runner._handle_bot_ping_command

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
    def _make_source(user_id="u1", chat_id="c1"):
        from gateway.config import Platform
        from gateway.session import SessionSource

        return SessionSource(
            platform=Platform.TELEGRAM,
            user_id=user_id,
            chat_id=chat_id,
            user_name="tester",
            chat_type="dm",
        )

    @classmethod
    def _session_key(cls, user_id="u1", chat_id="c1"):
        """The real routing key for the source — must match _session_key_for_source()."""
        from gateway.session import build_session_key

        return build_session_key(cls._make_source(user_id, chat_id))

    @classmethod
    def _make_event(cls, text: str, user_id="u1"):
        from gateway.platforms.base import MessageEvent

        return MessageEvent(
            text=text, source=cls._make_source(user_id), message_id="m1",
        )

    def _make_runner(self, *, running: bool = True, gated: bool = False, user_id="u1"):
        """Build a bare GatewayRunner.

        ``running=True`` seeds the *live* session-state slot (via the legacy
        ``_running_agents`` view) with a fresh ``started_ts`` so the runner takes its
        running-session fast-path; ``running=False`` leaves the session idle.
        ``gated=True`` enables an admin-only slash-access policy.
        """
        from gateway.config import GatewayConfig, Platform, PlatformConfig
        from gateway.run import GatewayRunner

        runner = object.__new__(GatewayRunner)
        runner.config = GatewayConfig(
            platforms={
                Platform.TELEGRAM: PlatformConfig(
                    enabled=True,
                    token="***",
                    extra=(
                        {"allow_admin_from": ["admin1"], "user_allowed_commands": []}
                        if gated else {}
                    ),
                )
            }
        )
        adapter = MagicMock()
        adapter.send = AsyncMock()
        adapter._pending_messages = {}
        runner.adapters = {Platform.TELEGRAM: adapter}
        runner._voice_mode = {}
        runner.hooks = SimpleNamespace(
            emit=AsyncMock(),
            emit_collect=AsyncMock(return_value=[]),
            loaded_hooks=False,
        )
        runner._session_model_overrides = {}
        runner._pending_model_notes = {}
        runner._background_tasks = set()
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
        key = self._session_key(user_id)
        if running:
            # Fresh timestamp: the runner evicts a running-agent slot that has been idle
            # past its timeout (a stale ts would silently demote this to the idle path).
            runner._running_agents = {key: MagicMock()}
            runner._running_agents_ts = {key: time.time()}
        else:
            runner._running_agents = {}
            runner._running_agents_ts = {}
        return runner

    def _running_agent(self, runner, user_id="u1"):
        """The live agent object parked in the session's running slot."""
        return runner._session_state(self._session_key(user_id)).turn.agent

    async def test_active_session_is_actually_running(self):
        """Precondition guard: the busy-path tests below must not silently run idle."""
        runner = self._make_runner()
        assert runner._is_session_running(self._session_key()) is True
        assert self._running_agent(runner) is not None

    async def test_active_session_returns_pong(self):
        """/bot-ping returns 'pong' while agent is running."""
        runner = self._make_runner()
        assert runner._is_session_running(self._session_key()) is True
        event = self._make_event("/bot-ping")
        result = await runner._handle_message(event)
        assert result == "pong"

    async def test_active_session_uses_busy_fast_path(self, monkeypatch):
        """The reply comes from the running-session fast-path, not the idle dispatch chain."""
        from gateway.run import GatewayRunner

        calls = []
        original = GatewayRunner._hm_busy_slash_or_photo

        async def _spy(self, event, source, quick_key):
            calls.append(quick_key)
            return await original(self, event, source, quick_key)

        monkeypatch.setattr(GatewayRunner, "_hm_busy_slash_or_photo", _spy)
        runner = self._make_runner()
        event = self._make_event("/bot-ping")
        assert await runner._handle_message(event) == "pong"
        assert calls == [self._session_key()]

    async def test_active_session_does_not_queue_message(self):
        """Verify bot-ping does not enqueue a pending message."""
        from gateway.config import Platform

        runner = self._make_runner()
        key = self._session_key()
        adapter = runner.adapters[Platform.TELEGRAM]
        event = self._make_event("/bot-ping")
        await runner._handle_message(event)
        assert key not in adapter._pending_messages
        assert runner._session_state(key).conversation.queued_events == []

    async def test_active_session_does_not_interrupt_agent(self):
        """Verify bot-ping does not call interrupt on the running agent."""
        runner = self._make_runner()
        running_agent = self._running_agent(runner)
        event = self._make_event("/bot-ping")
        await runner._handle_message(event)
        running_agent.interrupt.assert_not_called()

    async def test_active_session_does_not_trigger_llm_handler(self, monkeypatch):
        """bot-ping is dispatched directly, not sent as user text through the agent path."""
        from gateway.run import GatewayRunner

        calls = []

        async def _spy(self, event, source, quick_key):
            calls.append(quick_key)
            return None

        monkeypatch.setattr(GatewayRunner, "_handle_message_with_agent", _spy)
        runner = self._make_runner()
        event = self._make_event("/bot-ping")
        result = await runner._handle_message(event)
        assert result == "pong"
        assert calls == []  # never reached the LLM/agent processing path
        assert event.text == "/bot-ping"

    async def test_active_session_slash_access_denied(self):
        """bot-ping respects slash access policy when active session exists."""
        from gateway.config import Platform
        from gateway.platforms.base import MessageEvent

        runner = self._make_runner(gated=True, user_id="nonadmin")
        event = MessageEvent(
            text="/bot-ping", source=self._make_source("nonadmin"), message_id="m2",
        )
        result = await runner._handle_message(event)
        # Should be denied, not return pong
        assert result != "pong"
        assert "admin" in result.lower() or "⛔" in result or "denied" in result.lower()

    async def test_active_session_admin_still_gets_pong(self):
        """An admin under the same gated policy still gets 'pong' mid-run."""
        runner = self._make_runner(gated=True, user_id="admin1")
        event = self._make_event("/bot-ping", user_id="admin1")
        assert await runner._handle_message(event) == "pong"

    async def test_active_session_cold_path_returns_pong(self):
        """Cold path (no active agent) also returns pong."""
        runner = self._make_runner(running=False)
        assert runner._is_session_running(self._session_key()) is False
        event = self._make_event("/bot-ping")
        result = await runner._handle_message(event)
        assert result == "pong"

    async def test_cold_path_slash_access_denied(self):
        """bot-ping respects slash access policy on cold path (no active session)."""
        from gateway.platforms.base import MessageEvent

        runner = self._make_runner(running=False, gated=True, user_id="nonadmin")
        event = MessageEvent(
            text="/bot-ping", source=self._make_source("nonadmin"), message_id="m3",
        )
        result = await runner._handle_message(event)
        assert result != "pong"
        assert "admin" in result.lower() or "⛔" in result or "denied" in result.lower()
