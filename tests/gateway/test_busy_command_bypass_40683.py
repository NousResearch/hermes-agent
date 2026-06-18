"""Regression tests for slash commands silently dropped during busy state.

Covers #40683: when busy_text_mode == "queue" and effective_mode != "steer",
the busy handler returned False before parsing slash commands. This caused
/steer, /stop, /new, /queue, /status to be silently dropped to the cold
path where they were treated as plain text messages.

The fix adds a high-priority command bypass that resolves slash commands
BEFORE the early return. If the command is in ACTIVE_SESSION_BYPASS_COMMANDS,
the handler continues to the command handling below instead of returning False.

Also covers related issues: #26813 (/stop fed as steer text), #28503
(busy_input_mode: queue silently drops messages), #21790 (/steer not
recognized in Telegram DM).
"""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Minimal stubs so we can import gateway code without heavy deps
# ---------------------------------------------------------------------------
import sys, types

_tg = types.ModuleType("telegram")
_tg.constants = types.ModuleType("telegram.constants")
_ct = MagicMock()
_ct.SUPERGROUP = "supergroup"
_ct.GROUP = "group"
_ct.PRIVATE = "private"
_tg.constants.ChatType = _ct
sys.modules.setdefault("telegram", _tg)
sys.modules.setdefault("telegram.constants", _tg.constants)
sys.modules.setdefault("telegram.ext", types.ModuleType("telegram.ext"))

from gateway.platforms.base import (
    MessageEvent,
    MessageType,
    SessionSource,
    build_session_key,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_event(text="hello", chat_id="123", platform_val="telegram"):
    """Build a minimal MessageEvent."""
    source = SessionSource(
        platform=MagicMock(value=platform_val),
        chat_id=chat_id,
        chat_type="private",
        user_id="user1",
    )
    evt = MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=source,
        message_id="msg1",
    )
    return evt


def _make_runner():
    """Build a minimal GatewayRunner-like object for testing."""
    from gateway.run import GatewayRunner, _AGENT_PENDING_SENTINEL

    runner = object.__new__(GatewayRunner)
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._busy_ack_ts = {}
    runner._draining = False
    runner._busy_text_mode = "queue"  # The mode where the bug manifests
    runner._busy_input_mode = "interrupt"
    runner.adapters = {}
    runner.config = MagicMock()
    runner.session_store = None
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = True
    runner._is_user_authorized = lambda _source: True
    return runner, _AGENT_PENDING_SENTINEL


def _make_adapter(platform_val="telegram"):
    """Build a minimal adapter mock."""
    adapter = MagicMock()
    adapter._pending_messages = {}
    adapter._send_with_retry = AsyncMock()
    adapter.config = MagicMock()
    adapter.config.extra = {}
    adapter.platform = MagicMock(value=platform_val)
    adapter._text_debounce = {}
    adapter._busy_text_debounce_seconds = 0.6
    return adapter


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBusyCommandBypass40683:
    """Slash commands must be parsed BEFORE the busy_text_mode early return.

    Core regression for #40683: /steer, /stop, /new, /queue, /status arriving
    during busy+queue mode must NOT be silently dropped to the cold path.
    """

    @pytest.mark.asyncio
    async def test_steer_command_not_dropped_in_queue_mode(self):
        """#40683 core: /steer during busy+queue must not be silently dropped."""
        from gateway.run import GatewayRunner

        runner, _sentinel = _make_runner()
        runner._busy_text_mode = "queue"
        runner._busy_input_mode = "queue"
        adapter = _make_adapter()

        event = _make_event(text="/steer change approach to Y")
        sk = build_session_key(event.source)

        agent = MagicMock()
        agent.steer.return_value = True
        runner._running_agents[sk] = agent
        runner.adapters[event.source.platform] = adapter

        result = await GatewayRunner._handle_active_session_busy_message(runner, event, sk)

        # Should be handled (True), NOT dropped to cold path (False)
        assert result is True

    @pytest.mark.asyncio
    async def test_stop_command_not_dropped_in_queue_mode(self):
        """#26813: /stop during busy+queue must not be silently dropped."""
        from gateway.run import GatewayRunner

        runner, _sentinel = _make_runner()
        runner._busy_text_mode = "queue"
        runner._busy_input_mode = "queue"
        adapter = _make_adapter()

        event = _make_event(text="/stop")
        sk = build_session_key(event.source)

        agent = MagicMock()
        runner._running_agents[sk] = agent
        runner.adapters[event.source.platform] = adapter

        result = await GatewayRunner._handle_active_session_busy_message(runner, event, sk)

        # Should be handled (True), NOT dropped to cold path (False)
        assert result is True

    @pytest.mark.asyncio
    async def test_new_command_not_dropped_in_queue_mode(self):
        """#28503: /new during busy+queue must not be silently dropped."""
        from gateway.run import GatewayRunner

        runner, _sentinel = _make_runner()
        runner._busy_text_mode = "queue"
        runner._busy_input_mode = "queue"
        adapter = _make_adapter()

        event = _make_event(text="/new")
        sk = build_session_key(event.source)

        agent = MagicMock()
        runner._running_agents[sk] = agent
        runner.adapters[event.source.platform] = adapter

        result = await GatewayRunner._handle_active_session_busy_message(runner, event, sk)

        # Should be handled (True), NOT dropped to cold path (False)
        assert result is True

    @pytest.mark.asyncio
    async def test_queue_command_not_dropped_in_queue_mode(self):
        """#28503: /queue during busy+queue must not be silently dropped."""
        from gateway.run import GatewayRunner

        runner, _sentinel = _make_runner()
        runner._busy_text_mode = "queue"
        runner._busy_input_mode = "queue"
        adapter = _make_adapter()

        event = _make_event(text="/queue follow up task")
        sk = build_session_key(event.source)

        agent = MagicMock()
        runner._running_agents[sk] = agent
        runner.adapters[event.source.platform] = adapter

        result = await GatewayRunner._handle_active_session_busy_message(runner, event, sk)

        # Should be handled (True), NOT dropped to cold path (False)
        assert result is True

    @pytest.mark.asyncio
    async def test_status_command_not_dropped_in_queue_mode(self):
        """Status command during busy+queue must not be silently dropped."""
        from gateway.run import GatewayRunner

        runner, _sentinel = _make_runner()
        runner._busy_text_mode = "queue"
        runner._busy_input_mode = "queue"
        adapter = _make_adapter()

        event = _make_event(text="/status")
        sk = build_session_key(event.source)

        agent = MagicMock()
        agent.get_activity_summary.return_value = {
            "api_call_count": 5,
            "max_iterations": 90,
            "current_tool": "terminal",
        }
        runner._running_agents[sk] = agent
        runner._running_agents_ts[sk] = time.time() - 120
        runner.adapters[event.source.platform] = adapter

        result = await GatewayRunner._handle_active_session_busy_message(runner, event, sk)

        # Should be handled (True), NOT dropped to cold path (False)
        assert result is True

    @pytest.mark.asyncio
    async def test_plain_text_still_dropped_in_queue_mode(self):
        """Non-command text messages must still be dropped (returned False) in queue mode.

        This is the existing behavior that must be preserved — only slash commands
        get the bypass, not plain text.
        """
        from gateway.run import GatewayRunner

        runner, _sentinel = _make_runner()
        runner._busy_text_mode = "queue"
        runner._busy_input_mode = "queue"
        adapter = _make_adapter()

        event = _make_event(text="just a regular message")
        sk = build_session_key(event.source)

        agent = MagicMock()
        runner._running_agents[sk] = agent
        runner.adapters[event.source.platform] = adapter

        result = await GatewayRunner._handle_active_session_busy_message(runner, event, sk)

        # Should be dropped (False) — not a command
        assert result is False

    @pytest.mark.asyncio
    async def test_steer_text_still_works_in_steer_mode(self):
        """When effective_mode == 'steer', steer text should work regardless of busy_text_mode.

        This is the pre-existing behavior that must be preserved.
        """
        from gateway.run import GatewayRunner

        runner, _sentinel = _make_runner()
        runner._busy_text_mode = "queue"
        runner._busy_input_mode = "steer"  # steer mode bypasses the early return
        adapter = _make_adapter()

        event = _make_event(text="/steer try a different approach")
        sk = build_session_key(event.source)

        agent = MagicMock()
        agent.steer.return_value = True
        runner._running_agents[sk] = agent
        runner.adapters[event.source.platform] = adapter

        result = await GatewayRunner._handle_active_session_busy_message(runner, event, sk)

        assert result is True
        agent.steer.assert_called_once()

    @pytest.mark.asyncio
    async def test_interrupt_mode_still_works(self):
        """When effective_mode == 'interrupt' and busy_text_mode == 'queue',
        plain text is still dropped (returned False). Only high-priority
        slash commands bypass the early return.
        """
        from gateway.run import GatewayRunner

        runner, _sentinel = _make_runner()
        runner._busy_text_mode = "queue"
        runner._busy_input_mode = "interrupt"
        adapter = _make_adapter()

        event = _make_event(text="follow up")
        sk = build_session_key(event.source)

        agent = MagicMock()
        agent.get_activity_summary.return_value = {
            "api_call_count": 1,
            "max_iterations": 90,
            "current_tool": None,
        }
        runner._running_agents[sk] = agent
        runner._running_agents_ts[sk] = time.time()
        runner.adapters[event.source.platform] = adapter

        result = await GatewayRunner._handle_active_session_busy_message(runner, event, sk)

        # busy_text_mode == "queue" causes early return for non-command text
        assert result is False

    @pytest.mark.asyncio
    async def test_unknown_slash_command_still_dropped_in_queue_mode(self):
        """Commands not in ACTIVE_SESSION_BYPASS_COMMANDS should still be dropped.

        E.g., /model, /reasoning — these don't have dedicated busy handlers
        and should fall through to the cold path.
        """
        from gateway.run import GatewayRunner

        runner, _sentinel = _make_runner()
        runner._busy_text_mode = "queue"
        runner._busy_input_mode = "queue"
        adapter = _make_adapter()

        # /model is not in ACTIVE_SESSION_BYPASS_COMMANDS
        event = _make_event(text="/model anthropic/claude-sonnet-4")
        sk = build_session_key(event.source)

        agent = MagicMock()
        runner._running_agents[sk] = agent
        runner.adapters[event.source.platform] = adapter

        result = await GatewayRunner._handle_active_session_busy_message(runner, event, sk)

        # Should be dropped (False) — /model is not a high-priority command
        assert result is False
