"""Regression tests for #46866: approval responses misrouted as steered
mid-turn messages when the session is busy (agent running, blocked on
dangerous-command approval).

Before the fix, ``_handle_active_session_busy_message`` never checked for
pending tool approvals. Incoming approval/deny responses were steered or
queued instead of being delivered to the approval handler, so approvals
always timed out and auto-denied.
"""

import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


def _make_event(text: str) -> MessageEvent:
    return MessageEvent(
        text=text,
        source=_make_source(),
        message_id="m1",
    )


def _make_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    adapter = MagicMock()
    adapter.send = AsyncMock()
    adapter._send_with_retry = AsyncMock()
    adapter.resume_typing_for_chat = MagicMock()
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner.session_store = MagicMock()
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._background_tasks = set()
    runner._session_db = None
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._show_reasoning = False
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._draining = False
    runner._busy_input_mode = "steer"
    runner._busy_text_mode = "queue"
    runner._busy_ack_ts = {}
    runner._agent_has_active_subagents = lambda _agent: False
    runner._restart_requested = False
    return runner


def _clear_approval_state():
    from tools import approval as mod
    mod._gateway_queues.clear()
    mod._gateway_notify_cbs.clear()
    mod._session_approved.clear()
    mod._permanent_approved.clear()
    mod._pending.clear()


class TestBusySessionApprovalRouting:
    """Verify approval/deny responses are routed to the approval handler
    when the session is busy (agent running, blocked on approval)."""

    def setup_method(self):
        _clear_approval_state()

    @pytest.mark.asyncio
    async def test_plain_approve_resolves_pending_approval(self):
        """Typing 'approve' as plain text during a pending approval resolves it.
        Gateway users (Signal, WhatsApp) may not use slash syntax."""
        from tools.approval import (
            register_gateway_notify, _ApprovalEntry, _gateway_queues,
        )

        runner = _make_runner()
        source = _make_source()
        session_key = runner._session_key_for_source(source)

        register_gateway_notify(session_key, lambda d: None)
        entry = _ApprovalEntry({"command": "rm -rf /tmp/test"})
        _gateway_queues[session_key] = [entry]

        event = _make_event("approve")
        handled = await runner._handle_active_session_busy_message(event, session_key)

        assert handled is True, "Busy handler should return True (handled)"
        assert entry.event.is_set(), "Approval entry should be resolved"
        assert entry.result == "once"

    @pytest.mark.asyncio
    async def test_slash_approve_resolves_pending_approval(self):
        """Typing '/approve' during a pending approval resolves it."""
        from tools.approval import (
            register_gateway_notify, _ApprovalEntry, _gateway_queues,
        )

        runner = _make_runner()
        source = _make_source()
        session_key = runner._session_key_for_source(source)

        register_gateway_notify(session_key, lambda d: None)
        entry = _ApprovalEntry({"command": "ls -la"})
        _gateway_queues[session_key] = [entry]

        event = _make_event("/approve")
        handled = await runner._handle_active_session_busy_message(event, session_key)

        assert handled is True
        assert entry.event.is_set()
        assert entry.result == "once"

    @pytest.mark.asyncio
    async def test_plain_deny_denies_pending_approval(self):
        """Typing 'deny' as plain text during a pending approval denies it.
        Gateway users (Signal, WhatsApp) may not use slash syntax."""
        from tools.approval import (
            register_gateway_notify, _ApprovalEntry, _gateway_queues,
        )

        runner = _make_runner()
        source = _make_source()
        session_key = runner._session_key_for_source(source)

        register_gateway_notify(session_key, lambda d: None)
        entry = _ApprovalEntry({"command": "rm -rf /"})
        _gateway_queues[session_key] = [entry]

        event = _make_event("deny")
        handled = await runner._handle_active_session_busy_message(event, session_key)

        assert handled is True
        assert entry.event.is_set()
        assert entry.result == "deny"

    @pytest.mark.asyncio
    async def test_slash_deny_denies_pending_approval(self):
        """Typing '/deny' during a pending approval denies it."""
        from tools.approval import (
            register_gateway_notify, _ApprovalEntry, _gateway_queues,
        )

        runner = _make_runner()
        source = _make_source()
        session_key = runner._session_key_for_source(source)

        register_gateway_notify(session_key, lambda d: None)
        entry = _ApprovalEntry({"command": "dd if=/dev/zero of=/dev/sda"})
        _gateway_queues[session_key] = [entry]

        event = _make_event("/deny")
        handled = await runner._handle_active_session_busy_message(event, session_key)

        assert handled is True
        assert entry.event.is_set()
        assert entry.result == "deny"

    @pytest.mark.asyncio
    async def test_unrelated_message_not_intercepted(self):
        """When an approval is pending but the user sends an unrelated message,
        it should NOT be resolved — it should fall through to normal handling."""
        from tools.approval import (
            register_gateway_notify, _ApprovalEntry, _gateway_queues,
        )

        runner = _make_runner()
        source = _make_source()
        session_key = runner._session_key_for_source(source)

        register_gateway_notify(session_key, lambda d: None)
        entry = _ApprovalEntry({"command": "ls"})
        _gateway_queues[session_key] = [entry]

        # Set up a running agent sentinel so the handler proceeds past approval check
        runner._running_agents[session_key] = MagicMock()
        runner._running_agents[session_key].steer = MagicMock(return_value=True)

        event = _make_event("what's the weather?")
        handled = await runner._handle_active_session_busy_message(event, session_key)

        # Should be handled (True) but approval NOT resolved
        assert handled is True
        assert not entry.event.is_set(), "Unrelated message must not resolve approval"

    @pytest.mark.asyncio
    async def test_no_approval_pending_normal_flow(self):
        """When no approval is pending, messages are handled normally."""
        runner = _make_runner()
        source = _make_source()
        session_key = runner._session_key_for_source(source)

        # Set up a running agent
        runner._running_agents[session_key] = MagicMock()
        runner._running_agents[session_key].steer = MagicMock(return_value=True)

        event = _make_event("hello there")
        handled = await runner._handle_active_session_busy_message(event, session_key)

        # Should return True (handled by normal busy path)
        assert handled is True
