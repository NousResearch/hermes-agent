"""Regression test for #27352.

When a dangerous-command approval is pending (agent blocked inside
tools/approval.py) and the user sends a plain text follow-up instead of
``/approve`` / ``/deny``, the gateway must return an explanatory
``EphemeralReply`` telling the user what to do.  It must NOT silently route
the message into the busy-input pipeline (queue/steer/interrupt), because
none of those paths can resolve the approval — the agent thread is blocked
on a ``threading.Event`` that only ``/approve`` / ``/deny`` can signal, and
the user would experience the dangerous command timing out with the
confusing ``BLOCKED: Command timed out`` message even though they
responded promptly.
"""
import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Minimal telegram stubs so importing gateway.run works without optional deps.
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

from gateway.platforms.base import (  # noqa: E402
    MessageEvent,
    MessageType,
    SessionSource,
    build_session_key,
)


def _make_event(text="just go ahead", chat_id="123", platform_val="telegram"):
    source = SessionSource(
        platform=MagicMock(value=platform_val),
        chat_id=chat_id,
        chat_type="private",
        user_id="user1",
    )
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=source,
        message_id="msg1",
    )


def _make_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._busy_ack_ts = {}
    runner._draining = False
    runner.adapters = {}
    runner.config = MagicMock()
    runner.session_store = None
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner.hooks.emit_collect = AsyncMock(return_value=[])
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = True
    runner._is_user_authorized = lambda _source: True
    runner._check_slash_access = lambda _source, _cmd: None
    return runner


def _make_adapter(platform_val="telegram"):
    adapter = MagicMock()
    adapter._pending_messages = {}
    adapter._send_with_retry = AsyncMock()
    adapter.config = MagicMock()
    adapter.config.extra = {}
    adapter.platform = MagicMock(value=platform_val)
    return adapter


class TestPendingApprovalTextReply:
    """Plain text during pending approval returns explanatory EphemeralReply."""

    @pytest.mark.asyncio
    async def test_text_during_pending_approval_returns_explainer_no_interrupt(self):
        from gateway.run import EphemeralReply, GatewayRunner

        runner = _make_runner()
        adapter = _make_adapter()
        event = _make_event(text="yes, but do it another way")
        sk = build_session_key(event.source)

        running_agent = MagicMock()
        runner._busy_input_mode = "interrupt"
        runner._running_agents[sk] = running_agent
        runner.adapters[event.source.platform] = adapter

        # Pretend a dangerous-command approval is pending.
        with patch("tools.approval.has_blocking_approval", return_value=True):
            result = await GatewayRunner._handle_message(runner, event)

        # 1. The user gets an explicit reply (not silence, not None).
        assert isinstance(result, EphemeralReply)
        reply_text = str(result)
        assert "/approve" in reply_text
        assert "/deny" in reply_text

        # 2. The running agent was NOT interrupted — the approval thread
        #    needs to keep waiting on its event.
        running_agent.interrupt.assert_not_called()

        # 3. The text was NOT queued into the busy pipeline.  If it were,
        #    it would replay as a user turn after the approval times out.
        assert sk not in adapter._pending_messages
        assert sk not in runner._pending_messages

    @pytest.mark.asyncio
    async def test_text_without_pending_approval_falls_through_to_interrupt(self):
        """Sanity check: without a live approval, behavior is unchanged
        (interrupt mode still interrupts the running agent on text follow-ups).
        """
        from gateway.run import GatewayRunner

        runner = _make_runner()
        adapter = _make_adapter()
        event = _make_event(text="continue")
        sk = build_session_key(event.source)

        running_agent = MagicMock()
        runner._busy_input_mode = "interrupt"
        runner._running_agents[sk] = running_agent
        runner._running_agents_ts[sk] = 0  # past the telegram grace window
        runner.adapters[event.source.platform] = adapter

        with patch("tools.approval.has_blocking_approval", return_value=False):
            result = await GatewayRunner._handle_message(runner, event)

        # Falls through to the existing interrupt path.
        assert result is None
        running_agent.interrupt.assert_called_once_with("continue")
