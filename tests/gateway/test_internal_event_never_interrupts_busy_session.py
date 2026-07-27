"""Regression test: internal synthetic events must never interrupt a busy session.

Reported by @Heeervas (June 2026): an ``async_delegation`` completion from a
``delegate_task(background=true)`` subagent re-enters the originating gateway
session as an internal ``MessageEvent``. When that session was busy running a
turn, the completion was treated exactly like a user TEXT message and hit the
default ``busy_input_mode='interrupt'`` path — calling
``running_agent.interrupt()`` and aborting the active turn, plus sending a
"⚡ Interrupting current task" ack. The same shape affects background-process
completions (terminal ``notify_on_complete``), which also re-enter as internal
events.

The fix: ``_handle_active_session_busy_message`` returns ``False`` early for any
event with ``internal=True``, so the base adapter queues it silently (no
interrupt, no ack) and it cascades as a new turn after the current one finishes.
This preserves strict message-role alternation and the design invariant that a
completion surfaces as a NEW turn only when idle, never spliced into a running
turn.
"""

from __future__ import annotations

import sys
import threading
import types
from unittest.mock import AsyncMock, MagicMock

import pytest

# Minimal telegram stubs so gateway imports cleanly (mirrors sibling tests).
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
from gateway.run import GatewayRunner  # noqa: E402


def _make_internal_event(text: str = "[async delegation completed]") -> MessageEvent:
    source = SessionSource(
        platform=MagicMock(value="telegram"),
        chat_id="123",
        chat_type="private",
        user_id="user1",
    )
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=source,
        message_id="msg1",
        internal=True,
    )


def _make_runner() -> GatewayRunner:
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
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = True
    runner._is_user_authorized = lambda _source: True
    return runner


def _make_adapter() -> MagicMock:
    adapter = MagicMock()
    adapter._pending_messages = {}
    adapter._send_with_retry = AsyncMock()
    adapter.config = MagicMock()
    adapter.config.extra = {}
    adapter.platform = MagicMock(value="telegram")
    return adapter


def _make_running_parent() -> MagicMock:
    parent = MagicMock()
    parent._active_children = []  # no active subagents at completion time
    parent._active_children_lock = threading.Lock()
    parent.get_activity_summary.return_value = {
        "api_call_count": 4,
        "max_iterations": 60,
        "current_tool": "terminal",
    }
    return parent


@pytest.mark.asyncio
async def test_internal_event_does_not_interrupt_busy_session() -> None:
    """The async-delegation completion must not abort the active turn."""
    runner = _make_runner()
    runner._busy_input_mode = "interrupt"  # the default that caused the bug
    adapter = _make_adapter()
    event = _make_internal_event()
    sk = build_session_key(event.source)
    parent = _make_running_parent()
    runner._running_agents[sk] = parent
    runner.adapters[event.source.platform] = adapter

    handled = await runner._handle_active_session_busy_message(event, sk)

    # Returns False so the base adapter silently queues the internal event
    # as a cascading next turn — it must NOT be handled-with-interrupt here.
    assert handled is False
    # The active turn must survive.
    parent.interrupt.assert_not_called()
    # No "⚡ Interrupting current task" (or any) ack for a synthetic event.
    adapter._send_with_retry.assert_not_called()


@pytest.mark.asyncio
async def test_non_internal_event_still_interrupts() -> None:
    """Regression-guard the other direction: a real user message in interrupt
    mode with no subagents still interrupts (behaviour unchanged)."""
    runner = _make_runner()
    runner._busy_input_mode = "interrupt"
    adapter = _make_adapter()
    event = _make_internal_event(text="please stop")
    # Flip to a real user message.
    object.__setattr__(event, "internal", False)
    sk = build_session_key(event.source)
    parent = _make_running_parent()
    runner._running_agents[sk] = parent
    runner.adapters[event.source.platform] = adapter

    from unittest.mock import patch

    with patch("gateway.run.merge_pending_message_event"):
        handled = await runner._handle_active_session_busy_message(event, sk)

    assert handled is True
    parent.interrupt.assert_called_once_with("please stop")
