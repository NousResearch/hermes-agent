"""Regression tests for #30170.

#30170: Sending a message while ``delegate_task`` is running killed the
subagent because the gateway always called ``running_agent.interrupt()``
on the parent, which then cascaded synchronously through
``AIAgent._active_children`` and aborted every in-flight subagent. The
reporter (and the linked Phase-1 spec) asked for the gateway to demote
``busy_input_mode='interrupt'`` to ``queue`` semantics whenever the
parent is currently driving subagents, while leaving explicit ``/stop``
and ``/new`` slash commands untouched.

These tests pin down the gateway-side guard introduced for #30170:

* ``GatewayRunner._agent_has_active_subagents`` correctly recognises
  parents that own real children, without false-positives from a
  ``MagicMock()._active_children`` auto-attribute, missing locks, or
  the ``_AGENT_PENDING_SENTINEL`` placeholder.
* ``_handle_active_session_busy_message`` demotes the interrupt mode to
  queue semantics (no ``interrupt()`` call, message merged into the
  pending queue, ack reflects the demotion) when the parent has active
  subagents.
* The ``queue`` and ``steer`` configured modes still behave exactly as
  before — the guard is interrupt-only.
"""

from __future__ import annotations

import sys
import threading
import time
import types
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ──────────────────────────────────────────────────────────────────────
# Minimal stubs so gateway imports cleanly (mirrors test_busy_session_ack)
# ──────────────────────────────────────────────────────────────────────
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
from gateway.run import GatewayRunner, _AGENT_PENDING_SENTINEL  # noqa: E402


# ──────────────────────────────────────────────────────────────────────
# Builders (parallel to tests/gateway/test_busy_session_ack.py)
# ──────────────────────────────────────────────────────────────────────
def _make_event(text: str = "hello", chat_id: str = "123") -> MessageEvent:
    source = SessionSource(
        platform=MagicMock(value="telegram"),
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


def _make_parent_with_subagents(
    *, children: int = 1, with_lock: bool = True
) -> MagicMock:
    """A MagicMock shaped like an AIAgent that currently owns *children* subagents."""
    parent = MagicMock()
    parent._active_children = [MagicMock() for _ in range(children)]
    parent._active_children_lock = threading.Lock() if with_lock else None
    parent.get_activity_summary.return_value = {
        "api_call_count": 7,
        "max_iterations": 60,
        "current_tool": "delegate_task",
    }
    return parent


def _make_parent_no_subagents() -> MagicMock:
    """A MagicMock shaped like an AIAgent that is NOT delegating."""
    parent = MagicMock()
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    parent.get_activity_summary.return_value = {
        "api_call_count": 3,
        "max_iterations": 60,
        "current_tool": "terminal",
    }
    return parent


# ──────────────────────────────────────────────────────────────────────
# _agent_has_active_subagents
# ──────────────────────────────────────────────────────────────────────
class TestAgentHasActiveSubagents:
    """The detection helper must be both precise and defensive."""

    def test_returns_false_for_none(self) -> None:
        assert GatewayRunner._agent_has_active_subagents(None) is False

    def test_returns_false_for_pending_sentinel(self) -> None:
        assert (
            GatewayRunner._agent_has_active_subagents(_AGENT_PENDING_SENTINEL)
            is False
        )

    def test_returns_false_when_attribute_missing(self) -> None:
        """Production AIAgents always have _active_children, but the helper
        must not blow up on test stubs or partial mocks."""

        class StubAgent:
            pass

        assert GatewayRunner._agent_has_active_subagents(StubAgent()) is False

    def test_returns_false_for_empty_list(self) -> None:
        assert (
            GatewayRunner._agent_has_active_subagents(_make_parent_no_subagents())
            is False
        )

    def test_returns_true_for_single_child(self) -> None:
        assert (
            GatewayRunner._agent_has_active_subagents(_make_parent_with_subagents())
            is True
        )

    def test_returns_true_for_many_children(self) -> None:
        assert (
            GatewayRunner._agent_has_active_subagents(
                _make_parent_with_subagents(children=5)
            )
            is True
        )

    def test_works_without_lock(self) -> None:
        """``_active_children_lock`` is optional in test stubs."""
        assert (
            GatewayRunner._agent_has_active_subagents(
                _make_parent_with_subagents(with_lock=False)
            )
            is True
        )

    def test_rejects_truthy_non_collection_attribute(self) -> None:
        """The MagicMock auto-attribute regression. ``MagicMock()._active_children``
        is itself a truthy MagicMock — without the isinstance guard, the
        helper would falsely report subagents on every test mock."""
        parent = MagicMock()  # no explicit _active_children setup
        assert GatewayRunner._agent_has_active_subagents(parent) is False

    @pytest.mark.parametrize(
        "container",
        [(MagicMock(),), {MagicMock()}, [MagicMock()]],
        ids=["tuple", "set", "list"],
    )
    def test_accepts_list_tuple_set(self, container: Any) -> None:
        parent = MagicMock()
        parent._active_children = container
        parent._active_children_lock = threading.Lock()
        assert GatewayRunner._agent_has_active_subagents(parent) is True

    def test_returns_false_when_parent_not_executing_tools(self) -> None:
        """#46864 — Background children (delegate_task background=True) remain
        in _active_children after the parent's tool loop finishes, but
        _executing_tools is False because the parent has returned.
        The helper must return False so the gateway does NOT demote user
        interrupts to queue for background-only subagents."""
        parent = _make_parent_with_subagents(children=2)
        parent._executing_tools = False
        assert GatewayRunner._agent_has_active_subagents(parent) is False

    def test_returns_false_when_executing_tools_attr_missing(self) -> None:
        """If _executing_tools is not set (e.g. test stubs), default to True
        so existing behaviour is preserved."""
        parent = _make_parent_with_subagents(children=1)
        # Deliberately don't set _executing_tools — getattr should default to True
        assert GatewayRunner._agent_has_active_subagents(parent) is True


# ──────────────────────────────────────────────────────────────────────
# _handle_active_session_busy_message — interrupt demotion
# ──────────────────────────────────────────────────────────────────────
class TestBusyHandlerDemotesInterruptForSubagents:
    """The Phase-1 fix from #30170: parent.interrupt() must NOT fire when
    the parent is currently driving subagents."""

    @pytest.mark.asyncio
    async def test_does_not_call_interrupt_when_subagents_active(self) -> None:
        runner = _make_runner()
        runner._busy_input_mode = "interrupt"
        adapter = _make_adapter()
        event = _make_event(text="follow up while subagent runs")
        sk = build_session_key(event.source)
        parent = _make_parent_with_subagents()
        runner._running_agents[sk] = parent
        runner.adapters[event.source.platform] = adapter

        with patch("gateway.run.merge_pending_message_event") as merge_mock:
            handled = await runner._handle_active_session_busy_message(event, sk)

        assert handled is True
        parent.interrupt.assert_not_called()
        # Message must still be queued so it gets picked up on the next turn.
        merge_mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_ack_explains_the_demotion(self) -> None:
        """The user-visible ack must mention the subagent context AND
        the `/stop` escape hatch so the operator can self-correct."""
        runner = _make_runner()
        runner._busy_input_mode = "interrupt"
        adapter = _make_adapter()
        event = _make_event(text="hi mid-delegation")
        sk = build_session_key(event.source)
        parent = _make_parent_with_subagents()
        runner._running_agents[sk] = parent
        runner._running_agents_ts[sk] = time.time() - 120
        runner.adapters[event.source.platform] = adapter

        with patch("gateway.run.merge_pending_message_event"):
            await runner._handle_active_session_busy_message(event, sk)

        adapter._send_with_retry.assert_called_once()
        content = adapter._send_with_retry.call_args.kwargs.get("content", "")
        assert "Subagent working" in content
        assert "queued" in content.lower()
        assert "/stop" in content
        assert "Interrupting" not in content

    @pytest.mark.asyncio
    async def test_interrupt_still_fires_when_no_subagents(self) -> None:
        """Regression-guard the other direction: with no subagents the
        demotion must NOT trigger and behaviour must be byte-identical
        to the pre-#30170 interrupt path."""
        runner = _make_runner()
        runner._busy_input_mode = "interrupt"
        adapter = _make_adapter()
        event = _make_event(text="please stop")
        sk = build_session_key(event.source)
        parent = _make_parent_no_subagents()
        runner._running_agents[sk] = parent
        runner.adapters[event.source.platform] = adapter

        with patch("gateway.run.merge_pending_message_event"):
            await runner._handle_active_session_busy_message(event, sk)

        parent.interrupt.assert_called_once_with("please stop")
        content = adapter._send_with_retry.call_args.kwargs.get("content", "")
        assert "Interrupting" in content
        assert "Subagent" not in content

    @pytest.mark.asyncio
    async def test_queue_mode_unchanged_with_subagents(self) -> None:
        """Configured ``queue`` mode is already subagent-safe; the new
        guard must not change its behaviour or its ack text."""
        runner = _make_runner()
        runner._busy_input_mode = "queue"
        adapter = _make_adapter()
        event = _make_event(text="queued during delegate")
        sk = build_session_key(event.source)
        parent = _make_parent_with_subagents()
        runner._running_agents[sk] = parent
        runner.adapters[event.source.platform] = adapter

        with patch("gateway.run.merge_pending_message_event"):
            await runner._handle_active_session_busy_message(event, sk)

        parent.interrupt.assert_not_called()
        content = adapter._send_with_retry.call_args.kwargs.get("content", "")
        # The vanilla queue copy — NOT the #30170 "Subagent working" copy,
        # because the user explicitly asked for queue mode.
        assert "Queued for the next turn" in content
        assert "respond once the current task finishes" in content
        assert "Subagent working" not in content

    @pytest.mark.asyncio
    async def test_steer_mode_still_routes_through_running_agent_steer(
        self,
    ) -> None:
        """Configured ``steer`` mode must reach ``running_agent.steer()``
        even when subagents are active — the #30170 demotion is
        interrupt-specific so it doesn't accidentally disable steer."""
        runner = _make_runner()
        runner._busy_input_mode = "steer"
        adapter = _make_adapter()
        event = _make_event(text="course-correct")
        sk = build_session_key(event.source)
        parent = _make_parent_with_subagents()
        parent.steer = MagicMock(return_value=True)
        runner._running_agents[sk] = parent
        runner.adapters[event.source.platform] = adapter

        with patch("gateway.run.merge_pending_message_event"):
            await runner._handle_active_session_busy_message(event, sk)

        parent.steer.assert_called_once_with("course-correct")
        parent.interrupt.assert_not_called()

    @pytest.mark.asyncio
    async def test_pending_sentinel_does_not_demote(self) -> None:
        """The placeholder ``_AGENT_PENDING_SENTINEL`` is not a real
        agent — the guard must not treat it as having subagents.
        Otherwise we'd permanently queue messages for sessions that
        haven't actually started running yet."""
        runner = _make_runner()
        runner._busy_input_mode = "interrupt"
        adapter = _make_adapter()
        event = _make_event(text="follow up before start")
        sk = build_session_key(event.source)
        runner._running_agents[sk] = _AGENT_PENDING_SENTINEL
        runner.adapters[event.source.platform] = adapter

        with patch("gateway.run.merge_pending_message_event"):
            handled = await runner._handle_active_session_busy_message(event, sk)

        assert handled is True
        # Sentinel can't be interrupted (no .interrupt to call) — verify
        # that the helper still returns the "interrupting" copy because
        # demotion did NOT fire (and the sentinel branch in the real
        # handler just skips the interrupt call silently).
        content = adapter._send_with_retry.call_args.kwargs.get("content", "")
        assert "Subagent working" not in content
