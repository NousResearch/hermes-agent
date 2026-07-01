"""Regression tests for #56391.

When context compression is in flight (state.db compression lock held),
gateway ``busy_input_mode='interrupt'`` must demote to queue semantics so a
rapid message burst cannot start a follow-up turn against the pre-rotation
parent and fork orphaned compression siblings.
"""

from __future__ import annotations

import sys
import threading
import time
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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


def _make_runner(*, session_id: str = "parent-session") -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._busy_ack_ts = {}
    runner._draining = False
    runner.adapters = {}
    runner.config = MagicMock()
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = True
    runner._is_user_authorized = lambda _source: True
    runner._busy_input_mode = "interrupt"
    session_key = build_session_key(_make_event().source)
    entry = SimpleNamespace(session_key=session_key, session_id=session_id)
    session_store = SimpleNamespace(
        _lock=threading.Lock(),
        _entries={session_key: entry},
        switch_session=MagicMock(),
    )
    session_store._ensure_loaded_locked = lambda: None
    runner.session_store = session_store
    runner._session_db = MagicMock()
    runner._session_db._db = MagicMock()
    runner._session_db._db.get_compression_lock_holder.return_value = None
    return runner


def _make_adapter() -> MagicMock:
    adapter = MagicMock()
    adapter._pending_messages = {}
    adapter._send_with_retry = AsyncMock()
    adapter.config = MagicMock()
    adapter.config.extra = {}
    adapter.platform = MagicMock(value="telegram")
    return adapter


def _make_parent_no_subagents() -> MagicMock:
    parent = MagicMock()
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    parent.get_activity_summary.return_value = {
        "api_call_count": 3,
        "max_iterations": 60,
        "current_tool": "terminal",
    }
    return parent


class TestSessionHasCompressionInFlight:
    def test_returns_false_without_session_store(self) -> None:
        runner = _make_runner()
        runner.session_store = None
        assert runner._session_has_compression_in_flight("sk") is False

    def test_returns_true_when_lock_held(self) -> None:
        runner = _make_runner()
        sk = build_session_key(_make_event().source)
        runner._session_db._db.get_compression_lock_holder.return_value = "holder-1"
        assert runner._session_has_compression_in_flight(sk) is True

    def test_returns_false_when_lock_free(self) -> None:
        runner = _make_runner()
        sk = build_session_key(_make_event().source)
        runner._session_db._db.get_compression_lock_holder.return_value = None
        assert runner._session_has_compression_in_flight(sk) is False


class TestEnsureSessionEntryAtCompressionTip:
    def test_switches_to_canonical_tip(self) -> None:
        runner = _make_runner(session_id="parent-session")
        sk = build_session_key(_make_event().source)
        entry = runner.session_store._entries[sk]
        runner._session_db._db.get_compression_tip.return_value = "child-tip"
        switched = SimpleNamespace(session_key=sk, session_id="child-tip")
        runner.session_store.switch_session.return_value = switched

        result = runner._ensure_session_entry_at_compression_tip(sk, entry)

        assert result is switched
        runner.session_store.switch_session.assert_called_once_with(sk, "child-tip")

    def test_noop_when_already_at_tip(self) -> None:
        runner = _make_runner(session_id="live-tip")
        sk = build_session_key(_make_event().source)
        entry = runner.session_store._entries[sk]
        runner._session_db._db.get_compression_tip.return_value = "live-tip"

        result = runner._ensure_session_entry_at_compression_tip(sk, entry)

        assert result is entry
        runner.session_store.switch_session.assert_not_called()


class TestBusyHandlerDemotesInterruptForCompression:
    @pytest.mark.asyncio
    async def test_does_not_interrupt_when_compression_in_flight(self) -> None:
        runner = _make_runner()
        adapter = _make_adapter()
        event = _make_event(text="follow up during compression")
        sk = build_session_key(event.source)
        parent = _make_parent_no_subagents()
        runner._running_agents[sk] = parent
        runner.adapters[event.source.platform] = adapter
        runner._session_db._db.get_compression_lock_holder.return_value = "compressing"

        handled = await runner._handle_active_session_busy_message(event, sk)

        assert handled is True
        parent.interrupt.assert_not_called()
        assert adapter._pending_messages.get(sk) is event

    @pytest.mark.asyncio
    async def test_ack_explains_compression_demotion(self) -> None:
        runner = _make_runner()
        adapter = _make_adapter()
        event = _make_event(text="hi mid-compress")
        sk = build_session_key(event.source)
        parent = _make_parent_no_subagents()
        runner._running_agents[sk] = parent
        runner._running_agents_ts[sk] = time.time() - 120
        runner.adapters[event.source.platform] = adapter
        runner._session_db._db.get_compression_lock_holder.return_value = "compressing"

        with patch("gateway.run.merge_pending_message_event"):
            await runner._handle_active_session_busy_message(event, sk)

        adapter._send_with_retry.assert_called_once()
        content = adapter._send_with_retry.call_args.kwargs.get("content", "")
        assert "Compressing context" in content
        assert "queued" in content.lower()
        assert "/stop" in content
        assert "Interrupting" not in content

    @pytest.mark.asyncio
    async def test_interrupt_still_fires_without_compression_lock(self) -> None:
        runner = _make_runner()
        adapter = _make_adapter()
        event = _make_event(text="please stop")
        sk = build_session_key(event.source)
        parent = _make_parent_no_subagents()
        runner._running_agents[sk] = parent
        runner.adapters[event.source.platform] = adapter
        runner._session_db._db.get_compression_lock_holder.return_value = None

        with patch("gateway.run.merge_pending_message_event"):
            await runner._handle_active_session_busy_message(event, sk)

        parent.interrupt.assert_called_once_with("please stop")

    @pytest.mark.asyncio
    async def test_pending_sentinel_does_not_trigger_false_positive(self) -> None:
        runner = _make_runner()
        adapter = _make_adapter()
        event = _make_event(text="hello")
        sk = build_session_key(event.source)
        runner._running_agents[sk] = _AGENT_PENDING_SENTINEL
        runner.adapters[event.source.platform] = adapter
        runner._session_db._db.get_compression_lock_holder.return_value = "compressing"

        with patch("gateway.run.merge_pending_message_event"):
            handled = await runner._handle_active_session_busy_message(event, sk)

        assert handled is True
