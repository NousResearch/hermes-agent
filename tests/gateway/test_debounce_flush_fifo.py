"""Regression tests for the queue-mode debounce flush.

``busy_input_mode: queue`` routes active-session TEXT follow-ups through a
short debounce so a multi-tap thought stays one turn. The flush then handed
the burst to ``merge_pending_message_event(merge_text=True)``, which
newline-joins onto whatever already occupies the single pending slot. That
merge has no time bound, so every follow-up sent during a long turn collapsed
into one turn and the agent lost the message boundaries it needs to answer
each question separately.

The flush now hands the burst to the runner's FIFO instead, which is the same
entry point interrupt mode, steer-fallback and ``/queue`` already use.
"""

from __future__ import annotations

import asyncio
import sys
import types
from unittest.mock import AsyncMock, MagicMock

import pytest

# Minimal telegram stub so importing gateway.platforms.base does not pull
# in the real python-telegram-bot dependency.
_tg = sys.modules.get("telegram") or types.ModuleType("telegram")
_tg.constants = sys.modules.get("telegram.constants") or types.ModuleType("telegram.constants")
_ct = MagicMock()
_ct.PRIVATE = "private"
_ct.GROUP = "group"
_ct.SUPERGROUP = "supergroup"
_tg.constants.ChatType = _ct
sys.modules.setdefault("telegram", _tg)
sys.modules.setdefault("telegram.constants", _tg.constants)
sys.modules.setdefault("telegram.ext", types.ModuleType("telegram.ext"))

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)
from gateway.session import SessionSource, build_session_key


def _make_event(text: str) -> MessageEvent:
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="12345",
        chat_type="dm",
        user_id="u1",
    )
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=source,
        message_id=f"msg-{text[:8]}",
    )


class _DummyAdapter(BasePlatformAdapter):  # type: ignore[misc]
    async def connect(self, *, is_reconnect: bool = False):
        pass

    async def disconnect(self):
        pass

    async def get_chat_info(self, chat_id):
        return None

    async def send(self, *args, **kwargs):
        return SendResult(success=True, message_id="x")


def _make_adapter() -> BasePlatformAdapter:
    """Build a BasePlatformAdapter without running its heavy __init__."""
    adapter = object.__new__(_DummyAdapter)
    adapter.config = PlatformConfig(enabled=True, token="***")
    adapter.platform = Platform.TELEGRAM
    adapter._message_handler = AsyncMock(return_value=None)
    adapter._busy_session_handler = None
    adapter._active_sessions = {}
    adapter._pending_messages = {}
    adapter._session_tasks = {}
    adapter._background_tasks = set()
    adapter._post_delivery_callbacks = {}
    adapter._expected_cancelled_tasks = set()
    adapter._fatal_error_code = None
    adapter._fatal_error_message = None
    adapter._fatal_error_retryable = True
    adapter._fatal_error_handler = None
    adapter._running = True
    adapter._busy_text_mode = "queue"
    adapter._busy_text_debounce_seconds = 0.1
    adapter._busy_text_hard_cap_seconds = 1.0
    adapter._text_debounce = {}
    adapter._auto_tts_default = False
    adapter._auto_tts_enabled_chats = set()
    adapter._auto_tts_disabled_chats = set()
    adapter._typing_paused = set()
    return adapter


class _FifoRunner:
    """Stand-in for the gateway runner's FIFO entry point.

    The real runner installs its bound ``_handle_active_session_busy_message``
    on the adapter; the flush reaches ``_queue_or_replace_pending_event``
    through that binding, so no extra wiring is needed.
    """

    def __init__(self, adapter: BasePlatformAdapter) -> None:
        self._adapter = adapter
        self.queued: list[MessageEvent] = []

    async def busy_handler(self, event, session_key):
        return False

    def _adapter_for_source(self, source):
        return self._adapter

    def _queue_depth(self, session_key, *, adapter=None):
        # Mirrors GatewayRunner._queue_depth: overflow + the head slot.
        depth = len(self.queued)
        slot = getattr(adapter, '_pending_messages', None) if adapter is not None else None
        if slot is not None and session_key in slot:
            depth += 1
        return depth

    def _queue_or_replace_pending_event(self, session_key, event):
        slot = self._adapter._pending_messages
        if session_key in slot:
            self.queued.append(event)
        else:
            slot[session_key] = event


async def _busy_adapter_with(runner_attached: bool):
    adapter = _make_adapter()
    runner = None
    if runner_attached:
        runner = _FifoRunner(adapter)
        adapter._busy_session_handler = runner.busy_handler
    session_key = build_session_key(_make_event("probe").source)
    adapter._active_sessions[session_key] = asyncio.Event()
    return adapter, runner, session_key


@pytest.mark.asyncio
async def test_successive_bursts_each_get_their_own_turn():
    """Two bursts must arrive as two turns, not one newline-joined turn."""
    adapter, runner, session_key = await _busy_adapter_with(True)

    await adapter.handle_message(_make_event("one"))
    await adapter._flush_text_debounce_now(session_key)
    await adapter.handle_message(_make_event("two"))
    await adapter._flush_text_debounce_now(session_key)

    assert adapter._pending_messages[session_key].text == "one"
    assert [e.text for e in runner.queued] == ["two"]


@pytest.mark.asyncio
async def test_third_burst_appends_in_arrival_order():
    """Order must hold across more than two follow-ups."""
    adapter, runner, session_key = await _busy_adapter_with(True)

    for text in ("one", "two", "three"):
        await adapter.handle_message(_make_event(text))
        await adapter._flush_text_debounce_now(session_key)

    assert adapter._pending_messages[session_key].text == "one"
    assert [e.text for e in runner.queued] == ["two", "three"]
    assert "\n" not in adapter._pending_messages[session_key].text


@pytest.mark.asyncio
async def test_merge_preserved_when_no_runner_attached():
    """No runner means no FIFO to enqueue into, so the merge must still apply.

    This keeps the no-message-loss guarantee for adapters used standalone
    (TUI, tests) where no gateway runner is installed.
    """
    adapter, _runner, session_key = await _busy_adapter_with(False)
    assert adapter._busy_session_handler is None

    await adapter.handle_message(_make_event("one"))
    await adapter._flush_text_debounce_now(session_key)
    await adapter.handle_message(_make_event("two"))
    await adapter._flush_text_debounce_now(session_key)

    assert adapter._pending_messages[session_key].text == "one\ntwo"


@pytest.mark.asyncio
async def test_enqueue_failure_falls_back_to_merge():
    """A raising FIFO must not lose the burst."""
    adapter, runner, session_key = await _busy_adapter_with(True)

    def _boom(session_key, event):
        raise RuntimeError("fifo unavailable")

    runner._queue_or_replace_pending_event = _boom

    await adapter.handle_message(_make_event("one"))
    await adapter._flush_text_debounce_now(session_key)
    await adapter.handle_message(_make_event("two"))
    await adapter._flush_text_debounce_now(session_key)

    assert adapter._pending_messages[session_key].text == "one\ntwo"


@pytest.mark.asyncio
async def test_merge_preserved_when_adapter_unresolvable():
    """An unresolvable source must fall back to the merge, not vanish.

    _queue_or_replace_pending_event returns early WITHOUT queueing when
    _adapter_for_source yields nothing. Delegating into that path
    unconditionally would drop the burst where the previous merge kept it,
    so the flush only delegates once the runner resolves the source back to
    this adapter.
    """
    adapter, runner, session_key = await _busy_adapter_with(True)
    runner._adapter_for_source = lambda source: None

    await adapter.handle_message(_make_event("one"))
    await adapter._flush_text_debounce_now(session_key)
    await adapter.handle_message(_make_event("two"))
    await adapter._flush_text_debounce_now(session_key)

    assert not runner.queued
    assert adapter._pending_messages[session_key].text == "one\ntwo"


@pytest.mark.asyncio
async def test_merge_preserved_when_fifo_declines_silently():
    """A silent FIFO decline must fall back to the merge, not vanish.

    _queue_or_replace_pending_event returns WITHOUT queueing and WITHOUT
    raising once _BUSY_QUEUE_MAX_PENDING is reached. Treating that as
    success would drop the burst. The cap was also effectively unreachable
    before this change, since the old merge collapsed every follow-up into
    one slot instead of one entry each.
    """
    adapter, runner, session_key = await _busy_adapter_with(True)
    runner._queue_or_replace_pending_event = lambda session_key, event: None

    await adapter.handle_message(_make_event("one"))
    await adapter._flush_text_debounce_now(session_key)
    await adapter.handle_message(_make_event("two"))
    await adapter._flush_text_debounce_now(session_key)

    assert not runner.queued
    assert adapter._pending_messages[session_key].text == "one\ntwo"
