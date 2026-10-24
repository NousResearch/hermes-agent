"""Event-class merge separation: internal vs human boundary at the shared refusal helper.

The merge boundary must refuse to coalesce an internal event with a human event
even when both carry the same effective preservation decision.  The shared
``prompt_identity_conflict`` compares canonical ``event_preserves_prompt_pins``
AND independently requires the same ``internal`` class.  ``_can_merge_text_debounce_events``
must delegate to that one helper rather than duplicating its policy.

Two test functions:
1. Table-driven helper controls: ``prompt_identity_conflict`` and ``_can_merge_text_debounce_events``
   across every original case (same-class merge, internal↔human refusal in both orders,
   canonical-equal/different-internal countercontrols, different-preservation refusal).
2. Real admission: ``_hm_merge_pending_for_source`` and ``_queue_text_debounce`` +
   ``_flush_text_debounce_now`` through the real runner and adapter, exercising the
   line-113 occupant shape (internal pending A + different-sender human buffer X +
   same-sender human B) with real FIFO admission.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    prompt_identity_conflict,
)
from gateway.platforms.base_text_debounce import TextDebounceMixin
from gateway.platforms.event import event_preserves_prompt_pins
from gateway.session import SessionEntry, SessionSource

KEY = "agent:main:discord:group:1513247605675790346"
_ORIGIN = dict(
    platform=Platform.DISCORD,
    chat_id="1513247605675790346",
    chat_type="group",
    user_id="117431298246705156",
    scope_id="1480524732964278294",
)
_ALT_ORIGIN = dict(
    platform=Platform.DISCORD,
    chat_id="1513247605675790346",
    chat_type="group",
    user_id="999999999999999999",
    scope_id="1480524732964278294",
)


def _source() -> SessionSource:
    return SessionSource(**_ORIGIN, message_id="1552671843494666330")


def _alt_source() -> SessionSource:
    return SessionSource(**_ALT_ORIGIN, message_id="1552671843494666331")


def _event(
    text: str,
    *,
    internal: bool = False,
    preserve_prompt_pins: bool = False,
    source: SessionSource | None = None,
) -> MessageEvent:
    source = source or _source()
    return MessageEvent(
        text=text,
        source=source,
        message_id=None if internal else source.message_id,
        internal=internal,
        preserve_prompt_pins=preserve_prompt_pins,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Test 1: table-driven helper controls (all 26 original cases)
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    ("existing_internal", "event_internal", "existing_preserve", "event_preserve", "expected_conflict"),
    [
        # Same internal class, same effective preservation → no conflict
        (False, False, True, True, False),
        (True, True, True, True, False),
        (False, False, False, False, False),
        (True, True, False, False, False),
        # Same internal class, same effective preservation (both True via different raw flags) → no conflict
        (True, True, False, True, False),
        (True, True, True, False, False),
        # Different internal class, same effective preservation → CONFLICT
        # internal wake(True,False) vs noninternal synthetic(False,True): both effective True
        (True, False, False, True, True),
        (False, True, False, True, True),
        (True, False, True, True, True),
        (False, True, True, True, True),
        # Different effective preservation → conflict regardless of internal class
        (False, False, False, True, True),
        (False, False, True, False, True),
        # The reported raw-false/false wake/human hole, in both arrival orders.
        (True, False, False, False, True),
        (False, True, False, False, True),
    ],
)
def test_prompt_identity_conflict_and_debounce_delegate(
    existing_internal,
    event_internal,
    existing_preserve,
    event_preserve,
    expected_conflict,
):
    """prompt_identity_conflict compares canonical preservation AND internal class independently.

    _can_merge_text_debounce_events (same sender) returns the inverse of prompt_identity_conflict.
    """
    existing = _event(
        "existing",
        internal=existing_internal,
        preserve_prompt_pins=existing_preserve,
    )
    event = _event(
        "event",
        internal=event_internal,
        preserve_prompt_pins=event_preserve,
    )
    result = prompt_identity_conflict(existing, event)
    assert result is expected_conflict, (
        f"internal={existing_internal} vs {event_internal}, "
        f"preserve={existing_preserve} vs {event_preserve}: "
        f"expected conflict={expected_conflict}, got {result}"
    )
    # _can_merge_text_debounce_events with same sender = inverse of prompt_identity_conflict
    adapter = _StubAdapter()
    merge_result = adapter._can_merge_text_debounce_events(existing, event)
    assert merge_result is (not expected_conflict), (
        f"internal={existing_internal} vs {event_internal}, "
        f"preserve={existing_preserve} vs {event_preserve}: "
        f"expected merge={not expected_conflict}, got {merge_result}"
    )


# ═══════════════════════════════════════════════════════════════════════════
# Test 2: real admission through _hm_merge_pending_for_source + debounce
# ═══════════════════════════════════════════════════════════════════════════


class _StubAdapter(TextDebounceMixin):
    """Minimal adapter stub exposing the real _can_merge_text_debounce_events."""

    name = "stub"
    _busy_text_mode = "queue"
    _busy_text_debounce_seconds = 3.0
    _busy_text_hard_cap_seconds = 10.0
    _pending_messages: dict = {}
    gateway_runner = None


class _RealAdapter(BasePlatformAdapter):
    """Minimal real adapter for real runner admission tests."""

    async def connect(self, *, is_reconnect=False):
        return True

    async def disconnect(self):
        pass

    async def send(self, chat_id, text, **kwargs):
        pass

    async def get_chat_info(self, chat_id):
        return {}


def _make_runner():
    import gateway.run as gr

    runner = gr.GatewayRunner(GatewayConfig(group_sessions_per_user=False, multiplex_profiles=False))
    runner.adapters = {}
    runner._pending_messages = {}
    runner._is_user_authorized = lambda s: True
    runner._session_db = MagicMock()

    runner._cache_session_source = lambda k, s: None
    runner._set_session_env = lambda c: None
    runner._restart_requested = False
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner._turn_leases = None

    store = MagicMock()
    store.get_or_create_session.return_value = SessionEntry(
        session_key=KEY, session_id="sess-1",
        created_at=datetime(2026, 1, 1), updated_at=datetime(2026, 1, 2),
        platform=Platform.DISCORD, chat_type="group",
    )
    store.load_transcript.return_value = []
    store.has_platform_message_id.return_value = False
    runner.session_store = store

    adapter = _RealAdapter(PlatformConfig(enabled=True), Platform.DISCORD)
    adapter._pending_messages = {}
    adapter.gateway_runner = runner
    adapter._busy_text_mode = "queue"
    adapter._busy_text_debounce_seconds = 60.0
    adapter._busy_text_hard_cap_seconds = 120.0
    runner.adapters[Platform.DISCORD] = adapter
    assert runner._delivery_adapter_for(_source()) is adapter
    assert runner._session_key_for_source(_source()) == KEY
    assert runner._session_key_for_source(_alt_source()) == KEY
    return runner, adapter


def _overflow(runner) -> list:
    return runner._session_state(KEY).conversation.queued_events


@pytest.mark.parametrize("route", ["sentinel-internal-first", "sentinel-human-first", "debounce-occupied-buffer"])
@pytest.mark.asyncio
async def test_debounce_admission_real_runner_line113(route):
    """Real admission: internal pending A + human buffer X (different sender) + human B (same sender as A).

    Line-113 shape: A occupies the pending slot (internal). X is buffered via real
    _queue_text_debounce (human, different sender from A). B arrives via real
    _queue_text_debounce (human, same sender as A). The fix flushes X to the real
    FIFO and buffers/flushes B separately, so A, X, B all survive with distinct
    owners and unchanged fields.

    Also exercises _hm_merge_pending_for_source with real runner/adapter:
    internal event in slot + human event → refuse + admit to FIFO.
    """
    runner, adapter = _make_runner()
    tasks = []
    try:
        a = _event("internal A", internal=True)  # Ordinary wake: no opt-in flag.
        b = _event("human B")
        b.reply_to_message_id = "reply-B"
        if route.startswith("sentinel"):
            head, arrival = (a, b) if route.endswith("internal-first") else (b, a)
            assert runner._queue_or_replace_pending_event(KEY, head) is True
            assert runner._hm_merge_pending_for_source(arrival.source, KEY, arrival, merge_text=True) is True
            assert adapter._pending_messages[KEY] is head
            assert _overflow(runner) == [arrival]
            assert head._gateway_accepted is True and arrival._gateway_accepted is True
        else:
            x = _event("human X", source=_alt_source())
            x.reply_to_message_id = "reply-X"
            assert runner._queue_or_replace_pending_event(KEY, a) is True
            await adapter._queue_text_debounce(KEY, x)
            store = adapter._text_debounce_store()
            assert store[KEY].event is x
            tasks.append(store[KEY].task)
            # Unfixed code retains X, then folds this human B into internal A.
            await adapter._queue_text_debounce(KEY, b)
            assert store[KEY].event is b
            tasks.append(store[KEY].task)
            assert await adapter._flush_text_debounce_now(KEY) is True
            assert adapter._pending_messages[KEY] is a
            overflow = _overflow(runner)
            assert overflow == [x, b]
            assert x in overflow and b in overflow
            assert overflow.index(x) < overflow.index(b)
            assert x._gateway_accepted is True and b._gateway_accepted is True
            assert x.text == "human X" and x.internal is False
            assert x.message_id == "1552671843494666331"
            assert x.reply_to_message_id == "reply-X"
            assert event_preserves_prompt_pins(x) is False
        assert a.text == "internal A" and a.internal is True and a.preserve_prompt_pins is False
        assert a.message_id is None and event_preserves_prompt_pins(a) is True
        assert b.text == "human B" and b.internal is False and b.preserve_prompt_pins is False
        assert b.message_id == "1552671843494666330" and b.reply_to_message_id == "reply-B"
        assert event_preserves_prompt_pins(b) is False
    finally:
        for state in list(adapter._text_debounce_store().values()):
            tasks.append(state.task)
            state.cancel_timer()
        adapter._text_debounce_store().clear()
        outcomes = await asyncio.gather(*(task for task in tasks if task is not None), return_exceptions=True)
        assert all(value is None or isinstance(value, asyncio.CancelledError) for value in outcomes)
