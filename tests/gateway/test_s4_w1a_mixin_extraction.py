"""Regression tests for the s4 wave-1 mixin extraction of base.py.

Clusters lifted verbatim out of ``gateway/platforms/base.py``:

* c1 -> ``gateway/platforms/streaming_tts_mixin.py`` (``StreamingTTSMixin``):
  the streaming-TTS adapter contract defaults plus the per-turn
  whole-file suppression helpers.
* c9 -> ``gateway/platforms/text_debounce_mixin.py`` (``TextDebounceMixin``):
  queue-mode busy-text debounce (candidate gating, sender-attribution
  merging, bounded flush timer, discard).

These tests pin the observable behavior of the moved methods on the
adapter class (MRO-resolved through the mixins) using the established
bare-adapter pattern (``object.__new__`` + stubbed config), so a future
regression in the lift shows up as a behavior change here.
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
    AudioFormat,
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)
from gateway.platforms.streaming_tts_mixin import StreamingTTSMixin
from gateway.platforms.text_debounce_mixin import TextDebounceMixin
from gateway.session import SessionSource, build_session_key


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
    adapter._streaming_tts_completed_turns = set()
    return adapter


def _make_event(
    text: str,
    chat_id: str = "12345",
    *,
    chat_type: str = "dm",
    user_id: str = "u1",
    thread_id: str | None = None,
) -> MessageEvent:
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id=chat_id,
        chat_type=chat_type,
        user_id=user_id,
        thread_id=thread_id,
    )
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=source,
        message_id=f"msg-{text[:8]}",
    )


# ---------------------------------------------------------------------------
# Wiring: the adapter class must resolve the moved methods through the mixins
# ---------------------------------------------------------------------------


def test_adapter_inherits_both_mixins():
    assert issubclass(BasePlatformAdapter, StreamingTTSMixin)
    assert issubclass(BasePlatformAdapter, TextDebounceMixin)
    assert BasePlatformAdapter.__mro__[1] is StreamingTTSMixin
    assert BasePlatformAdapter.__mro__[2] is TextDebounceMixin


def test_moved_methods_resolve_through_mixins():
    assert BasePlatformAdapter.supports_streaming_tts.__module__ == "gateway.platforms.streaming_tts_mixin"
    assert BasePlatformAdapter._text_debounce_store.__module__ == "gateway.platforms.text_debounce_mixin"
    assert BasePlatformAdapter._queue_text_debounce.__module__ == "gateway.platforms.text_debounce_mixin"


# ---------------------------------------------------------------------------
# Streaming TTS contract defaults (cluster c1)
# ---------------------------------------------------------------------------


def test_streaming_tts_contract_defaults():
    adapter = _make_adapter()
    assert adapter.supports_streaming_tts("chat1", AudioFormat()) is False


@pytest.mark.asyncio
async def test_streaming_tts_lifecycle_defaults():
    adapter = _make_adapter()
    handle = await adapter.begin_streaming_tts("chat1", AudioFormat())
    assert handle is None
    # write/finish/abort are no-ops by default and must not raise.
    await adapter.write_streaming_tts(None, b"\x00\x00")
    await adapter.finish_streaming_tts(None)
    await adapter.abort_streaming_tts(None, error="boom")


def test_streaming_tts_turn_key_is_turn_scoped():
    adapter = _make_adapter()
    turn_one = adapter._streaming_tts_turn_key("chat-1", 101)
    turn_two = adapter._streaming_tts_turn_key("chat-1", 102)
    assert turn_one is not None and turn_one != turn_two
    assert turn_one == "chat-1:101"
    # No session key or marker -> no key (whole-file path stays available).
    assert adapter._streaming_tts_turn_key(None, None) is None


def test_streaming_tts_completed_turn_suppression():
    adapter = _make_adapter()
    assert adapter._streaming_tts_turn_completed("chat-1", 101) is False
    adapter._mark_streaming_tts_completed_turn("chat-1", 101)
    assert adapter._streaming_tts_turn_completed("chat-1", 101) is True
    # Same chat, next turn: not suppressed.
    assert adapter._streaming_tts_turn_completed("chat-1", 102) is False
    # Different chat, same marker: not suppressed.
    assert adapter._streaming_tts_turn_completed("chat-2", 101) is False


def test_streaming_tts_completed_turns_lazy_init():
    adapter = object.__new__(_DummyAdapter)  # no _streaming_tts_completed_turns
    assert adapter._streaming_tts_turn_completed("chat-1", 101) is False
    adapter._mark_streaming_tts_completed_turn("chat-1", 101)
    assert adapter._streaming_tts_turn_completed("chat-1", 101) is True


# ---------------------------------------------------------------------------
# Text debounce store + candidate gating (cluster c9)
# ---------------------------------------------------------------------------


def test_text_debounce_store_lazy_init():
    adapter = object.__new__(_DummyAdapter)
    store = adapter._text_debounce_store()
    assert store == {}
    assert adapter._text_debounce is store


def test_is_queue_text_debounce_candidate_gating():
    adapter = _make_adapter()
    assert adapter._is_queue_text_debounce_candidate(_make_event("hello")) is True
    # Empty text is not a candidate.
    assert adapter._is_queue_text_debounce_candidate(_make_event("")) is False
    # Commands are not debounced.
    assert adapter._is_queue_text_debounce_candidate(_make_event("/stop")) is False
    # Non-queue busy mode disables debounce.
    adapter._busy_text_mode = "interrupt"
    assert adapter._is_queue_text_debounce_candidate(_make_event("hello")) is False


def test_can_merge_text_debounce_events_same_sender():
    adapter = _make_adapter()
    existing = _make_event("first", user_id="u1")
    incoming = _make_event("second", user_id="u1")
    assert adapter._can_merge_text_debounce_events(existing, incoming) is True


def test_can_merge_text_debounce_events_different_sender():
    adapter = _make_adapter()
    existing = _make_event("first", user_id="u1")
    incoming = _make_event("second", user_id="u2")
    assert adapter._can_merge_text_debounce_events(existing, incoming) is False


def test_can_merge_text_debounce_events_dm_fallback():
    adapter = _make_adapter()
    existing = _make_event("first", chat_type="dm", user_id=None)
    incoming = _make_event("second", chat_type="dm", user_id=None)
    assert adapter._can_merge_text_debounce_events(existing, incoming) is True


def test_text_debounce_delay_bounded():
    adapter = _make_adapter()
    assert adapter._text_debounce_delay("s1") == 0.0
    adapter._text_debounce_store()["s1"] = types.SimpleNamespace(
        event=_make_event("x"),
        task=None,
        first_ts=asyncio.get_event_loop().time() - 5.0,
        last_ts=asyncio.get_event_loop().time() - 0.5,
    )
    delay = adapter._text_debounce_delay("s1")
    assert 0.0 <= delay <= adapter._busy_text_debounce_seconds + 0.01


@pytest.mark.asyncio
async def test_queue_and_flush_text_debounce():
    adapter = _make_adapter()
    await adapter._queue_text_debounce("s1", _make_event("first", user_id="u1"))
    await adapter._queue_text_debounce("s1", _make_event("second", user_id="u1"))
    # Same sender merges into one buffered event; the timer task is scheduled.
    state = adapter._text_debounce["s1"]
    assert state.event.text == "first\nsecond"

    flushed = await adapter._flush_text_debounce_now("s1")
    assert flushed is True
    assert "s1" not in adapter._text_debounce
    assert "s1" in adapter._pending_messages
    assert adapter._pending_messages["s1"].text == "first\nsecond"


@pytest.mark.asyncio
async def test_discard_text_debounce():
    adapter = _make_adapter()
    await adapter._queue_text_debounce("s1", _make_event("hello"))
    assert "s1" in adapter._text_debounce
    adapter._discard_text_debounce("s1")
    assert "s1" not in adapter._text_debounce
    assert "s1" not in adapter._pending_messages
