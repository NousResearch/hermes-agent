"""Tests for channel turn-coalescing.

A platform that mints a fresh per-message session for each top-level channel
message (Slack with ``reply_in_thread=true`` — the default — falls back to
``thread_ts = ts``) would otherwise let a rapid same-user follow-up race the
in-flight turn in its own session — the classic "agent answers a now-stale
question right after the real answer lands" bug. The coalescing index reattaches
such a follow-up to the author's active top-level turn so it is routed through
the normal busy handler (queue/cascade, or interrupt only a cheap turn).

Same-user / same-chat only: thread replies, other users, and DMs are untouched,
and under ``reply_in_thread=false`` the sessions already share a key so this
never fires.
"""

from __future__ import annotations

import asyncio
import sys
import types
from unittest.mock import AsyncMock, MagicMock

import pytest

# Minimal telegram stub so importing gateway.platforms.base does not pull in the
# real python-telegram-bot dependency (mirrors test_active_session_text_merge).
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


def _make_event(
    text: str,
    *,
    chat_id: str = "C1",
    chat_type: str = "channel",
    user_id: str = "u1",
    ts: str | None = None,
    thread_id: str | None = "__own__",
) -> MessageEvent:
    """Build an event. ``thread_id="__own__"`` mimics Slack's per-message
    top-level session (``thread_ts = ts``); pass an explicit value for a reply."""
    message_id = ts or f"ts-{text[:8]}"
    resolved_thread = message_id if thread_id == "__own__" else thread_id
    source = SessionSource(
        platform=Platform.SLACK,
        chat_id=chat_id,
        chat_type=chat_type,
        user_id=user_id,
        thread_id=resolved_thread,
    )
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=source,
        message_id=message_id,
    )


class _DummyAdapter(BasePlatformAdapter):  # type: ignore[misc]
    async def connect(self):
        pass

    async def disconnect(self):
        pass

    async def get_chat_info(self, chat_id):
        return None

    async def send(self, *args, **kwargs):
        return SendResult(success=True, message_id="x")


def _make_adapter() -> BasePlatformAdapter:
    adapter = object.__new__(_DummyAdapter)
    adapter.config = PlatformConfig(enabled=True, token="***")
    adapter.platform = Platform.SLACK
    adapter._message_handler = AsyncMock(return_value=None)
    adapter._busy_session_handler = None
    adapter._active_sessions = {}
    adapter._pending_messages = {}
    adapter._session_tasks = {}
    adapter._active_topfollow = {}
    adapter._background_tasks = set()
    adapter._post_delivery_callbacks = {}
    adapter._expected_cancelled_tasks = set()
    adapter._fatal_error_code = None
    adapter._fatal_error_message = None
    adapter._fatal_error_retryable = True
    adapter._fatal_error_handler = None
    adapter._running = True
    adapter._busy_text_mode = ""  # direct-merge, no debounce
    adapter._busy_text_debounce_seconds = 0.1
    adapter._busy_text_hard_cap_seconds = 1.0
    adapter._text_debounce = {}
    adapter._auto_tts_default = False
    adapter._auto_tts_enabled_chats = set()
    adapter._auto_tts_disabled_chats = set()
    adapter._typing_paused = set()
    return adapter


# ---- _topfollow_key ---------------------------------------------------------

def test_topfollow_key_for_top_level_channel():
    adapter = _make_adapter()
    ev = _make_event("hi", ts="ts1")
    assert adapter._topfollow_key(ev) == "C1\x1fu1"


def test_topfollow_key_none_for_dm():
    adapter = _make_adapter()
    assert adapter._topfollow_key(_make_event("hi", chat_type="dm")) is None


def test_topfollow_key_none_for_thread_reply():
    adapter = _make_adapter()
    # Anchor points at an earlier parent (!= this message's id) → real reply.
    ev = _make_event("hi", ts="ts2", thread_id="parent-ts")
    assert adapter._topfollow_key(ev) is None


def test_topfollow_key_none_without_user():
    adapter = _make_adapter()
    ev = _make_event("hi", ts="ts1")
    ev.source.user_id = None
    assert adapter._topfollow_key(ev) is None


# ---- coalescing in handle_message -------------------------------------------

@pytest.mark.asyncio
async def test_same_user_top_level_followup_coalesces_into_active_turn():
    adapter = _make_adapter()
    first = _make_event("which pipeline?", ts="ts1")
    active_key = build_session_key(first.source)
    # Simulate the first turn being in flight, registered in both structures.
    adapter._active_sessions[active_key] = asyncio.Event()
    adapter._active_topfollow["C1\x1fu1"] = active_key

    followup = _make_event("you there?", ts="ts2")  # new top-level → new key
    assert build_session_key(followup.source) != active_key

    await adapter.handle_message(followup)

    # No second concurrent session was started; it queued behind the active one.
    assert set(adapter._active_sessions) == {active_key}
    assert adapter._pending_messages[active_key].text == "you there?"


@pytest.mark.asyncio
async def test_other_user_top_level_does_not_coalesce():
    adapter = _make_adapter()
    first = _make_event("which pipeline?", ts="ts1", user_id="u1")
    active_key = build_session_key(first.source)
    adapter._active_sessions[active_key] = asyncio.Event()
    adapter._active_topfollow["C1\x1fu1"] = active_key

    other = _make_event("status?", ts="ts9", user_id="u2")
    await adapter.handle_message(other)

    # Different user → not folded into u1's turn; u1's pending stays empty.
    assert active_key not in adapter._pending_messages


@pytest.mark.asyncio
async def test_thread_reply_does_not_coalesce():
    adapter = _make_adapter()
    first = _make_event("which pipeline?", ts="ts1")
    active_key = build_session_key(first.source)
    adapter._active_sessions[active_key] = asyncio.Event()
    adapter._active_topfollow["C1\x1fu1"] = active_key

    reply = _make_event("reply in old thread", ts="ts2", thread_id="old-parent")
    await adapter.handle_message(reply)

    assert active_key not in adapter._pending_messages


# ---- index lifecycle --------------------------------------------------------

def test_release_session_guard_clears_topfollow_index():
    adapter = _make_adapter()
    adapter._active_sessions["k"] = asyncio.Event()
    adapter._active_topfollow["C1\x1fu1"] = "k"
    adapter._release_session_guard("k")
    assert adapter._active_topfollow == {}


def test_heal_stale_lock_clears_topfollow_index():
    adapter = _make_adapter()
    adapter._active_sessions["k"] = asyncio.Event()
    adapter._active_topfollow["C1\x1fu1"] = "k"

    class _DoneTask:
        def done(self):
            return True

    adapter._session_tasks["k"] = _DoneTask()
    assert adapter._heal_stale_session_lock("k") is True
    assert adapter._active_topfollow == {}
