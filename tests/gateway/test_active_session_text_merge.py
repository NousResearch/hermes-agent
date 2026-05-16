"""Regression test for #4469.

When the agent is actively running (session present in
``adapter._active_sessions``) and the user fires off multiple TEXT
follow-ups in rapid succession, the previous behaviour was a single-slot
replacement at ``gateway/platforms/base.py``:

    self._pending_messages[session_key] = event

So three rapid messages ``A``, ``B``, ``C`` arriving while the agent was
still working on the initial turn produced a pending slot containing only
``C``; ``A`` and ``B`` were silently dropped.

The fix routes the follow-up through ``merge_pending_message_event(...,
merge_text=True)`` so TEXT events accumulate into the existing pending
event's text instead of clobbering it.  Photo / media bursts continue to
merge through the same helper (they always did).
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
)
from gateway.session import SessionSource, build_session_key


def _make_event(text: str, chat_id: str = "12345") -> MessageEvent:
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id=chat_id,
        chat_type="dm",
        user_id="u1",
    )
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=source,
        message_id=f"msg-{text[:8]}",
    )


def _make_adapter() -> BasePlatformAdapter:
    """Build a BasePlatformAdapter without running its heavy __init__.

    We only need the bits ``handle_message`` touches on the active-session
    path: ``_active_sessions``, ``_pending_messages``,
    ``_message_handler``, ``_busy_session_handler``, ``config``, ``platform``.
    """

    class _DummyAdapter(BasePlatformAdapter):  # type: ignore[misc]
        async def connect(self):
            pass

        async def disconnect(self):
            pass

        async def get_chat_info(self, chat_id):
            return None

        async def send(self, *args, **kwargs):
            return MagicMock(success=True, message_id="x", retryable=False)

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
    adapter._auto_tts_default = False
    adapter._auto_tts_enabled_chats = set()
    adapter._auto_tts_disabled_chats = set()
    adapter._typing_paused = set()
    return adapter


@pytest.mark.asyncio
async def test_rapid_text_followups_accumulate_instead_of_replacing():
    """Three rapid TEXT follow-ups during an active session must all
    survive in ``adapter._pending_messages[session_key].text``."""
    adapter = _make_adapter()
    first = _make_event("part one")
    session_key = build_session_key(first.source)

    # Mark the session as active so subsequent messages take the
    # "already running" branch in handle_message.
    adapter._active_sessions[session_key] = asyncio.Event()

    second = _make_event("part two")
    third = _make_event("part three")

    await adapter.handle_message(second)
    await adapter.handle_message(third)

    # Both rapid follow-ups must be preserved, not just the last one.
    pending = adapter._pending_messages[session_key]
    assert pending.text == "part two\npart three", (
        f"expected accumulated text, got {pending.text!r}"
    )
    # Interrupt event must be signalled exactly like before.
    assert adapter._active_sessions[session_key].is_set()


@pytest.mark.asyncio
async def test_single_followup_is_stored_as_is():
    """One TEXT follow-up still lands as the event object itself
    (no spurious wrapping / mutation) — guards against the merge path
    breaking the simple case."""
    adapter = _make_adapter()
    first = _make_event("only one")
    session_key = build_session_key(first.source)

    adapter._active_sessions[session_key] = asyncio.Event()
    await adapter.handle_message(first)

    pending = adapter._pending_messages[session_key]
    assert pending is first
    assert pending.text == "only one"
    assert adapter._active_sessions[session_key].is_set()
