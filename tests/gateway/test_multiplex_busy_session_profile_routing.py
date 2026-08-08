"""Regression tests: a secondary profile's busy-session state must be keyed
under its own ``agent:<profile>`` namespace, not the default profile's
``agent:main`` bucket.

``BasePlatformAdapter.handle_message`` runs its active-session busy check
(and builds the session key used everywhere downstream) *before* it ever
calls ``self._message_handler``. ``_make_profile_message_handler`` (see
``gateway/run.py``) only stamps ``event.source.profile`` *inside* that
handler -- too late for the busy check above it. Without an adapter-level
``profile_name`` known synchronously at construction time, every secondary
profile's busy/pending/debounce state silently collided under ``agent:main``,
and any busy-session reply for that profile went out through the *default*
profile's adapter/bot instead of its own.
"""

from __future__ import annotations

import asyncio
import sys
import types
from unittest.mock import AsyncMock, MagicMock

import pytest

# Minimal telegram stub so importing gateway.platforms.base does not pull in
# the real python-telegram-bot dependency (mirrors test_active_session_text_merge.py).
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


def _make_event(text: str, chat_id: str = "12345") -> MessageEvent:
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id=chat_id,
        chat_type="dm",
        user_id="u1",
        user_name=None,
        thread_id=None,
    )
    return MessageEvent(
        text=text, message_type=MessageType.TEXT, source=source, message_id=f"msg-{text[:8]}"
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


def _make_adapter(profile_name: str | None) -> BasePlatformAdapter:
    """Build a BasePlatformAdapter without running its heavy __init__."""
    adapter = object.__new__(_DummyAdapter)
    adapter.config = PlatformConfig(enabled=True, token="***")
    adapter.platform = Platform.TELEGRAM
    adapter.profile_name = profile_name
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


@pytest.mark.asyncio
async def test_secondary_profile_source_profile_stamped_before_busy_check():
    """event.source.profile must be set synchronously -- before the busy
    check runs -- not deferred to inside self._message_handler."""
    adapter = _make_adapter("coder")
    event = _make_event("hello")
    assert event.source.profile is None  # unstamped, as a real inbound event is

    await adapter.handle_message(event)

    assert event.source.profile == "coder"


@pytest.mark.asyncio
async def test_secondary_profile_busy_session_keys_are_profile_scoped():
    """A secondary profile's busy-session state must live under its own
    agent:<profile> namespace, not collide with agent:main."""
    coder_adapter = _make_adapter("coder")
    coder_adapter._busy_text_mode = ""  # direct-merge, no debounce (see test_active_session_text_merge.py)

    probe_source = _make_event("first").source
    default_key = build_session_key(probe_source, profile=None)
    coder_key = build_session_key(probe_source, profile="coder")
    assert default_key != coder_key  # sanity: the two namespaces must differ

    # Simulate an in-flight turn on the coder profile only.
    coder_adapter._active_sessions[coder_key] = asyncio.Event()

    # A follow-up arrives on the coder bot while its session is busy.
    await coder_adapter.handle_message(_make_event("are you there?"))

    # Must be recognized as busy under the coder-scoped key -- not silently
    # dispatched straight to self._message_handler under the wrong (or no)
    # namespace, which is what happened before source.profile was stamped
    # early enough for this check to see it.
    assert coder_key in coder_adapter._pending_messages
    assert default_key not in coder_adapter._pending_messages
    coder_adapter._message_handler.assert_not_called()
