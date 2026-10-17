"""Gateway-internal events must never answer (or cancel) a pending clarify.

Reported on a live install, 2026-10-17 (×2, reproducible): a pending clarify card was posted to
Telegram and died with ``outcome=cancelled``, ``user_response: null`` within ~1-2
min — both times exactly when the next gateway-internal event (a kanban OOB
notification delivered as a wake) reached the same session. The wake is a synthetic
``MessageEvent(internal=True)`` admitted through the same inbound pipeline as human
messages, and two gates treated it as a candidate answer:

  * ``BasePlatformAdapter._handle_message_while_active`` routed it into the clarify
    text-intercept (``_dispatch_inline_reply``) instead of the busy handler, which
    queues internal events behind the turn (never interrupts or steers);
  * ``GatewayInboundMixin._hm_pending_reply_intercepts`` let it reach the clarify
    text-intercept, where unmatched prose on a choice prompt cancels the pending
    entry (``resolve_gateway_clarify(..., CANCELLED)``) — and on an open-ended
    prompt the notification text would have been accepted as the answer.

Both gates now require a non-internal event: a synthetic wake or notification can
never answer a prompt only the human owns. Human replies are unaffected.
"""

import asyncio
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource, build_session_key

WAKE_TEXT = "[kanban] Task t_1234abcd completed. Title: nightly build finished"


class _ClarifyBypassAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="test"), Platform.TELEGRAM)

    async def connect(self):
        return True

    async def disconnect(self):
        pass

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True, message_id="text")

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "private"}


def _event(text=WAKE_TEXT, *, internal=False):
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="12345",
            chat_type="private",
            user_id="user1",
        ),
        message_id="msg1",
        internal=internal,
    )


def _clear_clarify_state():
    from tools import clarify_gateway as cm

    with cm._lock:
        cm._entries.clear()
        cm._session_index.clear()
        cm._notify_cbs.clear()


def _bare_runner():
    """Bare GatewayRunner (object.__new__) with just what the intercepts touch."""
    from types import SimpleNamespace

    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._profile_adapters = {}
    runner.config = SimpleNamespace(multiplex_profiles=False)
    return runner


@pytest.mark.asyncio
async def test_internal_wake_does_not_cancel_pending_clarify():
    """The reported bug: a kanban wake delivered mid-prompt cancelled the card."""
    _clear_clarify_state()
    from tools import clarify_gateway as cm

    runner = _bare_runner()
    event = _event(internal=True)
    key = build_session_key(event.source)
    cm.register("clarify-1", key, "Push the wiki?", ["Yes", "No"])

    reply = await runner._hm_pending_reply_intercepts(event, event.source, key)

    assert reply is None  # falls through: the wake still runs its own turn
    entry = cm.get_pending_for_session(key, include_choice_prompts=True)
    assert entry is not None, "internal event consumed the pending clarify"
    assert not entry.event.is_set(), "internal event cancelled the pending clarify"
    assert entry.response is None


@pytest.mark.asyncio
async def test_internal_wake_does_not_answer_open_ended_clarify():
    """Open-ended prompts accept any text — the wake must not become the answer."""
    _clear_clarify_state()
    from tools import clarify_gateway as cm

    runner = _bare_runner()
    event = _event(internal=True)
    key = build_session_key(event.source)
    cm.register("clarify-1", key, "What should the title be?", None)

    await runner._hm_pending_reply_intercepts(event, event.source, key)

    entry = cm.get_pending_for_session(key, include_choice_prompts=True)
    assert entry is not None and not entry.event.is_set() and entry.response is None


@pytest.mark.asyncio
async def test_human_selection_still_resolves_pending_clarify():
    """Control: a typed answer from the human still resolves (and unblocks) it."""
    _clear_clarify_state()
    from tools import clarify_gateway as cm

    runner = _bare_runner()
    event = _event("2")
    key = build_session_key(event.source)
    cm.register("clarify-1", key, "Push the wiki?", ["Yes", "No"])

    reply = await runner._hm_pending_reply_intercepts(event, event.source, key)

    assert reply == ""
    entry = cm.get_pending_for_session(key, include_choice_prompts=True)
    assert entry is not None and entry.event.is_set()
    assert entry.response == "No"


@pytest.mark.asyncio
async def test_human_prose_on_choice_prompt_still_cancels():
    """Control: unmatched human prose keeps its existing cancel-and-route behavior."""
    _clear_clarify_state()
    from tools import clarify_gateway as cm

    runner = _bare_runner()
    event = _event("let's talk about something else")
    key = build_session_key(event.source)
    cm.register("clarify-1", key, "Push the wiki?", ["Yes", "No"])

    reply = await runner._hm_pending_reply_intercepts(event, event.source, key)

    assert reply is None
    entry = cm.get_pending_for_session(key, include_choice_prompts=True)
    assert entry is not None and entry.event.is_set()
    assert entry.response == cm.CANCELLED


@pytest.mark.asyncio
async def test_internal_wake_skips_clarify_bypass_and_is_queued_as_busy_followup():
    """Adapter gate: an internal wake must not be dispatched inline to the runner."""
    _clear_clarify_state()
    from tools import clarify_gateway as cm

    adapter = _ClarifyBypassAdapter()
    adapter._message_handler = AsyncMock(return_value="")
    adapter._busy_session_handler = AsyncMock(return_value=True)
    event = _event(internal=True)
    session_key = build_session_key(
        event.source,
        group_sessions_per_user=adapter.config.extra.get("group_sessions_per_user", True),
        thread_sessions_per_user=adapter.config.extra.get("thread_sessions_per_user", False),
    )
    adapter._active_sessions[session_key] = asyncio.Event()
    cm.register("clarify-1", session_key, "Push the wiki?", ["Yes", "No"])

    await adapter.handle_message(event)

    adapter._message_handler.assert_not_awaited()
    adapter._busy_session_handler.assert_awaited_once()
    entry = cm.get_pending_for_session(session_key, include_choice_prompts=True)
    assert entry is not None and not entry.event.is_set()


@pytest.mark.asyncio
async def test_human_reply_still_takes_the_clarify_bypass():
    """Control: a human message while blocked still reaches the text-intercept."""
    _clear_clarify_state()
    from tools import clarify_gateway as cm

    adapter = _ClarifyBypassAdapter()
    adapter._message_handler = AsyncMock(return_value="")
    adapter._busy_session_handler = AsyncMock(return_value=True)
    event = _event("2")
    session_key = build_session_key(
        event.source,
        group_sessions_per_user=adapter.config.extra.get("group_sessions_per_user", True),
        thread_sessions_per_user=adapter.config.extra.get("thread_sessions_per_user", False),
    )
    adapter._active_sessions[session_key] = asyncio.Event()
    cm.register("clarify-1", session_key, "Push the wiki?", ["Yes", "No"])

    await adapter.handle_message(event)

    adapter._message_handler.assert_awaited_once_with(event)
    adapter._busy_session_handler.assert_not_awaited()
