"""Regression tests for #125781: internal events must not answer pending human prompts.

An internal completion event (``internal=True``, e.g. a background-process or
delegation completion) carries ``allow_gateway_control=True`` so it can reach
gateway routing. Both pending-reply fast paths — the runner's
``_hm_pending_reply_intercepts`` (idle lane) and the adapter's busy-session
clarify bypass — checked only ``allow_gateway_control`` and not ``internal``,
so completion prose could cancel a pending choice clarify (rejected prose
resolves it with an empty response), become the answer to an open-ended
clarify, confirm/cancel a pending slash-confirm, or answer a pending update
prompt.

The fix: both fast paths exclude ``internal`` events; the completion then
falls through to the internal-safe routing (queued behind a busy turn, or a
new turn when idle) and the pending question stays armed for the human.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import AsyncMock, MagicMock, Mock

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

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    SendResult,
)
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource


def _clear_clarify_state():
    from tools import clarify_gateway as cm

    with cm._lock:
        cm._entries.clear()
        cm._session_index.clear()
        cm._notify_cbs.clear()


def _clear_confirm_state():
    from tools import slash_confirm as sc

    with sc._lock:
        sc._pending.clear()


def _source() -> SessionSource:
    return SessionSource(
        platform=MagicMock(value="telegram"),
        chat_id="123",
        chat_type="private",
        user_id="user1",
    )


def _event(text: str, *, internal: bool = False) -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=_source(),
        message_id="msg1",
        internal=internal,
    )


def _make_runner() -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner._peek_session_state = lambda _key: None
    runner._delivery_adapter_for = lambda _source: None
    runner._hm_write_update_response = Mock(return_value=None)
    return runner


@pytest.mark.asyncio
async def test_internal_event_does_not_cancel_pending_choice_clarify():
    """Completion prose must not resolve (cancel) a pending choice clarify."""
    _clear_clarify_state()
    from tools import clarify_gateway as cm

    runner = _make_runner()
    key = "telegram:123"
    entry = cm.register("clarify-1", key, "Continue?", ["Accept", "Cancel"])

    result = await runner._hm_pending_reply_intercepts(
        _event("background task completed", internal=True), _source(), key)

    assert result is None  # not consumed as a reply
    still = cm.get_pending_for_session(key, include_choice_prompts=True)
    assert still is not None and still.clarify_id == "clarify-1"
    assert not entry.event.is_set()

    # The human answer still resolves the ORIGINAL question, exactly once.
    assert cm.attempt_text_response_for_session(key, "Cancel") == cm.TEXT_RESOLVED
    assert entry.response == "Cancel"
    assert cm.attempt_text_response_for_session(key, "Cancel") == cm.TEXT_NO_PENDING


@pytest.mark.asyncio
async def test_internal_event_does_not_answer_open_ended_clarify():
    """Completion prose must not become the answer to an open-ended clarify."""
    _clear_clarify_state()
    from tools import clarify_gateway as cm

    runner = _make_runner()
    key = "telegram:123"
    entry = cm.register("clarify-2", key, "What should I deploy?", None)

    result = await runner._hm_pending_reply_intercepts(
        _event("background task completed", internal=True), _source(), key)

    assert result is None
    still = cm.get_pending_for_session(key, include_choice_prompts=True)
    assert still is not None and still.clarify_id == "clarify-2"
    assert not entry.event.is_set()

    assert cm.attempt_text_response_for_session(key, "deploy staging") == cm.TEXT_RESOLVED
    assert entry.response == "deploy staging"


@pytest.mark.asyncio
async def test_internal_event_does_not_resolve_pending_slash_confirm():
    """Internal text matching a confirm keyword must not run the confirm handler."""
    _clear_confirm_state()
    from tools import slash_confirm as sc

    runner = _make_runner()
    key = "telegram:123"
    handler = AsyncMock(return_value="confirmed")
    sc.register(key, "confirm-1", "reload-mcp", handler)

    result = await runner._hm_pending_reply_intercepts(
        _event("cancel", internal=True), _source(), key)

    assert result is None
    handler.assert_not_awaited()
    assert sc.get_pending(key) is not None

    # The human reply still resolves it.
    result = await runner._hm_pending_reply_intercepts(_event("cancel"), _source(), key)
    handler.assert_awaited_once_with("cancel")
    assert result == "confirmed"
    assert sc.get_pending(key) is None


@pytest.mark.asyncio
async def test_internal_event_does_not_answer_pending_update_prompt():
    """Internal text must not be written as the answer to a pending update prompt."""
    runner = _make_runner()
    key = "telegram:123"
    state = MagicMock()
    state.persistent.update_prompt_pending = True
    runner._peek_session_state = lambda _key: state

    result = await runner._hm_pending_reply_intercepts(
        _event("y", internal=True), _source(), key)

    assert result is None
    runner._hm_write_update_response.assert_not_called()
    assert state.persistent.update_prompt_pending is True


@pytest.mark.asyncio
async def test_human_reply_still_resolves_pending_clarify_via_intercepts():
    """Control: the internal exclusion must not narrow the human reply path."""
    _clear_clarify_state()
    from tools import clarify_gateway as cm

    runner = _make_runner()
    key = "telegram:123"
    entry = cm.register("clarify-3", key, "What should I deploy?", None)

    result = await runner._hm_pending_reply_intercepts(
        _event("deploy staging"), _source(), key)

    assert result == ""  # resolved replies return "" so adapters don't double-post
    assert entry.event.is_set()
    assert entry.response == "deploy staging"


# ── Adapter busy-session bypass ─────────────────────────────────────────────


class _ClarifyBypassAdapter(BasePlatformAdapter):
    """Real adapter with the minimum wired handlers (mirrors
    tests/gateway/test_clarify_active_session_bypass.py)."""

    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="test"), Platform.TELEGRAM)

    async def connect(self, *, is_reconnect: bool = False):
        return True

    async def disconnect(self):
        pass

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True, message_id="text")

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "private"}


def _make_busy_adapter():
    import asyncio

    from gateway.session import build_session_key

    adapter = _ClarifyBypassAdapter()
    adapter._message_handler = AsyncMock(return_value="")
    adapter._busy_session_handler = AsyncMock(return_value=True)
    event = _event("background task completed")
    key = build_session_key(
        event.source,
        group_sessions_per_user=adapter.config.extra.get("group_sessions_per_user", True),
        thread_sessions_per_user=adapter.config.extra.get("thread_sessions_per_user", False),
    )
    adapter._active_sessions[key] = asyncio.Event()
    return adapter, key


@pytest.mark.asyncio
async def test_busy_session_internal_event_skips_clarify_bypass():
    """While busy, an internal completion must not be routed to the clarify
    text-intercept inline dispatch — it belongs to the busy handler (queued)."""
    _clear_clarify_state()
    from tools import clarify_gateway as cm

    adapter, key = _make_busy_adapter()
    cm.register("clarify-busy", key, "Pick one", ["A", "B"])

    event = _event("background task completed", internal=True)
    await adapter.handle_message(event)

    adapter._message_handler.assert_not_awaited()
    adapter._busy_session_handler.assert_awaited_once()
    pending = cm.get_pending_for_session(key, include_choice_prompts=True)
    assert pending is not None and pending.clarify_id == "clarify-busy"


@pytest.mark.asyncio
async def test_busy_session_human_reply_still_reaches_clarify_bypass():
    """Control: a human reply while busy still reaches the clarify intercept."""
    _clear_clarify_state()
    from tools import clarify_gateway as cm

    adapter, key = _make_busy_adapter()
    cm.register("clarify-busy2", key, "Pick one", ["A", "B"])

    event = _event("None of those are valid options")
    await adapter.handle_message(event)

    adapter._message_handler.assert_awaited_once_with(event)
    adapter._busy_session_handler.assert_not_awaited()
