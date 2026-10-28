"""An inline bypass command never completes the outgoing turn's review ownership.

Every resolvable slash command typed while a turn runs bypasses the session guard and is
dispatched inline (``BasePlatformAdapter._dispatch_inline_reply``). The outgoing turn parks its
review-ownership completion on that guard only once ``begin_gateway_review_admission`` runs --
behind the ingress awaits and the unbounded wait for the previous review's cancellation -- so a
``/status`` typed in that window sees the completion appear on the guard DURING its own handler
call. Nothing an inline handler nests ever parks there (``/retry`` re-homes the nested turn's
completion onto the command event), so a completion that shows up on the guard mid-call is the
outgoing turn's own late park: taking it would release the live-turn token with the command's
send before the turn even ran, and the turn's candidate would be refused at capture with no
owner-tagged slug. The outgoing task reads its own completion off the guard the moment its
handler returns; the inline dispatch leaves the guard alone.
"""

from __future__ import annotations

import asyncio

import pytest

from agent import review_admission
from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run_turn_review import _GatewayReviewAdmission
from gateway.session import SessionSource

SESSION_ID = "inline-bypass-late-park-session"
SESSION_KEY = "inline-bypass-late-park-key"


def _event(text: str) -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.TELEGRAM, chat_id="chat", chat_type="dm", user_id="user"
        ),
    )


@pytest.mark.asyncio
async def test_inline_bypass_leaves_a_late_parking_outgoing_completion_on_the_guard(
    monkeypatch,
):
    monkeypatch.setattr(BasePlatformAdapter, "__abstractmethods__", frozenset())
    adapter = BasePlatformAdapter(
        PlatformConfig(enabled=True, token="***"), Platform.TELEGRAM
    )
    profile_key = review_admission.current_profile_key()
    guard = asyncio.Event()
    # The outgoing task started under this guard; its begin() has not parked yet.
    adapter._active_sessions[SESSION_KEY] = guard
    token = review_admission.note_turn_started(SESSION_ID, profile_key)
    outgoing = _GatewayReviewAdmission(SESSION_ID, profile_key, token)
    completed = []

    def _finish(**kwargs):
        completed.append(kwargs)
        return outgoing.finish(**kwargs)

    async def _handler(event):
        # The /status handler yields; meanwhile the outgoing turn's review-cancel wait ends and
        # ``begin_gateway_review_admission`` parks the turn's completion on the session guard.
        await asyncio.sleep(0)
        guard._gateway_review_delivery_complete = _finish
        await asyncio.sleep(0)
        return "status: busy"

    async def _send_with_retry(**kwargs):
        return SendResult(success=True, message_id="1")

    adapter._message_handler = _handler
    monkeypatch.setattr(adapter, "_send_with_retry", _send_with_retry)
    try:
        await adapter._dispatch_inline_reply(_event("/status"), session_key=SESSION_KEY)

        assert completed == [], (
            "the inline /status completed the outgoing turn's ownership"
        )
        assert review_admission.other_live_turn(SESSION_ID, None, profile_key), (
            "the outgoing turn's review token was released while its turn still runs"
        )
        assert getattr(guard, "_gateway_review_delivery_complete", None) is _finish, (
            "the outgoing turn's completion was taken off the session guard"
        )
        assert adapter._active_sessions[SESSION_KEY] is guard
    finally:
        review_admission.note_turn_finished(SESSION_ID, token, profile_key)
