"""A cancel that lands while ``_process_message_background`` is unwinding still releases the
turn's gateway review ownership.

The adapter task completes the review ownership ``gateway/run_turn_review.py`` parked on its
session Event (``_complete_review_delivery``) in its ``finally``, after it released the turn
marker, stopped the typing refresh and fired the post-delivery callback. Each of those is an
await (a DB marker CAS off-loop, platform round trips, a bounded callback), and the session
task stays registered until ``_finish_session_task``: a /stop, /new or /reset typed as the
reply lands (``_dispatch_active_session_command`` -> ``cancel_session_processing``) or an
adapter teardown cancels a task suspended exactly there. The ``CancelledError`` is raised at
that await, so without an inner guard the completion never runs, ``note_turn_finished`` is
never called, and the ``(profile, session)`` bucket in ``agent.review_admission`` keeps the
token for the process lifetime: every later automatic review on the session is refused as
``live_turn_active``.
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

SESSION_ID = "finally-cancel-session"
SESSION_KEY = "finally-cancel-key"
# The finally's awaits, in order; the cancel lands at each of them.
_FINALLY_AWAITS = (
    "_release_turn_marker",
    "_stop_typing_refresh",
    "_fire_post_delivery_callback",
)


def _event(text: str = "hello") -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.TELEGRAM, chat_id="chat", chat_type="dm", user_id="user"
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("block_at", _FINALLY_AWAITS)
async def test_cancel_during_the_adapter_finally_still_completes_review_ownership(
    monkeypatch, block_at
):
    monkeypatch.setattr(BasePlatformAdapter, "__abstractmethods__", frozenset())
    adapter = BasePlatformAdapter(
        PlatformConfig(enabled=True, token="***"), Platform.TELEGRAM
    )
    monkeypatch.setattr(adapter.config, "typing_indicator", False, raising=False)
    profile_key = review_admission.current_profile_key()
    token = review_admission.note_turn_started(SESSION_ID, profile_key)
    admission = _GatewayReviewAdmission(SESSION_ID, profile_key, token)
    outcomes = []

    def _finish(*, delivery_succeeded, cause=None, outer=None):
        outcomes.append((delivery_succeeded, cause))
        return admission.finish(delivery_succeeded=delivery_succeeded, cause=cause)

    event = _event()

    async def _handler(_event):
        _event._gateway_review_delivery_complete = _finish
        return "visible response"

    async def _send(*_args, **_kwargs):
        return SendResult(success=True, message_id="sent")

    adapter.set_message_handler(_handler)
    adapter.send = _send
    adapter._active_sessions[SESSION_KEY] = asyncio.Event()

    # The handler returned and the reply went out: the task is past its processing-complete
    # hook, so the next call of the blocked helper is the finally's own.
    unwinding = asyncio.Event()
    orig_hook = adapter._run_processing_hook

    async def _run_processing_hook(name, *args, **kwargs):
        if name == "on_processing_complete":
            unwinding.set()
        return await orig_hook(name, *args, **kwargs)

    blocked, gate = asyncio.Event(), asyncio.Event()
    orig_blocked = getattr(adapter, block_at)

    async def _suspend_in_finally(*args, **kwargs):
        if unwinding.is_set() and not blocked.is_set():
            blocked.set()
            await gate.wait()
        return await orig_blocked(*args, **kwargs)

    monkeypatch.setattr(adapter, "_run_processing_hook", _run_processing_hook)
    monkeypatch.setattr(adapter, block_at, _suspend_in_finally)

    task = asyncio.create_task(adapter._process_message_background(event, SESSION_KEY))
    adapter._track_session_task(SESSION_KEY, task)
    await asyncio.wait_for(blocked.wait(), 5)
    try:
        # /stop or /new arrives now: the session command cancels the suspended task.
        await adapter.cancel_session_processing(
            SESSION_KEY, release_guard=False, discard_pending=False
        )
        assert task.done()
        live_after = review_admission.other_live_turn(SESSION_ID, None, profile_key)
    finally:
        gate.set()
        # Balance the registry whatever happened, so other tests are unaffected.
        review_admission.note_turn_finished(SESSION_ID, token, profile_key)
    assert outcomes == [(True, None)], (
        f"review ownership was not completed with the delivery (cancel at {block_at})"
    )
    assert live_after is False, "the review token outlived the cancelled task"
