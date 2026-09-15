"""Regression: an interrupt-and-clear must not leave a clarify prompt armed.

A turn parked in ``clarify_gateway.wait_for_response`` is blocked on a
``threading.Event``, so the cooperative ``agent.interrupt()`` issued by
``/stop`` (and ``/new``) cannot reach it and the turn's ``finally`` — the only
place that cancelled the clarify — never runs at interrupt time.  The entry
therefore survives the stop, and the user's NEXT message is intercepted by the
gateway's clarify hook and routed into the already-invalidated turn, whose
reply is suppressed as stale.  Net effect: the message is silently swallowed
and the user has to send it twice.

``_interrupt_and_clear_session`` is the shared chokepoint for every such path,
so the cancellation belongs there.
"""

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.run import GatewayRunner
from gateway.session import SessionSource
from gateway.platforms.base import Platform
from tools import clarify_gateway


SESSION_KEY = "agent:main:discord:thread:12345:12345"


def _source():
    return SessionSource(
        platform=Platform.DISCORD,
        chat_type="group",
        chat_id="12345",
        thread_id="12345",
        user_id="u1",
    )


def _bare_runner():
    runner = object.__new__(GatewayRunner)
    runner._sessions = {}
    runner.session_store = MagicMock()
    runner.session_store._entries = {}
    runner._restore_pending_one_turn_model_override = MagicMock()
    runner._adapter_for_source = MagicMock(return_value=None)
    runner._thread_metadata_for_source = MagicMock(return_value=None)
    runner._release_running_agent_state = MagicMock()
    runner._evict_cached_agent = MagicMock()
    return runner


@pytest.fixture(autouse=True)
def _clean_clarify_state():
    clarify_gateway.clear_session(SESSION_KEY)
    yield
    clarify_gateway.clear_session(SESSION_KEY)


@pytest.mark.asyncio
async def test_interrupt_cancels_pending_clarify_and_unblocks_the_waiter():
    """The user-visible contract: after a stop, no clarify can intercept.

    Asserts both halves of the symptom — the entry is gone (so the next
    inbound message reaches the agent normally instead of being swallowed),
    and the parked agent thread is released rather than waiting out the full
    clarify timeout.
    """
    entry = clarify_gateway.register(
        clarify_id="stopclr0001",
        session_key=SESSION_KEY,
        question="rotate now or defer?",
        choices=[],
    )
    assert clarify_gateway.get_pending_for_session(SESSION_KEY) is not None

    runner = _bare_runner()
    await runner._interrupt_and_clear_session(
        SESSION_KEY,
        _source(),
        interrupt_reason="stop_command",
        invalidation_reason="stop_command",
    )

    # The next inbound message cannot be intercepted as an answer to the
    # dead turn — this is the swallowed-message symptom.
    assert clarify_gateway.get_pending_for_session(SESSION_KEY) is None
    assert not clarify_gateway.has_pending(SESSION_KEY)
    # The blocked agent thread was released via the cancellation sentinel.
    assert entry.event.is_set()


@pytest.mark.asyncio
async def test_multi_choice_clarify_is_cancelled_too():
    """A choice-prompt clarify is armed on the same interception path.

    ``get_pending_for_session(include_choice_prompts=True)`` is what the
    gateway consults when the user types instead of tapping a choice, so a
    surviving choice entry swallows the next message just as an open-ended
    one does.
    """
    clarify_gateway.register(
        clarify_id="stopclr0002",
        session_key=SESSION_KEY,
        question="pick one",
        choices=["a", "b"],
    )
    assert clarify_gateway.get_pending_for_session(
        SESSION_KEY, include_choice_prompts=True
    ) is not None

    runner = _bare_runner()
    await runner._interrupt_and_clear_session(
        SESSION_KEY,
        _source(),
        interrupt_reason="stop_command",
        invalidation_reason="stop_command",
    )

    assert clarify_gateway.get_pending_for_session(
        SESSION_KEY, include_choice_prompts=True
    ) is None


@pytest.mark.asyncio
async def test_clarify_cancelled_on_the_release_state_false_path():
    """Same contract on the sibling call path.

    ``_interrupt_and_clear_session`` is also invoked with
    ``release_running_state=False``; the cancellation must not be tucked
    inside that branch, or the bug stays reachable through it.
    """
    clarify_gateway.register(
        clarify_id="stopclr0003",
        session_key=SESSION_KEY,
        question="pick one",
        choices=["a", "b"],
    )
    runner = _bare_runner()
    await runner._interrupt_and_clear_session(
        SESSION_KEY,
        _source(),
        interrupt_reason="new_command",
        invalidation_reason="new_command",
        release_running_state=False,
    )
    assert not clarify_gateway.has_pending(SESSION_KEY)


@pytest.mark.asyncio
async def test_no_pending_clarify_is_a_clean_noop():
    """The common case (nothing pending) must not raise or misbehave."""
    runner = _bare_runner()
    await runner._interrupt_and_clear_session(
        SESSION_KEY,
        _source(),
        interrupt_reason="stop_command",
        invalidation_reason="stop_command",
    )
    assert clarify_gateway.get_pending_for_session(SESSION_KEY) is None
    assert not clarify_gateway.has_pending(SESSION_KEY)


@pytest.mark.asyncio
async def test_other_sessions_are_untouched():
    """Cancellation is scoped to the interrupted session only.

    A gateway serves many concurrent sessions; stopping one must not cancel
    another's in-flight question.
    """
    other_key = "agent:main:discord:thread:99999:99999"
    clarify_gateway.clear_session(other_key)
    try:
        clarify_gateway.register(
            clarify_id="stopclr0004",
            session_key=SESSION_KEY,
            question="mine",
            choices=[],
        )
        other = clarify_gateway.register(
            clarify_id="stopclr0005",
            session_key=other_key,
            question="theirs",
            choices=[],
        )

        runner = _bare_runner()
        await runner._interrupt_and_clear_session(
            SESSION_KEY,
            _source(),
            interrupt_reason="stop_command",
            invalidation_reason="stop_command",
        )

        assert not clarify_gateway.has_pending(SESSION_KEY)
        assert clarify_gateway.has_pending(other_key)
        assert not other.event.is_set()
    finally:
        clarify_gateway.clear_session(other_key)


@pytest.mark.asyncio
@pytest.mark.parametrize("release", [True, False])
@pytest.mark.parametrize("mode", ["open", "other", "choice"])
@pytest.mark.parametrize("sentinel", [True, False])
async def test_interrupted_waiter_and_successor_ownership(release, mode, sentinel):
    from gateway.run import _AGENT_PENDING_SENTINEL
    from gateway.run_turn_runner import TurnRunner

    runner = _bare_runner()
    state = runner._session_state(SESSION_KEY)
    state.turn.agent = _AGENT_PENDING_SENTINEL if sentinel else MagicMock(
        _gateway_turn_process_task_id="", _gateway_turn_process_baseline=None)
    old_turn = object.__new__(TurnRunner)
    old_turn._ctx = SimpleNamespace(session_key=SESSION_KEY)
    # Force the real conversation wrapper through its finally without running a model.
    old_turn._native_image_run_message = MagicMock(side_effect=RuntimeError("finish old turn"))
    entry = clarify_gateway.register("owned-old", SESSION_KEY, "old?",
                                     None if mode == "open" else ["a", "b"], owner=old_turn)
    if mode == "other":
        assert clarify_gateway.mark_awaiting_text(entry.clarify_id)
    waiting = threading.Event()
    original_wait = entry.event.wait

    def observed_wait(timeout=None):
        waiting.set()
        return original_wait(timeout)

    entry.event.wait = observed_wait
    waiter = asyncio.create_task(asyncio.to_thread(clarify_gateway.wait_for_response, entry.clarify_id, 10))
    assert await asyncio.to_thread(waiting.wait, 3)
    paused, resume = asyncio.Event(), asyncio.Event()

    class Adapter:
        async def interrupt_session_activity(self, *args):
            paused.set()
            await resume.wait()

    runner._adapter_for_source.return_value = Adapter()
    source = _source()
    event = SimpleNamespace(source=source)
    if not release:
        operation = runner._interrupt_and_clear_session(
            SESSION_KEY, source, interrupt_reason="stop_command",
            invalidation_reason="stop_command", release_running_state=False)
    elif mode == "open":
        operation = runner._busy_stop_command(event, SESSION_KEY, source)
    elif mode == "other":
        runner._handle_reset_command = AsyncMock(return_value="reset")
        operation = runner._busy_new_command(event, SESSION_KEY, source)
    else:
        runner._async_session_store = SimpleNamespace(
            _store=runner.session_store,
            get_or_create_session=AsyncMock(return_value=SimpleNamespace(session_key="caller")))
        runner._sibling_thread_run_keys = MagicMock(return_value=[SESSION_KEY])
        runner._is_user_authorized_for_source = MagicMock(return_value=True)
        operation = runner._handle_stop_command(event)
    interrupt = asyncio.create_task(operation)
    try:
        await asyncio.wait_for(paused.wait(), 3)
        assert await asyncio.wait_for(waiter, 3) == ""
        assert clarify_gateway.attempt_text_response_for_session(SESSION_KEY, "a") == clarify_gateway.TEXT_NO_PENDING
        successor = clarify_gateway.register("owned-new", SESSION_KEY, "new?", None, owner=object())
        runner._invalidate_session_run_generation(SESSION_KEY, reason="successor")
        resume.set()
        await interrupt
        with pytest.raises(RuntimeError, match="finish old turn"):
            old_turn._run_conversation_with_approval(None, [], None, None, None)
        assert not successor.event.is_set()
        assert clarify_gateway.resolve_text_response_for_session(SESSION_KEY, "successor answer")
        assert clarify_gateway.wait_for_response(successor.clarify_id, 1) == "successor answer"
    finally:
        resume.set()
        clarify_gateway.clear_session(SESSION_KEY)
        await interrupt
        await waiter


@pytest.mark.parametrize("failed", [True, False])
def test_delayed_send_failure_does_not_cancel_successor(failed):
    from gateway.platforms.base import SendResult
    from gateway.run_turn_runner import TurnRunner

    old_turn = object.__new__(TurnRunner)
    old_turn._ctx = SimpleNamespace(
        session_key=SESSION_KEY, _status_adapter=MagicMock(), _run_still_current=lambda: True,
        _status_chat_id="12345", _status_thread_metadata=None)
    old_turn._close_native_stream_boundary = MagicMock()
    old_turn._stream_consumer = MagicMock(return_value=None)
    successor = None

    class DelayedSend:
        def result(self, timeout):
            nonlocal successor
            clarify_gateway.clear_session(SESSION_KEY)
            successor = clarify_gateway.register("send-new", SESSION_KEY, "new?", None, owner=object())
            if failed:
                raise RuntimeError("definitive transport failure")
            return SendResult(success=False, error="egress declined: destination refused")

    old_turn._schedule = MagicMock(return_value=DelayedSend())
    result = old_turn._clarify_callback_sync("old?", None)
    assert result.startswith("[clarify prompt could not be delivered")
    assert successor is not None and not successor.event.is_set()
    assert clarify_gateway.resolve_text_response_for_session(SESSION_KEY, "new answer")


@pytest.mark.asyncio
async def test_interrupted_generation_cannot_register_again_or_overwrite_answer():
    from gateway.run_turn_runner import TurnRunner

    runner = _bare_runner()
    generation = runner._invalidate_session_run_generation(SESSION_KEY, reason="start")
    turn = object.__new__(TurnRunner)
    turn._ctx = SimpleNamespace(
        session_key=SESSION_KEY, _status_adapter=MagicMock(),
        _run_still_current=lambda: runner._is_session_run_current(SESSION_KEY, generation))
    entry = clarify_gateway.register("won-answer", SESSION_KEY, "question?", None, owner=turn)
    assert clarify_gateway.resolve_gateway_clarify(entry.clarify_id, "already won")
    await runner._interrupt_and_clear_session(
        SESSION_KEY, _source(), interrupt_reason="stop_command", invalidation_reason="stop_command")
    assert entry.response == "already won"
    assert turn._clarify_callback_sync("late question?", None) == "[Clarify cancelled]"
    assert not clarify_gateway.has_pending(SESSION_KEY)
