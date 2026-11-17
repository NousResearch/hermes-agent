"""The boot auto-resume's pre-claimed slot belongs to its own synthetic event.

``_schedule_resume_pending_sessions`` claims the session slot (the pending sentinel) before it spawns
``_run_startup_resume_event``, so an inbound message that arrives first queues instead of building a
second agent. The resume event's own ``_handle_message`` then met that sentinel at the busy-session
guard and bounced into the pending queue; the turn only ran when the adapter's late-arrival drain
re-dispatched it in a task nobody waited for. So the startup wrapper, whose job is to hold inbound
until the resumed turn finishes, returned while the turn was still running, and its ``finally``
released whatever pending claim the slot held, even one taken by another event.
"""
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
from gateway.platforms.base import SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner, _AGENT_PENDING_SENTINEL
from gateway.session import SessionEntry
from tests.gateway.restart_test_helpers import make_restart_runner, make_restart_source


def _pending(runner, chat_id):
    source = make_restart_source(chat_id=chat_id)
    key = runner._session_key_for_source(source)
    runner.session_store._entries = {key: SessionEntry(
        session_key=key, session_id="sid", created_at=datetime.now(), updated_at=datetime.now(),
        origin=source, platform=Platform.TELEGRAM, chat_type="dm", resume_pending=True,
        resume_reason="restart_interrupted", last_resume_marked_at=datetime.now(),
    )}
    return source, key


def _real_pipeline(runner, adapter, run_agent):
    """The production dispatch chain: adapter.handle_message -> _process_message_background ->
    GatewayRunner._handle_message, with only the agent run and the leaf sends stubbed."""
    runner._handle_message = GatewayRunner._handle_message.__get__(runner, GatewayRunner)
    runner._release_running_agent_state = GatewayRunner._release_running_agent_state.__get__(runner, GatewayRunner)
    runner._check_slash_access = lambda *a, **k: None
    runner._begin_session_run_generation = lambda key: 1
    runner._is_session_run_current = lambda key, generation: True
    runner._invalidate_session_run_generation = lambda *a, **k: 0
    runner._claim_active_session_slot = lambda key, source: (object(), None)
    runner._post_turn_goal_continuation = AsyncMock()
    runner.session_store.get_or_create_session.return_value = None
    runner._handle_message_with_agent = run_agent
    adapter.set_message_handler(runner._handle_message)
    adapter.send = AsyncMock()
    adapter._keep_typing = AsyncMock()
    adapter._stop_typing_refresh = AsyncMock()
    adapter._send_with_retry = AsyncMock(return_value=SendResult(success=True, message_id="1"))
    adapter._run_processing_hook = AsyncMock()


@pytest.mark.asyncio
async def test_the_startup_wrapper_waits_for_the_resumed_turn():
    runner, adapter = make_restart_runner()
    runner._startup_restore_in_progress = True
    runner._startup_restore_tasks = []
    _source, key = _pending(runner, "resume-blocked")
    entered, release = asyncio.Event(), asyncio.Event()
    runs = []

    async def blocked(event, source, quick_key, generation):
        runs.append(event)
        entered.set()
        await release.wait()
        return "RESUMED OK"
    _real_pipeline(runner, adapter, blocked)

    assert runner._schedule_resume_pending_sessions() == 1
    wrapper = runner._startup_restore_tasks[0]
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        for _ in range(10):
            await asyncio.sleep(0)  # let anything already runnable (a finishing wrapper) run
        assert not wrapper.done(), "the startup wrapper finished while the resumed turn was still running"
        assert key not in getattr(adapter, "_pending_messages", {}), "the resume event bounced off its own claim"
    finally:
        release.set()
    await asyncio.wait_for(wrapper, timeout=5)
    assert len(runs) == 1 and getattr(runs[0], "_startup_resume_claim", False)
    assert key not in runner._running_agents


@pytest.mark.asyncio
async def test_the_startup_wrapper_never_releases_another_events_pending_claim():
    runner, adapter = make_restart_runner()
    runner._release_running_agent_state = GatewayRunner._release_running_agent_state.__get__(runner, GatewayRunner)
    source, key = _pending(runner, "resume-handover")
    other = MessageEvent(text="hello", message_type=MessageType.TEXT, source=source)

    async def handed_over(event):
        # The resume's turn ended and another event claimed the slot before the wrapper's finally.
        runner._release_running_agent_state(key)
        state = runner._session_state(key)
        state.turn.agent = _AGENT_PENDING_SENTINEL
        state.turn.event = other
    adapter.handle_message = handed_over

    assert runner._schedule_resume_pending_sessions() == 1
    await asyncio.gather(*list(runner._background_tasks))

    state = runner._peek_session_state(key)
    assert state is not None and state.turn.agent is _AGENT_PENDING_SENTINEL and state.turn.event is other


@pytest.mark.asyncio
async def test_an_unrelated_inbound_message_still_queues_behind_the_resume_claim():
    runner, adapter = make_restart_runner()
    source, key = _pending(runner, "resume-inbound")
    gate = asyncio.Event()

    async def held(event):
        await gate.wait()
    adapter.handle_message = held
    assert runner._schedule_resume_pending_sessions() == 1
    try:
        inbound = MessageEvent(text="hi", message_type=MessageType.TEXT, source=source)
        assert not runner._owns_startup_resume_claim(key, inbound)
        assert runner._is_session_running(key)
        state = runner._peek_session_state(key)
        assert runner._owns_startup_resume_claim(key, state.turn.event)
    finally:
        gate.set()
    await asyncio.gather(*list(runner._background_tasks))
