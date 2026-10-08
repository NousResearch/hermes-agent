"""Busy follow-ups keep one authorized route from admission through queue drain."""

from __future__ import annotations

import asyncio
import dataclasses
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.event import AuthorizedQueueEnvelope, MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource, build_session_key
from tests.gateway.test_active_session_text_merge import _make_adapter


def _source(profile: str = "alpha") -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="chat-7",
        chat_type="dm",
        user_id="user-7",
        profile=profile,
    )


def _event(text: str, *, profile: str = "alpha", message_id: str = "m-7") -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=_source(profile),
        message_id=message_id,
    )


def _runner_and_adapter():
    runner: Any = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(group_sessions_per_user=True, thread_sessions_per_user=False)
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._busy_ack_ts = {}
    runner._queued_events = {}
    runner._draining = False
    runner._restart_requested = False
    runner._busy_input_mode = "queue"
    runner._busy_text_mode = "interrupt"
    runner.session_store = None
    runner._is_user_authorized_for_source = lambda _source: True
    runner._admit_bot_message_for_source = lambda _source: True
    runner._effective_busy_input_mode = lambda _source: "queue"
    runner._effective_busy_text_mode = lambda _source: "interrupt"
    runner._session_key_for_source = lambda source: build_session_key(source, profile=source.profile)
    runner._compose_busy_ack_message = lambda *args, **kwargs: "queued"

    adapter = _make_adapter()
    adapter._busy_text_mode = ""
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._delivery_adapter_for = lambda _source: adapter
    return runner, adapter


@pytest.mark.asyncio
async def test_busy_queue_keeps_the_authorized_route_and_session_after_source_changes():
    """A queued event is an immutable admission snapshot, not a route re-resolution at drain."""
    runner, adapter = _runner_and_adapter()
    event = _event("follow the admitted route")
    admitted_key = runner._session_key_for_source(event.source)
    runner._running_agents[admitted_key] = MagicMock()
    runner._running_agents_ts[admitted_key] = 1.0

    ack_entered = asyncio.Event()
    release_ack = asyncio.Event()

    async def _hold_ack(*_args, **_kwargs):
        ack_entered.set()
        await release_ack.wait()

    runner._send_busy_ack_reply = _hold_ack
    admission = asyncio.create_task(runner._handle_active_session_busy_message(event, admitted_key))
    await asyncio.wait_for(ack_entered.wait(), timeout=1)

    envelope = getattr(event, "_authorized_queue_envelope", None)
    assert isinstance(envelope, AuthorizedQueueEnvelope)
    assert dataclasses.is_dataclass(envelope)
    assert envelope.session_key == admitted_key
    assert envelope.source.profile == "alpha"
    source_copy = envelope.source
    source_copy.profile = "beta"
    assert envelope.source.profile == "alpha"
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(envelope, "session_key", "agent:beta:telegram:dm:chat-7")

    # A route/config refresh after the wire update was accepted must not move the queued turn.
    event.source.profile = "beta"
    release_ack.set()
    assert await admission is True
    rerouted_key = runner._session_key_for_source(event.source)
    assert rerouted_key != admitted_key
    assert runner._bind_authorized_queue_envelope(event, rerouted_key) is None

    runner._scale_to_zero_note_real_inbound = lambda: None
    runner._hm_pre_gateway_dispatch_hook = AsyncMock(return_value=event)
    readmitted = await runner._hm_admit_event(event)
    assert readmitted is not None
    _, readmitted_source, _ = readmitted
    assert readmitted_source.profile == "alpha"
    assert runner._session_key_for_source(readmitted_source) == admitted_key

    pending_event, pending = await runner._run_agent_drain_pending(
        {"final_response": "first", "messages": []}, adapter, _source("alpha"), admitted_key
    )
    assert pending_event is event
    assert pending == "follow the admitted route"

    runner._MAX_INTERRUPT_DEPTH = 8
    runner._run_agent = AsyncMock(return_value={"final_response": "done", "messages": []})
    runner._is_goal_continuation_event = MagicMock(return_value=False)
    runner._prepare_profile_scoped_inbound_message_text = AsyncMock(return_value=pending)
    runner._reply_anchor_for_event = MagicMock(return_value=None)
    runner._pinned_channel_inputs = lambda key, prompt, source, *, internal: (prompt, source)
    runner._persist_prompt_pins = AsyncMock()
    runner._delivery_adapter_for = MagicMock(return_value=None)
    runner._intake_adapter_for = MagicMock(return_value=None)
    runner._refresh_agent_cache_message_count = AsyncMock()
    turn_ctx = SimpleNamespace(
        source=_source("alpha"),
        session_id="sid-alpha",
        session_key=admitted_key,
        run_generation=3,
        _interrupt_depth=0,
        history=[],
        _status_thread_metadata=None,
        context_prompt=None,
        result_holder=[None],
        channel_prompt=None,
    )

    await runner._run_agent_queued_followup(
        turn_ctx,
        adapter=None,
        pending=pending,
        pending_event=pending_event,
        response="first",
        result={"interrupted": True, "messages": []},
        stream_task=None,
    )

    followup = runner._run_agent.await_args.kwargs
    assert followup["session_key"] == admitted_key
    assert followup["session_id"] == "sid-alpha"
    assert followup["source"].profile == "alpha"


@pytest.mark.asyncio
async def test_restart_barrier_drains_admitted_event_and_never_claims_a_failed_queue():
    """Restart waits for an accepted envelope; a full queue gets a rejection, never a queued ack."""
    runner, adapter = _runner_and_adapter()
    event = _event("survive restart", message_id="m-restart")
    session_key = runner._session_key_for_source(event.source)
    adapter._active_sessions[session_key] = asyncio.Event()
    runner._running_agents[session_key] = MagicMock()

    handler_entered = asyncio.Event()
    release_admission = asyncio.Event()

    async def _barrier(event_arg, key_arg):
        handler_entered.set()
        await release_admission.wait()
        return await runner._handle_active_session_busy_message(event_arg, key_arg)

    adapter.set_busy_session_handler(_barrier)
    runner._send_busy_reply = AsyncMock()

    inbound = asyncio.create_task(adapter.handle_message(event))
    await asyncio.wait_for(handler_entered.wait(), timeout=1)
    runner._draining = True
    runner._restart_requested = True
    release_admission.set()
    await inbound

    assert event._gateway_accepted is True
    queued_notice = runner._send_busy_reply.await_args.args[2]
    assert "queued" in queued_notice.lower()

    pending_event, pending = await runner._run_agent_drain_pending(
        {"final_response": "first", "messages": []}, adapter, event.source, session_key
    )
    assert pending_event is event
    assert pending == "survive restart"

    # Queue admission is transactional with its acknowledgment. At capacity, no queued claim is sent.
    runner._BUSY_QUEUE_MAX_PENDING = 1
    adapter._pending_messages[session_key] = _event("already pending", message_id="m-existing")
    rejected = _event("cannot fit", message_id="m-rejected")
    runner._send_busy_reply.reset_mock()

    assert await runner._handle_active_session_busy_message(rejected, session_key) is True
    rejected_notice = runner._send_busy_reply.await_args.args[2]
    assert "queued" not in rejected_notice.lower()
    assert "not accepting" in rejected_notice.lower()
    assert rejected._gateway_accepted is False
