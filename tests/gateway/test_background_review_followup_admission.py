"""Gateway turns expose queued same-session messages to background-review admission."""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import threading
import time
import types
from typing import cast
from unittest.mock import MagicMock

import pytest

from agent import background_review
from agent import review_admission
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult, TextDebounceState
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.run_turn_runner import TurnRunner
from gateway.session import SessionSource


def _wire_with_adapter(adapter, session_key: str = "session-key", overflow_probe=None):
    agent = types.SimpleNamespace()
    ctx = types.SimpleNamespace(
        progress_callback=None,
        native_tool_start_callback=None,
        voice_ack_callback=None,
        _voice_ack_guild=[None],
        _native_slack_task_cards=False,
        native_tool_complete_callback=None,
        _step_callback_sync=None,
        _hooks_ref=types.SimpleNamespace(loaded_hooks=[]),
        _status_callback_sync=None,
        _event_callback_sync=None,
        _status_adapter=adapter,
        _status_thread_metadata={},
        session_key=session_key,
        user_config={},
        source=types.SimpleNamespace(platform="telegram"),
        mute_notification_reply=False,
        _thinking_enabled=False,
        agent_holder=[None],
        tools_holder=[None],
        process_task_id=None,
        process_baseline=None,
        run_generation=1,
    )
    holder = types.SimpleNamespace(
        _ctx=ctx,
        _runner=types.SimpleNamespace(
            _service_tier=None,
            _consume_pending_turn_sidecar_notes=lambda key: [],
            _overflow_queue=overflow_probe or (lambda _key: []),
        ),
        _make_bg_review_callbacks=lambda: (lambda message: None, lambda: None),
        _merge_turn_request_overrides=TurnRunner._merge_turn_request_overrides,
        _clarify_callback_sync=lambda *a, **k: None,
        _notice_callback_sync=lambda *a, **k: None,
        _attach_session_title_callback=lambda agent, ctx: None,
    )
    TurnRunner._wire_turn_agent_callbacks(
        cast(TurnRunner, holder), agent, {}, None, None, None, False
    )
    return agent


def test_followup_probe_tracks_the_current_session_without_consuming_it(monkeypatch):
    monkeypatch.setattr(BasePlatformAdapter, "__abstractmethods__", frozenset())
    adapter = object.__new__(BasePlatformAdapter)
    adapter._post_delivery_callbacks = {}
    agent = _wire_with_adapter(adapter)

    assert adapter.has_pending_message("session-key") is False
    assert agent.followup_pending_callback() is False

    queued = object()
    adapter._pending_messages = {"session-key": queued}
    assert adapter.has_pending_message("session-key") is True
    assert agent.followup_pending_callback() is True
    assert adapter._pending_messages["session-key"] is queued

    assert adapter.get_pending_message("session-key") is queued
    assert adapter.has_pending_message("session-key") is False
    assert agent.followup_pending_callback() is False

    debounced = object()
    adapter._text_debounce = {"session-key": debounced}
    assert adapter.has_pending_message("session-key") is True
    assert agent.followup_pending_callback() is True
    assert adapter._text_debounce["session-key"] is debounced


@pytest.mark.parametrize("broken_probe", ["has_pending_message", "overflow_queue"])
def test_raising_adapter_probe_reads_as_admission_failure(monkeypatch, broken_probe):
    """The gateway-installed probe must propagate an adapter failure, never swallow it as "no
    follow-up": unknown foreground state reads as ``admission_probe_failed`` and blocks the
    automatic review instead of authorizing a full-transcript request."""
    monkeypatch.setattr(BasePlatformAdapter, "__abstractmethods__", frozenset())
    adapter = object.__new__(BasePlatformAdapter)
    adapter._post_delivery_callbacks = {}
    adapter._pending_messages = {}

    def _boom(_key):
        raise RuntimeError(f"{broken_probe} failed")

    if broken_probe == "has_pending_message":
        adapter.has_pending_message = _boom
        agent = _wire_with_adapter(adapter)
    else:
        agent = _wire_with_adapter(adapter, overflow_probe=_boom)

    with pytest.raises(RuntimeError):
        agent.followup_pending_callback()
    assert (
        review_admission.foreground_block_reason(agent)
        == review_admission.REASON_ADMISSION_FAILURE
    )
    assert review_admission.followup_pending(agent) is True


def test_recursive_gateway_turn_drops_nonterminal_review_candidate():
    """``bind_agent`` runs once per turn of an in-band chain — the recursive ``_run_agent``
    carries the outer admission (``test_queued_followup_turn_carries_the_outer_review_admission``)
    — so a first turn's non-terminal candidate never outlives the follow-up that superseded it."""
    from gateway.run_turn import _GatewayReviewAdmission

    session_id = "gateway-recursive-candidate"
    profile_key = review_admission.current_profile_key()
    token = review_admission.note_turn_started(session_id, profile_key)
    admission = _GatewayReviewAdmission(session_id, profile_key, token)
    agent = types.SimpleNamespace(
        session_id=session_id,
        _spawn_background_review=MagicMock(),
    )

    admission.bind_agent(agent)
    admission.capture_candidate(
        agent,
        [{"role": "assistant", "content": "nonterminal response"}],
        review_memory=True,
        review_skills=False,
    )
    admission.bind_agent(agent)
    admission.finish(delivery_succeeded=True)

    agent._spawn_background_review.assert_not_called()
    assert not review_admission.other_live_turn(session_id, None, profile_key)


def test_gateway_admission_bind_agent_aliases_rotated_session():
    """Session hygiene may rotate the session BEFORE the agent is bound (the gateway token is
    registered ahead of ``_hmwa_prepare_turn``). Binding must alias the rotated child so the
    delivery window keeps one live owner on it, and finish must release both keys."""
    from gateway.run_turn import _GatewayReviewAdmission

    profile_key = "/profiles/p"
    token = review_admission.note_turn_started("S1", profile_key)
    admission = _GatewayReviewAdmission("S1", profile_key, token)
    agent = types.SimpleNamespace(session_id="S2")

    try:
        admission.bind_agent(agent)

        assert review_admission.other_live_turn("S2", None, profile_key) is True
        assert admission.session_id == "S2"
    finally:
        admission.finish(delivery_succeeded=False)

    assert review_admission.other_live_turn("S1", None, profile_key) is False
    assert review_admission.other_live_turn("S2", None, profile_key) is False


@pytest.mark.asyncio
async def test_gateway_owns_review_admission_from_prepare_through_delivery(monkeypatch):
    runner = object.__new__(GatewayRunner)
    session_id = "gateway-lifecycle-session"
    session_key = "gateway-lifecycle-key"
    source = _event().source
    event = _event()
    profile_key = review_admission.current_profile_key()
    parent = types.SimpleNamespace(
        session_id=session_id,
        _background_review_agent=None,
        _background_review_run=None,
        _background_review_lock=threading.Lock(),
    )
    review_run = background_review.prepare_background_review_run(
        parent, session_id=session_id, profile_key=profile_key
    )
    assert review_run is not None
    assert review_run.begin_request(object()) is True

    def _interrupt_and_ack(_review_agent, **_kwargs):
        background_review.finish_background_review_run(parent, review_run)

    monkeypatch.setattr(
        background_review, "_interrupt_background_review", _interrupt_and_ack
    )

    async def _resolve(_event, _source):
        return source, types.SimpleNamespace(session_id=session_id), session_key

    async def _prepare(*_args):
        assert review_run.request_done.is_set()
        assert review_admission.other_live_turn(session_id, None, profile_key)
        return "prepared reply", []

    runner._hmwa_resolve_session = _resolve
    runner._hmwa_prepare_turn = _prepare

    result = await runner._handle_message_with_agent(
        event, source, session_key, run_generation=1
    )

    assert result == "prepared reply"
    assert review_admission.other_live_turn(session_id, None, profile_key)
    event._gateway_review_delivery_complete(delivery_succeeded=True)
    assert not review_admission.other_live_turn(session_id, None, profile_key)


@pytest.mark.asyncio
async def test_gateway_review_cancellation_wait_does_not_block_event_loop(monkeypatch):
    runner = object.__new__(GatewayRunner)
    session_id = "gateway-nonblocking-cancel-wait"
    session_key = "gateway-nonblocking-key"
    source = _event().source
    event = _event()
    profile_key = review_admission.current_profile_key()
    parent = types.SimpleNamespace(
        session_id=session_id,
        _background_review_agent=None,
        _background_review_run=None,
        _background_review_lock=threading.Lock(),
    )
    review_run = background_review.prepare_background_review_run(
        parent, session_id=session_id, profile_key=profile_key
    )
    assert review_run is not None
    assert review_run.begin_request(object()) is True
    cancel_seen = threading.Event()
    loop_advanced = threading.Event()
    observed = []

    monkeypatch.setattr(
        background_review,
        "_interrupt_background_review",
        lambda _review_agent, **_kwargs: cancel_seen.set(),
    )

    def _ack_after_loop_progress():
        assert cancel_seen.wait(timeout=10.0)
        observed.append(loop_advanced.wait(timeout=10.0))
        background_review.finish_background_review_run(parent, review_run)

    finisher = threading.Thread(target=_ack_after_loop_progress)
    finisher.start()

    async def _resolve(_event, _source):
        return source, types.SimpleNamespace(session_id=session_id), session_key

    async def _prepare(*_args):
        return "prepared reply", []

    async def _mark_loop_progress():
        await asyncio.sleep(0)
        loop_advanced.set()

    runner._hmwa_resolve_session = _resolve
    runner._hmwa_prepare_turn = _prepare
    marker = asyncio.create_task(_mark_loop_progress())

    result = await runner._handle_message_with_agent(
        event, source, session_key, run_generation=1
    )
    await marker
    finisher.join(timeout=10.0)
    event._gateway_review_delivery_complete(delivery_succeeded=False)

    assert result == "prepared reply"
    assert finisher.is_alive() is False
    assert observed == [True]


@pytest.mark.asyncio
async def test_cancelled_gateway_admission_wait_releases_live_turn_owner(monkeypatch):
    from gateway.run_turn import _GatewayReviewAdmission

    session_id = "gateway-cancelled-admission"
    profile_key = review_admission.current_profile_key()
    wait_entered = threading.Event()
    release_wait = threading.Event()

    monkeypatch.setattr(
        background_review,
        "cancel_background_review_for_live_turn",
        lambda *_args, **_kwargs: object(),
    )

    def _blocked_wait(_review_run):
        wait_entered.set()
        assert release_wait.wait(timeout=10.0)

    monkeypatch.setattr(
        background_review,
        "wait_for_background_review_cancellation",
        _blocked_wait,
    )

    task = asyncio.create_task(
        _GatewayReviewAdmission.begin(object(), session_id, profile_key)
    )
    assert await asyncio.to_thread(wait_entered.wait, 10.0)
    assert review_admission.other_live_turn(session_id, None, profile_key)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    release_wait.set()

    assert not review_admission.other_live_turn(session_id, None, profile_key)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "send_succeeded, handoff, expected",
    [
        (True, False, (True, None)),
        (False, False, (False, None)),
        (True, True, (False, review_admission.REASON_PENDING_HANDOFF)),
    ],
    ids=["sent", "send_failed", "handoff"],
)
async def test_base_delivery_completes_gateway_review_with_actual_outcome(
    monkeypatch, send_succeeded, handoff, expected
):
    """Ownership completes with the ACTUAL outcome: a failed send or a queued follow-up
    handed to the drain task is not a confirmed terminal delivery, so no review may spawn.
    A drain handoff names itself as the cause so the skip line can say so."""
    monkeypatch.setattr(BasePlatformAdapter, "__abstractmethods__", frozenset())
    adapter = BasePlatformAdapter(
        PlatformConfig(enabled=True, token="***"), Platform.TELEGRAM
    )
    event = _event()
    session_key = "gateway-delivery-outcome"
    outcomes = []

    async def _handler(_event):
        _event._gateway_review_delivery_complete = (
            lambda *, delivery_succeeded, cause=None: outcomes.append((
                delivery_succeeded,
                cause,
            ))
        )
        return "visible response"

    async def _send(*_args, **_kwargs):
        return SendResult(success=send_succeeded, message_id="sent")

    adapter.set_message_handler(_handler)
    adapter.send = _send
    adapter._active_sessions[session_key] = asyncio.Event()
    if handoff:
        adapter._pending_messages[session_key] = _event(text="queued follow-up")
        adapter._spawn_drain_task = lambda _pending, _key, **_kwargs: None

    await adapter._process_message_background(event, session_key)

    assert outcomes == [expected]


@pytest.mark.asyncio
async def test_drain_handoff_completes_only_its_own_review_ownership(monkeypatch):
    """Turn N's cleanup completes turn N's ownership, never the drain follow-up's.

    A queued follow-up is handed to a fresh task on the SAME session Event, and that task's
    ``_handle_message_with_agent`` attaches its own delivery-complete callback to the Event
    while turn N is still unwinding (stop-typing / post-delivery awaits). Reading the Event in
    turn N's finally would release turn N+1's live-turn token mid-turn (re-opening the overlap
    window for a deferred review) and leak turn N's token, so every later automatic review for
    the session would be skipped as ``live_turn_active``.
    """
    monkeypatch.setattr(BasePlatformAdapter, "__abstractmethods__", frozenset())
    adapter = BasePlatformAdapter(
        PlatformConfig(enabled=True, token="***"), Platform.TELEGRAM
    )
    monkeypatch.setattr(adapter.config, "typing_indicator", False, raising=False)
    session_key = "gateway-drain-handoff"
    guard = asyncio.Event()
    adapter._active_sessions[session_key] = guard
    queued = _event(text="queued follow-up")
    adapter._pending_messages[session_key] = queued
    finished_n, finished_n1, drained, attached = [], [], [], []

    def finish_n(*, delivery_succeeded, cause=None):
        finished_n.append(delivery_succeeded)

    def finish_n1(*, delivery_succeeded, cause=None):
        finished_n1.append(delivery_succeeded)

    async def _handler(_event):
        guard._gateway_review_delivery_complete = finish_n
        return "reply"

    async def _send(*_args, **_kwargs):
        return SendResult(success=True, message_id="sent")

    async def _stop_typing(*_args, **_kwargs):
        # First cleanup await after the handoff: the drain task's turn N+1 has already landed
        # its own callback on the shared Event.
        if drained and not attached:
            attached.append(True)
            guard._gateway_review_delivery_complete = finish_n1

    adapter.set_message_handler(_handler)
    adapter.send = _send
    adapter._spawn_drain_task = lambda pending, _key, **_kwargs: drained.append(pending)
    adapter._stop_typing_refresh = _stop_typing

    await adapter._process_message_background(_event(), session_key)

    assert drained == [queued]
    assert finished_n == [False]
    assert finished_n1 == []
    assert guard._gateway_review_delivery_complete is finish_n1


def test_read_only_pending_probes_do_not_recreate_cleaned_admission_state(monkeypatch):
    monkeypatch.setattr(BasePlatformAdapter, "__abstractmethods__", frozenset())
    adapter = BasePlatformAdapter(
        PlatformConfig(enabled=True, token="***"), Platform.TELEGRAM
    )
    guard = asyncio.Event()
    adapter._active_sessions["cleaned"] = guard
    agent = _wire_with_adapter(adapter, "cleaned")

    adapter._cleanup_finished_session_task("cleaned", guard)
    assert adapter._followup_admission == {}

    for _ in range(5):
        assert adapter.has_pending_message("cleaned") is False
        assert agent.followup_pending_callback() is False
    assert adapter._followup_admission == {}

    adapter._pending_messages["legacy"] = object()
    assert adapter.has_pending_message("legacy") is True
    assert adapter._followup_admission == {}


class _TrackingRLock:
    """RLock test double that exposes ownership without platform-private methods."""

    def __init__(self):
        self._lock = threading.RLock()
        self._owner = None
        self._depth = 0
        self.acquisitions = 0

    def __enter__(self):
        self._lock.acquire()
        self.acquisitions += 1
        owner = threading.get_ident()
        if self._owner == owner:
            self._depth += 1
        else:
            self._owner = owner
            self._depth = 1
        return self

    def __exit__(self, *_args):
        self._depth -= 1
        if self._depth == 0:
            self._owner = None
        self._lock.release()

    def held_by_current_thread(self) -> bool:
        return self._owner == threading.get_ident()


def _event(*, message_type=MessageType.TEXT, user_id="user", text="follow up"):
    return MessageEvent(
        text=text,
        message_type=message_type,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="chat",
            chat_type="dm",
            user_id=user_id,
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("queue_path", ["generic", "photo", "merge", "debounce"])
async def test_queue_mutation_and_review_cancellation_are_one_critical_section(
    monkeypatch, queue_path
):
    """A request contender reaches admission while queue insertion owns the shared lock."""
    monkeypatch.setattr(BasePlatformAdapter, "__abstractmethods__", frozenset())
    monkeypatch.setattr(
        background_review,
        "_interrupt_background_review",
        lambda _agent, **_kwargs: None,
    )
    adapter = object.__new__(BasePlatformAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter._post_delivery_callbacks = {}
    adapter._pending_messages = {}
    adapter._text_debounce = {}
    adapter._busy_session_handler = None
    adapter._busy_text_mode = (
        "queue" if queue_path in {"merge", "debounce"} else "interrupt"
    )
    adapter._busy_text_debounce_seconds = 60.0
    adapter._busy_text_hard_cap_seconds = 60.0
    admission = adapter.followup_admission_state("session-key")
    admission.lock = tracked_lock = _TrackingRLock()
    agent = _wire_with_adapter(adapter)
    agent._background_review_agent = None
    agent._background_review_run = None
    agent._background_review_lock = threading.Lock()
    run = background_review.prepare_background_review_run(
        agent, admission_lock=agent.followup_pending_lock
    )
    assert run is not None

    request_started = threading.Event()
    request_done = threading.Event()
    admitted = []
    contender_threads = []
    callback_observations = []
    original_cancel = background_review.cancel_background_review_for_pending_followup

    def force_request_contender(parent):
        def begin_request():
            request_started.set()
            admitted.append(run.begin_request(object()))
            request_done.set()

        contender = threading.Thread(target=begin_request)
        contender_threads.append(contender)
        contender.start()
        callback_observations.append(request_started.wait(timeout=10.0))
        callback_observations.append(tracked_lock.held_by_current_thread())
        callback_observations.append(adapter.has_pending_message("session-key"))
        if not tracked_lock.held_by_current_thread():
            callback_observations.append(request_done.wait(timeout=10.0))
        original_cancel(parent)

    monkeypatch.setattr(
        background_review,
        "cancel_background_review_for_pending_followup",
        force_request_contender,
    )

    if queue_path == "merge":
        adapter._pending_messages["session-key"] = _event(text="pending")
        adapter._text_debounce["session-key"] = TextDebounceState(
            event=_event(user_id="other", text="other sender"),
            task=None,
            first_ts=0.0,
            last_ts=0.0,
        )
    event = _event(
        message_type=MessageType.PHOTO if queue_path == "photo" else MessageType.TEXT
    )

    await adapter._handle_message_while_active(event, "session-key")

    assert callback_observations[:3] == [True, True, True]
    assert len(contender_threads) == 1
    contender_threads[0].join(timeout=10.0)
    assert contender_threads[0].is_alive() is False
    assert admitted == [False]
    assert run.cancel_requested.is_set()
    assert adapter.has_pending_message("session-key") is True
    for debounce_state in adapter._text_debounce.values():
        if debounce_state.task is not None:
            debounce_state.task.cancel()
            await asyncio.gather(debounce_state.task, return_exceptions=True)
    background_review.finish_background_review_run(agent, run)


def _busy_runner(adapter, agent, session_key):
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._queued_events = {}
    runner._running_agents = {session_key: agent}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._busy_ack_ts = {}
    runner._busy_input_mode = "interrupt"
    runner._busy_text_mode = "interrupt"
    runner._draining = False
    runner.session_store = None
    runner.hooks = types.SimpleNamespace(emit=lambda *_args, **_kwargs: None)
    runner._is_user_authorized = lambda _source: True
    return runner


def test_runner_dropped_at_queue_cap_does_not_publish_followup(monkeypatch):
    monkeypatch.setattr(BasePlatformAdapter, "__abstractmethods__", frozenset())
    adapter = BasePlatformAdapter(
        PlatformConfig(enabled=True, token="***"), Platform.TELEGRAM
    )
    session_key = "capped-session"
    state = adapter.followup_admission_state(session_key)
    cancellations = []
    adapter.register_followup_review_cancel(
        session_key, lambda: cancellations.append(True)
    )
    adapter._pending_messages[session_key] = _event(text="head")
    runner = _busy_runner(adapter, types.SimpleNamespace(), session_key)
    runner._BUSY_QUEUE_MAX_PENDING = 1
    dropped = _event(text="dropped")

    runner._queue_or_replace_pending_event(session_key, dropped)

    assert state.epoch == 0
    assert cancellations == []
    assert dropped._gateway_accepted is False
    assert not runner._overflow_queue(session_key)


def test_goal_clear_and_empty_promotion_keep_overflow_fenced_and_visible(monkeypatch):
    monkeypatch.setattr(BasePlatformAdapter, "__abstractmethods__", frozenset())
    adapter = BasePlatformAdapter(
        PlatformConfig(enabled=True, token="***"), Platform.TELEGRAM
    )
    session_key = "goal-promotion-session"
    admission = adapter.followup_admission_state(session_key)
    admission.lock = tracked_lock = _TrackingRLock()
    fenced_under_admission = []
    adapter.register_followup_review_cancel(
        session_key,
        lambda: fenced_under_admission.append(tracked_lock.held_by_current_thread()),
    )
    runner = _busy_runner(adapter, types.SimpleNamespace(), session_key)
    goal = _event(text="[Continuing toward your standing goal]\nGoal: ship")
    first = _event(text="first real follow-up")
    second = _event(text="second real follow-up")
    adapter._pending_messages[session_key] = goal
    runner._session_state(session_key).conversation.queued_events.extend([
        first,
        second,
    ])

    assert runner._clear_goal_pending_continuations(session_key, adapter) == 1
    assert admission.epoch == 1
    assert runner._promote_queued_event(session_key, adapter, None) is first
    assert admission.epoch == 2
    assert fenced_under_admission == [True, True]

    agent = _wire_with_adapter(
        adapter, session_key, overflow_probe=runner._overflow_queue
    )
    assert adapter.has_pending_message(session_key) is False
    assert [event.text for event in runner._overflow_queue(session_key)] == [
        "second real follow-up"
    ]
    assert agent.followup_pending_callback() is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fast_path", ["photo_burst", "telegram_grace", "max_interrupt_depth"]
)
async def test_busy_fast_path_merges_publish_the_followup_fence(monkeypatch, fast_path):
    """Every remaining direct pending-slot write queues the next live turn too.

    The PRIORITY busy fast path and the interrupt-depth cap write the adapter's pending slot
    directly, so without the shared fence a review sampling "no follow-up" a moment earlier still
    publishes its full-transcript request.
    """
    monkeypatch.setattr(BasePlatformAdapter, "__abstractmethods__", frozenset())
    adapter = BasePlatformAdapter(
        PlatformConfig(enabled=True, token="***"), Platform.TELEGRAM
    )
    session_key = "fast-path-session"
    state = adapter.followup_admission_state(session_key)
    state.lock = tracked_lock = _TrackingRLock()
    fenced_under_admission = []
    adapter.register_followup_review_cancel(
        session_key,
        lambda: fenced_under_admission.append(tracked_lock.held_by_current_thread()),
    )
    runner = _busy_runner(adapter, types.SimpleNamespace(), session_key)

    if fast_path == "photo_burst":
        event = _event(message_type=MessageType.PHOTO)
        assert await runner._hm_busy_slash_or_photo(
            event, event.source, session_key
        ) == (True, None)
    elif fast_path == "telegram_grace":
        event = _event(message_type=MessageType.TEXT)
        runner._session_state(session_key).turn.started_ts = time.time()
        assert (
            runner._hm_busy_telegram_grace_queue(
                event, event.source, session_key, "interrupt"
            )
            is True
        )
    else:
        event = _event(message_type=MessageType.TEXT)
        turn_ctx = types.SimpleNamespace(
            source=event.source,
            session_id="session-id",
            session_key=session_key,
            run_generation=1,
            _interrupt_depth=runner._MAX_INTERRUPT_DEPTH,
            history=[],
            _status_thread_metadata={},
            result_holder=[None],
        )
        assert await runner._run_agent_queued_followup(
            turn_ctx,
            adapter,
            "queued text",
            event,
            "response",
            {"interrupted": True},
            None,
        ) == {"final_response": "response", "messages": []}

    assert adapter._pending_messages[session_key] is event
    assert fenced_under_admission == [True]
    assert state.epoch == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "runner_path", ["first_slot", "overflow", "media_merge", "promotion"]
)
async def test_runner_busy_queue_mutations_share_review_admission_lock(
    monkeypatch, runner_path
):
    """Force request admission against real busy-handler FIFO/media paths without sleeps."""
    monkeypatch.setattr(BasePlatformAdapter, "__abstractmethods__", frozenset())
    monkeypatch.setattr(
        background_review,
        "_interrupt_background_review",
        lambda _agent, **_kwargs: None,
    )
    monkeypatch.setenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", "false")
    adapter = BasePlatformAdapter(
        PlatformConfig(enabled=True, token="***"), Platform.TELEGRAM
    )
    session_key = "runner-session"
    admission = adapter.followup_admission_state(session_key)
    admission.lock = tracked_lock = _TrackingRLock()
    agent = _wire_with_adapter(adapter, session_key)
    agent._background_review_agent = None
    agent._background_review_run = None
    agent._background_review_lock = threading.Lock()
    agent._active_children = []
    agent.interrupt = lambda *_args, **_kwargs: None
    runner = _busy_runner(adapter, agent, session_key)
    run = background_review.prepare_background_review_run(
        agent, admission_lock=agent.followup_pending_lock
    )
    assert run is not None

    request_started = threading.Event()
    admitted = []
    contender_threads = []
    callback_holds_admission = []
    original_cancel = background_review.cancel_background_review_for_pending_followup

    def force_request_contender(parent):
        def begin_request():
            request_started.set()
            admitted.append(run.begin_request(object()))

        contender = threading.Thread(target=begin_request)
        contender_threads.append(contender)
        contender.start()
        assert request_started.wait(timeout=10.0)
        callback_holds_admission.append(tracked_lock.held_by_current_thread())
        original_cancel(parent)

    monkeypatch.setattr(
        background_review,
        "cancel_background_review_for_pending_followup",
        force_request_contender,
    )

    incoming = _event(message_type=MessageType.TEXT, text="incoming")
    if runner_path == "overflow":
        adapter._pending_messages[session_key] = _event(text="head")
    elif runner_path == "media_merge":
        adapter._pending_messages[session_key] = _event(
            message_type=MessageType.PHOTO, text="head"
        )
        incoming = _event(message_type=MessageType.PHOTO, text="incoming")
    elif runner_path == "promotion":
        runner._session_state(session_key).conversation.queued_events.append(incoming)

    if runner_path == "promotion":
        current = _event(text="current")
        assert runner._promote_queued_event(session_key, adapter, current) is current
    else:
        assert (
            await runner._handle_active_session_busy_message(incoming, session_key)
            is True
        )

    assert callback_holds_admission == [True]
    assert len(contender_threads) == 1
    contender_threads[0].join(timeout=10.0)
    assert contender_threads[0].is_alive() is False
    assert admitted == [False]
    assert run.cancel_requested.is_set()
    if runner_path == "overflow":
        assert [event.text for event in runner._overflow_queue(session_key)] == [
            "incoming"
        ]
    else:
        assert session_key in adapter._pending_messages
    background_review.finish_background_review_run(agent, run)


@pytest.mark.asyncio
async def test_gateway_refine_reports_an_occupied_review_slot(monkeypatch):
    """A /refine that cannot start (canonical slot occupied) must say so, never the success text."""
    from run_agent import AIAgent

    monkeypatch.setattr(
        background_review, "prepare_background_review_run", lambda *_a, **_k: None
    )
    agent = object.__new__(AIAgent)
    agent.session_id = "refine-occupied"
    agent._delegate_depth = 0
    agent.valid_tool_names = {"memory", "skill_manage"}
    agent._session_messages = [{"role": "user", "content": "hi"}]
    runner = object.__new__(GatewayRunner)
    runner._running_agents = {}
    runner._session_key_for_source = lambda _source: "refine-key"
    runner._cached_agent_for = lambda _key: agent

    reply = await runner._handle_refine_command(_event(text="/refine"))

    assert reply.startswith("/refine failed to start")
    assert "already running" in reply
    assert "Reviewing this conversation" not in reply


@pytest.mark.parametrize(
    "cause, expected_slug",
    [
        (None, review_admission.REASON_DELIVERY_UNCONFIRMED),
        (
            review_admission.REASON_PENDING_HANDOFF,
            review_admission.REASON_PENDING_HANDOFF,
        ),
    ],
    ids=["unconfirmed", "handoff"],
)
def test_unconfirmed_delivery_logs_owner_and_reason(caplog, cause, expected_slug):
    """Dropping a captured candidate is a skip decision: one owner-tagged, body-free line."""
    from gateway.run_turn import _GatewayReviewAdmission

    session_id = "gateway-unconfirmed-delivery"
    profile_key = review_admission.current_profile_key()
    token = review_admission.note_turn_started(session_id, profile_key)
    admission = _GatewayReviewAdmission(session_id, profile_key, token)
    agent = types.SimpleNamespace(
        session_id=session_id, _spawn_background_review=MagicMock()
    )
    admission.bind_agent(agent)
    admission.capture_candidate(
        agent,
        [{"role": "assistant", "content": "private reply body"}],
        review_memory=True,
        review_skills=False,
    )

    with caplog.at_level("INFO"):
        admission.finish(delivery_succeeded=False, cause=cause)

    lines = [
        record.getMessage()
        for record in caplog.records
        if expected_slug in record.getMessage()
    ]
    assert len(lines) == 1, caplog.text
    assert review_admission.owner_tag(profile_key, session_id) in lines[0]
    assert session_id not in lines[0]
    assert "private reply body" not in lines[0]
    agent._spawn_background_review.assert_not_called()
    assert review_admission.other_live_turn(session_id, None, profile_key) is False


@pytest.mark.asyncio
async def test_gateway_spawns_the_review_once_after_confirmed_delivery():
    """The positive gateway lifecycle: capture -> confirmed delivery -> exactly one spawn, with
    the frozen owner identity, inside the bound Context, after ownership release, and OFF the
    event-loop thread (the spawn pipeline is O(transcript))."""
    from gateway.run_turn import _GatewayReviewAdmission

    marker = contextvars.ContextVar("review_spawn_marker", default="unset")
    session_id = "gateway-positive-spawn"
    profile_key = review_admission.current_profile_key()
    loop_thread = threading.get_ident()
    spawned = threading.Event()
    calls = []

    def _spawn(**kwargs):
        try:
            asyncio.get_running_loop()
            has_loop = True
        except RuntimeError:
            has_loop = False
        calls.append({
            "kwargs": kwargs,
            "marker": marker.get(),
            "live": review_admission.other_live_turn(session_id, None, profile_key),
            "thread": threading.get_ident(),
            "has_loop": has_loop,
        })
        spawned.set()

    agent = types.SimpleNamespace(
        session_id=session_id, _spawn_background_review=_spawn
    )
    token = review_admission.note_turn_started(session_id, profile_key)
    admission = _GatewayReviewAdmission(session_id, profile_key, token)
    reset = marker.set("bound-turn")
    try:
        admission.bind_agent(agent)
    finally:
        marker.reset(reset)
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "ask"}]},
        {"role": "assistant", "content": "ok"},
    ]
    admission.capture_candidate(
        agent, messages, review_memory=True, review_skills=False
    )

    handle = admission.finish(delivery_succeeded=True)
    assert await asyncio.to_thread(spawned.wait, 10.0)
    if isinstance(handle, threading.Thread):
        await asyncio.to_thread(handle.join, 10.0)
    admission.finish(delivery_succeeded=True)  # the latch: never a second spawn

    assert len(calls) == 1
    call = calls[0]
    kwargs = call["kwargs"]
    assert kwargs["_spawning_turn_token"] == token
    assert kwargs["_review_profile_key"] == profile_key
    assert kwargs["_review_session_id"] == session_id
    assert (kwargs["review_memory"], kwargs["review_skills"]) == (True, False)
    kwargs["messages_snapshot"][0]["content"][0]["text"] = "mutated by the fork"
    assert messages[0]["content"][0]["text"] == "ask"  # structural clone
    assert call["marker"] == "bound-turn"  # ran inside the Context bound at bind time
    assert call["live"] is False  # ownership released before the spawn
    assert call["thread"] != loop_thread
    assert call["has_loop"] is False
    assert agent._gateway_review_admission is None


@pytest.mark.asyncio
async def test_queued_followup_turn_carries_the_outer_review_admission(monkeypatch):
    """The in-band follow-up is bound to the OUTER admission like the first turn: the recursive
    ``_run_agent`` carries it, so ``bind_agent`` runs per turn (candidate reset, Context and
    rotated-session alias refreshed) instead of the follow-up inheriting the first turn's
    non-terminal candidate through the cached agent."""
    monkeypatch.setattr(BasePlatformAdapter, "__abstractmethods__", frozenset())
    adapter = BasePlatformAdapter(
        PlatformConfig(enabled=True, token="***"), Platform.TELEGRAM
    )
    session_key = "followup-carries-admission"
    runner = _busy_runner(adapter, types.SimpleNamespace(), session_key)
    admission = object()
    captured = {}

    async def _fake_run_agent(
        message, context_prompt, history, source, session_id, **kwargs
    ):
        captured.update(kwargs)
        return {"final_response": "follow-up reply", "messages": list(history)}

    async def _noop_refresh(*_args, **_kwargs):
        return None

    runner._run_agent = _fake_run_agent
    runner._refresh_agent_cache_message_count = _noop_refresh
    event = _event()
    turn_ctx = types.SimpleNamespace(
        source=event.source,
        session_id="session-id",
        session_key=session_key,
        run_generation=1,
        _interrupt_depth=0,
        history=[],
        _status_thread_metadata={},
        result_holder=[None],
        context_prompt="",
        channel_prompt=None,
        gateway_review_admission=admission,
    )

    result = await runner._run_agent_queued_followup(
        turn_ctx, adapter, "queued text", None, "response", {"interrupted": True}, None
    )

    assert result["final_response"] == "follow-up reply"
    assert captured["gateway_review_admission"] is admission


@pytest.mark.asyncio
async def test_queued_followup_runs_unowned_when_the_turn_context_carries_no_admission(
    monkeypatch,
):
    """Upstream drives this seam with turn-context doubles carrying exactly the fields main's
    function reads; the admission is this branch's field. The seam reads it the way
    ``TurnRunner._wire_turn_agent_callbacks`` does: absent means an unowned follow-up (None),
    never an AttributeError that fails an untouched upstream test."""
    monkeypatch.setattr(BasePlatformAdapter, "__abstractmethods__", frozenset())
    adapter = BasePlatformAdapter(
        PlatformConfig(enabled=True, token="***"), Platform.TELEGRAM
    )
    session_key = "followup-without-admission"
    runner = _busy_runner(adapter, types.SimpleNamespace(), session_key)
    captured = {}

    async def _fake_run_agent(
        message, context_prompt, history, source, session_id, **kwargs
    ):
        captured.update(kwargs)
        return {"final_response": "follow-up reply", "messages": list(history)}

    async def _noop_refresh(*_args, **_kwargs):
        return None

    runner._run_agent = _fake_run_agent
    runner._refresh_agent_cache_message_count = _noop_refresh
    event = _event()
    turn_ctx = types.SimpleNamespace(
        source=event.source,
        session_id="session-id",
        session_key=session_key,
        run_generation=1,
        _interrupt_depth=0,
        history=[],
        _status_thread_metadata={},
        result_holder=[None],
        context_prompt="",
        channel_prompt=None,
    )

    result = await runner._run_agent_queued_followup(
        turn_ctx, adapter, "queued text", None, "response", {"interrupted": True}, None
    )

    assert result["final_response"] == "follow-up reply"
    assert "gateway_review_admission" in captured
    assert captured["gateway_review_admission"] is None


@pytest.mark.asyncio
async def test_review_completion_failure_is_logged_with_owner_and_reason(
    monkeypatch, caplog
):
    """``finish`` releases ownership and drops the candidate before the spawn hop, so an
    exception inside it silently disables the review: the adapter's delivery finally must log
    one owner-tagged, body-free line and still complete the turn's cleanup."""
    from gateway.run_turn import _GatewayReviewAdmission

    monkeypatch.setattr(BasePlatformAdapter, "__abstractmethods__", frozenset())
    adapter = BasePlatformAdapter(
        PlatformConfig(enabled=True, token="***"), Platform.TELEGRAM
    )
    session_id = "gateway-completion-error-session"
    profile_key = review_admission.current_profile_key()
    token = review_admission.note_turn_started(session_id, profile_key)
    admission = _GatewayReviewAdmission(session_id, profile_key, token)

    def _boom(self, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(_GatewayReviewAdmission, "finish", _boom)

    async def _handler(_event):
        _event._gateway_review_delivery_complete = admission.finish
        return "visible response"

    async def _send(*_args, **_kwargs):
        return SendResult(success=True, message_id="sent")

    adapter.set_message_handler(_handler)
    adapter.send = _send
    session_key = "gateway-completion-error"
    adapter._active_sessions[session_key] = asyncio.Event()
    try:
        with caplog.at_level("WARNING"):
            await adapter._process_message_background(_event(), session_key)
    finally:
        review_admission.note_turn_finished(session_id, token, profile_key)

    lines = [
        record.getMessage()
        for record in caplog.records
        if review_admission.REASON_COMPLETION_ERROR in record.getMessage()
    ]
    assert len(lines) == 1, caplog.text
    assert review_admission.owner_tag(profile_key, session_id) in lines[0]
    assert session_id not in lines[0]


def test_finish_logs_a_spawn_failure_with_owner_and_reason(monkeypatch, caplog):
    """A spawn thread that cannot start must not raise out of the delivery finally nor vanish
    silently: ownership is released, the candidate is dropped, and one owner-tagged line
    names the reason."""
    from gateway import run_turn as run_turn_module
    from gateway.run_turn import _GatewayReviewAdmission

    session_id = "gateway-spawn-failure"
    profile_key = review_admission.current_profile_key()
    token = review_admission.note_turn_started(session_id, profile_key)
    admission = _GatewayReviewAdmission(session_id, profile_key, token)
    agent = types.SimpleNamespace(
        session_id=session_id, _spawn_background_review=MagicMock()
    )
    admission.bind_agent(agent)
    admission.capture_candidate(
        agent,
        [{"role": "assistant", "content": "private reply body"}],
        review_memory=True,
        review_skills=False,
    )

    class _UnstartableThread:
        def __init__(self, *_args, **_kwargs):
            pass

        def start(self):
            raise RuntimeError("can't start new thread")

    monkeypatch.setattr(
        run_turn_module,
        "threading",
        types.SimpleNamespace(
            Thread=_UnstartableThread, Lock=threading.Lock, Event=threading.Event
        ),
    )
    with caplog.at_level("WARNING"):
        assert admission.finish(delivery_succeeded=True) is None

    lines = [
        record.getMessage()
        for record in caplog.records
        if review_admission.REASON_COMPLETION_ERROR in record.getMessage()
    ]
    assert len(lines) == 1, caplog.text
    assert review_admission.owner_tag(profile_key, session_id) in lines[0]
    assert session_id not in lines[0]
    assert "private reply body" not in lines[0]
    agent._spawn_background_review.assert_not_called()
    assert review_admission.other_live_turn(session_id, None, profile_key) is False


def test_stale_probe_state_and_fresh_fence_state_cannot_deadlock(monkeypatch):
    """A review admitting under the state its turn captured must never wedge against the fence
    of a later activation of the same session key.

    Turn N wires its follow-up probe against the ``FollowupAdmissionState`` live at the time; the
    session goes idle (cleanup pops the store entry) and turn N+1 — on a rotated session id, so
    the live-turn registry does not refuse the review first — wires a fresh state with its own
    cancel fence. The review holds turn N's state lock and, on the old code, the process-wide
    registry lock while its probe reaches ``has_pending_message``, which took the CURRENT store
    entry's lock; the gateway loop, inside a fenced slot write under that fresh lock, ran the
    cancel callback into the registry lock. Two threads, two locks, opposite order: the gateway
    event loop froze for every session the process serves.
    """
    monkeypatch.setattr(BasePlatformAdapter, "__abstractmethods__", frozenset())
    monkeypatch.setattr(
        background_review,
        "_interrupt_background_review",
        lambda _agent, **_kwargs: None,
    )
    adapter = BasePlatformAdapter(
        PlatformConfig(enabled=True, token="***"), Platform.TELEGRAM
    )
    session_key = "rotated-session-key"
    profile_key = review_admission.current_profile_key()

    old_state = adapter.followup_admission_state(session_key)
    agent_n = _wire_with_adapter(adapter, session_key)
    agent_n.session_id = "sid-old"
    agent_n._background_review_agent = None
    agent_n._background_review_run = None
    agent_n._background_review_lock = threading.Lock()
    assert agent_n.followup_pending_lock is old_state.lock

    guard = asyncio.Event()
    adapter._active_sessions[session_key] = guard
    adapter._cleanup_finished_session_task(session_key, guard)
    assert adapter._existing_followup_admission_state(session_key) is None

    agent_n1 = _wire_with_adapter(adapter, session_key)
    agent_n1.session_id = "sid-new"
    new_state = adapter.followup_admission_state(session_key)
    assert new_state is not old_state and callable(new_state.cancel_review)

    run = background_review.prepare_background_review_run(
        agent_n,
        admission_gate=lambda: review_admission.foreground_block_reason(
            agent_n, None, profile_key, "sid-old"
        ),
        admission_lock=old_state.lock,
        foreground_admission_lock=review_admission.admission_lock(),
        session_id="sid-old",
        profile_key=profile_key,
    )
    assert run is not None

    review_in_probe = threading.Event()
    gateway_holds_fresh_state = threading.Event()
    original_probe = adapter.has_pending_message

    def _probe(key):
        review_in_probe.set()
        gateway_holds_fresh_state.wait(timeout=5.0)
        return original_probe(key)

    adapter.has_pending_message = _probe

    def _mutation():
        gateway_holds_fresh_state.set()
        review_in_probe.wait(timeout=5.0)
        return True

    admitted, fenced = [], []
    review_thread = threading.Thread(
        target=lambda: admitted.append(run.begin_request(object())),
        name="review-begin_request",
        daemon=True,
    )
    gateway_thread = threading.Thread(
        target=lambda: fenced.append(
            adapter.apply_followup_queue_mutation(session_key, _mutation)
        ),
        name="gateway-slot-write",
        daemon=True,
    )
    review_thread.start()
    gateway_thread.start()
    review_thread.join(timeout=5.0)
    gateway_thread.join(timeout=5.0)
    try:
        assert not review_thread.is_alive(), "review wedged inside the host probe"
        assert not gateway_thread.is_alive(), (
            "gateway slot write wedged behind the review"
        )
        assert fenced == [True]
        # The rotated session's live turn is another registry owner: the review is admitted.
        assert admitted == [True]
    finally:
        # A wedged run still owns the registry lock; finishing it would hang the test too.
        if not (review_thread.is_alive() or gateway_thread.is_alive()):
            background_review.finish_background_review_run(agent_n, run)


def test_pending_probe_takes_no_lock_from_a_replaced_admission_state(monkeypatch):
    """``has_pending_message`` acquires nothing: the per-turn probe closure already runs under the
    state lock its turn captured, and the store entry may since have been replaced by a later
    activation of the same key. Taking that entry's lock from a stale probe is a lock-order
    inversion against the gateway fence (state lock, then registry lock)."""
    monkeypatch.setattr(BasePlatformAdapter, "__abstractmethods__", frozenset())
    adapter = BasePlatformAdapter(
        PlatformConfig(enabled=True, token="***"), Platform.TELEGRAM
    )
    session_key = "replaced-admission-state"
    old_state = adapter.followup_admission_state(session_key)
    old_state.lock = old_lock = _TrackingRLock()
    agent = _wire_with_adapter(adapter, session_key)
    adapter._followup_admission_store().pop(session_key)
    new_state = adapter.followup_admission_state(session_key)
    new_state.lock = new_lock = _TrackingRLock()
    adapter._pending_messages[session_key] = _event(text="queued")

    with old_lock:
        assert adapter.has_pending_message(session_key) is True
        assert agent.followup_pending_callback() is True
    assert new_lock.acquisitions == 0
    assert old_lock.held_by_current_thread() is False


@pytest.mark.asyncio
async def test_inline_dispatched_nested_turn_completes_its_review_ownership(
    monkeypatch,
):
    """/retry typed while the outgoing turn's reply is on the wire is dispatched inline (every
    recognised command bypasses the active-session guard) and nests a second agent turn through
    the runner's idle path. That turn parks its review-ownership completion on the live session
    Event — which the outgoing task already read once, the moment its own handler returned — so
    the inline dispatcher must complete it with its send outcome. Otherwise the session's
    live-turn token leaks for the process lifetime and every later automatic review is refused
    as ``live_turn_active``.
    """
    from gateway.run_turn import _GatewayReviewAdmission

    monkeypatch.setattr(BasePlatformAdapter, "__abstractmethods__", frozenset())
    adapter = BasePlatformAdapter(
        PlatformConfig(enabled=True, token="***"), Platform.TELEGRAM
    )
    monkeypatch.setattr(adapter.config, "typing_indicator", False, raising=False)
    runner = object.__new__(GatewayRunner)
    session_id = "inline-nested-session"
    session_key = "inline-nested-key"
    profile_key = review_admission.current_profile_key()
    adapter._active_sessions[session_key] = asyncio.Event()
    finishes, sends = [], []

    async def _resolve(event, _source):
        return event.source, types.SimpleNamespace(session_id=session_id), session_key

    async def _prepare(*_args):
        return "reply", []

    runner._hmwa_resolve_session = _resolve
    runner._hmwa_prepare_turn = _prepare
    runner._delivery_adapter_for = lambda _source: adapter
    original_finish = _GatewayReviewAdmission.finish

    def _finish(self, *, delivery_succeeded, cause=None, outer=None):
        finishes.append((self.token, delivery_succeeded, cause))
        return original_finish(
            self, delivery_succeeded=delivery_succeeded, cause=cause, outer=outer
        )

    monkeypatch.setattr(_GatewayReviewAdmission, "finish", _finish)

    async def _handler(event):
        turn = (
            _event(text="retried prompt")
            if (event.text or "").startswith("/retry")
            else event
        )
        return await runner._handle_message_with_agent(
            turn, turn.source, session_key, run_generation=len(sends) + 1
        )

    async def _send(chat_id, content, reply_to=None, metadata=None):
        sends.append(content)
        if len(sends) == 1:
            # The user's /retry lands while this (outgoing) reply is being sent.
            await adapter._handle_message_while_active(
                _event(text="/retry"), session_key
            )
        return SendResult(success=True, message_id=f"sent-{len(sends)}")

    adapter.set_message_handler(_handler)
    adapter.send = _send

    await adapter._process_message_background(_event(text="first"), session_key)

    assert sends == ["reply", "reply"]
    assert [ok for _token, ok, _cause in finishes] == [True, True]
    assert len({token for token, _ok, _cause in finishes}) == 2
    assert review_admission.other_live_turn(session_id, None, profile_key) is False


@pytest.mark.asyncio
async def test_direct_handoff_completes_ownership_from_its_own_event_beside_a_live_task(
    monkeypatch,
):
    """The CLI->gateway handoff is a direct-call ingress: it runs the synthetic turn inline
    and completes review ownership from its OWN event. The destination chat may have a live
    adapter task at that moment, so the runner must not park the handoff's completion on that
    task's session Event: nobody reads it there (the task took its own callback the moment its
    handler returned), or it overwrites the callback the task still has parked. Either way a
    live-turn token leaks and every later automatic review on the session is refused as
    ``live_turn_active``.
    """
    monkeypatch.setattr(BasePlatformAdapter, "__abstractmethods__", frozenset())
    adapter = BasePlatformAdapter(
        PlatformConfig(enabled=True, token="***"), Platform.TELEGRAM
    )
    runner = object.__new__(GatewayRunner)
    session_id = "handoff-beside-live-task"
    session_key = "handoff-beside-live-task-key"
    profile_key = review_admission.current_profile_key()
    guard = asyncio.Event()
    adapter._active_sessions[session_key] = guard
    sends = []

    def live_task_finish(*, delivery_succeeded, cause=None):
        raise AssertionError("the live adapter task's own completion must stay parked")

    guard._gateway_review_delivery_complete = live_task_finish

    async def _resolve(event, _source):
        return event.source, types.SimpleNamespace(session_id=session_id), session_key

    async def _prepare(*_args):
        return "handoff reply", []

    async def _handle_message(event):
        return await runner._handle_message_with_agent(
            event, event.source, session_key, 1
        )

    async def _send(_platform, _chat_id, text, _metadata=None):
        sends.append(text)
        return SendResult(success=True, message_id="sent")

    destination = types.SimpleNamespace(
        source=_event().source,
        platform=Platform.TELEGRAM,
        platform_name="telegram",
        home=types.SimpleNamespace(chat_id="chat"),
        effective_thread_id=None,
        transport=types.SimpleNamespace(send=_send),
    )

    async def _resolve_destination(_row, _profile_name):
        return destination

    async def _get_or_create_session(_source):
        return None

    async def _switch_session(_key, _cli_session_id, **_kwargs):
        return object()

    store = types.SimpleNamespace()
    runner.session_store = store
    runner._async_session_store = types.SimpleNamespace(
        _store=store,
        get_or_create_session=_get_or_create_session,
        switch_session=_switch_session,
    )
    runner._hmwa_resolve_session = _resolve
    runner._hmwa_prepare_turn = _prepare
    runner._delivery_adapter_for = lambda _source: adapter
    runner._handle_message = _handle_message
    runner._handoff_resolve_destination = _resolve_destination
    runner._handoff_session_key = lambda _dest, _profile_name: session_key
    runner._evict_cached_agent = lambda _key: None
    runner._release_running_agent_state = lambda _key, **_kwargs: True

    await runner._process_handoff({
        "id": "cli-session",
        "title": "work",
        "handoff_platform": "telegram",
    })

    assert sends == ["handoff reply"]
    assert review_admission.other_live_turn(session_id, None, profile_key) is False
    assert guard._gateway_review_delivery_complete is live_task_finish


@pytest.mark.asyncio
async def test_nested_retry_review_supersedes_the_retracted_outgoing_candidate(
    monkeypatch, caplog
):
    """/retry typed while the outgoing turn's reply is on the wire rewinds the transcript and
    nests the retried turn inline (previous test). The outgoing turn's live-turn token is
    released only when ITS send completes, so the nested turn's candidate cannot spawn when the
    inline dispatch completes it — and the outgoing turn's own candidate is the transcript
    ``/retry`` just retracted. The terminal delivery owner must spawn the freshest candidate
    exactly once, after every token is released: the retried turn's, never the retracted one.
    """
    import gateway.run_turn as run_turn_module

    monkeypatch.setattr(BasePlatformAdapter, "__abstractmethods__", frozenset())
    adapter = BasePlatformAdapter(
        PlatformConfig(enabled=True, token="***"), Platform.TELEGRAM
    )
    monkeypatch.setattr(adapter.config, "typing_indicator", False, raising=False)
    runner = object.__new__(GatewayRunner)
    session_id = "nested-retry-session"
    session_key = "nested-retry-key"
    profile_key = review_admission.current_profile_key()
    guard = asyncio.Event()
    adapter._active_sessions[session_key] = guard
    sends, spawns = [], []

    def _spawn(**kwargs):
        spawns.append((
            kwargs["messages_snapshot"][0]["content"],
            review_admission.other_live_turn(
                session_id, kwargs["_spawning_turn_token"], profile_key
            ),
        ))

    agent = types.SimpleNamespace(
        session_id=session_id, _spawn_background_review=_spawn
    )

    class _InlineThread:
        def __init__(self, *, target, args=(), kwargs=None, daemon=None, name=None):
            self._run = lambda: target(*args, **(kwargs or {}))

        def start(self):
            self._run()

    monkeypatch.setattr(
        run_turn_module,
        "threading",
        types.SimpleNamespace(
            Thread=_InlineThread, Lock=threading.Lock, Event=threading.Event
        ),
    )

    async def _resolve(event, _source):
        return event.source, types.SimpleNamespace(session_id=session_id), session_key

    async def _prepare(event, *_args):
        # The turn ran: its terminal candidate is frozen on this turn's delivery owner, which
        # the runner parked on the live session Event before preparing the turn.
        admission = guard._gateway_review_delivery_complete.__self__
        admission.bind_agent(agent)
        admission.capture_candidate(
            agent,
            [{"role": "user", "content": event.text}],
            review_memory=True,
            review_skills=False,
        )
        return "reply", []

    runner._hmwa_resolve_session = _resolve
    runner._hmwa_prepare_turn = _prepare
    runner._delivery_adapter_for = lambda _source: adapter

    async def _handler(event):
        turn = (
            _event(text="retried prompt")
            if (event.text or "").startswith("/retry")
            else event
        )
        return await runner._handle_message_with_agent(
            turn, turn.source, session_key, run_generation=len(sends) + 1
        )

    async def _send(chat_id, content, reply_to=None, metadata=None):
        sends.append(content)
        if len(sends) == 1:
            # The user's /retry lands while this (outgoing) reply is being sent.
            await adapter._handle_message_while_active(
                _event(text="/retry"), session_key
            )
        return SendResult(success=True, message_id=f"sent-{len(sends)}")

    adapter.set_message_handler(_handler)
    adapter.send = _send

    with caplog.at_level("INFO"):
        await adapter._process_message_background(_event(text="first"), session_key)

    assert sends == ["reply", "reply"]
    assert spawns == [("retried prompt", False)]
    assert review_admission.other_live_turn(session_id, None, profile_key) is False
    superseded = [
        record.getMessage()
        for record in caplog.records
        if review_admission.REASON_CANDIDATE_SUPERSEDED in record.getMessage()
    ]
    assert len(superseded) == 1, caplog.text
    assert review_admission.owner_tag(profile_key, session_id) in superseded[0]
    assert session_id not in superseded[0]
    assert "first" not in superseded[0]


@pytest.mark.asyncio
async def test_inline_retry_completes_ownership_after_the_outgoing_task_released_the_guard(
    monkeypatch,
):
    """The runner's ``/retry`` handler nests the retried turn through a synthetic event it builds
    itself. The inline dispatch captured the session guard before calling the handler, and the
    nested turn reaches the runner's carrier lookup only after several awaits — by then the
    outgoing task's send may have completed and its unwind released that guard. The nested
    completion then lands on the synthetic event, which no delivery path reads: the live-turn
    token would stay registered for the process lifetime, every later automatic review on the
    session refused as ``live_turn_active``, and the retried turn's own candidate lost. The
    handler must leave its nested completion where the delivery owner reads it: on the command
    event, completed with the inline send's outcome.
    """
    import gateway.run_turn as run_turn_module

    monkeypatch.setattr(BasePlatformAdapter, "__abstractmethods__", frozenset())
    adapter = BasePlatformAdapter(
        PlatformConfig(enabled=True, token="***"), Platform.TELEGRAM
    )
    monkeypatch.setattr(adapter.config, "typing_indicator", False, raising=False)
    runner = object.__new__(GatewayRunner)
    session_id = "released-guard-retry-session"
    session_key = "released-guard-retry-key"
    profile_key = review_admission.current_profile_key()
    guard = asyncio.Event()
    adapter._active_sessions[session_key] = guard  # the outgoing turn: reply on the wire
    sends, spawns = [], []

    def _spawn(**kwargs):
        spawns.append((
            kwargs["messages_snapshot"][0]["content"],
            review_admission.other_live_turn(
                session_id, kwargs["_spawning_turn_token"], profile_key
            ),
        ))

    agent = types.SimpleNamespace(
        session_id=session_id, _spawn_background_review=_spawn
    )

    class _InlineThread:
        def __init__(self, *, target, args=(), kwargs=None, daemon=None, name=None):
            self._run = lambda: target(*args, **(kwargs or {}))

        def start(self):
            self._run()

    monkeypatch.setattr(
        run_turn_module,
        "threading",
        types.SimpleNamespace(
            Thread=_InlineThread, Lock=threading.Lock, Event=threading.Event
        ),
    )

    async def _resolve(event, _source):
        # The outgoing task's send completed while /retry was rewinding the transcript: its
        # unwind released the guard before the nested turn reached the carrier lookup.
        adapter._cleanup_finished_session_task(session_key, guard)
        return event.source, types.SimpleNamespace(session_id=session_id), session_key

    async def _prepare(event, *_args):
        # No guard is left: the runner parked this turn's completion on the nested event.
        admission = event._gateway_review_delivery_complete.__self__
        admission.bind_agent(agent)
        admission.capture_candidate(
            agent,
            [{"role": "user", "content": event.text}],
            review_memory=True,
            review_skills=False,
        )
        return "reply", []

    async def _get_or_create_session(_source):
        return types.SimpleNamespace(session_id=session_id, last_prompt_tokens=0)

    async def _load_transcript(_session_id):
        return [
            {"role": "user", "content": "retried prompt"},
            {"role": "assistant", "content": "retracted reply"},
        ]

    async def _rewrite_transcript(_session_id, _history, **_kwargs):
        return True

    async def _nested(event):
        return await runner._handle_message_with_agent(
            event, event.source, session_key, 2
        )

    runner._hmwa_resolve_session = _resolve
    runner._hmwa_prepare_turn = _prepare
    runner._delivery_adapter_for = lambda _source: adapter
    runner._record_model_friction = lambda *_args, **_kwargs: None
    runner.session_store = store = types.SimpleNamespace()
    runner._async_session_store = types.SimpleNamespace(
        _store=store,
        get_or_create_session=_get_or_create_session,
        load_transcript=_load_transcript,
        rewrite_transcript=_rewrite_transcript,
    )
    runner._handle_message = _nested

    async def _handler(event):
        return await runner._handle_retry_command(event)

    async def _send(chat_id, content, reply_to=None, metadata=None):
        sends.append(content)
        return SendResult(success=True, message_id=f"sent-{len(sends)}")

    adapter.set_message_handler(_handler)
    adapter.send = _send

    await adapter._handle_message_while_active(_event(text="/retry"), session_key)

    assert sends == ["reply"]
    assert adapter._active_sessions == {}
    assert review_admission.other_live_turn(session_id, None, profile_key) is False
    assert spawns == [("retried prompt", False)]


@pytest.mark.asyncio
async def test_reset_command_guard_swap_does_not_strand_the_turns_review_ownership(
    monkeypatch,
):
    """/stop (like /new and /reset) swaps the session guard for a command-scoped Event while
    the runner handles it. A turn whose task started under the old guard but reaches the
    runner's carrier lookup after the swap parks its review-ownership completion on the
    COMMAND guard: the command's inline dispatch reads that guard only while its own handler
    runs, and the turn's task reads the guard it started under. The command's lifecycle must
    complete what it strands, or the session's live-turn token leaks for the process lifetime
    and every later automatic review is refused as ``live_turn_active``.
    """
    monkeypatch.setattr(BasePlatformAdapter, "__abstractmethods__", frozenset())
    adapter = BasePlatformAdapter(
        PlatformConfig(enabled=True, token="***"), Platform.TELEGRAM
    )
    monkeypatch.setattr(adapter.config, "typing_indicator", False, raising=False)
    runner = object.__new__(GatewayRunner)
    session_id = "stop-swap-session"
    session_key = "stop-swap-key"
    profile_key = review_admission.current_profile_key()
    resolving, swapped, parked = asyncio.Event(), asyncio.Event(), asyncio.Event()
    sends, guards = [], {}

    async def _resolve(event, _source):
        resolving.set()
        await swapped.wait()  # /stop lands while the turn is resolving its session
        return event.source, types.SimpleNamespace(session_id=session_id), session_key

    async def _prepare(*_args):
        guards["at_park"] = adapter._active_sessions.get(session_key)
        parked.set()
        return "stale", []  # no turn: the interrupt made this generation stale

    runner._hmwa_resolve_session = _resolve
    runner._hmwa_prepare_turn = _prepare
    runner._delivery_adapter_for = lambda _source: adapter

    async def _handler(event):
        if (event.text or "").startswith("/stop"):
            # The busy /stop path: interrupt + ack, never an agent turn.
            return "stopped"
        return await runner._handle_message_with_agent(
            event, event.source, session_key, 1
        )

    async def _send(chat_id, content, reply_to=None, metadata=None):
        sends.append(content)
        if content == "stopped":
            # The ack is on the wire: the turn resumes and parks its completion meanwhile.
            swapped.set()
            await parked.wait()
        return SendResult(success=True, message_id=f"sent-{len(sends)}")

    adapter.set_message_handler(_handler)
    adapter.send = _send

    first_guard = asyncio.Event()
    adapter._active_sessions[session_key] = first_guard
    task = asyncio.create_task(
        adapter._process_message_background(_event(text="first"), session_key)
    )
    adapter._track_session_task(session_key, task)
    await resolving.wait()

    await adapter._handle_message_while_active(_event(text="/stop"), session_key)
    with contextlib.suppress(asyncio.CancelledError):
        await task

    assert sends[0] == "stopped"
    assert guards["at_park"] is not first_guard, "the turn parked on the command guard"
    assert adapter._active_sessions == {}
    assert review_admission.other_live_turn(session_id, None, profile_key) is False
