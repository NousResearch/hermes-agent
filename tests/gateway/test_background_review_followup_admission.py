"""Gateway turns expose queued same-session messages to background-review admission."""

from __future__ import annotations

import asyncio
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

    def _interrupt_and_ack(_review_agent):
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
        lambda _review_agent: cancel_seen.set(),
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
        adapter._spawn_drain_task = lambda _pending, _key: None

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
    adapter._spawn_drain_task = lambda pending, _key: drained.append(pending)
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

    def __enter__(self):
        self._lock.acquire()
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
        background_review, "_interrupt_background_review", lambda _agent: None
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
        background_review, "_interrupt_background_review", lambda _agent: None
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
        gateway_review_admission=admission,
    )

    result = await runner._run_agent_queued_followup(
        turn_ctx, adapter, "queued text", None, "response", {"interrupted": True}, None
    )

    assert result["final_response"] == "follow-up reply"
    assert captured["gateway_review_admission"] is admission


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
