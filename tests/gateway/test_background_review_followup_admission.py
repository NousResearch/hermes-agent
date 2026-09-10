"""Gateway turns expose queued same-session messages to background-review admission."""

from __future__ import annotations

import asyncio
import threading
import time
import types
from typing import cast

import pytest

from agent import background_review
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, TextDebounceState
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.run_turn_runner import TurnRunner
from gateway.session import SessionSource


def _wire_with_adapter(adapter, session_key: str = "session-key"):
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
        callback_observations.append(request_started.wait(timeout=1.0))
        callback_observations.append(tracked_lock.held_by_current_thread())
        callback_observations.append(adapter.has_pending_message("session-key"))
        if not tracked_lock.held_by_current_thread():
            callback_observations.append(request_done.wait(timeout=1.0))
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
    contender_threads[0].join(timeout=1.0)
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
        assert request_started.wait(timeout=1.0)
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
    contender_threads[0].join(timeout=1.0)
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
