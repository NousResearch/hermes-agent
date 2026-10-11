"""Processing-hook parity for queued follow-up turns.

A message that arrives while a turn is already running is parked in the
adapter's ``_pending_messages`` slot and drained *in-band* by
``GatewayRunner._run_agent`` rather than by
``BasePlatformAdapter._process_message_background``.  The runner-side drain
must still fire the ``on_processing_start`` / ``on_processing_complete``
lifecycle hooks, otherwise every platform that renders a read-receipt
reaction from those hooks (Slack 👀, Discord, Telegram, Feishu, Matrix,
Signal, ...) silently skips the acknowledgement for mid-turn messages.
"""

import asyncio
import importlib
import sys
import threading
import types
from types import SimpleNamespace

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    ProcessingOutcome,
    SendResult,
)
from gateway.session import SessionSource


class HookRecordingAdapter(BasePlatformAdapter):
    """Adapter that records the processing-hook lifecycle it is driven through."""

    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="***"), Platform.TELEGRAM)
        self.started: list = []
        self.completed: list = []

    async def connect(self) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        return SendResult(success=True, message_id="sent-1")

    async def send_typing(self, chat_id, metadata=None) -> None:
        return None

    async def stop_typing(self, chat_id) -> None:
        return None

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}

    async def on_processing_start(self, event: MessageEvent) -> None:
        self.started.append(getattr(event, "message_id", None))

    async def on_processing_complete(self, event, outcome) -> None:
        self.completed.append((getattr(event, "message_id", None), outcome))


class _TwoTurnAgent:
    calls: list = []

    def __init__(self, **kwargs):
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None, **_kwargs):
        type(self).calls.append(message)
        return {
            "final_response": f"done-{len(type(self).calls)}",
            "messages": [],
            "api_calls": 1,
        }


class _RaisingSecondTurnAgent:
    calls: list = []

    def __init__(self, **kwargs):
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None, **_kwargs):
        type(self).calls.append(message)
        if len(type(self).calls) >= 2:
            raise RuntimeError("boom in the queued follow-up turn")
        return {
            "final_response": "done-1",
            "messages": [],
            "api_calls": 1,
        }


def _make_runner(adapter):
    gateway_run = importlib.import_module("gateway.run")
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {adapter.platform: adapter}
    runner._voice_mode = {}
    runner._prefill_messages = []
    runner._ephemeral_system_prompt = ""
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._session_db = None
    runner._running_agents = {}
    runner._session_run_generation = {}
    runner.hooks = SimpleNamespace(loaded_hooks=False)
    runner.config = SimpleNamespace(
        thread_sessions_per_user=False,
        group_sessions_per_user=False,
        stt_enabled=False,
    )
    runner._model = "openai/gpt-4.1-mini"
    runner._base_url = None
    return runner


def _install_fake_agent(monkeypatch, tmp_path, agent_cls):
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = agent_cls
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )


SESSION_KEY = "agent:main:telegram:dm:4242"


@pytest.mark.asyncio
@pytest.mark.parametrize("diagnostic_last", [False, True])
async def test_queue_terminal_presentation_belongs_to_last_turn(monkeypatch, tmp_path, diagnostic_last):
    _TwoTurnAgent.calls = []
    _install_fake_agent(monkeypatch, tmp_path, _TwoTurnAgent)
    (tmp_path / "config.yaml").write_text(
        "display: {suppress_warning_notifications: true}", encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = HookRecordingAdapter()
    runner = _make_runner(adapter)
    adapter._pending_messages[SESSION_KEY] = MessageEvent(
        text="follow-up", source=_source(), internal=diagnostic_last,
        metadata={"notification_category": "diagnostic"} if diagnostic_last else {}, message_id="queued")
    result = await runner._run_agent(message="first", context_prompt="", history=[], source=_source(),
        session_id="queue-policy", session_key=SESSION_KEY,
        persist_user_display_metadata=None if diagnostic_last else {"notification_category": "diagnostic"})
    assert _TwoTurnAgent.calls == ["first", "follow-up"]
    assert result["final_response"] == "done-2"
    assert result["_notification_reply_muted"] is diagnostic_last
    from gateway.warning_notifications import diagnostic_wake_muted
    outer = MessageEvent(text="first", source=_source(), internal=not diagnostic_last)
    outer._notification_reply_muted = result["_notification_reply_muted"]
    assert diagnostic_wake_muted(outer) is diagnostic_last


def _source():
    return SessionSource(platform=Platform.TELEGRAM, chat_id="4242", chat_type="dm")


@pytest.mark.asyncio
async def test_queued_followup_fires_processing_hooks(monkeypatch, tmp_path):
    """The runner-drained follow-up gets the same start/complete hooks as a
    message that arrives while the session is idle."""
    _TwoTurnAgent.calls = []
    _install_fake_agent(monkeypatch, tmp_path, _TwoTurnAgent)

    adapter = HookRecordingAdapter()
    runner = _make_runner(adapter)

    adapter._pending_messages[SESSION_KEY] = MessageEvent(
        text="the follow-up",
        message_type=MessageType.TEXT,
        source=_source(),
        message_id="queued-1",
    )

    result = await runner._run_agent(
        message="the first turn",
        context_prompt="",
        history=[],
        source=_source(),
        session_id="sess-hooks",
        session_key=SESSION_KEY,
    )

    # The follow-up really did run in-band.
    assert result["final_response"] == "done-2"
    assert _TwoTurnAgent.calls == ["the first turn", "the follow-up"]

    # ...and it was acknowledged through the lifecycle hooks.
    assert adapter.started == ["queued-1"]
    assert adapter.completed == [("queued-1", ProcessingOutcome.SUCCESS)]


@pytest.mark.asyncio
async def test_queued_followup_failure_completes_the_hook(monkeypatch, tmp_path):
    """A follow-up turn that blows up still closes its hook, so a platform
    never strands a 'still working' marker on the user's message."""
    _RaisingSecondTurnAgent.calls = []
    _install_fake_agent(monkeypatch, tmp_path, _RaisingSecondTurnAgent)

    adapter = HookRecordingAdapter()
    runner = _make_runner(adapter)

    adapter._pending_messages[SESSION_KEY] = MessageEvent(
        text="the doomed follow-up",
        message_type=MessageType.TEXT,
        source=_source(),
        message_id="queued-2",
    )

    with pytest.raises(RuntimeError):
        await runner._run_agent(
            message="the first turn",
            context_prompt="",
            history=[],
            source=_source(),
            session_id="sess-hooks-failure",
            session_key=SESSION_KEY,
        )

    assert adapter.started == ["queued-2"]
    assert adapter.completed == [("queued-2", ProcessingOutcome.FAILURE)]


@pytest.mark.asyncio
async def test_synthetic_followup_is_not_acknowledged(monkeypatch, tmp_path):
    """Drains with no inbound platform message — /goal continuations, wake-ups,
    CLI hand-offs — carry no message_id and must stay silent: there is nothing
    on the platform to react to."""
    _TwoTurnAgent.calls = []
    _install_fake_agent(monkeypatch, tmp_path, _TwoTurnAgent)

    adapter = HookRecordingAdapter()
    runner = _make_runner(adapter)

    adapter._pending_messages[SESSION_KEY] = MessageEvent(
        text="synthetic continuation",
        message_type=MessageType.TEXT,
        source=_source(),
        message_id=None,
    )

    result = await runner._run_agent(
        message="the first turn",
        context_prompt="",
        history=[],
        source=_source(),
        session_id="sess-hooks-synthetic",
        session_key=SESSION_KEY,
    )

    # It still ran — we only suppressed the acknowledgement, not the turn.
    assert result["final_response"] == "done-2"
    assert _TwoTurnAgent.calls == ["the first turn", "synthetic continuation"]

    assert adapter.started == []
    assert adapter.completed == []


@pytest.mark.asyncio
async def test_raw_envelope_only_followup_is_acknowledged(monkeypatch, tmp_path):
    """Signal never sets message_id — its hook keys off the raw envelope
    (sender + timestamp_ms) — and Discord's reads raw_message. An event
    carrying only a raw envelope is still a real inbound message."""
    _TwoTurnAgent.calls = []
    _install_fake_agent(monkeypatch, tmp_path, _TwoTurnAgent)

    adapter = HookRecordingAdapter()
    runner = _make_runner(adapter)

    adapter._pending_messages[SESSION_KEY] = MessageEvent(
        text="signal-shaped follow-up",
        message_type=MessageType.TEXT,
        source=_source(),
        message_id=None,
        raw_message={"sender": "+15550100", "timestamp_ms": 1700000000000},
    )

    await runner._run_agent(
        message="the first turn",
        context_prompt="",
        history=[],
        source=_source(),
        session_id="sess-hooks-raw",
        session_key=SESSION_KEY,
    )

    assert adapter.started == [None]
    assert adapter.completed == [(None, ProcessingOutcome.SUCCESS)]


class CompleteOnlyAdapter(HookRecordingAdapter):
    """Google Chat and webhook implement on_processing_complete WITHOUT
    on_processing_start; theirs is end-of-cycle teardown (reap the typing
    card / end the delivery session), not a reaction."""

    on_processing_start = BasePlatformAdapter.on_processing_start


@pytest.mark.asyncio
async def test_complete_only_adapter_is_left_alone(monkeypatch, tmp_path):
    """We bracket, so both halves must belong to us. An adapter that only
    implements the completion half must not be handed a completion here: at
    this point the follow-up's reply has not been delivered yet, so its
    teardown would fire against a live turn."""
    _TwoTurnAgent.calls = []
    _install_fake_agent(monkeypatch, tmp_path, _TwoTurnAgent)

    adapter = CompleteOnlyAdapter()
    runner = _make_runner(adapter)

    adapter._pending_messages[SESSION_KEY] = MessageEvent(
        text="the follow-up",
        message_type=MessageType.TEXT,
        source=_source(),
        message_id="queued-3",
    )

    result = await runner._run_agent(
        message="the first turn",
        context_prompt="",
        history=[],
        source=_source(),
        session_id="sess-hooks-complete-only",
        session_key=SESSION_KEY,
    )

    assert result["final_response"] == "done-2"
    assert adapter.completed == []


@pytest.mark.asyncio
async def test_fifo_depth_limit_preserves_every_accepted_message(monkeypatch, tmp_path):
    """The recursion cap hands the oldest event back to the adapter, not over its successor."""
    first_started = threading.Event()
    release_first = threading.Event()
    model_calls = []

    class BlockingFirstAgent(_TwoTurnAgent):
        def run_conversation(self, message, conversation_history=None, task_id=None, **kwargs):
            model_calls.append(message)
            if message == "M1":
                first_started.set()
                if not release_first.wait(5):
                    raise AssertionError("first turn was not released")
            return {"final_response": f"done-{message}", "messages": [], "api_calls": 1}

    _install_fake_agent(monkeypatch, tmp_path, BlockingFirstAgent)
    adapter = HookRecordingAdapter()
    runner = _make_runner(adapter)
    runner._queued_events = {}
    turns = []
    outer_ids = []
    real_run_agent = runner._run_agent

    async def record_turn(**kwargs):
        turns.append((kwargs.get("inbound_message_id"), kwargs.get("_interrupt_depth", 0)))
        return await real_run_agent(**kwargs)

    monkeypatch.setattr(runner, "_run_agent", record_turn)

    async def handler(event):
        outer_ids.append(event.message_id)
        await runner._run_agent(
            message=event.text, context_prompt="", history=[], source=event.source,
            session_id="fifo-depth", session_key=SESSION_KEY,
            inbound_message_id=event.message_id, event_message_id=event.message_id,
        )
        return None

    adapter.set_message_handler(handler)
    events = [MessageEvent(text=f"M{i}", source=_source(), message_id=f"M{i}")
              for i in range(1, 11)]
    task = asyncio.create_task(adapter._process_message_background(events[0], SESSION_KEY))
    try:
        assert await asyncio.to_thread(first_started.wait, 5)
        assert SESSION_KEY in adapter._active_sessions
        for event in events[1:]:
            runner._enqueue_fifo(SESSION_KEY, event, adapter)
        accepted_ids = [events[0].message_id] + [
            event.message_id for event in events[1:] if event._gateway_accepted]
        assert runner._queue_depth(SESSION_KEY, adapter=adapter) == len(events) - 1
    finally:
        release_first.set()
    try:
        await asyncio.wait_for(task, 10)
        # Exercise the real adapter's fresh-task handoff as well as the runner's recursion.
        while adapter._background_tasks:
            await asyncio.wait_for(asyncio.gather(*list(adapter._background_tasks)), 10)
        remaining_depth = runner._queue_depth(SESSION_KEY, adapter=adapter)
        guard_live = SESSION_KEY in adapter._active_sessions
    finally:
        await adapter.cancel_background_tasks()

    assert [message_id for message_id, _ in turns] == accepted_ids
    assert model_calls == accepted_ids
    cap = runner._MAX_INTERRUPT_DEPTH
    assert [depth for _, depth in turns] == [i % (cap + 1) for i in range(len(events))]
    assert outer_ids == accepted_ids[::cap + 1]
    assert adapter.started == accepted_ids
    assert sorted(adapter.completed) == sorted(
        (message_id, ProcessingOutcome.SUCCESS) for message_id in accepted_ids)
    assert remaining_depth == 0
    assert not guard_live


@pytest.mark.asyncio
@pytest.mark.parametrize("message_type", [MessageType.TEXT, MessageType.PHOTO, MessageType.DOCUMENT])
@pytest.mark.parametrize("overflow_tail", [False, True])
async def test_depth_limit_restores_events_without_media_merging(
    message_type, overflow_tail,
):
    """Put-back preserves event identity/payload, including when promotion emptied overflow."""
    adapter = HookRecordingAdapter()
    runner = _make_runner(adapter)
    runner._queued_events = {}
    source = _source()
    head = MessageEvent(
        text="unrun head", source=source, message_id="head", message_type=message_type,
        media_urls=[] if message_type == MessageType.TEXT else ["head-media"],
        media_types=[] if message_type == MessageType.TEXT else ["application/octet-stream"],
    )
    successor = MessageEvent(
        text="successor", source=source, message_id="successor", message_type=message_type,
        media_urls=[] if message_type == MessageType.TEXT else ["successor-media"],
        media_types=[] if message_type == MessageType.TEXT else ["application/octet-stream"],
    )
    tail = MessageEvent(text="tail", source=source, message_id="tail")
    queued = [head, successor] + ([tail] if overflow_tail else [])
    original_payloads = [(event.text, tuple(event.media_urls), tuple(event.media_types))
                         for event in queued]
    for event in queued:
        runner._enqueue_fifo(SESSION_KEY, event, adapter)
    result = {"final_response": "done", "messages": []}
    pending_event, pending = await runner._run_agent_drain_pending(
        result, adapter, source, SESSION_KEY)
    assert pending_event is head
    assert adapter._pending_messages[SESSION_KEY] is successor
    # An arrival during an awaited media drain must remain behind the already-staged successor.
    late = MessageEvent(text="late", source=source, message_id="late")
    runner._enqueue_fifo(SESSION_KEY, late, adapter)
    turn_ctx = SimpleNamespace(
        source=source, session_id="depth-media", session_key=SESSION_KEY,
        run_generation=None, _interrupt_depth=runner._MAX_INTERRUPT_DEPTH,
        history=[], _status_thread_metadata=None, result_holder=[result],
    )
    returned = await runner._run_agent_queued_followup(
        turn_ctx, adapter, pending, pending_event, result, result, None)
    assert returned is result
    assert adapter._pending_messages[SESSION_KEY] is head
    assert runner._overflow_queue(SESSION_KEY) == queued[1:] + [late]
    assert [(event.text, tuple(event.media_urls), tuple(event.media_types))
            for event in queued] == original_payloads
