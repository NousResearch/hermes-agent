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
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock
from urllib.parse import quote

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


class _ThreeTurnAgent:
    calls: list = []
    adapter = None

    def __init__(self, **kwargs):
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None, **_kwargs):
        type(self).calls.append(message)
        if len(type(self).calls) == 2:
            type(self).adapter._pending_messages[SESSION_KEY] = MessageEvent(
                text="the final follow-up",
                message_type=MessageType.TEXT,
                source=_source(),
                message_id="queued-2",
            )
        return {
            "final_response": f"done-{len(type(self).calls)}",
            "messages": [],
            "api_calls": 1,
        }


class _ThreeTurnRaisingAgent(_ThreeTurnAgent):
    def run_conversation(self, message, conversation_history=None, task_id=None, **_kwargs):
        type(self).calls.append(message)
        if len(type(self).calls) == 2:
            type(self).adapter._pending_messages[SESSION_KEY] = MessageEvent(
                text="the doomed final follow-up",
                message_type=MessageType.TEXT,
                source=_source(),
                message_id="queued-2",
            )
        if len(type(self).calls) == 3:
            raise RuntimeError("boom in the nested queued follow-up turn")
        return {
            "final_response": f"done-{len(type(self).calls)}",
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
    (tmp_path / "config.yaml").write_text("display: {suppress_warning_notifications: true}")
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


async def _complete_deferred(adapter, result, outcome):
    outer_event = MessageEvent(text="opening", source=_source(), message_id="opening-1")
    outer_event._queued_followup_processing_tickets = result.pop(
        "_queued_followup_processing_tickets")
    await adapter._complete_queued_followup_processing(outer_event, outcome)


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

    # The runner can start the queued event, but SUCCESS is delivery-aware: the
    # terminal reply is still returned to BasePlatformAdapter for its final send.
    assert adapter.started == ["queued-1"]
    assert adapter.completed == []
    await _complete_deferred(adapter, result, ProcessingOutcome.SUCCESS)
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

    result = await runner._run_agent(
        message="the first turn",
        context_prompt="",
        history=[],
        source=_source(),
        session_id="sess-hooks-raw",
        session_key=SESSION_KEY,
    )

    assert adapter.started == [None]
    assert adapter.completed == []
    await _complete_deferred(adapter, result, ProcessingOutcome.SUCCESS)
    assert adapter.completed == [(None, ProcessingOutcome.SUCCESS)]


@pytest.mark.asyncio
async def test_multiple_queued_followups_complete_once_in_arrival_order(monkeypatch, tmp_path):
    _ThreeTurnAgent.calls = []
    _install_fake_agent(monkeypatch, tmp_path, _ThreeTurnAgent)

    # Opening and first queued responses deliver in-band; the terminal queued
    # response is returned to Base and deliberately refused by the transport.
    adapter = DeliveryRecordingAdapter(send_success=False, send_results=[True, True])
    _ThreeTurnAgent.adapter = adapter
    runner = _make_runner(adapter)
    adapter._pending_messages[SESSION_KEY] = MessageEvent(
        text="the first follow-up",
        source=_source(),
        message_id="queued-1",
    )

    opening_event = MessageEvent(text="the first turn", source=_source(), message_id="opening-1")

    async def _handler(_event):
        result = await runner._run_agent(
            message="the first turn",
            context_prompt="",
            history=[],
            source=_source(),
            session_id="sess-hooks-chain",
            session_key=SESSION_KEY,
        )
        _event._queued_followup_processing_tickets = result.pop(
            "_queued_followup_processing_tickets")
        return result["final_response"]

    adapter.set_message_handler(_handler)
    adapter._active_sessions[SESSION_KEY] = asyncio.Event()
    await adapter._process_message_background(opening_event, SESSION_KEY)

    assert _ThreeTurnAgent.calls == [
        "the first turn", "the first follow-up", "the final follow-up"]
    assert adapter.started == ["opening-1", "queued-1", "queued-2"]
    assert [item for item in adapter.completed if item[0] != "opening-1"] == [
        ("queued-1", ProcessingOutcome.SUCCESS),
        ("queued-2", ProcessingOutcome.FAILURE),
    ]
    assert adapter.timeline.index(("send", "done-2", True)) < adapter.timeline.index(
        ("complete", "queued-1", ProcessingOutcome.SUCCESS))
    assert adapter.timeline.index(("complete", "queued-1", ProcessingOutcome.SUCCESS)) < (
        adapter.timeline.index(("send", "done-3", False)))
    assert adapter.completed.count(("queued-1", ProcessingOutcome.SUCCESS)) == 1
    assert adapter.completed.count(("queued-2", ProcessingOutcome.FAILURE)) == 1


@pytest.mark.asyncio
async def test_nested_failure_does_not_overwrite_delivered_intermediate_success(
    monkeypatch, tmp_path,
):
    _ThreeTurnRaisingAgent.calls = []
    _install_fake_agent(monkeypatch, tmp_path, _ThreeTurnRaisingAgent)

    adapter = DeliveryRecordingAdapter(send_results=[True, True])
    _ThreeTurnRaisingAgent.adapter = adapter
    runner = _make_runner(adapter)
    adapter._pending_messages[SESSION_KEY] = MessageEvent(
        text="the first follow-up", source=_source(), message_id="queued-1")

    with pytest.raises(RuntimeError, match="nested queued follow-up"):
        await runner._run_agent(
            message="the first turn",
            context_prompt="",
            history=[],
            source=_source(),
            session_id="sess-hooks-nested-failure",
            session_key=SESSION_KEY,
        )

    assert _ThreeTurnRaisingAgent.calls == [
        "the first turn", "the first follow-up", "the doomed final follow-up"]
    assert adapter.completed == [
        ("queued-1", ProcessingOutcome.SUCCESS),
        ("queued-2", ProcessingOutcome.FAILURE),
    ]
    assert adapter.completed.count(("queued-1", ProcessingOutcome.SUCCESS)) == 1


@pytest.mark.asyncio
async def test_depth_cap_preserves_current_terminal_ticket(monkeypatch, tmp_path):
    _ThreeTurnAgent.calls = []
    _install_fake_agent(monkeypatch, tmp_path, _ThreeTurnAgent)

    adapter = HookRecordingAdapter()
    _ThreeTurnAgent.adapter = adapter
    runner = _make_runner(adapter)
    runner._MAX_INTERRUPT_DEPTH = 1
    adapter._pending_messages[SESSION_KEY] = MessageEvent(
        text="the first follow-up", source=_source(), message_id="queued-1")

    result = await runner._run_agent(
        message="the first turn",
        context_prompt="",
        history=[],
        source=_source(),
        session_id="sess-hooks-depth-cap",
        session_key=SESSION_KEY,
    )

    assert _ThreeTurnAgent.calls == ["the first turn", "the first follow-up"]
    assert adapter.started == ["queued-1"]
    assert adapter.completed == []
    tickets = result["_queued_followup_processing_tickets"]
    assert len(tickets) == 1
    assert tickets[0]["event"].message_id == "queued-1"
    assert adapter._pending_messages[SESSION_KEY].message_id == "queued-2"

    await _complete_deferred(adapter, result, ProcessingOutcome.SUCCESS)
    assert adapter.completed == [("queued-1", ProcessingOutcome.SUCCESS)]


class DeliveryRecordingAdapter(HookRecordingAdapter):
    def __init__(self, *, send_success: bool = True, send_results=None):
        super().__init__()
        self.send_success = send_success
        self.send_results = list(send_results or [])
        self.timeline = []

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        success = self.send_results.pop(0) if self.send_results else self.send_success
        self.timeline.append(("send", content, success))
        return SendResult(
            success=success,
            message_id="sent-1" if success else None,
            error=None if success else "delivery refused",
        )

    async def on_processing_complete(self, event, outcome) -> None:
        self.timeline.append(("complete", getattr(event, "message_id", None), outcome))
        await super().on_processing_complete(event, outcome)


async def _run_base_delivery(adapter, *, response, queued_message_ids):
    opening_event = MessageEvent(text="opening", source=_source(), message_id="opening-1")
    tickets = []
    for message_id in queued_message_ids:
        queued_event = MessageEvent(text="queued", source=_source(), message_id=message_id)
        await adapter.on_processing_start(queued_event)
        tickets.append({"adapter": adapter, "event": queued_event, "closed": False})

    async def _handler(event):
        event._queued_followup_processing_tickets = tickets
        return response

    adapter.set_message_handler(_handler)
    adapter._active_sessions[SESSION_KEY] = asyncio.Event()
    await adapter._process_message_background(opening_event, SESSION_KEY)
    return tickets


@pytest.mark.asyncio
async def test_terminal_delivery_failure_completes_queued_hook_after_send():
    adapter = DeliveryRecordingAdapter(send_success=False)

    tickets = await _run_base_delivery(
        adapter, response="terminal answer", queued_message_ids=["queued-final"])

    queued_completion = (
        "complete", "queued-final", ProcessingOutcome.FAILURE)
    assert queued_completion in adapter.timeline
    assert adapter.timeline.index(("send", "terminal answer", False)) < adapter.timeline.index(
        queued_completion)
    assert adapter.completed.count(("queued-final", ProcessingOutcome.FAILURE)) == 1
    assert tickets[0]["closed"] is True


@pytest.mark.asyncio
async def test_no_final_send_after_tool_delivery_still_completes_successfully():
    adapter = DeliveryRecordingAdapter(send_success=False)

    await _run_base_delivery(
        adapter, response=None, queued_message_ids=["queued-tool-delivered"])

    assert not any(item[0] == "send" for item in adapter.timeline)
    assert adapter.completed.count(
        ("queued-tool-delivered", ProcessingOutcome.SUCCESS)) == 1


class BlockingCompletionAdapter(DeliveryRecordingAdapter):
    def __init__(self):
        super().__init__(send_success=True)
        self.completion_attempts = []
        self.first_queued_completion_entered = asyncio.Event()
        self.release_first_queued_completion = asyncio.Event()

    async def on_processing_complete(self, event, outcome) -> None:
        message_id = getattr(event, "message_id", None)
        self.completion_attempts.append((message_id, outcome))
        if message_id == "queued-1":
            self.first_queued_completion_entered.set()
            await self.release_first_queued_completion.wait()
        await super().on_processing_complete(event, outcome)


@pytest.mark.asyncio
async def test_cancel_mid_completion_keeps_later_tickets_and_delivery_outcome():
    adapter = BlockingCompletionAdapter()
    opening_event = MessageEvent(text="opening", source=_source(), message_id="opening-1")
    tickets = []
    for message_id in ("queued-1", "queued-2", "queued-3"):
        queued_event = MessageEvent(text="queued", source=_source(), message_id=message_id)
        tickets.append({"adapter": adapter, "event": queued_event, "closed": False})

    async def _handler(event):
        event._queued_followup_processing_tickets = tickets
        return "terminal answer"

    adapter.set_message_handler(_handler)
    adapter._active_sessions[SESSION_KEY] = asyncio.Event()
    task = asyncio.create_task(adapter._process_message_background(opening_event, SESSION_KEY))
    await asyncio.wait_for(adapter.first_queued_completion_entered.wait(), timeout=5)
    adapter._expected_cancelled_tasks.add(task)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # The first ticket was claimed exactly once when cancellation landed.  Later
    # tickets were still attached to the event and receive the already-proven
    # successful delivery verdict before cancellation propagates.
    queued_attempts = [item for item in adapter.completion_attempts if item[0].startswith("queued-")]
    assert queued_attempts == [
        ("queued-1", ProcessingOutcome.SUCCESS),
        ("queued-2", ProcessingOutcome.SUCCESS),
        ("queued-3", ProcessingOutcome.SUCCESS),
    ]
    assert adapter.completed == [
        ("opening-1", ProcessingOutcome.SUCCESS),
        ("queued-2", ProcessingOutcome.SUCCESS),
        ("queued-3", ProcessingOutcome.SUCCESS),
        ("opening-1", ProcessingOutcome.CANCELLED),
    ]
    assert all(ticket["closed"] for ticket in tickets)
    assert opening_event._queued_followup_processing_tickets == []


@pytest.mark.asyncio
async def test_stale_suppressed_terminal_queued_response_is_cancelled():
    adapter = DeliveryRecordingAdapter(send_success=True)
    opening_event = MessageEvent(text="opening", source=_source(), message_id="opening-1")
    queued_event = MessageEvent(text="queued", source=_source(), message_id="queued-stale")
    ticket = {"adapter": adapter, "event": queued_event, "closed": False}

    async def _handler(event):
        event._queued_followup_processing_tickets = [ticket]
        adapter._pending_messages[SESSION_KEY] = MessageEvent(
            text="newer", source=_source(), message_id="newer-1")
        adapter._active_sessions[SESSION_KEY].set()
        return "stale terminal answer"

    adapter.set_message_handler(_handler)
    adapter._active_sessions[SESSION_KEY] = asyncio.Event()
    adapter._spawn_drain_task = lambda *_args, **_kwargs: None
    await adapter._process_message_background(opening_event, SESSION_KEY)

    assert not any(item[0] == "send" for item in adapter.timeline)
    assert adapter.completed.count(
        ("queued-stale", ProcessingOutcome.CANCELLED)) == 1
    assert ticket["closed"] is True


@pytest.mark.asyncio
async def test_stale_generation_real_handler_cancels_terminal_queued_ticket():
    """A /new or /stop generation bump after the queued chain returns must not
    turn the handler's stale ``None`` into a successful queued acknowledgement."""
    adapter = DeliveryRecordingAdapter(send_success=True)
    runner = _make_runner(adapter)
    source = _source()
    opening_event = MessageEvent(text="opening", source=source, message_id="opening-1")
    queued_event = MessageEvent(text="terminal queued", source=source, message_id="queued-stale-gen")
    ticket = {"adapter": adapter, "event": queued_event, "closed": False}
    session_entry = SimpleNamespace(session_id="sess-stale-generation")
    prepared = runner._PreparedTurn(
        history=[],
        context_prompt="",
        message_text="opening",
        persist_user_message="opening",
        persist_user_timestamp=None,
        persist_user_display_kind=None,
        persistence_session_id=session_entry.session_id,
        persistence_owner="handler-path-test",
    )
    runner._hmwa_resolve_session = AsyncMock(
        return_value=(source, session_entry, SESSION_KEY))
    runner._hmwa_prepare_turn = AsyncMock(return_value=(prepared, []))
    runner._hmwa_stop_typing_for_turn = AsyncMock()
    runner._clear_session_env = lambda _tokens: None
    runner.hooks = SimpleNamespace(emit=AsyncMock())
    runner._run_agent = AsyncMock(return_value={
        "final_response": "stale terminal answer",
        "messages": [],
        "_queued_followup_processing_tickets": [ticket],
    })
    runner._is_session_run_current = lambda _key, _generation: False

    async def _real_handler(event):
        return await runner._handle_message_with_agent(
            event, source, SESSION_KEY, 7)

    adapter.set_message_handler(_real_handler)
    adapter._active_sessions[SESSION_KEY] = asyncio.Event()
    await adapter._process_message_background(opening_event, SESSION_KEY)

    runner._run_agent.assert_awaited_once()
    assert not any(item[0] == "send" for item in adapter.timeline)
    assert adapter.completed.count(
        ("queued-stale-gen", ProcessingOutcome.CANCELLED)) == 1
    assert not any(
        message_id == "queued-stale-gen" and outcome is ProcessingOutcome.SUCCESS
        for message_id, outcome in adapter.completed
    )
    assert ticket["closed"] is True


@pytest.mark.asyncio
async def test_filtered_attachment_only_response_returns_failed_delivery_verdict(monkeypatch):
    adapter = DeliveryRecordingAdapter(send_success=True)
    runner = _make_runner(adapter)
    adapter.extract_media = lambda _response: ([("C:/outside/not-approved.pdf", False)], "")
    monkeypatch.setattr(
        BasePlatformAdapter, "filter_media_delivery_paths", staticmethod(lambda _media: []))

    verdict = await runner._deliver_queued_first_response(
        "MEDIA:C:/outside/not-approved.pdf",
        source=_source(),
        adapter=adapter,
        deliver_media=True,
    )

    assert verdict == {"expected": True, "attempted": False, "succeeded": False}
    assert adapter.timeline == []


@pytest.mark.asyncio
async def test_attachment_send_result_failure_returns_failed_delivery_verdict(monkeypatch):
    adapter = DeliveryRecordingAdapter(send_success=True)
    runner = _make_runner(adapter)
    media = [("C:/approved/report.pdf", False)]
    adapter.extract_media = lambda _response: (media, "")
    monkeypatch.setattr(
        BasePlatformAdapter, "filter_media_delivery_paths", staticmethod(lambda paths: paths))

    async def _fail_document(**_kwargs):
        return SendResult(success=False, error="upload refused")

    adapter.send_document = _fail_document
    verdict = await runner._deliver_queued_first_response(
        "MEDIA:C:/approved/report.pdf",
        source=_source(),
        adapter=adapter,
        deliver_media=True,
    )

    assert verdict == {"expected": True, "attempted": True, "succeeded": False}
    assert adapter.timeline == []


@pytest.mark.asyncio
async def test_one_successful_attachment_does_not_hide_a_failed_sibling(monkeypatch):
    adapter = DeliveryRecordingAdapter(send_success=True)
    runner = _make_runner(adapter)
    media = [("C:/approved/one.pdf", False), ("C:/approved/two.pdf", False)]
    adapter.extract_media = lambda _response: (media, "")
    monkeypatch.setattr(
        BasePlatformAdapter, "filter_media_delivery_paths", staticmethod(lambda paths: paths))
    outcomes = iter((True, False))

    async def _send_document(**_kwargs):
        return SendResult(success=next(outcomes))

    adapter.send_document = _send_document
    verdict = await runner._deliver_queued_first_response(
        "MEDIA:C:/approved/one.pdf\nMEDIA:C:/approved/two.pdf",
        source=_source(), adapter=adapter, deliver_media=True,
    )

    assert verdict == {"expected": True, "attempted": True, "succeeded": False}
    assert verdict.media_expected is True
    assert verdict.media_succeeded is False


@pytest.mark.asyncio
async def test_delivered_text_keeps_media_retry_enabled_after_partial_failure():
    from gateway.run_notifications import _DeliveryVerdict

    adapter = DeliveryRecordingAdapter(send_success=True)
    runner = _make_runner(adapter)
    verdict = _DeliveryVerdict({"expected": True, "attempted": True, "succeeded": False})
    verdict.text_expected = verdict.text_succeeded = True
    verdict.media_expected = True
    verdict.media_succeeded = False
    runner._deliver_queued_first_response = AsyncMock(return_value=verdict)
    turn_ctx = SimpleNamespace(
        session_key=SESSION_KEY, stream_consumer_holder=[None], mute_notification_reply=False,
        persist_user_display_kind=None, source=_source(), _status_thread_metadata=None,
        event_message_id=None, inbound_message_id="queued-media", run_generation=1,
        _queued_processing_ticket=None,
    )
    result = {"final_response": "answer with MEDIA", "failed": False}

    await runner._run_agent_deliver_first_response(turn_ctx, adapter, "answer", result, None)

    assert result["already_sent"] is True
    assert result["media_already_delivered"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "filename",
    [
        "hello world-صورة.png",
        "image #1-100%.png",
        r"literal\name image.png",
    ],
)
async def test_attachment_file_url_preserves_native_path_contract(
    monkeypatch, tmp_path, filename,
):
    # A backslash is a separator on Windows but part of a filename on POSIX.
    # Exercise the real host path rather than imposing one OS's rules on the other.
    media = tmp_path / filename
    media.parent.mkdir(parents=True, exist_ok=True)
    media.write_bytes(b"test image")
    media_path = str(media)
    expected_url = f"file://{quote(media.as_posix(), safe='/:')}"
    adapter = DeliveryRecordingAdapter(send_success=True)
    runner = _make_runner(adapter)
    adapter.extract_media = lambda _response: ([(media_path, False)], "")
    monkeypatch.setattr(
        BasePlatformAdapter, "filter_media_delivery_paths", staticmethod(lambda paths: paths))
    captured = []

    async def _capture_images(**kwargs):
        captured.extend(kwargs["images"])
        return SendResult(success=True, message_id="image-1")

    adapter.send_multiple_images = _capture_images
    verdict = await runner._deliver_queued_first_response(
        f"MEDIA:{media_path}",
        source=_source(),
        adapter=adapter,
        deliver_media=True,
    )

    assert captured == [(expected_url, "")]
    assert verdict == {"expected": True, "attempted": True, "succeeded": True}


@pytest.mark.asyncio
async def test_declined_reconcile_edit_returns_failure_without_duplicate_send():
    adapter = DeliveryRecordingAdapter(send_success=True)
    runner = _make_runner(adapter)

    async def _decline_edit(**_kwargs):
        return SendResult(
            success=False,
            error="discord egress declined: target is not an approved destination",
        )

    adapter.edit_message = _decline_edit
    verdict = await runner._deliver_queued_first_response(
        "sensitive answer",
        source=_source(),
        adapter=adapter,
        deliver_media=False,
        stream_consumer=SimpleNamespace(message_id="draft-1", _turn_split_delivery=False),
    )

    assert verdict == {"expected": True, "attempted": True, "succeeded": False}
    assert adapter.timeline == []


class BlockingStartAdapter(HookRecordingAdapter):
    def __init__(self):
        super().__init__()
        self.start_entered = asyncio.Event()
        self.release_start = asyncio.Event()

    async def on_processing_start(self, event: MessageEvent) -> None:
        self.started.append(getattr(event, "message_id", None))
        self.start_entered.set()
        await self.release_start.wait()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("expected_cancel", "outcome"),
    [
        (True, ProcessingOutcome.CANCELLED),
        (False, ProcessingOutcome.FAILURE),
    ],
)
async def test_cancel_during_start_uses_background_task_owner(
    monkeypatch, tmp_path, expected_cancel, outcome,
):
    _TwoTurnAgent.calls = []
    _install_fake_agent(monkeypatch, tmp_path, _TwoTurnAgent)

    owner = HookRecordingAdapter()
    hook_adapter = BlockingStartAdapter()
    runner = _make_runner(owner)
    runner._intake_adapter_for = (
        lambda source: hook_adapter if source.profile == "secondary" else owner)
    queued_source = _source()
    queued_source.profile = "secondary"
    owner._pending_messages[SESSION_KEY] = MessageEvent(
        text="the cancellable follow-up",
        source=queued_source,
        message_id="queued-cancel",
    )

    task = asyncio.create_task(runner._run_agent(
        message="the first turn",
        context_prompt="",
        history=[],
        source=_source(),
        session_id="sess-hooks-cancel",
        session_key=SESSION_KEY,
    ))
    await asyncio.wait_for(hook_adapter.start_entered.wait(), timeout=5)
    if expected_cancel:
        owner._expected_cancelled_tasks.add(task)
    assert task not in hook_adapter._expected_cancelled_tasks
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert hook_adapter.completed == [("queued-cancel", outcome)]


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
