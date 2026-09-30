"""Processing-lifecycle contracts that every bracketing adapter observes.

Adapters render ``on_processing_start`` and ``on_processing_complete`` as
in-progress and outcome markers (reactions, read receipts, typing cards), so
every started message must complete exactly once, with an outcome that
describes what happened to that message.
"""

import asyncio
import time
from dataclasses import replace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, ProcessingOutcome, SendResult
from gateway.run import _INTERRUPT_REASON_STOP, GatewayRunner
from gateway.session import SessionSource
from gateway.turn_context import TurnContext
from tests.gateway.test_queued_followup_processing_hooks import (
    SESSION_KEY,
    HookRecordingAdapter,
    _install_fake_agent,
    _make_runner,
    _source,
)


class LifecycleLogAdapter(HookRecordingAdapter):
    """Records starts, completions and sends in one ordered log."""

    def __init__(self):
        super().__init__()
        self.log: list = []
        self.refused: set = set()

    async def on_processing_start(self, event):
        self.log.append(("start", event.message_id))

    async def on_processing_complete(self, event, outcome):
        self.log.append(("complete", event.message_id, outcome))

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.log.append(("send", content))
        if any(refused in content for refused in self.refused):
            return SendResult(success=False, error="refused")
        return SendResult(success=True, message_id="sent")


class _ScriptedAgent:
    """Returns one scripted result per call, in order."""

    results: list = []
    calls: list = []

    def __init__(self, **kwargs):
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None, **_kwargs):
        type(self).calls.append(message)
        return type(self).results[len(type(self).calls) - 1]


def _interrupted(message):
    return {"final_response": "Operation interrupted.", "messages": [], "api_calls": 1,
            "interrupted": True, "completed": False, "interrupt_message": message}


def _done(text):
    return {"final_response": text, "messages": [], "api_calls": 1}


async def _run_chain(monkeypatch, tmp_path, results, queued_text="the voice note"):
    _ScriptedAgent.calls, _ScriptedAgent.results = [], results
    _install_fake_agent(monkeypatch, tmp_path, _ScriptedAgent)
    adapter = LifecycleLogAdapter()
    runner = _make_runner(adapter)
    adapter._pending_messages[SESSION_KEY] = MessageEvent(
        text=queued_text, source=_source(), message_id="queued-1")

    async def respond(event):
        result = await runner._run_agent(
            message=event.text, context_prompt="", history=[], source=event.source,
            session_id="sess-lifecycle", session_key=SESSION_KEY, processing_event=event)
        return result["final_response"]

    adapter.set_message_handler(respond)
    await adapter.handle_message(MessageEvent(text="the first turn", source=_source(), message_id="first-1"))
    await asyncio.gather(*adapter._background_tasks)
    return adapter


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("interrupt_message", "expected"),
    [
        (
            "the voice note",
            [("start", "first-1"), ("start", "queued-1"), ("send", "done-2"),
             ("complete", "queued-1", ProcessingOutcome.SUCCESS),
             ("complete", "first-1", ProcessingOutcome.SUCCESS)],
        ),
        (
            _INTERRUPT_REASON_STOP,
            [("start", "first-1"), ("complete", "first-1", ProcessingOutcome.CANCELLED),
             ("start", "queued-1"), ("send", "done-2"),
             ("complete", "queued-1", ProcessingOutcome.SUCCESS)],
        ),
    ],
    ids=["superseded-by-input", "stopped"],
)
async def test_interrupted_turn_outcome_depends_on_what_interrupted_it(
    monkeypatch, tmp_path, interrupt_message, expected
):
    """New input supersedes the turn, so the message completes with the chain that answers it; a stop
    cancels it."""
    adapter = await _run_chain(monkeypatch, tmp_path, [_interrupted(interrupt_message), _done("done-2")])

    assert (_ScriptedAgent.calls, adapter.log) == (["the first turn", "the voice note"], expected)


def _priority_runner(monkeypatch, mode):
    monkeypatch.setenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", "false")
    adapter = LifecycleLogAdapter()
    adapter.platform = Platform.SLACK
    runner = GatewayRunner(config=GatewayConfig())
    runner.adapters[Platform.SLACK] = adapter
    runner._busy_input_mode = mode
    monkeypatch.setattr(runner, "_is_user_authorized_for_source", lambda source: True)
    adapter.set_message_handler(runner._handle_message)
    return runner, adapter


_AGENT_VERBS = {"steer", "redirect", "interrupt"}
_START_RUNNING = ("start", "running-1")
_START_CORR = ("start", "corr-1")
_DONE_RUNNING = ("complete", "running-1", ProcessingOutcome.SUCCESS)
_DONE_CORR = ("complete", "corr-1", ProcessingOutcome.SUCCESS)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "redirects", "tracked", "leftover", "expected"),
    [
        ("steer", False, True, False, ("steer", None, [_START_RUNNING, _START_CORR, _DONE_CORR, _DONE_RUNNING])),
        ("interrupt", True, True, False,
         ("redirect", None, [_START_RUNNING, _START_CORR, _DONE_CORR, _DONE_RUNNING])),
        ("interrupt", False, True, False,
         ("interrupt", None, [_START_RUNNING, _START_CORR, _DONE_CORR, _DONE_RUNNING])),
        ("steer", False, True, True, ("steer", "corr-1", [_START_RUNNING, _START_CORR, _DONE_RUNNING, _DONE_CORR])),
        ("steer", False, False, False, ("steer", None, [_START_CORR, _DONE_CORR, _DONE_RUNNING])),
        ("steer", False, False, True, ("steer", None, [_START_CORR, _DONE_CORR, _DONE_RUNNING])),
    ],
    ids=["steer", "redirect", "interrupt", "leftover-steer", "untracked-turn", "untracked-turn-leftover"],
)
async def test_priority_path_input_has_one_lifecycle(monkeypatch, mode, redirects, tracked, leftover, expected):
    """The adapter can be idle while the runner still owns a turn for the session. The adapter then
    starts the message itself before the runner folds it into the running turn. The message completes
    once: with the turn that consumed it, or with its own turn when the model returns it as a leftover
    steer. When the hooks do not track the running turn, the message completes when its handler
    returns, and a leftover steer then runs as plain text."""
    from gateway.run_turn_followup_ack import _run_followup_processing_hook

    runner, adapter = _priority_runner(monkeypatch, mode)
    source, key, receiver, running = _running_slack_turn(runner, finished=False)
    receiver._supports_active_turn_redirect = redirects
    if tracked:
        await adapter._run_processing_hook("on_processing_start", running)

    await adapter.handle_message(MessageEvent(text="correction", source=source, message_id="corr-1"))
    await asyncio.gather(*adapter._background_tasks)
    after_handler = list(adapter.log)
    result = {"final_response": "reply"}
    if leftover:
        result["pending_steer"] = receiver.steer.call_args.args[0]
    pending_event, _pending = await runner._run_agent_drain_pending(
        result, adapter, source, key, processing_event=running)
    # The queued lane completes the running turn after its reply, then runs the follow-up.
    await adapter._run_processing_hook("on_processing_complete", running, ProcessingOutcome.SUCCESS)
    await _run_followup_processing_hook(adapter, pending_event, "on_processing_start")
    await _run_followup_processing_hook(adapter, pending_event, "on_processing_complete", ProcessingOutcome.SUCCESS)

    verbs = [name for name, *_ in receiver.mock_calls if name in _AGENT_VERBS]
    handler_log = [_START_RUNNING, _START_CORR] if tracked else [_START_CORR, _DONE_CORR]
    assert (verbs, getattr(pending_event, "message_id", None), after_handler, adapter.log) == (
        [expected[0]], expected[1], handler_log, expected[2])


@pytest.mark.asyncio
async def test_two_leftover_steers_complete_with_the_turn_that_reads_them(monkeypatch):
    """Both steers arrived after the agent's last tool batch, so the agent returns their texts as one
    ``pending_steer`` and a single follow-up turn reads them. Neither message completes with the
    running turn; both complete with the follow-up's outcome."""
    from gateway.run_turn_followup_ack import _run_followup_processing_hook

    runner, adapter = _priority_runner(monkeypatch, "steer")
    source, key, receiver, running = _running_slack_turn(runner, finished=False)
    await adapter._run_processing_hook("on_processing_start", running)
    for message_id, text in (("corr-1", "first correction"), ("corr-2", "second correction")):
        await adapter.handle_message(MessageEvent(text=text, source=source, message_id=message_id))
        await asyncio.gather(*adapter._background_tasks)
    pending_steer = "\n".join(invocation.args[0] for invocation in receiver.steer.call_args_list)

    pending_event, _pending = await runner._run_agent_drain_pending(
        {"final_response": "reply", "pending_steer": pending_steer}, adapter, source, key,
        processing_event=running)
    await adapter._run_processing_hook("on_processing_complete", running, ProcessingOutcome.SUCCESS)
    await _run_followup_processing_hook(adapter, pending_event, "on_processing_start")
    await _run_followup_processing_hook(adapter, pending_event, "on_processing_complete", ProcessingOutcome.FAILURE)

    assert (pending_event.message_id, adapter.log) == ("corr-2", [
        _START_RUNNING, _START_CORR, ("start", "corr-2"), _DONE_RUNNING,
        ("complete", "corr-1", ProcessingOutcome.FAILURE), ("complete", "corr-2", ProcessingOutcome.FAILURE),
    ])


class _CompleteOnlyAdapter(LifecycleLogAdapter):
    """Overrides only on_processing_complete, as A2A, Google Chat and the webhook adapter do."""

    on_processing_start = BasePlatformAdapter.on_processing_start


@pytest.mark.asyncio
@pytest.mark.parametrize("leftover", [False, True])
async def test_priority_path_input_completes_once_without_a_start_hook(monkeypatch, leftover):
    """The queued lane runs no hooks for an adapter without its own start hook. A steered message
    therefore completes when its handler returns, and a leftover steer for it runs as plain text."""
    from gateway.run_turn_followup_ack import _run_followup_processing_hook

    runner, _adapter = _priority_runner(monkeypatch, "steer")
    adapter = _CompleteOnlyAdapter()
    adapter.platform = Platform.SLACK
    runner.adapters[Platform.SLACK] = adapter
    adapter.set_message_handler(runner._handle_message)
    source, key, receiver, running = _running_slack_turn(runner, finished=False)
    await adapter._run_processing_hook("on_processing_start", running)

    await adapter.handle_message(MessageEvent(text="correction", source=source, message_id="corr-1"))
    await asyncio.gather(*adapter._background_tasks)
    result = {"final_response": "reply"}
    if leftover:
        result["pending_steer"] = receiver.steer.call_args.args[0]
    pending_event, _pending = await runner._run_agent_drain_pending(
        result, adapter, source, key, processing_event=running)
    await adapter._run_processing_hook("on_processing_complete", running, ProcessingOutcome.SUCCESS)
    await _run_followup_processing_hook(adapter, pending_event, "on_processing_start")
    await _run_followup_processing_hook(adapter, pending_event, "on_processing_complete", ProcessingOutcome.SUCCESS)

    assert (pending_event, adapter.log) == (None, [_DONE_CORR, _DONE_RUNNING])


class _BlockingSendAdapter(LifecycleLogAdapter):
    """Holds the send of ``blocked_content`` until the test releases it."""

    def __init__(self, blocked_content):
        super().__init__()
        self.blocked_content = blocked_content
        self.send_started, self.send_release = asyncio.Event(), asyncio.Event()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        if content == self.blocked_content:
            self.send_started.set()
            await self.send_release.wait()
        return await super().send(chat_id, content, reply_to=reply_to, metadata=metadata)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("final_delivery", "queued_outcome"),
    [("delivered", ProcessingOutcome.SUCCESS), ("refused", ProcessingOutcome.FAILURE),
     ("cancelled", ProcessingOutcome.CANCELLED)],
)
async def test_each_queued_turn_completes_after_its_own_reply(monkeypatch, tmp_path, final_delivery, queued_outcome):
    """A turn completes once its reply is delivered, before the queued follow-up starts, so cancelling
    the follow-up cannot change the earlier outcome. The terminal follow-up's reply goes out through
    the adapter's final delivery, which decides that follow-up's outcome."""
    _ScriptedAgent.calls, _ScriptedAgent.results = [], [_done("done-1"), _done("done-2")]
    _install_fake_agent(monkeypatch, tmp_path, _ScriptedAgent)
    adapter = _BlockingSendAdapter("done-2")
    if final_delivery == "refused":
        adapter.refused.add("done-2")
    runner = _make_runner(adapter)
    adapter._pending_messages[SESSION_KEY] = MessageEvent(text="follow-up", source=_source(), message_id="queued-1")

    async def respond(event):
        result = await runner._run_agent(
            message=event.text, context_prompt="", history=[], source=event.source,
            session_id="sess-lifecycle", session_key=SESSION_KEY, processing_event=event)
        return result["final_response"]

    adapter.set_message_handler(respond)
    await adapter.handle_message(MessageEvent(text="first", source=_source(), message_id="first-1"))
    await asyncio.wait_for(adapter.send_started.wait(), 5)
    if final_delivery == "cancelled":
        await adapter.cancel_session_processing(SESSION_KEY)
    else:
        adapter.send_release.set()
        await asyncio.gather(*adapter._background_tasks)

    sends = [entry[1] for entry in adapter.log if entry[0] == "send"]
    assert (
        [entry for entry in adapter.log if entry[0] != "send"],
        adapter.log[:3],
        sends[1:2],
        adapter.log[-2][0],
    ) == (
        [("start", "first-1"), ("complete", "first-1", ProcessingOutcome.SUCCESS),
         ("start", "queued-1"), ("complete", "queued-1", queued_outcome)],
        [("start", "first-1"), ("send", "done-1"), ("complete", "first-1", ProcessingOutcome.SUCCESS)],
        [] if final_delivery == "cancelled" else ["done-2"],
        "start" if final_delivery == "cancelled" else "send",
    )



def _running_slack_turn(runner, *, finished):
    source = SessionSource(platform=Platform.SLACK, chat_id="C1", chat_type="dm", user_id="U1")
    receiver = MagicMock(_supports_active_turn_redirect=True, _active_children=[])
    receiver.steer.return_value = receiver.redirect.return_value = True
    receiver.get_activity_summary.return_value = {"seconds_since_activity": 0}
    running = MessageEvent(text="running", source=source, message_id="running-1")
    key = runner._session_key_for_source(source)
    turn = runner._session_state(key).turn
    turn.agent, turn.event, turn.processing_event, turn.started_ts = receiver, running, running, time.time()
    turn.ctx = TurnContext(
        session_key=key, event_message_id="running-1", inbound_message_id="running-1",
        result_holder=[_done("reply") if finished else None])
    return source, key, receiver, running


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "text", "entry"),
    [("steer", "late", "busy"), ("interrupt", "late", "busy"), ("steer", "late", "priority"),
     ("interrupt", "late", "priority"), ("steer", "/steer late", "priority")],
)
async def test_input_after_the_model_result_waits_for_its_own_turn(monkeypatch, mode, text, entry):
    """Once the running turn has its model result, its agent cannot consume more input. Steering or
    redirecting would attach the message to a reply that never saw it, so the message waits in the
    FIFO and runs as its own turn."""
    runner, adapter = _priority_runner(monkeypatch, mode)
    source, key, receiver, _running = _running_slack_turn(runner, finished=True)
    late = MessageEvent(text=text, source=source, message_id="late-1")

    if entry == "busy":
        assert await runner._handle_active_session_busy_message(late, key) is True
    else:
        await runner._handle_message(late)
    queued = adapter._pending_messages.get(key)

    assert (
        [name for name, *_ in receiver.mock_calls if name in _AGENT_VERBS],
        (queued.message_id, queued.text),
    ) == ([], ("late-1", "late"))


@pytest.mark.asyncio
@pytest.mark.parametrize("queued_first", [False, True])
async def test_a_leftover_steer_runs_as_its_own_message(monkeypatch, queued_first):
    """A steer that arrives after the agent's last tool batch comes back as ``pending_steer``. It runs
    as its own turn for the message that sent it, so that message's lifecycle hooks and reply anchor
    apply. A follow-up queued earlier runs first, and the steer waits behind it instead of being
    dropped."""
    runner, adapter = _priority_runner(monkeypatch, "steer")
    source, key, receiver, running = _running_slack_turn(runner, finished=False)
    queued = MessageEvent(text="queued", source=source, message_id="queued-1")
    if queued_first:
        runner._enqueue_fifo(key, queued, adapter)
    late = MessageEvent(text="late", source=source, message_id="late-1")
    assert await runner._handle_active_session_busy_message(late, key) is True
    (admitted,) = receiver.steer.call_args.args

    drained = [await runner._run_agent_drain_pending(
        {"final_response": "reply", "pending_steer": admitted}, adapter, source, key, processing_event=running)]
    if queued_first:
        drained.append(await runner._run_agent_drain_pending(
            {"final_response": "queued reply"}, adapter, source, key, processing_event=queued))

    assert drained == [(queued, "queued")] * queued_first + [(replace(late, text=admitted), admitted)]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["failed-result", "handler-error"])
async def test_a_failed_turn_completes_with_its_delivery_outcome(monkeypatch, tmp_path, failure):
    """A failed turn still delivers a failure notice to the user. The completion outcome reports
    that delivery, as it does for every other reply."""
    runner, adapter = _priority_runner(monkeypatch, "queue")
    monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
    monkeypatch.setattr("gateway.run._resolve_runtime_agent_kwargs", lambda: {"api_key": "test-key"})
    runner._run_agent = (
        AsyncMock(side_effect=RuntimeError("Provider failed")) if failure == "handler-error"
        else AsyncMock(return_value={"final_response": "Provider unavailable", "messages": [],
                                     "failed": True, "completed": False})
    )
    source = SessionSource(platform=Platform.SLACK, chat_id="C1", chat_type="dm", user_id="U1")

    await adapter.handle_message(MessageEvent(text="first", source=source, message_id="first-1"))
    await asyncio.gather(*adapter._background_tasks)

    assert (
        [entry for entry in adapter.log if entry[0] != "send"],
        len([entry for entry in adapter.log if entry[0] == "send"]) >= 1,
    ) == ([("start", "first-1"), ("complete", "first-1", ProcessingOutcome.SUCCESS)], True)
