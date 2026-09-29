"""Processing-lifecycle contracts that every bracketing adapter observes.

Adapters render ``on_processing_start`` and ``on_processing_complete`` as
in-progress and outcome markers (reactions, read receipts, typing cards), so
every started message must complete exactly once, with an outcome that
describes what happened to that message.
"""

import asyncio
import time
from unittest.mock import MagicMock

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import MessageEvent, ProcessingOutcome, SendResult
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
        self.send_results: list = []

    async def on_processing_start(self, event):
        self.log.append(("start", event.message_id))

    async def on_processing_complete(self, event, outcome):
        self.log.append(("complete", event.message_id, outcome))

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.log.append(("send", content))
        return self.send_results.pop(0) if self.send_results else SendResult(success=True, message_id="sent")


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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "redirects", "verb"),
    [("steer", False, "steer"), ("interrupt", True, "redirect"), ("interrupt", False, "interrupt")],
)
async def test_priority_path_completes_input_that_the_adapter_started(monkeypatch, mode, redirects, verb):
    """The adapter can be idle while the runner still owns a turn for the session. The adapter then
    starts the message itself, and the runner folds it into the running turn without running it
    again, so the message must complete when its handler returns."""
    runner, adapter = _priority_runner(monkeypatch, mode)
    source = SessionSource(platform=Platform.SLACK, chat_id="C1", chat_type="dm", user_id="U1")
    receiver = MagicMock(_supports_active_turn_redirect=redirects, _active_children=[])
    receiver.steer.return_value = receiver.redirect.return_value = True
    receiver.get_activity_summary.return_value = {"seconds_since_activity": 0}
    running = MessageEvent(text="running", source=source, message_id="running-1")
    key = runner._session_key_for_source(source)
    turn = runner._session_state(key).turn
    turn.agent, turn.event, turn.processing_event, turn.started_ts = receiver, running, running, time.time()
    turn.ctx = TurnContext(session_key=key, event_message_id="running-1", inbound_message_id="running-1")

    await adapter.handle_message(MessageEvent(text="correction", source=source, message_id="corr-1"))
    await asyncio.gather(*adapter._background_tasks)

    assert ([name for name, *_ in receiver.mock_calls if name in {"steer", "redirect", "interrupt"}], adapter.log) == (
        [verb], [("start", "corr-1"), ("complete", "corr-1", ProcessingOutcome.SUCCESS)],
    )
