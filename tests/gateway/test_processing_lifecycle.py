"""Processing-lifecycle contracts that every bracketing adapter observes.

Adapters render ``on_processing_start`` and ``on_processing_complete`` as
in-progress and outcome markers (reactions, read receipts, typing cards), so
every started message must complete exactly once, with an outcome that
describes what happened to that message.
"""

import asyncio

import pytest

from gateway.platforms.base import MessageEvent, ProcessingOutcome, SendResult
from gateway.run import _INTERRUPT_REASON_STOP
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
