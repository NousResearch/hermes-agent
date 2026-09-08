"""Tests for the save-before-external-streaming gate."""

import asyncio
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

import pytest

from gateway.run import GatewayRunner
from gateway.run_turn_runner import TurnRunner
from gateway.turn_context import TurnContext


class _FakeConsumer:
    def __init__(self):
        self.started = asyncio.Event()
        self.finished = asyncio.Event()
        self.finish_calls = []

    def finish(self, final_text=None):
        self.finish_calls.append(final_text)

    async def run(self):
        self.started.set()
        await self.finished.wait()


class _FakeStreamingTTS:
    def __init__(self):
        self._task = None
        self.done = False
        self.suppress_whole_file = False
        self.started = False
        self.finished = False
        self.abort_reason = None

    def start(self):
        self.started = True

    def finish(self):
        self.finished = True

    def abort(self, reason):
        self.abort_reason = reason
        self.done = True

    async def wait_complete(self, timeout=10.0):
        return self.done


@pytest.mark.asyncio
async def test_stream_consumer_does_not_start_before_persistence_release():
    runner = GatewayRunner.__new__(GatewayRunner)
    consumer = _FakeConsumer()
    holder = [consumer]
    release = asyncio.Event()

    task = asyncio.create_task(runner._run_agent_stream_consumer_task(holder, release))
    await asyncio.sleep(0)
    assert not consumer.started.is_set()

    release.set()
    await asyncio.sleep(0)
    assert consumer.started.is_set()

    consumer.finished.set()
    await task


@pytest.mark.asyncio
async def test_stream_consumer_waits_when_persistence_fails_closed():
    runner = GatewayRunner.__new__(GatewayRunner)
    consumer = _FakeConsumer()
    release = asyncio.Event()

    task = asyncio.create_task(runner._run_agent_stream_consumer_task([consumer], release))
    await asyncio.sleep(0.02)
    assert not consumer.started.is_set()
    assert not task.done()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_executor_release_is_scheduled_on_the_owning_event_loop():
    """The real executor→loop crossing must not call Event.set() on the worker."""
    loop = asyncio.get_running_loop()
    release = asyncio.Event()
    consumer = _FakeConsumer()
    runner = TurnRunner.__new__(TurnRunner)
    runner._ctx = TurnContext(
        result_holder=[None],
        stream_release_event=release,
        _loop_for_step=loop,
    )
    result = {
        "final_response": "answer",
        "completed": True,
        "failed": False,
        "interrupted": False,
        "persistence_confirmed": True,
    }

    gateway_runner = GatewayRunner.__new__(GatewayRunner)
    stream_task = asyncio.create_task(
        gateway_runner._run_agent_stream_consumer_task([consumer], release),
    )
    await asyncio.sleep(0)
    assert not consumer.started.is_set()

    await loop.run_in_executor(
        None, runner._finish_stream_consumer, result, [], consumer,
    )
    await asyncio.wait_for(release.wait(), timeout=1.0)
    await asyncio.wait_for(consumer.started.wait(), timeout=1.0)
    assert consumer.finish_calls == ["answer"]

    consumer.finished.set()
    await stream_task


@pytest.mark.asyncio
async def test_unknown_persistence_result_does_not_release_stream():
    loop = asyncio.get_running_loop()
    release = asyncio.Event()
    consumer = _FakeConsumer()
    runner = TurnRunner.__new__(TurnRunner)
    runner._ctx = TurnContext(
        result_holder=[None],
        stream_release_event=release,
        _loop_for_step=loop,
    )

    await loop.run_in_executor(
        None, runner._finish_stream_consumer,
        {"final_response": "answer", "completed": True, "failed": False}, [], consumer,
    )
    await asyncio.sleep(0)
    assert not release.is_set()


@pytest.mark.asyncio
async def test_unknown_persistence_result_does_not_start_streaming_tts():
    tts = _FakeStreamingTTS()
    runner = GatewayRunner.__new__(GatewayRunner)
    turn_ctx = TurnContext(streaming_tts_consumer_holder=[tts])

    await runner._run_agent_finalize_streaming_tts(
        turn_ctx, adapter=None,
        result={"final_response": "answer", "completed": True, "failed": False},
    )

    assert tts.started is False
    assert tts.finished is False
    assert tts.abort_reason == "canonical persistence not confirmed before streaming TTS start"


@pytest.mark.asyncio
@pytest.mark.parametrize("result", [
    {"failed": True},
    {"interrupted": True, "completed": True, "final_response": "partial"},
    {"completed": False, "final_response": "partial"},
    {"completed": True, "final_response": ""},
    {"completed": True, "final_response": "(empty)"},
    {"completed": True, "final_response": "answer", "persistence_confirmed": False},
])
async def test_streaming_tts_aborts_without_start_when_result_is_not_persisted(result):
    runner = GatewayRunner.__new__(GatewayRunner)
    consumer = MagicMock()
    consumer._task = None
    turn_ctx = cast(
        TurnContext,
        SimpleNamespace(
            streaming_tts_consumer_holder=[consumer],
            session_key="session-1",
            run_generation=1,
        ),
    )

    await runner._run_agent_finalize_streaming_tts(
        turn_ctx,
        adapter=MagicMock(),
        result=result,
    )

    consumer.abort.assert_called_once_with("canonical persistence not confirmed before streaming TTS start")
    consumer.start.assert_not_called()
    consumer.finish.assert_not_called()
