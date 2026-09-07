"""Tests for the save-before-external-streaming gate."""

import asyncio
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

import pytest

from gateway.run import GatewayRunner
from gateway.turn_context import TurnContext


class _FakeConsumer:
    def __init__(self):
        self.started = asyncio.Event()
        self.finished = asyncio.Event()

    async def run(self):
        self.started.set()
        await self.finished.wait()


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
async def test_streaming_tts_aborts_without_start_when_persistence_fails():
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
        result={"failed": True},
    )

    consumer.abort.assert_called_once_with("persistence failed before streaming TTS start")
    consumer.start.assert_not_called()
    consumer.finish.assert_not_called()
