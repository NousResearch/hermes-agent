"""Cancelled turns must not spend the normal text-flush budget or lose cancellation."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.run_turn import GatewayTurnMixin
from gateway.turn_context import TurnContext


class _TurnFixture(GatewayTurnMixin):
    """Use the real turn/flush/cleanup path with controlled worker and transport I/O."""

    def __init__(self, *, consumer=True, hold_progress=False):
        self._draining = False
        self.started = asyncio.Event()
        self.consumer_started = asyncio.Event()
        self.flushing = asyncio.Event()
        self.stream_settled = asyncio.Event()
        self.finish = asyncio.Event()
        self.progress_settling = asyncio.Event()
        self.release_progress = asyncio.Event()
        self.hold_progress = hold_progress
        self.tasks = set()
        self.released = []
        self.delivered = []
        self.tts_aborts = []
        self.worker_future = asyncio.get_running_loop().create_future()
        self.ctx = TurnContext(session_key="cancel-fixture", run_generation=7)
        self.ctx.streaming_tts_consumer_holder[0] = SimpleNamespace(
            done=False, suppress_whole_file=False, finish=self._finish_tts, abort=self.tts_aborts.append,
            wait_complete=self._wait_tts,
        )
        if consumer:
            self.ctx.stream_consumer_holder[0] = SimpleNamespace(run=self._consume)
        self._run_agent_mark_streamed_delivery = AsyncMock()

    async def _consume(self):
        self.consumer_started.set()
        await self.finish.wait()
        self.delivered.append("complete answer")

    async def _park(self, *args):
        self.tasks.add(asyncio.current_task())
        await asyncio.Event().wait()

    _run_agent_write_tool_log = _park
    _run_agent_track_agent = _park
    _run_agent_monitor_for_interrupt = _park
    _run_agent_notify_long_running = _park

    async def _progress(self):
        try:
            await self._park()
        finally:
            if self.hold_progress:
                self.progress_settling.set()
                await self.release_progress.wait()

    async def _run_agent_stream_consumer_task(self, holder):
        self.stream_task = asyncio.current_task()
        self.tasks.add(self.stream_task)
        await super()._run_agent_stream_consumer_task(holder)

    async def _await_stream_task(self, task):
        self.flushing.set()
        await super()._await_stream_task(task)
        self.stream_settled.set()

    def _get_proxy_url(self):
        return None

    def _run_agent_display_settings(self, source):
        return SimpleNamespace(
            needs_progress_queue=True, log_mode_enabled=True, log_queue=None,
            _native_slack_task_cards=False,
        )

    def _run_agent_build_turn_context(self, *args, **kwargs):
        runner = SimpleNamespace(run_sync=lambda: None, send_progress_messages=self._progress)
        return self.ctx, runner, None

    def _run_agent_bind_turn_wiring(self, *args):
        return None

    def _run_agent_start_streaming_tts(self, *args):
        pass

    def _run_agent_start_turn_worker(self, *args):
        self.started.set()
        return SimpleNamespace(executor_task=self.worker_future, agent_timeout=None)

    def _adapter_for_source(self, source):
        return None

    def _release_running_agent_state(self, key, *, run_generation):
        self.released.append((key, run_generation))

    def _run_agent_schedule_bubble_cleanup(self, *args):
        pass

    def _finish_tts(self):
        self.ctx.streaming_tts_consumer_holder[0].done = True

    async def _wait_tts(self, **kwargs):
        self._finish_tts()

    def complete_worker(self):
        result = {"completed": True, "final_response": "complete answer", "messages": []}
        self.ctx.result_holder[0] = result
        self.worker_future.set_result(result)

    def start(self):
        return asyncio.create_task(self._run_agent_inner(
            "fixture", "", [], SimpleNamespace(chat_id="fixture"), "fixture-session",
            session_key=self.ctx.session_key, run_generation=self.ctx.run_generation,
        ))

    async def dispose(self, task):
        self.finish.set()
        self.release_progress.set()
        self.worker_future.cancel()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        for background in self.tasks:
            background.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)

    def assert_cleanup(self):
        assert self.released == [(self.ctx.session_key, self.ctx.run_generation)]
        assert self.tasks and all(task.done() for task in self.tasks)
        assert self.ctx.streaming_tts_consumer_holder[0].done


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["worker", "no_consumer", "flush", "settlement"])
async def test_cancelled_turn_settles_stream_and_preserves_cancellation(phase):
    fixture = _TurnFixture(consumer=phase != "no_consumer", hold_progress=phase == "settlement")
    task = fixture.start()
    try:
        await asyncio.wait_for(fixture.started.wait(), 2)
        if phase != "no_consumer":
            await asyncio.wait_for(fixture.consumer_started.wait(), 2)
        if phase in {"flush", "settlement"}:
            fixture.complete_worker()
            await asyncio.wait_for(fixture.flushing.wait(), 2)
        if phase == "settlement":
            fixture.finish.set()
            await asyncio.wait_for(fixture.progress_settling.wait(), 2)
            await asyncio.wait_for(fixture.stream_settled.wait(), 2)
        task.cancel()
        # Event synchronization above identifies the phase. This generous bound only
        # rejects the unrelated five-second normal-flush budget on an aborted turn.
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(asyncio.shield(task), 2)
        fixture.assert_cleanup()
        fixture._run_agent_mark_streamed_delivery.assert_not_awaited()
        assert fixture.delivered == (["complete answer"] if phase == "settlement" else [])
        if phase in {"worker", "no_consumer"}:
            assert fixture.tts_aborts == ["cleanup"]
    finally:
        await fixture.dispose(task)


@pytest.mark.asyncio
@pytest.mark.parametrize("consumer_cancelled", [False, True])
async def test_normal_turn_waits_for_flush_and_tolerates_child_cancellation(consumer_cancelled):
    fixture = _TurnFixture()
    task = fixture.start()
    try:
        await asyncio.wait_for(fixture.consumer_started.wait(), 2)
        fixture.complete_worker()
        await asyncio.wait_for(fixture.flushing.wait(), 2)
        assert not task.done()
        assert not fixture.delivered
        if consumer_cancelled:
            fixture.stream_task.cancel()
        else:
            fixture.finish.set()
        result = await asyncio.wait_for(task, 2)
        assert result["final_response"] == "complete answer"
        assert fixture.delivered == ([] if consumer_cancelled else ["complete answer"])
        fixture._run_agent_mark_streamed_delivery.assert_awaited_once_with(result, fixture.ctx)
        fixture.assert_cleanup()
        assert not fixture.tts_aborts
    finally:
        await fixture.dispose(task)
