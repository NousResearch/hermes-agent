"""Tests for the save-before-external-streaming gate."""

import asyncio
from types import SimpleNamespace

import pytest

from gateway.run import GatewayRunner
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


class _E2EFakeConsumer(_FakeConsumer):
    def finish(self, final_text=None):
        super().finish(final_text)
        self.finished.set()


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
    from gateway.run_turn_runner import TurnRunner

    loop = asyncio.get_running_loop()
    release = asyncio.Event()
    consumer = _FakeConsumer()
    runner = TurnRunner.__new__(TurnRunner)
    runner._ctx = TurnContext(
        result_holder=[None],
        stream_consumer_holder=[consumer],
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
    await asyncio.sleep(0)
    assert not release.is_set()
    assert not consumer.started.is_set()

    gateway_runner._run_agent_release_stream_consumer(runner._ctx, result)
    await asyncio.wait_for(release.wait(), timeout=1.0)
    await asyncio.wait_for(consumer.started.wait(), timeout=1.0)
    assert consumer.finish_calls == ["answer"]

    consumer.finished.set()
    await stream_task


@pytest.mark.asyncio
async def test_unknown_persistence_result_does_not_release_stream():
    from gateway.run_turn_runner import TurnRunner

    loop = asyncio.get_running_loop()
    release = asyncio.Event()
    consumer = _FakeConsumer()
    runner = TurnRunner.__new__(TurnRunner)
    runner._ctx = TurnContext(
        result_holder=[None],
        stream_consumer_holder=[consumer],
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
async def test_pre_delivery_gate_runs_before_stream_release_and_preserves_shadow_response():
    runner = GatewayRunner.__new__(GatewayRunner)
    events = []
    consumer = _FakeConsumer()
    release = asyncio.Event()
    ctx = TurnContext(
        stream_consumer_holder=[consumer],
        stream_release_event=release,
        pre_delivery_gate=lambda result, _ctx: (
            events.append(("gate", result["final_response"])),
            {"decision": "blocked", "side_effect_status": "not_attempted"},
        )[1],
    )
    result = {
        "final_response": "original answer",
        "completed": True,
        "failed": False,
        "interrupted": False,
        "persistence_confirmed": True,
    }

    stream_task = asyncio.create_task(
        runner._run_agent_stream_consumer_task(ctx.stream_consumer_holder, release),
    )
    result = await runner._run_agent_apply_pre_delivery_gate(ctx, result)
    events.append(("before_release", consumer.started.is_set()))
    runner._run_agent_release_stream_consumer(ctx, result)
    await asyncio.wait_for(consumer.started.wait(), timeout=1.0)

    assert events == [("gate", "original answer"), ("before_release", False)]
    assert result["final_response"] == "original answer"
    assert result["pre_delivery_gate_result"]["decision"] == "blocked"
    assert consumer.finish_calls == ["original answer"]
    consumer.finished.set()
    await stream_task


@pytest.mark.asyncio
async def test_pre_delivery_gate_exception_is_inconclusive_but_shadow_continues():
    runner = GatewayRunner.__new__(GatewayRunner)
    ctx = TurnContext(pre_delivery_gate=lambda _result, _ctx: (_ for _ in ()).throw(RuntimeError("boom")))
    result = {
        "final_response": "original answer",
        "completed": True,
        "failed": False,
        "interrupted": False,
        "persistence_confirmed": True,
    }

    result = await runner._run_agent_apply_pre_delivery_gate(ctx, result)

    assert result["final_response"] == "original answer"
    assert result["pre_delivery_gate_result"] == {
        "decision": "inconclusive",
        "error_code": "pre_delivery_gate_error",
    }


@pytest.mark.asyncio
async def test_real_run_agent_inner_orders_gate_before_fake_platform_stream(monkeypatch):
    """Exercise the real GatewayRunner orchestration with only leaf effects faked."""
    runner = GatewayRunner.__new__(GatewayRunner)
    consumer = _E2EFakeConsumer()
    events = []
    result = {
        "final_response": "original answer",
        "completed": True,
        "failed": False,
        "interrupted": False,
        "persistence_confirmed": True,
    }
    source = SimpleNamespace(platform="fake", chat_id="chat", thread_id=None)
    ctx = TurnContext(
        source=source,
        session_key="session",
        stream_consumer_holder=[consumer],
        stream_release_event=asyncio.Event(),
        result_holder=[result],
        pre_delivery_gate=lambda value, _ctx: (
            events.append(("gate", value["final_response"])),
            {"decision": "blocked", "side_effect_status": "not_attempted"},
        )[1],
    )
    turn_runner = SimpleNamespace(
        send_progress_messages=lambda: asyncio.sleep(0),
        run_sync=lambda: None,
    )
    display = SimpleNamespace(
        needs_progress_queue=False,
        log_mode_enabled=False,
        log_queue=None,
        _native_slack_task_cards=False,
    )

    runner._run_agent_display_settings = lambda _source: display
    runner._run_agent_build_turn_context = lambda *_args, **_kwargs: (ctx, turn_runner, None)
    runner._run_agent_bind_turn_wiring = lambda *_args: None
    runner._run_agent_start_streaming_tts = lambda *_args: None
    runner._run_agent_track_agent = lambda *_args: asyncio.sleep(0)
    runner._run_agent_monitor_for_interrupt = lambda *_args: asyncio.sleep(0)
    runner._run_agent_notify_long_running = lambda *_args: asyncio.sleep(0)
    runner._run_agent_start_turn_worker = lambda *_args: SimpleNamespace(executor_task=None)
    runner._run_agent_await_turn_worker = lambda *_args: _async_return(result)
    runner._run_agent_evict_on_fallback = lambda *_args: None
    runner._run_agent_drain_pending = lambda *_args: _async_return((None, None))
    runner._run_agent_mark_streamed_delivery = lambda *_args: _async_return(None)
    runner._run_agent_schedule_bubble_cleanup = lambda *_args: None
    runner._run_agent_cleanup_turn_tasks = lambda *_args, **_kwargs: _async_return(None)
    runner._adapter_for_source = lambda _source: None

    returned = await runner._run_agent_inner(
        message="hello", context_prompt="", history=[], source=source,
        session_id="session", session_key="session",
    )
    await asyncio.wait_for(consumer.started.wait(), timeout=1.0)

    assert returned is result
    assert events == [("gate", "original answer")]
    assert result["final_response"] == "original answer"
    assert consumer.finish_calls == ["original answer"]


async def _async_return(value):
    return value
