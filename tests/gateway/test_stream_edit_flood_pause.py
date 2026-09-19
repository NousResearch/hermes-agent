"""Streaming edits must respect a bounded server wait without losing buffered text."""

import asyncio
from collections import deque
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gateway.platforms.base import SendResult
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig


class Clock:
    def __init__(self):
        self.now = 100.0
        self.waits = []
        self.ticks = deque()

    def monotonic(self):
        return self.now

    async def sleep(self, seconds):
        if seconds == 0.05:
            if not self.ticks:
                pytest.fail("Consumer ran past the scripted ticks")
            self.ticks.popleft()()
        else:
            self.waits.append(seconds)
            self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    clock = Clock()
    for module in ("gateway.stream_consumer", "gateway.stream_consumer_transport"):
        monkeypatch.setattr(f"{module}.time", clock)
    monkeypatch.setattr("gateway.stream_consumer.asyncio.sleep", clock.sleep)
    return clock


def flood(wait=9.0):
    return SendResult(success=False, error=f"flood_control:{wait}", retry_after=wait)


def make_consumer(clock, **config):
    results = deque()
    edit_times = []

    async def edit(**kwargs):
        edit_times.append(clock.now)
        return results.popleft() if results else SendResult(
            success=True, message_id=kwargs["message_id"])

    adapter = SimpleNamespace(
        MAX_MESSAGE_LENGTH=4096,
        send=AsyncMock(return_value=SendResult(success=True, message_id="preview")),
        edit_message=AsyncMock(side_effect=edit),
        delete_message=AsyncMock(return_value=True),
        edit_results=results,
        edit_times=edit_times,
    )
    cfg = StreamConsumerConfig(edit_interval=0.25, buffer_threshold=1, cursor=" ▉")
    for key, value in config.items():
        setattr(cfg, key, value)
    return GatewayStreamConsumer(adapter, "chat", cfg), adapter


@pytest.mark.asyncio
@pytest.mark.parametrize("buffer_threshold", [1, 10_000])
async def test_pause_gates_ticks_and_resumes_with_all_buffered_text(clock, buffer_threshold):
    consumer, adapter = make_consumer(clock, buffer_threshold=buffer_threshold)
    await consumer._send_or_edit("First ▉")
    adapter.edit_results.append(flood())
    consumer.on_delta("First plus")
    snapshots = []

    def during_pause():
        snapshots.append((consumer._flood_strikes, consumer._edit_supported,
                          consumer._fallback_final_send))
        consumer.on_delta(" buffered text")
        clock.now = 108.99

    def just_before_deadline():
        snapshots.append(adapter.edit_message.await_count)
        clock.now = 109.0

    def finish():
        snapshots.append(adapter.edit_message.call_args.kwargs["content"])
        consumer.finish()

    clock.ticks.extend([during_pause, just_before_deadline, finish])
    await consumer.run()

    assert snapshots == [(0, True, False), 1, "First plus buffered text ▉"]
    assert adapter.edit_times == [100.0, 109.0, 109.0]
    assert adapter.edit_message.call_args.kwargs["content"] == "First plus buffered text"
    assert adapter.edit_message.call_args.kwargs["finalize"] is True
    assert consumer.final_content_delivered is True
    assert consumer.delivered_final_matches("First plus buffered text")
    assert adapter.send.await_count == 1
    assert clock.waits == []


@pytest.mark.asyncio
@pytest.mark.parametrize("retry_after", [None, 0.0, -1.0, 271.0])
async def test_unhonoured_wait_keeps_legacy_strikes_and_doubling(clock, retry_after):
    consumer, adapter = make_consumer(clock)
    await consumer._send_or_edit("First ▉")
    adapter.edit_results.extend([SendResult(
        success=False, error="flood_control:9.0", retry_after=retry_after)] * 4)

    for attempt, interval in enumerate([0.5, 1.0, 2.0], start=1):
        assert not await consumer._send_or_edit(f"First plus {attempt} ▉")
        assert consumer._flood_strikes == attempt
        assert consumer._current_edit_interval == interval
        assert consumer._fallback_final_send is (attempt == 3)
        assert consumer._edit_supported is (attempt < 3)
        assert getattr(consumer, "_flood_pause_until", 0.0) == 0.0

    # The third strike still makes the existing best-effort cursor-strip attempt.
    assert adapter.edit_message.await_count == 4
    assert clock.waits == []


@pytest.mark.asyncio
@pytest.mark.parametrize("cap, expected_strikes", [(8.0, 1), (9.0, 0)])
async def test_configured_cap_controls_whether_wait_is_honoured(clock, cap, expected_strikes):
    consumer, adapter = make_consumer(clock, flood_pause_cap_seconds=cap)
    await consumer._send_or_edit("First ▉")
    adapter.edit_results.append(flood())
    await consumer._send_or_edit("First plus ▉")

    assert consumer._flood_strikes == expected_strikes
    assert consumer._current_edit_interval == (0.5 if expected_strikes else 0.25)
    assert getattr(consumer, "_flood_pause_until", 0.0) == (0.0 if expected_strikes else 109.0)


@pytest.mark.asyncio
async def test_three_refusals_after_honoured_waits_enter_fallback(clock):
    consumer, adapter = make_consumer(clock)
    await consumer._send_or_edit("First ▉")
    adapter.edit_results.extend([flood()] * 5)

    for strike in range(4):
        clock.now = 100.0 + strike * 9.0
        await consumer._send_or_edit(f"First plus {strike} ▉")
        assert consumer._flood_strikes == strike
        assert consumer._fallback_final_send is (strike == 3)
        assert consumer._edit_supported is (strike < 3)


@pytest.mark.asyncio
@pytest.mark.parametrize("retry_succeeds", [True, False])
async def test_turn_final_waits_and_retries_only_once(clock, retry_succeeds):
    consumer, adapter = make_consumer(clock)
    await consumer._send_or_edit("First ▉")
    adapter.edit_results.append(flood())
    if not retry_succeeds:
        adapter.edit_results.append(flood())
    consumer.finish("First plus the final answer")
    await consumer.run()

    assert clock.waits == [9.0]
    assert adapter.edit_times == [100.0, 109.0]
    assert all(call.kwargs["finalize"] for call in adapter.edit_message.call_args_list)
    assert consumer.final_content_delivered is True
    assert adapter.send.await_count == (1 if retry_succeeds else 2)
    if not retry_succeeds:
        assert adapter.send.call_args.kwargs["content"] == "plus the final answer"
        assert consumer._edit_supported is False


@pytest.mark.asyncio
async def test_turn_final_gets_one_retry_when_refusal_reaches_strike_limit(clock):
    consumer, adapter = make_consumer(clock)
    await consumer._send_or_edit("First ▉")
    adapter.edit_results.extend([flood()] * 4)
    for attempt in range(3):
        clock.now = 100.0 + attempt * 9.0
        await consumer._send_or_edit(f"First plus {attempt} ▉")
    assert consumer._flood_strikes == 2
    clock.now = 127.0
    consumer.finish("First plus the final answer")
    await consumer.run()

    assert clock.waits == [9.0]
    assert adapter.edit_times == [100.0, 109.0, 118.0, 127.0, 136.0]
    assert adapter.edit_message.call_args.kwargs["finalize"] is True
    assert adapter.send.await_count == 1
    assert consumer.final_content_delivered is True


@pytest.mark.asyncio
@pytest.mark.parametrize("segment_break", [False, True])
async def test_boundary_waits_for_active_pause_before_finalizing(clock, segment_break):
    consumer, adapter = make_consumer(clock)
    await consumer._send_or_edit("First ▉")
    adapter.edit_results.append(flood())
    await consumer._send_or_edit("First plus ▉")
    clock.now = 102.0
    consumer.on_delta("First plus buffered tail")
    if segment_break:
        consumer.on_segment_break()

        def next_segment():
            consumer.on_delta("Next segment answer")
            consumer.finish()

        clock.ticks.append(next_segment)
    else:
        consumer.finish()
    await consumer.run()

    assert clock.waits == [7.0]
    assert adapter.edit_times == [100.0, 109.0]
    assert adapter.edit_message.call_args.kwargs["content"] == "First plus buffered tail"
    assert adapter.edit_message.call_args.kwargs["finalize"] is True
    assert adapter.send.await_count == (2 if segment_break else 1)
    assert consumer.final_content_delivered is True


@pytest.mark.asyncio
async def test_segment_refusal_flushes_unseen_tail_after_wait(clock):
    consumer, adapter = make_consumer(clock)
    await consumer._send_or_edit("First ▉")
    adapter.edit_results.append(flood())
    consumer.on_delta("First plus segment tail")
    consumer.on_segment_break()

    def next_segment():
        consumer.on_delta("Next segment answer")
        consumer.finish()

    clock.ticks.append(next_segment)
    await consumer.run()

    assert clock.waits == [9.0]
    assert adapter.edit_times == [100.0, 109.0]
    tail = adapter.send.call_args_list[1].kwargs
    assert tail["content"] == "plus segment tail"
    assert tail["metadata"]["_interim_send"] is True
    assert consumer.final_content_delivered is True


@pytest.mark.asyncio
async def test_success_resets_pause_and_strikes(clock):
    consumer, adapter = make_consumer(clock)
    await consumer._send_or_edit("First ▉")
    adapter.edit_results.extend([flood(), flood()])
    await consumer._send_or_edit("First plus ▉")
    clock.now = 109.0
    await consumer._send_or_edit("First plus more ▉")
    assert consumer._flood_strikes == 1
    clock.now = 118.0
    assert await consumer._send_or_edit("First plus more text ▉")
    assert consumer._flood_strikes == 0
    assert consumer._flood_pause_until == 0.0

    adapter.edit_results.append(flood())
    await consumer._send_or_edit("First plus more text again ▉")
    assert consumer._flood_strikes == 0
    assert consumer._edit_supported is True


@pytest.mark.asyncio
async def test_final_flood_opt_in_still_falls_back_immediately(clock):
    consumer, adapter = make_consumer(clock)
    adapter.FALLBACK_ON_FINAL_EDIT_FLOOD = True
    await consumer._send_or_edit("First ▉")
    adapter.edit_results.append(flood())
    consumer.finish("First plus the final answer")
    await consumer.run()

    assert clock.waits == []
    assert adapter.edit_times == [100.0]
    assert adapter.send.await_count == 2
    assert consumer.final_content_delivered is True


@pytest.mark.asyncio
async def test_cursor_only_failure_keeps_final_content_delivered(clock):
    consumer, adapter = make_consumer(clock)
    await consumer._send_or_edit("Complete answer ▉")
    adapter.edit_results.append(flood())
    seen_during_wait = []
    sleep = clock.sleep

    async def observe_wait(seconds):
        seen_during_wait.append(consumer.final_content_delivered)
        await sleep(seconds)

    with patch("gateway.stream_consumer.asyncio.sleep", observe_wait):
        consumer.finish("Complete answer")
        await consumer.run()

    assert seen_during_wait == [True]
    assert consumer.final_content_delivered is True
    assert consumer.delivered_final_matches("Complete answer")
    assert adapter.send.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("pause_already_active", [False, True])
async def test_turn_reset_during_wait_does_not_deliver_stale_final(clock, pause_already_active):
    consumer, adapter = make_consumer(clock)
    await consumer._send_or_edit("First ▉")
    adapter.edit_results.append(flood())
    if pause_already_active:
        await consumer._send_or_edit("First plus ▉")
    sleep = clock.sleep

    async def reset_during_wait(seconds):
        await sleep(seconds)
        consumer._run_still_current = lambda: False

    with patch("gateway.stream_consumer.asyncio.sleep", reset_during_wait):
        consumer.finish("First plus the final answer")
        await consumer.run()

    assert clock.waits == [9.0]
    assert adapter.edit_times == [100.0]
    assert adapter.send.await_count == 1
    assert consumer.final_content_delivered is False


@pytest.mark.asyncio
async def test_a_cancel_during_the_wait_aborts_instead_of_being_swallowed(clock):
    """The join budget belongs to the gateway, which grants the flood wait up front (see the
    next two tests). A cancel arriving here is therefore a real abort — a session reset or a
    shutdown — and catching it would silently defeat a timeout this module does not own."""
    consumer, adapter = make_consumer(clock)
    await consumer._send_or_edit("First \u2589")
    adapter.edit_results.append(flood())

    sleep = clock.sleep
    cancelled = False

    async def cancel_once(seconds):
        # Exactly one cancel, the shape of the gateway's join timeout firing mid-wait.
        nonlocal cancelled
        if not cancelled:
            cancelled = True
            clock.now += 5.0
            raise asyncio.CancelledError
        await sleep(seconds)

    with patch("gateway.stream_consumer.asyncio.sleep", cancel_once):
        consumer.finish("First plus the final answer")
        await consumer.run()   # run() owns cancellation: it finalizes best-effort, never re-raises

    # The refused edit only. No retry at 109.0, and teardown did not sit out the rest of the ban.
    assert adapter.edit_times == [100.0]
    assert clock.waits == []
    assert consumer.final_content_delivered is False


@pytest.mark.asyncio
async def test_the_outstanding_wait_is_published_for_the_join_budget(clock):
    consumer, adapter = make_consumer(clock)
    assert consumer.flood_pause_remaining == 0.0

    await consumer._send_or_edit("First \u2589")
    adapter.edit_results.append(flood())
    await consumer._send_or_edit("First plus \u2589")
    assert consumer.flood_pause_remaining == 9.0

    clock.now += 4.0
    assert consumer.flood_pause_remaining == 5.0
    clock.now += 9.0
    assert consumer.flood_pause_remaining == 0.0   # never negative once the ban has passed


@pytest.mark.asyncio
async def test_the_gateway_join_grants_the_outstanding_flood_wait():
    """Without this the consumer would have to outlive the cancel to finish its final edit."""
    from gateway.run import GatewayRunner

    budgets = []

    async def fake_wait_for(task, timeout):
        budgets.append(timeout)
        task.cancel()
        raise asyncio.TimeoutError

    async def _never():
        await asyncio.sleep(60)

    with patch("gateway.run_turn.asyncio.wait_for", fake_wait_for):
        for consumer in (None,
                         SimpleNamespace(flood_pause_remaining=9.0),
                         SimpleNamespace(flood_pause_remaining=0.0)):
            await GatewayRunner._await_stream_task(asyncio.create_task(_never()), consumer)

    assert budgets == [5.0, 14.0, 5.0]
