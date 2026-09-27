"""An exhausted Slack posting quota must not amplify optional progress traffic."""
import asyncio
import queue
from types import SimpleNamespace

import pytest

from gateway.config import Platform
from gateway.platforms.base import SendResult
from gateway.run_turn_runner import TurnRunner


class RecordingAdapter:
    MAX_MESSAGE_LENGTH = 150
    name = "slack"

    def __init__(self, error="message_limit_exceeded"):
        self.error = error
        self.calls = []

    async def send_typing(self, *args, **kwargs):
        pass

    async def send(self, **kwargs):
        self.calls.append(("send", kwargs["content"]))
        return SendResult(success=False, error=self.error)

    async def edit_message(self, **kwargs):
        self.calls.append(("edit", kwargs["content"]))
        return SendResult(success=False, error=self.error)


def make_runner(adapter, platform=Platform.SLACK):
    ctx = SimpleNamespace(
        source=SimpleNamespace(chat_id="C1", platform=platform),
        progress_grouping="grouped", _progress_metadata={"_interim_send": True},
        _progress_reply_to=None, progress_queue=queue.Queue(),
        last_progress_msg=[None], repeat_count=[0],
        _native_slack_task_cards=False, _run_still_current=lambda: True,
        agent_holder=[None],
    )
    runner = TurnRunner(SimpleNamespace(_delivery_adapter_for=lambda _: adapter), ctx)
    runner._track_progress_result = lambda result: None
    return runner, ctx


@pytest.mark.asyncio
async def test_public_progress_worker_stops_after_quota_refusal():
    adapter = RecordingAdapter()
    runner, ctx = make_runner(adapter)
    consumed = asyncio.Event()

    class ObservedQueue(queue.Queue):
        def get_nowait(self):
            item = super().get_nowait()
            if item == "second":
                consumed.set()
            return item

    ctx.progress_queue = ObservedQueue()
    ctx.progress_queue.put("first")
    ctx.progress_queue.put("second")
    task = asyncio.create_task(runner.send_progress_messages())
    try:
        await asyncio.wait_for(consumed.wait(), timeout=3)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    assert adapter.calls == [("send", "first")]
    assert ctx.progress_queue.empty()


@pytest.mark.asyncio
@pytest.mark.parametrize("first_operation", ["send", "edit", "overflow"])
async def test_quota_stop_survives_reset_overflow_and_cancel(first_operation):
    adapter = RecordingAdapter('Slack API error: message_limit_exceeded')
    runner, ctx = make_runner(adapter)
    state = runner._progress_edit_state(adapter)
    state.progress_lines = ["first"]
    if first_operation == "edit":
        state.progress_msg_id = "1"
    if first_operation == "overflow":
        state.progress_lines = ["x" * 60, "y" * 60, "z" * 60]
        await runner._roll_progress_overflow_if_needed(state)
    else:
        await runner._progress_send_or_edit(state, "first")
    assert len(adapter.calls) == 1
    runner._reset_progress_bubble(state)
    state.progress_lines = ["a" * 60, "b" * 60]
    await runner._roll_progress_overflow_if_needed(state)
    await runner._progress_send_or_edit(state, "later")
    ctx.progress_queue.put(("__reset__",))
    ctx.progress_queue.put("c" * 60)
    await runner._drain_progress_on_cancel(state)
    assert len(adapter.calls) == 1
    # Not an adapter-wide breaker: final replies and a fresh turn still try.
    await adapter.send(content="final")
    fresh = runner._progress_edit_state(adapter)
    fresh.progress_lines = ["next turn"]
    await runner._progress_send_or_edit(fresh, "next turn")
    assert adapter.calls[-2:] == [("send", "final"), ("send", "next turn")]


@pytest.mark.asyncio
@pytest.mark.parametrize("error", ["", None, "ratelimited", "msg_too_long", "not_message_limit_exceeded"])
async def test_other_errors_preserve_edit_fallback(error):
    adapter = RecordingAdapter(error)
    runner, _ = make_runner(adapter)
    state = runner._progress_edit_state(adapter)
    state.progress_msg_id = "1"
    state.progress_lines = ["first"]
    await runner._progress_send_or_edit(state, "first")
    assert adapter.calls == [("edit", "first"), ("send", "first")]


@pytest.mark.asyncio
async def test_success_and_other_platform_do_not_trip_quota_stop():
    adapter = RecordingAdapter()
    runner, _ = make_runner(adapter, platform=Platform.TELEGRAM)
    state = runner._progress_edit_state(adapter)
    state.progress_lines = ["first"]
    await runner._progress_send_or_edit(state, "first")
    await runner._progress_send_or_edit(state, "second")
    assert len(adapter.calls) == 2
    runner, _ = make_runner(adapter)
    state = runner._progress_edit_state(adapter)
    runner._observe_progress_quota(state, SendResult(success=True, error="message_limit_exceeded"))
    assert state.quota_failure is None


@pytest.mark.asyncio
async def test_raised_transport_error_does_not_trip_quota_stop():
    adapter = RecordingAdapter()
    runner, _ = make_runner(adapter)
    state = runner._progress_edit_state(adapter)

    async def broken(**kwargs):
        raise TimeoutError("unknown delivery outcome")

    adapter.send = broken
    with pytest.raises(TimeoutError):
        await runner._send_progress_text(state, "first")
    assert state.quota_failure is None
