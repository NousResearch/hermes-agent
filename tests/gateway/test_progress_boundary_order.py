"""Progress ordering across asynchronous producer and delivery boundaries."""
import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from gateway.progress_events import (
    ContentBoundaryBuffer, DurableContentBoundary, DurableContentSource,
    ProvisionalContentBoundary, RetractedContentBoundary,
)
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig


def test_overlapping_preview_outcomes_preserve_progress_order():
    buffer = ContentBoundaryBuffer()
    first = ProvisionalContentBoundary("first")
    second = ProvisionalContentBoundary("second")
    durable = DurableContentBoundary("second", DurableContentSource.STREAM_FINALIZED)
    assert buffer.feed("before") == ["before"]
    for item in [first, "middle", second, "after", durable, durable]:
        assert buffer.feed(item) == []
    assert buffer.feed(RetractedContentBoundary("first")) == ["middle", durable, "after"]
    assert buffer.feed(RetractedContentBoundary("first")) == []
    assert buffer.finish() == []


@pytest.mark.asyncio
async def test_worker_queues_multiple_segments_before_platform_delivery():
    adapter = MagicMock()
    adapter.MAX_MESSAGE_LENGTH = 4096
    adapter.send = AsyncMock(side_effect=[
        SimpleNamespace(success=True, message_id="first"),
        SimpleNamespace(success=True, message_id="second"),
    ])
    adapter.edit_message = AsyncMock(return_value=SimpleNamespace(success=True))
    events = []
    consumer = GatewayStreamConsumer(adapter, "chat", StreamConsumerConfig(),
        on_content_boundary=events.append)
    consumer.on_delta("first segment")
    consumer.on_segment_break()
    consumer.on_delta("second segment")
    consumer.finish()
    provisional = list(events)
    assert all(isinstance(item, ProvisionalContentBoundary) for item in provisional)
    assert len(provisional) == 2
    await consumer.run()
    durable = [item for item in events if isinstance(item, DurableContentBoundary)]
    assert [item.boundary_id for item in durable] == [item.boundary_id for item in provisional]
    assert [item.message_id for item in durable] == ["first", "second"]
    assert [call.kwargs["content"] for call in adapter.send.call_args_list] == [
        "first segment", "second segment"]


def test_dedup_after_a_durable_seal_becomes_first_line():
    from gateway.run_turn_runner import TurnRunner
    state = SimpleNamespace(progress_lines=[])
    text = TurnRunner._progress_absorb(None, state, ("__dedup__", "same tool", 1))
    assert state.progress_lines == [text] == ["same tool"]


@pytest.mark.asyncio
async def test_ambiguous_commentary_is_not_declared_deleted():
    adapter = MagicMock()
    adapter.MAX_MESSAGE_LENGTH = 4096
    adapter.send = AsyncMock(return_value=SimpleNamespace(success=False,
        error="read timeout", retryable=False))
    events = []
    consumer = GatewayStreamConsumer(adapter, "chat", StreamConsumerConfig(),
        on_content_boundary=events.append)
    consumer.on_commentary("possibly delivered")
    consumer.finish()
    await consumer.run()
    assert len(events) == 2
    assert isinstance(events[0], ProvisionalContentBoundary)
    assert isinstance(events[1], DurableContentBoundary)
    assert events[0].boundary_id == events[1].boundary_id


@pytest.mark.asyncio
async def test_persistent_draft_releases_live_progress_before_turn_end():
    from tests.gateway.test_stream_consumer_draft import _make_draft_capable_adapter
    adapter = _make_draft_capable_adapter()
    adapter.draft_stream_is_message = True
    buffer = ContentBoundaryBuffer()
    ready = []
    consumer = GatewayStreamConsumer(adapter, "chat", StreamConsumerConfig(
        transport="draft", chat_type="dm", cursor=""),
        on_content_boundary=lambda event: ready.extend(buffer.feed(event)))
    consumer.on_delta("first segment")
    consumer._drain_queue()
    await consumer._start_transports()
    assert await consumer._send_or_edit("first segment")
    assert buffer.feed("tool started") == ["tool started"]
    consumer.on_segment_break()
    consumer.on_delta(" continued")
    tick = consumer._drain_queue()
    await consumer._end_segment(tick)
    consumer._drain_queue()
    assert await consumer._send_or_edit("first segment continued")
    assert len([event for event in ready if isinstance(event, DurableContentBoundary)]) == 1
    assert consumer._accumulated == "first segment continued"


@pytest.mark.asyncio
async def test_progress_cleanup_failure_cannot_leak_the_session_slot():
    import asyncio
    from gateway.run import GatewayRunner
    started = asyncio.Event()
    async def failing_drain():
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            raise RuntimeError("transport disconnected")
    progress = asyncio.create_task(failing_drain())
    tracking = asyncio.create_task(asyncio.Event().wait())
    await started.wait()
    runner = object.__new__(GatewayRunner)
    runner._release_running_agent_state = MagicMock()
    runner._draining = False
    ctx = SimpleNamespace(stream_consumer_holder=[None], session_key="owned",
        streaming_tts_consumer_holder=[None], run_generation=1)
    await runner._run_agent_cleanup_turn_tasks(ctx, progress_task=progress,
        log_task=None, interrupt_monitor=None, _notify_task=None,
        tracking_task=tracking, stream_task=None)
    assert tracking.cancelled()
    runner._release_running_agent_state.assert_called_once_with("owned", run_generation=1)
