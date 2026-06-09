"""Tests for approval boundary handling in WeCom native streaming."""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig


@pytest.fixture
def mock_adapter():
    """Create a mock WeCom adapter with native streaming support."""
    from gateway.platforms.base import BasePlatformAdapter
    MockAdapter = type("MockAdapter", (BasePlatformAdapter,), {
        "MAX_MESSAGE_LENGTH": 4096,
        "SUPPORTS_MESSAGE_EDITING": False,
        "SUPPORTS_NATIVE_STREAMING": True,
    })
    MockAdapter.__abstractmethods__ = frozenset()
    adapter = MockAdapter.__new__(MockAdapter)
    adapter._typing_paused = set()
    adapter.send_stream_frame = AsyncMock(return_value=True)
    adapter.send = AsyncMock(return_value=MagicMock(success=True, message_id="msg"))
    adapter.supports_native_streaming = lambda chat_type=None, metadata=None: True
    return adapter


@pytest.fixture
def consumer_config():
    """Create a minimal consumer config."""
    return StreamConsumerConfig(
        chat_type="dm", cursor="",
        edit_interval=0.01, buffer_threshold=5,
    )


@pytest.mark.asyncio
async def test_approval_boundary_finalizes_and_disables_native(mock_adapter, consumer_config):
    """Approval boundary must finalize the current stream (creating a stable
    message for pre-approval content) and disable native streaming so
    post-approval output goes through reliable send()."""
    consumer = GatewayStreamConsumer(
        adapter=mock_adapter,
        chat_id="test_chat",
        config=consumer_config,
    )

    consumer._use_native_streaming = True
    consumer._native_stream_opened = True
    consumer._turn_id = "turn_123"
    consumer._initial_reply_to_id = "msg_456"
    consumer._accumulated = "下面我来执行：先确认最新分区"

    # Signal approval boundary
    boundary_result = consumer.close_for_approval_prompt()
    if isinstance(boundary_result, tuple):
        boundary_future, _ = boundary_result
    else:
        boundary_future = boundary_result

    consumer_task = asyncio.create_task(consumer.run())
    result = await asyncio.wait_for(boundary_future, timeout=1.0)

    assert result is True

    # Stream must be finalized (stable message created)
    finalize_calls = [
        call for call in mock_adapter.send_stream_frame.call_args_list
        if call.kwargs.get("finalize") is True
    ]
    assert len(finalize_calls) == 1, "Must finalize the stream"
    finalize_text = finalize_calls[0].args[0]
    assert finalize_text == "下面我来执行：先确认最新分区"

    # Native streaming must be disabled for post-approval output
    assert consumer._use_native_streaming is False, (
        "Native streaming must be disabled — post-approval goes via send()"
    )
    assert consumer._native_stream_opened is False

    consumer.finish()
    await asyncio.sleep(0.05)
    consumer_task.cancel()
    try:
        await consumer_task
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_approval_boundary_uses_placeholder_when_no_accumulated(mock_adapter, consumer_config):
    """When there's no accumulated text, finalize with a visible placeholder."""
    consumer = GatewayStreamConsumer(
        adapter=mock_adapter,
        chat_id="test_chat",
        config=consumer_config,
    )

    consumer._use_native_streaming = True
    consumer._native_stream_opened = True
    consumer._turn_id = "turn_123"
    consumer._accumulated = ""  # No text accumulated

    boundary_result = consumer.close_for_approval_prompt()
    if isinstance(boundary_result, tuple):
        boundary_future, _ = boundary_result
    else:
        boundary_future = boundary_result

    consumer_task = asyncio.create_task(consumer.run())
    result = await asyncio.wait_for(boundary_future, timeout=1.0)
    assert result is True

    finalize_calls = [
        call for call in mock_adapter.send_stream_frame.call_args_list
        if call.kwargs.get("finalize") is True
    ]
    assert len(finalize_calls) == 1
    finalize_text = finalize_calls[0].args[0]
    assert finalize_text == "⏸ 等待审批中..."

    consumer.finish()
    await asyncio.sleep(0.05)
    consumer_task.cancel()
    try:
        await consumer_task
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_approval_boundary_post_approval_one_shot_send(mock_adapter, consumer_config):
    """After approval boundary, post-approval content must:
    1. Set buffer_only=True (no mid-stream flushes)
    2. Accumulate all deltas without sending
    3. Deliver everything via one adapter.send() call on finish()"""
    consumer = GatewayStreamConsumer(
        adapter=mock_adapter,
        chat_id="test_chat",
        config=consumer_config,
    )

    consumer._use_native_streaming = True
    consumer._native_stream_opened = True
    consumer._turn_id = "turn_123"
    consumer._accumulated = "Pre-approval text"

    boundary_result = consumer.close_for_approval_prompt()
    if isinstance(boundary_result, tuple):
        boundary_future, _ = boundary_result
    else:
        boundary_future = boundary_result

    consumer_task = asyncio.create_task(consumer.run())
    await asyncio.wait_for(boundary_future, timeout=1.0)

    # Verify buffer_only is set
    assert consumer.cfg.buffer_only is True, "Must set buffer_only after boundary"
    assert consumer._use_native_streaming is False

    # Send post-approval content — should NOT trigger any immediate send
    mock_adapter.send_stream_frame.reset_mock()
    mock_adapter.send.reset_mock()

    consumer.on_delta("Post-approval result text here")
    await asyncio.sleep(0.1)

    # Before finish(): no send() or stream frame calls
    assert mock_adapter.send.call_count == 0, (
        "buffer_only: no send before finish()"
    )
    stream_calls = [
        call for call in mock_adapter.send_stream_frame.call_args_list
        if not call.kwargs.get("finalize")
    ]
    assert len(stream_calls) == 0, (
        "Post-approval must NOT use native streaming"
    )

    # Now finish — should deliver via send()
    consumer.finish()
    await asyncio.sleep(0.1)

    # adapter.send should have been called with the full post-approval text
    send_calls = mock_adapter.send.call_args_list
    assert len(send_calls) >= 1, "finish() must deliver via send()"
    # The delivered text should contain our post-approval content
    delivered = send_calls[-1].args[1] if len(send_calls[-1].args) > 1 else send_calls[-1].kwargs.get("content", "")
    assert "Post-approval result" in delivered

    consumer_task.cancel()
    try:
        await consumer_task
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_approval_boundary_stream_not_opened_at_boundary_time(mock_adapter, consumer_config):
    """When native streaming is active but _native_stream_opened is still False
    at the time boundary processes (e.g., seed succeeded but stream was closed by
    a prior error before boundary arrives), no finalize is sent."""
    consumer = GatewayStreamConsumer(
        adapter=mock_adapter,
        chat_id="test_chat",
        config=consumer_config,
    )

    # Let run() do its normal seed (which sets _native_stream_opened=True)
    # Then manually close it before boundary processes
    consumer._use_native_streaming = True
    consumer._native_stream_opened = True
    consumer._turn_id = "turn_123"
    consumer._accumulated = "Some text"

    # Queue boundary
    boundary_result = consumer.close_for_approval_prompt()
    if isinstance(boundary_result, tuple):
        boundary_future, _ = boundary_result
    else:
        boundary_future = boundary_result

    # Simulate: stream was closed by error BEFORE consumer processes boundary
    consumer._native_stream_opened = False

    consumer_task = asyncio.create_task(consumer.run())
    result = await asyncio.wait_for(boundary_future, timeout=1.0)

    assert result is True
    # Native streaming should be disabled after boundary
    assert consumer._use_native_streaming is False

    consumer.finish()
    await asyncio.sleep(0.05)
    consumer_task.cancel()
    try:
        await consumer_task
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_approval_boundary_finalize_fails_fallback_send_succeeds(mock_adapter, consumer_config):
    """When stream finalize fails but fallback send() succeeds, boundary is True."""
    consumer = GatewayStreamConsumer(
        adapter=mock_adapter,
        chat_id="test_chat",
        config=consumer_config,
    )

    consumer._use_native_streaming = True
    consumer._native_stream_opened = True
    consumer._turn_id = "turn_123"
    consumer._accumulated = "Pre-approval text"

    # Finalize fails (returns False)
    mock_adapter.send_stream_frame = AsyncMock(return_value=False)
    # Fallback send succeeds
    mock_adapter.send = AsyncMock(return_value=MagicMock(success=True, message_id="msg"))

    boundary_result = consumer.close_for_approval_prompt()
    if isinstance(boundary_result, tuple):
        boundary_future, _ = boundary_result
    else:
        boundary_future = boundary_result

    consumer_task = asyncio.create_task(consumer.run())
    result = await asyncio.wait_for(boundary_future, timeout=1.0)

    assert result is True, "Fallback send succeeded → boundary should be True"
    mock_adapter.send.assert_awaited_once_with("test_chat", "Pre-approval text")

    consumer.finish()
    await asyncio.sleep(0.05)
    consumer_task.cancel()
    try:
        await consumer_task
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_approval_boundary_finalize_and_fallback_both_fail(mock_adapter, consumer_config):
    """When both stream finalize and fallback send() fail, boundary is False."""
    consumer = GatewayStreamConsumer(
        adapter=mock_adapter,
        chat_id="test_chat",
        config=consumer_config,
    )

    consumer._use_native_streaming = True
    consumer._native_stream_opened = True
    consumer._turn_id = "turn_123"
    consumer._accumulated = "Pre-approval text"

    # Finalize fails (raises)
    mock_adapter.send_stream_frame = AsyncMock(side_effect=RuntimeError("stream dead"))
    # Fallback send also fails
    mock_adapter.send = AsyncMock(return_value=MagicMock(success=False, error="timeout"))

    boundary_result = consumer.close_for_approval_prompt()
    if isinstance(boundary_result, tuple):
        boundary_future, _ = boundary_result
    else:
        boundary_future = boundary_result

    consumer_task = asyncio.create_task(consumer.run())
    result = await asyncio.wait_for(boundary_future, timeout=1.0)

    assert result is False, "Both finalize and fallback failed → boundary should be False"

    consumer.finish()
    await asyncio.sleep(0.05)
    consumer_task.cancel()
    try:
        await consumer_task
    except asyncio.CancelledError:
        pass
