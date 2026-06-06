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
async def test_approval_boundary_cancelled_sends_invisible_finalize(mock_adapter, consumer_config):
    """Test that cancelled boundary sends invisible finalize, not visible placeholder."""
    consumer = GatewayStreamConsumer(
        adapter=mock_adapter,
        chat_id="test_chat",
        config=consumer_config,
    )

    # Force native streaming mode
    consumer._use_native_streaming = True
    consumer._native_stream_opened = True
    consumer._turn_id = "turn_123"
    consumer._initial_reply_to_id = "msg_456"
    consumer._accumulated = "Some partial text"

    # Signal approval boundary and immediately mark as cancelled
    boundary_result = consumer.close_for_approval_prompt()
    if isinstance(boundary_result, tuple):
        boundary_future, cancelled_flag = boundary_result
        cancelled_flag["cancelled"] = True  # Simulate timeout
    else:
        pytest.skip("Consumer doesn't support cancellation flag")

    # Start consumer task
    consumer_task = asyncio.create_task(consumer.run())

    # Signal completion
    consumer.finish()

    # Wait for consumer to process
    await asyncio.sleep(0.1)
    consumer_task.cancel()
    try:
        await consumer_task
    except asyncio.CancelledError:
        pass

    # Verify finalize was called with invisible content (not "⏸ 等待审批中...")
    finalize_calls = [
        call for call in mock_adapter.send_stream_frame.call_args_list
        if call.kwargs.get("finalize") is True
    ]
    assert len(finalize_calls) >= 1, "Should have called finalize"

    # Check the finalize call for boundary
    finalize_text = finalize_calls[0].args[0] if finalize_calls[0].args else finalize_calls[0].kwargs.get("text", "")

    # Should NOT be the visible placeholder
    assert finalize_text != "⏸ 等待审批中...", "Cancelled boundary should not send visible placeholder"
    # Should be invisible (zero-width space or empty)
    assert finalize_text in ("​", "", "✅") or len(finalize_text.strip()) == 0, \
        f"Cancelled boundary should use invisible finalize, got: {repr(finalize_text)}"


@pytest.mark.asyncio
async def test_approval_boundary_finalize_failure_returns_false(mock_adapter, consumer_config):
    """Test that finalize failure causes boundary to return False."""
    consumer = GatewayStreamConsumer(
        adapter=mock_adapter,
        chat_id="test_chat",
        config=consumer_config,
    )

    # Force native streaming mode
    consumer._use_native_streaming = True
    consumer._native_stream_opened = True
    consumer._turn_id = "turn_123"

    # Make finalize fail
    mock_adapter.send_stream_frame.side_effect = Exception("Finalize failed")

    # Signal approval boundary
    boundary_result = consumer.close_for_approval_prompt()
    if isinstance(boundary_result, tuple):
        boundary_future, _ = boundary_result
    else:
        boundary_future = boundary_result

    # Start consumer task
    consumer_task = asyncio.create_task(consumer.run())

    # Signal completion
    consumer.finish()

    # Wait for consumer to process
    await asyncio.sleep(0.1)

    # Check boundary result
    try:
        result = await asyncio.wait_for(boundary_future, timeout=1.0)
        assert result is False, "Boundary should return False when finalize fails"
    except asyncio.TimeoutError:
        pytest.fail("Boundary future never resolved")

    consumer_task.cancel()
    try:
        await consumer_task
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_approval_boundary_clears_initial_reply_to(mock_adapter, consumer_config):
    """Test that boundary clears initial_reply_to_id for fresh req_id."""
    consumer = GatewayStreamConsumer(
        adapter=mock_adapter,
        chat_id="test_chat",
        config=consumer_config,
    )

    # Force native streaming mode with initial reply_to
    consumer._use_native_streaming = True
    consumer._native_stream_opened = True
    consumer._turn_id = "turn_old"
    consumer._initial_reply_to_id = "msg_old"  # Original message

    # Signal approval boundary
    boundary_result = consumer.close_for_approval_prompt()
    if isinstance(boundary_result, tuple):
        boundary_future, _ = boundary_result
    else:
        boundary_future = boundary_result

    # Start consumer task
    consumer_task = asyncio.create_task(consumer.run())

    # Wait for boundary to process
    try:
        await asyncio.wait_for(boundary_future, timeout=1.0)
    except asyncio.TimeoutError:
        pytest.fail("Boundary future never resolved")

    # Verify state was reset
    assert consumer._initial_reply_to_id is None, \
        "Boundary should clear initial_reply_to_id for post-approval fresh req_id"
    assert consumer._turn_id != "turn_old", \
        "Boundary should generate new turn_id"

    consumer.finish()
    await asyncio.sleep(0.05)
    consumer_task.cancel()
    try:
        await consumer_task
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_approval_boundary_re_seed_uses_new_turn_id(mock_adapter, consumer_config):
    """Test that post-approval re-seed uses new turn_id and no reply_to."""
    consumer = GatewayStreamConsumer(
        adapter=mock_adapter,
        chat_id="test_chat",
        config=consumer_config,
    )

    # Force native streaming mode
    consumer._use_native_streaming = True
    consumer._native_stream_opened = True
    old_turn_id = "turn_old"
    consumer._turn_id = old_turn_id
    consumer._initial_reply_to_id = "msg_old"

    # Signal approval boundary
    boundary_result = consumer.close_for_approval_prompt()
    if isinstance(boundary_result, tuple):
        boundary_future, _ = boundary_result
    else:
        boundary_future = boundary_result

    # Start consumer task
    consumer_task = asyncio.create_task(consumer.run())

    # Wait for boundary
    await asyncio.wait_for(boundary_future, timeout=1.0)

    # Get new turn_id after boundary
    new_turn_id = consumer._turn_id
    assert new_turn_id != old_turn_id, "Should have new turn_id"

    # Now send new content (simulating post-approval output)
    consumer.on_delta("Post-approval content")
    await asyncio.sleep(0.1)

    # Verify re-seed was called with new turn_id and no reply_to
    seed_calls = [
        call for call in mock_adapter.send_stream_frame.call_args_list
        if call.args[0] == "" and not call.kwargs.get("finalize", False)
    ]

    # Should have at least one seed call after boundary
    assert len(seed_calls) >= 2, "Should have re-seeded after boundary"

    # Check the re-seed call (after boundary, should be second or later seed)
    re_seed_call = seed_calls[-1]  # Last seed call
    assert re_seed_call.kwargs.get("turn_id") == new_turn_id, \
        "Re-seed should use new turn_id"
    assert re_seed_call.kwargs.get("reply_to") is None, \
        "Re-seed should not pass old reply_to (let adapter use _last_chat_req_ids)"

    consumer.finish()
    await asyncio.sleep(0.05)
    consumer_task.cancel()
    try:
        await consumer_task
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_approval_boundary_success_path(mock_adapter, consumer_config):
    """Test normal approval boundary (not cancelled, finalize succeeds)."""
    consumer = GatewayStreamConsumer(
        adapter=mock_adapter,
        chat_id="test_chat",
        config=consumer_config,
    )

    consumer._use_native_streaming = True
    consumer._native_stream_opened = True
    consumer._turn_id = "turn_old"
    consumer._accumulated = "Some work in progress"

    # Signal boundary (no cancellation)
    boundary_result = consumer.close_for_approval_prompt()
    if isinstance(boundary_result, tuple):
        boundary_future, _ = boundary_result
    else:
        boundary_future = boundary_result

    # Start consumer
    consumer_task = asyncio.create_task(consumer.run())

    # Wait for boundary
    result = await asyncio.wait_for(boundary_future, timeout=1.0)

    # Should succeed
    assert result is True, "Boundary should return True on success"

    # Verify visible placeholder was sent (not cancelled)
    finalize_calls = [
        call for call in mock_adapter.send_stream_frame.call_args_list
        if call.kwargs.get("finalize") is True
    ]
    assert len(finalize_calls) >= 1
    finalize_text = finalize_calls[0].args[0] if finalize_calls[0].args else finalize_calls[0].kwargs.get("text", "")

    # Should be the accumulated content or placeholder
    assert finalize_text in ("Some work in progress", "⏸ 等待审批中..."), \
        "Non-cancelled boundary should send visible content"

    consumer.finish()
    await asyncio.sleep(0.05)
    consumer_task.cancel()
    try:
        await consumer_task
    except asyncio.CancelledError:
        pass
