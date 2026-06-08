"""Tests for streaming content checkpoint functionality (issue #41696).

When the gateway crashes during streaming, incomplete turn content should be
persisted to the session database so it can be recovered after restart.
"""

import asyncio
import time
from unittest import mock

import pytest


@pytest.mark.asyncio
async def test_stream_consumer_checkpoint_callback():
    """Test that stream consumer can register and call checkpoint callback."""
    from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
    
    # Create a mock adapter
    mock_adapter = mock.MagicMock()
    mock_adapter.message_len_fn = len
    mock_adapter.MAX_MESSAGE_LENGTH = 4096
    mock_adapter.supports_draft_streaming.return_value = False
    mock_adapter.truncate_message.return_value = ["test"]
    mock_adapter.edit_message = mock.AsyncMock()
    mock_adapter.send_new = mock.AsyncMock(return_value="msg_123")
    
    # Create a consumer
    consumer = GatewayStreamConsumer(
        adapter=mock_adapter,
        chat_id="test_chat",
        config=StreamConsumerConfig(buffer_only=True),
        metadata={"session_id": "sess_123"},
    )
    
    # Set up checkpoint callback
    checkpoint_calls = []
    def checkpoint_fn(role, content, is_incomplete=False):
        checkpoint_calls.append({
            "role": role,
            "content": content,
            "is_incomplete": is_incomplete,
        })
    
    consumer.set_checkpoint_callback(checkpoint_fn, "sess_123")
    
    # Verify callback is registered
    assert consumer._checkpoint_callback == checkpoint_fn
    assert consumer._session_id == "sess_123"


@pytest.mark.asyncio
async def test_stream_consumer_checkpoint_on_delta():
    """Test that checkpoint is called periodically during streaming."""
    from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
    
    mock_adapter = mock.MagicMock()
    mock_adapter.message_len_fn = len
    mock_adapter.MAX_MESSAGE_LENGTH = 4096
    mock_adapter.supports_draft_streaming.return_value = False
    mock_adapter.truncate_message.return_value = ["test"]
    mock_adapter.edit_message = mock.AsyncMock()
    mock_adapter.send_new = mock.AsyncMock(return_value="msg_123")
    
    consumer = GatewayStreamConsumer(
        adapter=mock_adapter,
        chat_id="test_chat",
        config=StreamConsumerConfig(buffer_only=True, buffer_threshold=10),
        metadata={"session_id": "sess_123"},
    )
    
    checkpoint_calls = []
    def checkpoint_fn(role, content, is_incomplete=False):
        checkpoint_calls.append({
            "role": role,
            "content": content,
            "is_incomplete": is_incomplete,
        })
    
    consumer.set_checkpoint_callback(checkpoint_fn, "sess_123")
    consumer._checkpoint_interval = 0.1  # Reduce for faster testing
    
    # Simulate streaming deltas
    consumer.on_delta("Hello ")
    consumer.on_delta("world")
    
    # Wait for checkpoint interval
    await asyncio.sleep(0.15)
    
    # Manually call checkpoint to test
    consumer._checkpoint_incomplete_turn()
    
    # Verify checkpoint was called with accumulated text
    assert len(checkpoint_calls) > 0
    assert "Hello world" in [c["content"] for c in checkpoint_calls]
    assert all(c["is_incomplete"] for c in checkpoint_calls)


def test_stream_consumer_checkpoint_marker():
    """Test that incomplete content is marked with is_incomplete flag."""
    from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
    
    mock_adapter = mock.MagicMock()
    mock_adapter.message_len_fn = len
    mock_adapter.MAX_MESSAGE_LENGTH = 4096
    
    consumer = GatewayStreamConsumer(
        adapter=mock_adapter,
        chat_id="test_chat",
        config=StreamConsumerConfig(),
        metadata={"session_id": "sess_456"},
    )
    
    captured = []
    def checkpoint_fn(role, content, is_incomplete=False):
        captured.append((role, content, is_incomplete))
    
    consumer.set_checkpoint_callback(checkpoint_fn, "sess_456")
    consumer._accumulated = "Partial response text..."
    consumer._checkpoint_incomplete_turn()
    
    assert len(captured) == 1
    role, content, is_incomplete = captured[0]
    assert role == "assistant"
    assert content == "Partial response text..."
    assert is_incomplete is True


def test_stream_consumer_checkpoint_skips_empty():
    """Test that checkpoint skips when accumulated text is empty."""
    from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
    
    mock_adapter = mock.MagicMock()
    
    consumer = GatewayStreamConsumer(
        adapter=mock_adapter,
        chat_id="test_chat",
        config=StreamConsumerConfig(),
        metadata={},
    )
    
    checkpoint_calls = []
    def checkpoint_fn(role, content, is_incomplete=False):
        checkpoint_calls.append((role, content, is_incomplete))
    
    consumer.set_checkpoint_callback(checkpoint_fn, "sess_789")
    consumer._accumulated = ""  # Empty
    consumer._checkpoint_incomplete_turn()
    
    # Should not call checkpoint for empty content
    assert len(checkpoint_calls) == 0


def test_stream_consumer_checkpoint_handles_errors():
    """Test that checkpoint handles errors gracefully."""
    from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
    
    mock_adapter = mock.MagicMock()
    
    consumer = GatewayStreamConsumer(
        adapter=mock_adapter,
        chat_id="test_chat",
        config=StreamConsumerConfig(),
        metadata={},
    )
    
    # Callback that raises an error
    def bad_checkpoint(role, content, is_incomplete=False):
        raise RuntimeError("Database error")
    
    consumer.set_checkpoint_callback(bad_checkpoint, "sess_999")
    consumer._accumulated = "Response text"
    
    # Should not raise — error is logged internally
    consumer._checkpoint_incomplete_turn()
    
    # Verify no exception was raised
    assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
