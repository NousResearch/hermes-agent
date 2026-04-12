"""Tests for GatewayStreamConsumer — accumulated text preservation on send failure.

When _send_new_chunk fails during the chunk-splitting path (lines 188-193),
_accumulated is cleared unconditionally even if sends failed, causing text
loss.  The fix: only clear _accumulated after confirming all chunks were
delivered.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig, _DONE


def _make_adapter(max_len=4096, send_fn=None, truncate_fn=None):
    """Create a mock adapter for stream consumer tests."""
    adapter = MagicMock()
    adapter.MAX_MESSAGE_LENGTH = max_len
    if send_fn:
        adapter.send = AsyncMock(side_effect=send_fn)
    else:
        adapter.send = AsyncMock(return_value=SimpleNamespace(success=True, message_id="msg_1"))
    adapter.edit_message = AsyncMock(return_value=SimpleNamespace(success=True))
    if truncate_fn:
        adapter.truncate_message = truncate_fn
    else:
        adapter.truncate_message = lambda text, limit: [text]
    return adapter


class TestAccumulatedTextPreservation:
    @pytest.mark.asyncio
    async def test_chunk_send_failure_preserves_accumulated(self):
        """When _send_new_chunk fails during the chunk-splitting path,
        _accumulated must not be cleared — the text would be lost."""
        call_count = 0

        async def failing_send(**kwargs):
            nonlocal call_count
            call_count += 1
            raise Exception("network error")

        adapter = _make_adapter(
            max_len=4096,
            send_fn=failing_send,
            truncate_fn=lambda text, limit: [text[:limit], text[limit:]],
        )

        cfg = StreamConsumerConfig(buffer_threshold=10)
        consumer = GatewayStreamConsumer(adapter, "chat_123", config=cfg)

        # _safe_limit = max(500, 4096 - len(cursor) - 100) = 3991
        # Text must exceed _safe_limit to trigger the chunk-splitting path
        long_text = "A" * 4000
        consumer._queue.put(long_text)
        consumer._queue.put(_DONE)

        await consumer.run()

        # After the fix, _accumulated should be preserved when sends fail
        assert consumer._accumulated != "" or call_count == 0, (
            "_accumulated should not be cleared when all _send_new_chunk calls fail"
        )

    @pytest.mark.asyncio
    async def test_partial_chunk_failure_preserves_unsent_text(self):
        """If the first chunk succeeds but a later one fails, the unsent
        portion should be preserved in _accumulated."""
        call_count = 0

        async def partial_fail_send(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return SimpleNamespace(success=True, message_id="msg_1")
            raise Exception("network error on second chunk")

        adapter = _make_adapter(
            max_len=4096,
            send_fn=partial_fail_send,
            truncate_fn=lambda text, limit: [text[:limit], text[limit:]],
        )

        cfg = StreamConsumerConfig(buffer_threshold=10)
        consumer = GatewayStreamConsumer(adapter, "chat_123", config=cfg)

        long_text = "A" * 4000
        consumer._queue.put(long_text)
        consumer._queue.put(_DONE)

        await consumer.run()

        # After the fix, the unsent portion (second chunk) should remain
        # in _accumulated — not silently lost
        assert consumer._accumulated != "", (
            "Unsent text should be preserved when a chunk fails"
        )
