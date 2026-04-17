"""Tests for stream consumer no_edit_mode (signal-streaming-split-patch).

Covers: no_edit_mode prevents mid-sentence fragmentation, paragraph boundary
splitting, short response single-message delivery, and long response
paragraph-based chunking.

Based on hermes-patches/signal-streaming-split-patch.md.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_consumer(no_edit_mode=True, cursor="", buffer_threshold=40, max_msg_len=4096):
    """Create a consumer with a mock adapter (Signal-like: no editing)."""
    adapter = MagicMock()
    # Signal: send returns message_id=None
    send_result = SimpleNamespace(success=True, message_id=None)
    adapter.send = AsyncMock(return_value=send_result)
    adapter.edit_message = AsyncMock(
        return_value=SimpleNamespace(success=False, error="Not supported")
    )
    adapter.MAX_MESSAGE_LENGTH = max_msg_len
    adapter.truncate_message = MagicMock(side_effect=lambda text, limit: [text])

    config = StreamConsumerConfig(
        edit_interval=0.01,  # Fast for tests
        no_edit_mode=no_edit_mode,
        cursor=cursor,
        buffer_threshold=buffer_threshold,
    )
    consumer = GatewayStreamConsumer(adapter, "chat_123", config)
    return consumer, adapter


def _all_sent_texts(adapter):
    """Collect all text passed to adapter.send() calls."""
    texts = []
    for call in adapter.send.call_args_list:
        content = call.kwargs.get("content", "")
        if content:
            texts.append(content)
    return texts


# ===========================================================================
# Core: short responses arrive as single message
# ===========================================================================

class TestNoEditModeShortResponse:
    """Short responses should arrive as a single message, not fragmented.
    
    This is the core fix from signal-streaming-split-patch: before the patch,
    responses like "It's live: https://example.com" got split into two messages
    because the stream consumer flushed after the timer/buffer threshold.
    """

    @pytest.mark.asyncio
    async def test_short_response_single_message(self):
        """A short response should be delivered as one message."""
        consumer, adapter = _make_consumer(no_edit_mode=True)

        consumer.on_delta("It's ")
        consumer.on_delta("live: ")
        consumer.on_delta("https://example.com")
        consumer.finish()

        await consumer.run()

        assert adapter.send.call_count == 1
        sent = _all_sent_texts(adapter)
        assert len(sent) == 1
        assert "It's live: https://example.com" in sent[0]

    @pytest.mark.asyncio
    async def test_short_tokens_not_fragmented(self):
        """Individual character tokens should not cause fragmentation."""
        consumer, adapter = _make_consumer(no_edit_mode=True, buffer_threshold=10)

        # Feed character by character
        for c in "Hello, world!":
            consumer.on_delta(c)
        consumer.finish()

        await consumer.run()

        assert adapter.send.call_count == 1
        sent = _all_sent_texts(adapter)
        assert "Hello, world!" in sent[0]


# ===========================================================================
# Paragraph boundary splitting
# ===========================================================================

class TestNoEditModeParagraphSplitting:
    """Long responses split at paragraph boundaries (\\n\\n), not mid-sentence.
    
    Paragraph splitting only fires mid-stream (not on got_done). So to test it,
    we must feed text gradually without calling finish() until after the consumer
    has had a chance to process the paragraph break.
    """

    @pytest.mark.asyncio
    async def test_splits_at_paragraph_boundary_midstream(self):
        """Mid-stream: buffer with \\n\\n and >= min_buffer chars splits at break."""
        import asyncio

        consumer, adapter = _make_consumer(no_edit_mode=True)
        consumer._no_edit_min_buffer = 100

        para1 = "A" * 120
        para2 = "B" * 60

        # Feed first paragraph + separator, but don't finish yet
        consumer.on_delta(f"{para1}\n\n{para2}")

        # Run consumer briefly (it will process the queue, see paragraph
        # break >= min_buffer, and flush para1)
        run_task = asyncio.ensure_future(consumer.run())
        await asyncio.sleep(0.1)  # Let consumer process

        # Now finish
        consumer.finish()
        await run_task

        # Para1 should have been flushed mid-stream, para2 on finish
        assert adapter.send.call_count >= 2
        all_text = " ".join(_all_sent_texts(adapter))
        assert "A" * 120 in all_text
        assert "B" * 60 in all_text

    @pytest.mark.asyncio
    async def test_no_split_below_min_buffer(self):
        """Even with \\n\\n, text below min_buffer is not split."""
        consumer, adapter = _make_consumer(no_edit_mode=True)
        consumer._no_edit_min_buffer = 500  # High threshold

        consumer.on_delta("Short paragraph.\n\nAnother short one.")
        consumer.finish()

        await consumer.run()

        assert adapter.send.call_count == 1

    @pytest.mark.asyncio
    async def test_no_split_without_paragraph_break(self):
        """Long continuous text without \\n\\n stays as one message."""
        consumer, adapter = _make_consumer(no_edit_mode=True)
        consumer._no_edit_min_buffer = 50

        consumer.on_delta("A" * 300)  # Long but no paragraph breaks
        consumer.finish()

        await consumer.run()

        assert adapter.send.call_count == 1

    @pytest.mark.asyncio
    async def test_all_text_delivered_on_completion(self):
        """Even without mid-stream splitting, all text is delivered on finish."""
        consumer, adapter = _make_consumer(no_edit_mode=True)
        consumer._no_edit_min_buffer = 100

        para1 = "X" * 100
        para2 = "Y" * 100
        # Feed everything and finish immediately — no mid-stream split,
        # but all text should still arrive
        consumer.on_delta(f"{para1}\n\n{para2}")
        consumer.finish()

        await consumer.run()

        all_text = "".join(_all_sent_texts(adapter))
        assert "X" * 100 in all_text
        assert "Y" * 100 in all_text


# ===========================================================================
# Segment breaks (tool boundaries)
# ===========================================================================

class TestNoEditModeSegmentBreaks:
    """Segment breaks should still trigger flushes in no_edit_mode."""

    @pytest.mark.asyncio
    async def test_segment_break_flushes_buffer(self):
        """A tool boundary (segment break) flushes even short text."""
        consumer, adapter = _make_consumer(no_edit_mode=True)

        consumer.on_delta("Before tool call.")
        consumer.on_segment_break()
        consumer.on_delta("After tool call.")
        consumer.finish()

        await consumer.run()

        # At least 2 messages — one before and one after segment
        assert adapter.send.call_count >= 2
        texts = _all_sent_texts(adapter)
        assert any("Before tool call" in t for t in texts)
        assert any("After tool call" in t for t in texts)


# ===========================================================================
# No cursor artifacts
# ===========================================================================

class TestNoEditModeNoCursor:
    """No_edit_mode should not leak streaming cursor into messages."""

    @pytest.mark.asyncio
    async def test_no_cursor_in_output(self):
        """Signal messages should never contain the cursor character."""
        consumer, adapter = _make_consumer(no_edit_mode=True, cursor=" ▉")

        consumer.on_delta("Hello ")
        consumer.on_delta("world!")
        consumer.finish()

        await consumer.run()

        for call in adapter.send.call_args_list:
            content = call.kwargs.get("content", "")
            assert "▉" not in content, f"Cursor leaked: {content!r}"


# ===========================================================================
# Completion always delivers remaining text
# ===========================================================================

class TestNoEditModeCompletion:
    """Stream completion should always deliver remaining text."""

    @pytest.mark.asyncio
    async def test_finish_delivers_all_remaining(self):
        """finish() flushes whatever is buffered."""
        consumer, adapter = _make_consumer(no_edit_mode=True)

        consumer.on_delta("Just a tiny response")
        consumer.finish()

        await consumer.run()

        assert adapter.send.call_count == 1
        sent = _all_sent_texts(adapter)
        assert "Just a tiny response" in sent[0]

    @pytest.mark.asyncio
    async def test_finish_after_paragraph_split_sends_remainder(self):
        """After a paragraph split mid-stream, the trailing text is sent on finish."""
        consumer, adapter = _make_consumer(no_edit_mode=True)
        consumer._no_edit_min_buffer = 100

        para1 = "A" * 120
        para2 = "B" * 50
        consumer.on_delta(f"{para1}\n\n{para2}")
        consumer.finish()

        await consumer.run()

        # Both parts delivered
        all_text = "".join(_all_sent_texts(adapter))
        assert "A" * 120 in all_text
        assert "B" * 50 in all_text


# ===========================================================================
# Config flag
# ===========================================================================

class TestNoEditModeConfig:
    """Verify the config flag is wired correctly."""

    def test_config_defaults_to_false(self):
        config = StreamConsumerConfig()
        assert config.no_edit_mode is False

    def test_config_propagates_to_consumer(self):
        consumer, _ = _make_consumer(no_edit_mode=True)
        assert consumer._no_edit_mode is True

    def test_config_false_disables(self):
        consumer, _ = _make_consumer(no_edit_mode=False)
        assert consumer._no_edit_mode is False
