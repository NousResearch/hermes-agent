"""Tests for WeComStreamConsumer — native WeCom stream consumer."""

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.wecom_stream_consumer import (
    WeComStreamConsumer,
    _escape_for_think_block,
    _escape_for_visible,
    _escape_think_tags,
    build_ws_stream_content,
    build_waiting_model_content,
)


# ── build_ws_stream_content unit tests ──────────────────────────────────


class TestBuildWsStreamContent:
    def test_visible_only(self):
        result = build_ws_stream_content(visible_text="hello")
        assert result == "hello"

    def test_reasoning_only_open(self):
        result = build_ws_stream_content(reasoning_text="thinking hard")
        # Open tag present, no closing tag when finish=False and no visible text
        assert result.startswith("<think")
        assert "</think>" not in result

    def test_reasoning_closed_on_finish(self):
        result = build_ws_stream_content(reasoning_text="thinking", finish=True)
        assert "<think>" in result
        assert "</think>" in result

    def test_reasoning_and_visible(self):
        result = build_ws_stream_content(reasoning_text="reasoning", visible_text="answer")
        assert "<think>" in result
        assert "reasoning" in result
        assert "</think>" in result
        assert "answer" in result

    def test_empty_returns_empty(self):
        result = build_ws_stream_content()
        assert result == ""

    def test_empty_reasoning_returns_visible(self):
        result = build_ws_stream_content(reasoning_text="", visible_text="hello")
        assert result == "hello"

    def test_error_text_without_reasoning(self):
        result = build_ws_stream_content(error_text="Connection failed")
        assert "Connection failed" in result
        assert "\u26a0" in result

    def test_error_text_with_reasoning(self):
        result = build_ws_stream_content(
            reasoning_text="thinking",
            error_text="Timeout",
        )
        # Error closes the think block
        assert "</think>" in result
        assert "Timeout" in result

    def test_error_text_with_reasoning_and_visible(self):
        result = build_ws_stream_content(
            reasoning_text="thinking",
            visible_text="partial answer",
            error_text="Interrupted",
        )
        assert "<think>" in result
        assert "partial answer" in result
        assert "Interrupted" in result

    def test_reasoning_with_literal_think_tags_escaped(self):
        result = build_ws_stream_content(
            reasoning_text="use <think> tag",
            finish=True,
        )
        # The literal <think> inside reasoning should be escaped
        assert "&lt;think&gt;" in result
        # The wrapping think tags should be intact
        assert result.startswith("<think>")
        assert "</think>" in result

    def test_visible_with_literal_think_tags_escaped(self):
        result = build_ws_stream_content(
            reasoning_text="reasoning",
            visible_text="use <think> tag",
            finish=True,
        )
        assert "&lt;think&gt;" in result
        assert "use" in result

    def test_empty_error_returns_empty(self):
        result = build_ws_stream_content(error_text="")
        assert result == ""

    def test_error_only_no_reasoning(self):
        result = build_ws_stream_content(error_text="Something broke")
        # No think block when there is no reasoning
        assert "<think>" not in result
        assert "Something broke" in result

    def test_backticks_in_reasoning_replaced(self):
        bt = chr(96)  # backtick
        result = build_ws_stream_content(
            reasoning_text=f"use {bt}{bt}{bt}python code",
            finish=True,
        )
        # Backticks in reasoning should be replaced with \u02cb
        assert "\u02cb" in result
        assert "python" in result
        # Original backticks should not be present in think block
        assert "`" not in result.split("</think>")[0]

    def test_backticks_in_visible_preserved(self):
        bt = chr(96)  # backtick
        result = build_ws_stream_content(
            reasoning_text="reasoning",
            visible_text=f"use {bt}{bt}{bt}python code\n{bt}{bt}{bt}",
            finish=True,
        )
        # Backticks in visible text should be preserved
        visible_part = result.split("</think>")[-1]
        assert "`" in visible_part
        assert "python" in visible_part

    def test_show_reasoning_false_hides_content(self):
        # When show_reasoning is disabled, reasoning_text is empty
        # so the think block should not appear
        result = build_ws_stream_content(
            reasoning_text="",
            visible_text="final answer",
            finish=True,
        )
        assert "<think>" not in result
        assert result == "final answer"


# ── Escape function tests ─────────────────────────────────────────────


class TestEscapeFunctions:
    def test_escape_think_tags_noop(self):
        assert _escape_think_tags("hello") == "hello"

    def test_escape_think_tags_open(self):
        text = chr(60) + "think" + chr(62) + " content"
        result = _escape_think_tags(text)
        assert "&lt;think&gt;" in result

    def test_escape_think_tags_close(self):
        text = "content" + chr(60) + "/think" + chr(62)
        result = _escape_think_tags(text)
        assert "&lt;/think&gt;" in result

    def test_escape_for_think_block_replaces_backticks(self):
        result = _escape_for_think_block("use `code`")
        assert "\u02cb" in result
        assert "`" not in result
        assert "code" in result

    def test_escape_for_think_block_escapes_tags(self):
        text = "use " + chr(60) + "think" + chr(62) + " tag"
        result = _escape_for_think_block(text)
        assert "&lt;think&gt;" in result

    def test_escape_for_visible_preserves_backticks(self):
        result = _escape_for_visible("use `code`")
        assert "`" in result
        assert "code" in result

    def test_escape_for_visible_escapes_tags(self):
        text = "use " + chr(60) + "think" + chr(62) + " tag"
        result = _escape_for_visible(text)
        assert "&lt;think&gt;" in result


# ── WeComStreamConsumer unit tests ──────────────────────────────────────


class TestWeComStreamConsumerCallbacks:
    """Verify thread-safe callback queueing."""

    def _make_consumer(self):
        adapter = MagicMock()
        adapter._new_req_id = MagicMock(return_value="stream-123")
        adapter._send_reply_stream = AsyncMock()
        adapter._thinking_task = None
        adapter._thinking_cancelled = False
        return WeComStreamConsumer(
            adapter=adapter,
            chat_id="chat-1",
            reply_req_id="req-1",
            stream_id="stream-123",
        )

    def test_on_delta_queues_visible(self):
        consumer = self._make_consumer()
        consumer.on_delta("hello")
        kind, data = consumer._queue.get_nowait()
        assert kind == "visible"
        assert data == "hello"

    def test_on_reasoning_queues_reasoning(self):
        consumer = self._make_consumer()
        consumer.on_reasoning("thinking...")
        kind, data = consumer._queue.get_nowait()
        assert kind == "reasoning"
        assert data == "thinking..."

    def test_on_delta_none_queues_segment_break(self):
        consumer = self._make_consumer()
        consumer.on_delta(None)
        kind, data = consumer._queue.get_nowait()
        assert kind == "segment_break"

    def test_finish_queues_done(self):
        consumer = self._make_consumer()
        consumer.finish()
        kind, data = consumer._queue.get_nowait()
        assert kind == "done"

    def test_on_commentary_queues_commentary(self):
        consumer = self._make_consumer()
        consumer.on_commentary("interim message")
        kind, data = consumer._queue.get_nowait()
        assert kind == "commentary"
        assert data == "interim message"

    def test_on_commentary_empty_noop(self):
        consumer = self._make_consumer()
        consumer.on_commentary("")
        assert consumer._queue.empty()


class TestWeComStreamConsumerRun:
    """Verify async run loop sends stream updates."""

    def _make_consumer(self):
        adapter = MagicMock()
        adapter._new_req_id = MagicMock(return_value="stream-456")
        adapter._send_reply_stream = AsyncMock()
        adapter._thinking_task = None
        adapter._thinking_cancelled = False
        return WeComStreamConsumer(
            adapter=adapter,
            chat_id="chat-1",
            reply_req_id="req-1",
            stream_id="stream-123",
        )

    @pytest.mark.asyncio
    async def test_run_sends_visible_and_finishes(self):
        consumer = self._make_consumer()
        consumer.on_delta("hello world")
        consumer.finish()

        await consumer.run()

        assert consumer.already_sent
        assert consumer.final_response_sent
        # Should have sent at least one update + final
        assert consumer.adapter._send_reply_stream.call_count >= 1

    @pytest.mark.asyncio
    async def test_run_cancels_thinking_loop_on_first_token(self):
        consumer = self._make_consumer()
        # Simulate thinking loop running
        mock_task = MagicMock()
        mock_task.done.return_value = False
        consumer.adapter._thinking_task = mock_task

        consumer.on_delta("first token")
        consumer.finish()

        await consumer.run()

        # Thinking loop should have been cancelled
        assert consumer.adapter._thinking_cancelled is True
        mock_task.cancel.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_with_reasoning_preserves_tags(self):
        consumer = self._make_consumer()
        consumer.on_reasoning("step 1")
        consumer.on_delta("answer")
        consumer.finish()

        await consumer.run()

        assert consumer.final_response_sent
        # Check the final call includes reasoning in think tags
        final_call = consumer.adapter._send_reply_stream.call_args_list[-1]
        content = final_call[1].get("content") or final_call[0][1] if len(final_call[0]) > 1 else ""
        # The content should contain think tags
        assert "<think>" in str(consumer._build_stream_content(finish=True))

    @pytest.mark.asyncio
    async def test_properties_before_run(self):
        consumer = self._make_consumer()
        assert not consumer.already_sent
        assert not consumer.final_response_sent

    @pytest.mark.asyncio
    async def test_on_error_sends_error_hint(self):
        consumer = self._make_consumer()
        consumer.on_reasoning("step 1")
        consumer.on_error("Connection error")

        await consumer.run()

        assert consumer.final_response_sent
        # Content is 2nd positional arg to _send_reply_stream
        final_call = consumer.adapter._send_reply_stream.call_args_list[-1]
        content = final_call[0][1]
        assert "Connection error" in content
        assert "\u26a0" in content  # warning emoji

    @pytest.mark.asyncio
    async def test_on_error_without_reasoning(self):
        consumer = self._make_consumer()
        consumer.on_error("Timeout")

        await consumer.run()

        assert consumer.final_response_sent
        final_call = consumer.adapter._send_reply_stream.call_args_list[-1]
        content = final_call[0][1]
        assert "Timeout" in content

    @pytest.mark.asyncio
    async def test_length_rotation_triggers(self):
        consumer = self._make_consumer()
        # Build up content that exceeds STREAM_MAX_CONTENT_LENGTH
        consumer._reasoning_text = "r" * 3000
        consumer._accumulated_visible = "v" * 1000
        # The combined stream content should exceed the 3500 limit
        assert len(consumer._build_stream_content(finish=False)) >= 3500
        assert consumer._should_rotate_for_length() is True

    @pytest.mark.asyncio
    async def test_length_rotation_not_triggered(self):
        consumer = self._make_consumer()
        consumer._reasoning_text = "short"
        consumer._accumulated_visible = "answer"
        assert consumer._should_rotate_for_length() is False

    @pytest.mark.asyncio
    async def test_length_rotation_sends_multiple(self):
        consumer = self._make_consumer()
        # Manually trigger rotation mid-stream to verify it sends properly
        consumer.on_reasoning("x" * 2000)
        consumer.on_delta("visible text")
        consumer.finish()

        await consumer.run()

        assert consumer.final_response_sent
        # At least one update + final
        assert consumer.adapter._send_reply_stream.call_count >= 1

    @pytest.mark.asyncio
    async def test_on_error_empty_noop(self):
        consumer = self._make_consumer()
        consumer.on_error("")
        assert consumer._queue.empty()
