"""Tests for GatewayStreamConsumer iteration boundary behaviour.

Covers the bug where resumed text after tool calls was concatenated
directly onto prior streamed content with no separator (issue #2177).
"""

import queue
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig, _DONE


def _make_consumer(accumulated=""):
    adapter = MagicMock()
    adapter.MAX_MESSAGE_LENGTH = 4096
    adapter.send = AsyncMock(return_value=MagicMock(success=True, message_id="msg_1"))
    adapter.edit_message = AsyncMock(return_value=MagicMock(success=True))
    consumer = GatewayStreamConsumer(
        adapter=adapter,
        chat_id="chat_123",
        config=StreamConsumerConfig(edit_interval=0.0, buffer_threshold=1),
    )
    consumer._accumulated = accumulated
    return consumer


class TestOnDeltaBasic:
    def test_text_enqueued(self):
        consumer = _make_consumer()
        consumer.on_delta("hello")
        assert consumer._queue.get_nowait() == "hello"

    def test_empty_string_not_enqueued(self):
        consumer = _make_consumer()
        consumer.on_delta("")
        assert consumer._queue.empty()

    def test_none_no_separator_when_empty(self):
        consumer = _make_consumer(accumulated="")
        consumer.on_delta(None)
        assert consumer._queue.empty()


class TestOnDeltaBoundary:
    def test_none_injects_newline_when_no_trailing_newline(self):
        """Core regression test for issue #2177."""
        consumer = _make_consumer(accumulated="Let me check the issues.")
        consumer.on_delta(None)
        item = consumer._queue.get_nowait()
        assert item == "\n", f"Expected newline separator, got {item!r}"

    def test_none_no_newline_when_already_ends_with_newline(self):
        consumer = _make_consumer(accumulated="Let me check.\n")
        consumer.on_delta(None)
        assert consumer._queue.empty()

    def test_resumed_text_follows_separator(self):
        consumer = _make_consumer(accumulated="Let me check the issues.")
        consumer.on_delta(None)
        consumer.on_delta("Found 3 issues.")
        newline = consumer._queue.get_nowait()
        resumed = consumer._queue.get_nowait()
        assert newline == "\n"
        assert resumed == "Found 3 issues."


class TestFinish:
    def test_finish_puts_done_sentinel(self):
        consumer = _make_consumer()
        consumer.finish()
        assert consumer._queue.get_nowait() is _DONE


class TestAlreadySent:
    def test_initially_false(self):
        consumer = _make_consumer()
        assert consumer.already_sent is False
