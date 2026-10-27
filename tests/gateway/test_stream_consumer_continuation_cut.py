"""Regression tests for the word-boundary fallback continuation cut (#116312).

Streaming edits fire on a throttle tick, not at a word boundary, so the prefix
(the last text a successful edit put on screen) can end inside a word.  When
edits then stop working, _continuation_text must back the cut up to the last
space or newline so the fresh continuation re-sends the broken word's tail and
reads as an ordinary continuation instead of starting mid-word.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig


def _make_consumer() -> GatewayStreamConsumer:
    """Minimal consumer whose send/edit surface is only exercised if a bug
    sends on the wrong field."""
    adapter = MagicMock()
    adapter.REQUIRES_EDIT_FINALIZE = False
    adapter.MAX_MESSAGE_LENGTH = 4096
    adapter.send = AsyncMock(return_value=SimpleNamespace(success=True, message_id="m"))
    adapter.edit_message = AsyncMock(return_value=SimpleNamespace(success=True, message_id="m"))
    return GatewayStreamConsumer(
        adapter=adapter,
        chat_id="chat",
        config=StreamConsumerConfig(cursor=" ▉"),
    )


class TestContinuationWordBoundary:
    """#116312 — back the fallback cut up to the last space/newline."""

    def test_mid_word_prefix_backs_up_to_last_space(self):
        # Issue reproduction: the visible preview ends mid-word ("рядкам" of
        # "рядками"); the continuation must start at the word boundary.
        consumer = _make_consumer()
        consumer._last_sent_text = "Правлю точними живими рядкам ▉"
        final_text = "Правлю точними живими рядками (мій попередній EN-паттерн…"
        assert consumer._continuation_text(final_text) == (
            "рядками (мій попередній EN-паттерн…")

    def test_one_long_token_keeps_original_cut(self):
        # No boundary to back up to: the original cut stands rather than
        # re-sending the whole message.
        consumer = _make_consumer()
        consumer._last_sent_text = "у" * 100
        final_text = ("у" * 100) + ("д" * 20)
        assert consumer._continuation_text(final_text) == "д" * 20

    def test_whole_word_prefix_is_not_resent(self):
        # The preview ends exactly on a word; the continuation must not repeat it.
        consumer = _make_consumer()
        consumer._last_sent_text = "Here is the answer ▉"
        assert consumer._continuation_text("Here is the answer to your question.") == "to your question."
