"""Streaming intentional-silence suppression.

When the agent chooses not to reply it emits a bare control marker
(``NO_REPLY`` / ``[SILENT]`` / …).  The gateway's whole-response filter
(``gateway/response_filters.is_intentional_silence_agent_result``) suppresses
this on the non-streaming delivery path, but the *streaming* path
(``GatewayStreamConsumer``) previously had no silence awareness: it edited the
raw marker onto the screen delta-by-delta and finalized it *before* the
whole-response filter could run.  On any streaming-capable adapter (Slack,
Telegram, Discord, …) users saw a literal ``NO_REPLY`` bubble.

These tests pin the two halves of the fix:

* ``is_partial_silence_marker`` — the mid-stream hold-back predicate.
* ``GatewayStreamConsumer`` — an exact-marker final buffer is suppressed and
  any already-shown preview is retracted, while substantive prose that merely
  mentions a marker is delivered normally.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.response_filters import (
    is_intentional_silence_response,
    is_partial_silence_marker,
)
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig


# --------------------------------------------------------------------------
# is_partial_silence_marker — mid-stream hold-back predicate
# --------------------------------------------------------------------------

# Buffers that could still resolve to a marker → held back while streaming.
PARTIAL_POSITIVE = [
    "N",
    "NO",
    "NO_",
    "NO_REP",
    "NO_REPLY",      # exact marker, not yet terminated by stream-end
    "NO REPLY",
    "no reply",      # canonicalized (case/space-insensitive)
    "  no_reply  ",  # surrounding whitespace stripped
    "[",
    "[SIL",
    "[SILENT]",
    "SILENT",
    "sil",
]

# Buffers that have already diverged from every marker → stream normally.
PARTIAL_NEGATIVE = [
    "",
    "   ",
    "No reply needed — here is the plan",   # diverged past the marker
    "NO_REPLYING",                           # superset, not a prefix
    "Nope",
    "Hello there",
    "The NO_REPLY token means silence",      # marker mentioned mid-prose
    "x" * 65,                                # over the 64-char cap
    "silence is golden",                     # 'SILENCE...' is not a marker prefix
]


def test_partial_predicate_agrees_with_exact_on_full_markers():
    """Every exact silence marker is also a (trivial) partial of itself."""
    from gateway.response_filters import LIVE_GATEWAY_SILENT_MARKERS

    for marker in LIVE_GATEWAY_SILENT_MARKERS:
        assert is_partial_silence_marker(marker) is True
        assert is_intentional_silence_response(marker) is True


# --------------------------------------------------------------------------
# GatewayStreamConsumer — end-to-end suppression through run()
# --------------------------------------------------------------------------

def _make_adapter(*, supports_delete: bool = True) -> MagicMock:
    """Minimal MagicMock adapter wired for send/edit/delete."""
    adapter = MagicMock()
    adapter.REQUIRES_EDIT_FINALIZE = False
    adapter.MAX_MESSAGE_LENGTH = 4096
    adapter.send = AsyncMock(return_value=SimpleNamespace(
        success=True, message_id="preview_1",
    ))
    adapter.edit_message = AsyncMock(return_value=SimpleNamespace(
        success=True, message_id="preview_1",
    ))
    if supports_delete:
        adapter.delete_message = AsyncMock(return_value=True)
    else:
        del adapter.delete_message  # type: ignore[attr-defined]
    return adapter


def _sent_and_edited(adapter):
    texts = []
    for call in adapter.send.call_args_list:
        texts.append(call.kwargs.get("content", ""))
    if getattr(adapter, "edit_message", None) is not None:
        for call in adapter.edit_message.call_args_list:
            texts.append(call.kwargs.get("content", ""))
    return texts


class TestStreamedSilenceSuppression:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("boundary", ["segment", "commentary"])
    @pytest.mark.parametrize("text", ["No", "\u200b\ufeff", "[SILENT]"])
    async def test_boundaries_preserve_words_but_do_not_flush_non_content(self, boundary, text):
        adapter = _make_adapter()
        consumer = GatewayStreamConsumer(
            adapter, "chat_1", StreamConsumerConfig(edit_interval=0.01, buffer_threshold=1),
        )
        consumer.on_delta(text)
        if boundary == "segment":
            consumer.on_segment_break()
        else:
            consumer.on_commentary("Checking now")
        consumer.on_delta("Final answer")
        consumer.finish("Final answer")
        await consumer.run()

        expected = ["No"] if text == "No" else []
        if boundary == "commentary":
            expected.append("Checking now")
        assert _sent_and_edited(adapter) == [*expected, "Final answer"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("transport", ["draft", "native"])
    async def test_invisible_final_keeps_cumulative_preamble(self, transport):
        from tests.gateway.test_stream_final_contract import _make_draft_adapter
        from tests.gateway.test_stream_consumer_wecom_native import _make_native_streaming_adapter

        adapter = _make_draft_adapter() if transport == "draft" else _make_native_streaming_adapter()
        consumer = GatewayStreamConsumer(
            adapter, "chat_1", StreamConsumerConfig(transport="auto", edit_interval=0.01, buffer_threshold=1, cursor=""),
        )
        consumer.on_delta("Visible preamble")
        consumer.on_segment_break()
        consumer.on_delta("\u200b")
        consumer.finish("\u200b")
        await consumer.run()

        from gateway.run import GatewayRunner

        result = {"final_response": "\u200b", "response_previewed": True}
        runner = GatewayRunner.__new__(GatewayRunner)
        await runner._run_agent_mark_streamed_delivery(result, SimpleNamespace(
            source=SimpleNamespace(chat_id="chat_1"), session_key="key",
            stream_consumer_holder=[consumer],
        ))
        assert not result.get("already_sent")
        if transport == "draft":
            assert adapter.send_calls[-1]["content"].startswith("Visible preamble")
            assert adapter.edit_calls == []
        else:
            assert adapter.frames[-1]["finalize"] is True
            assert adapter.frames[-1]["text"].startswith("Visible preamble")

    @pytest.mark.asyncio
    async def test_format_only_overflow_tail_does_not_erase_substantive_heads(self):
        import asyncio
        from tests.gateway.test_stream_final_contract import _make_draft_adapter

        adapter = _make_draft_adapter()
        adapter.MAX_MESSAGE_LENGTH = 600
        adapter.delete_message = AsyncMock(return_value=True)
        consumer = GatewayStreamConsumer(
            adapter, "chat_1", StreamConsumerConfig(edit_interval=0.01, buffer_threshold=1, cursor=""),
        )
        # The first 500 characters fill the platform budget; only format controls
        # remain in the active tail after the substantive head is sent.
        text = "x" * 500 + "\u200b" * 5
        consumer.on_delta(text[:400])
        task = asyncio.create_task(consumer.run())
        try:
            async with asyncio.timeout(2):
                while not adapter.send_calls:
                    await asyncio.sleep(0.01)
                consumer.on_delta(text[400:])
                while not adapter.edit_calls:
                    await asyncio.sleep(0.01)
            consumer.finish(text)
            await task
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        adapter.delete_message.assert_not_awaited()
        assert adapter.edit_calls[0]["content"] == "x" * 500
        assert consumer.final_content_delivered
        assert consumer.delivered_final_matches(text)

    @pytest.mark.asyncio
    async def test_invisible_only_stream_is_fully_suppressed(self):
        """Format-only output must never become a blank platform message."""
        adapter = _make_adapter()
        consumer = GatewayStreamConsumer(
            adapter, "chat_1",
            StreamConsumerConfig(edit_interval=0.01, buffer_threshold=1),
        )
        consumer.on_delta("\u200b\ufeff")
        consumer.on_segment_break()
        consumer.finish()
        await consumer.run()

        assert _sent_and_edited(adapter) == []
        assert consumer.final_response_sent is False
        assert consumer.final_content_delivered is False
        assert consumer.already_sent is False

    @pytest.mark.asyncio
    async def test_invisible_final_segment_preserves_visible_preamble(self):
        adapter = _make_adapter()
        consumer = GatewayStreamConsumer(
            adapter, "chat_1",
            StreamConsumerConfig(edit_interval=0.01, buffer_threshold=1),
        )
        consumer.on_delta("Visible preamble")
        consumer.on_segment_break()
        consumer.on_delta("\u200b")
        consumer.finish()
        await consumer.run()

        assert _sent_and_edited(adapter) == ["Visible preamble"]
        adapter.delete_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_reply_only_stream_is_fully_suppressed(self):
        """A stream whose entire content is NO_REPLY sends nothing visible."""
        adapter = _make_adapter()
        consumer = GatewayStreamConsumer(
            adapter, "chat_1",
            StreamConsumerConfig(edit_interval=0.01, buffer_threshold=1),
        )
        consumer.on_delta("NO_REPLY")
        consumer.finish()
        await consumer.run()

        # No marker text ever reached the platform.
        for text in _sent_and_edited(adapter):
            assert "NO_REPLY" not in text, f"marker leaked: {text!r}"

        # Delivery flags stay False so the gateway does not treat the marker
        # as a delivered reply (its whole-response filter then drops it too).
        assert consumer.final_response_sent is False
        assert consumer.final_content_delivered is False
        assert consumer.already_sent is False

    @pytest.mark.asyncio
    async def test_partial_marker_preview_is_retracted(self):
        """A marker flushed mid-stream as a preview is deleted on completion."""
        adapter = _make_adapter()
        consumer = GatewayStreamConsumer(
            adapter, "chat_1",
            StreamConsumerConfig(edit_interval=0.01, buffer_threshold=1),
        )
        # Force a mid-stream preview: pretend "NO_REPLY" was already put on
        # screen (the pre-fix behaviour) before got_done runs.
        consumer._message_id = "preview_1"
        consumer._preview_message_ids = {"preview_1"}
        consumer._already_sent = True

        consumer.on_delta("NO_REPLY")
        consumer.finish()
        await consumer.run()

        # The stale preview was best-effort deleted.
        adapter.delete_message.assert_awaited_once_with("chat_1", "preview_1")
        assert consumer.final_content_delivered is False
        assert consumer.already_sent is False
