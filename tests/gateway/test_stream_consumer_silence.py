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


@pytest.mark.parametrize("text", PARTIAL_POSITIVE)
def test_partial_silence_marker_positive(text):
    assert is_partial_silence_marker(text) is True


@pytest.mark.parametrize("text", PARTIAL_NEGATIVE)
def test_partial_silence_marker_negative(text):
    assert is_partial_silence_marker(text) is False


def test_partial_silence_marker_none_safe():
    assert is_partial_silence_marker(None) is False


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

    @pytest.mark.asyncio
    async def test_suppression_without_delete_support_is_best_effort(self):
        """Adapter lacking delete_message still suppresses (leaves no new send)."""
        adapter = _make_adapter(supports_delete=False)
        consumer = GatewayStreamConsumer(
            adapter, "chat_1",
            StreamConsumerConfig(edit_interval=0.01, buffer_threshold=1),
        )
        consumer.on_delta("NO_REPLY")
        consumer.finish()
        await consumer.run()

        for text in _sent_and_edited(adapter):
            assert "NO_REPLY" not in text
        assert consumer.final_content_delivered is False

    @pytest.mark.asyncio
    async def test_bracket_silent_marker_suppressed(self):
        """The [SILENT] marker is suppressed just like NO_REPLY."""
        adapter = _make_adapter()
        consumer = GatewayStreamConsumer(
            adapter, "chat_1",
            StreamConsumerConfig(edit_interval=0.01, buffer_threshold=1),
        )
        consumer.on_delta("[SILENT]")
        consumer.finish()
        await consumer.run()

        for text in _sent_and_edited(adapter):
            assert "[SILENT]" not in text
        assert consumer.final_content_delivered is False

    @pytest.mark.asyncio
    async def test_prose_mentioning_marker_is_delivered(self):
        """Substantive prose that merely mentions NO_REPLY is NOT suppressed."""
        adapter = _make_adapter()
        consumer = GatewayStreamConsumer(
            adapter, "chat_1",
            StreamConsumerConfig(edit_interval=0.01, buffer_threshold=5),
        )
        body = "The NO_REPLY token tells the gateway to stay silent."
        consumer.on_delta(body)
        consumer.finish()
        await consumer.run()

        delivered = "".join(_sent_and_edited(adapter))
        assert "NO_REPLY" in delivered
        assert consumer.final_content_delivered is True

    @pytest.mark.asyncio
    async def test_marker_prefix_then_prose_is_delivered(self):
        """A reply that starts marker-like but continues is delivered whole.

        "NO REPLY needed …" passes through the mid-stream hold-back while the
        buffer is still a marker prefix, then flushes normally once it diverges.
        The final text is NOT an exact marker, so got_done does not suppress it.
        """
        adapter = _make_adapter()
        consumer = GatewayStreamConsumer(
            adapter, "chat_1",
            StreamConsumerConfig(edit_interval=0.01, buffer_threshold=1),
        )
        consumer.on_delta("NO REPLY")
        consumer.on_delta(" needed — the build is already green.")
        consumer.finish()
        await consumer.run()

        delivered = "".join(_sent_and_edited(adapter))
        assert "the build is already green" in delivered
        assert consumer.final_content_delivered is True
