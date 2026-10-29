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


# --------------------------------------------------------------------------
# Reply written beside a housekeeping tool, then a bare marker follow-up
# --------------------------------------------------------------------------

ANSWER = "Your meeting is at 3pm."


def _numbered_adapter():
    """Edit-transport adapter whose sends get fresh ids, so segments are distinct messages."""
    adapter = _make_adapter()
    counter = iter(range(1, 100))

    async def _send(**kwargs):
        return SimpleNamespace(success=True, message_id=f"m{next(counter)}")

    adapter.send = AsyncMock(side_effect=_send)
    return adapter


def _visible(adapter) -> list:
    shown = {}
    for call in adapter.send.call_args_list:
        shown[f"m{len(shown) + 1}"] = call.kwargs.get("content", "")
    for call in adapter.edit_message.call_args_list:
        shown[call.kwargs.get("message_id")] = call.kwargs.get("content", "")
    for call in adapter.delete_message.call_args_list:
        shown.pop(call.args[1], None)
    return list(shown.values())


async def _stream_answer_then_marker(final_text):
    adapter = _numbered_adapter()
    consumer = GatewayStreamConsumer(
        adapter, "chat_1", StreamConsumerConfig(edit_interval=0.01, buffer_threshold=1),
    )
    consumer.on_delta(ANSWER)
    consumer.on_delta(None)  # housekeeping tool boundary
    consumer.on_delta("NO_REPLY")
    consumer.finish(final_text)
    await consumer.run()
    return adapter, consumer


class TestMarkerAfterStreamedReply:
    @pytest.mark.asyncio
    async def test_reply_final_after_marker_shows_the_answer_once(self):
        """The gateway hands back the already-streamed reply line as the final: it stays
        visible exactly once, and the held-back marker segment does not repeat it."""
        adapter, consumer = await _stream_answer_then_marker(ANSWER)

        assert _visible(adapter) == [ANSWER]
        adapter.delete_message.assert_not_awaited()
        assert not any("NO_REPLY" in text for text in _sent_and_edited(adapter))
        # The gateway's dedupe sees the reply as delivered and sends nothing more.
        assert consumer.has_durably_delivered_text(ANSWER) is True

    @pytest.mark.asyncio
    async def test_marker_final_still_retracts_the_turn(self):
        """A turn that stays silent (final is the marker) still retracts its streamed
        preamble; the gateway's whole-response filter owns what goes out next."""
        adapter, consumer = await _stream_answer_then_marker("NO_REPLY")

        assert _visible(adapter) == []
        assert consumer.final_content_delivered is False


def _draft_adapter():
    from tests.gateway.test_stream_consumer_draft import _make_draft_capable_adapter

    adapter = _make_draft_capable_adapter()
    adapter.answer_copies = lambda: [c.kwargs.get("content") for c in adapter.send.call_args_list].count(ANSWER)
    adapter.marker_shown = lambda: any(
        "NO_REPLY" in (text or "") for text in
        [c.kwargs.get("content") for c in adapter.send.call_args_list] + [d["content"] for d in adapter.draft_calls])
    return adapter


def _native_adapter():
    from tests.gateway.test_stream_consumer_wecom_native import _make_native_streaming_adapter

    adapter = _make_native_streaming_adapter()
    adapter.answer_copies = lambda: sum(1 for f in adapter.frames if f["finalize"] and f["text"] == ANSWER)
    adapter.marker_shown = lambda: any("NO_REPLY" in f["text"] for f in adapter.frames)
    return adapter


@pytest.mark.asyncio
@pytest.mark.parametrize("make_adapter", [_draft_adapter, _native_adapter], ids=["draft", "native"])
@pytest.mark.parametrize("final_text", [ANSWER, "NO_REPLY"], ids=["reply-final", "marker-final"])
async def test_marker_after_reply_on_draft_and_native_transports(make_adapter, final_text):
    """Draft and native transports: a reply final after the held-back marker is
    delivered once; a marker final keeps today's silent-turn handling."""
    adapter = make_adapter()
    consumer = GatewayStreamConsumer(
        adapter, "chat_1", StreamConsumerConfig(transport="auto", chat_type="dm", cursor="",
                                                edit_interval=0.01, buffer_threshold=1),
    )
    consumer.on_delta(ANSWER)
    consumer.on_delta(None)  # housekeeping tool boundary
    consumer.on_delta("NO_REPLY")
    consumer.finish(final_text)
    await consumer.run()

    assert not adapter.marker_shown()
    if final_text == ANSWER:
        assert adapter.answer_copies() == 1
        assert consumer.has_durably_delivered_text(ANSWER) is True
    else:
        assert consumer.final_content_delivered is False
