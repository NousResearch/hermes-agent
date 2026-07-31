"""Regression tests for the fresh-final-for-long-lived-previews path.

Ported from openclaw/openclaw#72038.  When a streamed preview has been
visible long enough that the platform's edit timestamp would be
noticeably stale by completion time, the stream consumer delivers the
final reply as a brand-new message and best-effort deletes the old
preview.  This makes Telegram's visible timestamp reflect completion
time instead of first-token time.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig


def _make_adapter(*, supports_delete: bool = True) -> MagicMock:
    """Build a minimal MagicMock adapter wired for send/edit/delete."""
    adapter = MagicMock()
    adapter.REQUIRES_EDIT_FINALIZE = False
    adapter.MAX_MESSAGE_LENGTH = 4096
    adapter.send = AsyncMock(return_value=SimpleNamespace(
        success=True, message_id="initial_preview",
    ))
    adapter.edit_message = AsyncMock(return_value=SimpleNamespace(
        success=True, message_id="initial_preview",
    ))
    if supports_delete:
        adapter.delete_message = AsyncMock(return_value=True)
    else:
        # Adapter without the optional delete_message method — fresh-final
        # should still work, it just leaves the stale preview in place.
        del adapter.delete_message  # type: ignore[attr-defined]
    return adapter


class TestFreshFinalForLongLivedPreviews:
    """openclaw#72038 port — send fresh final when preview is old."""

    @pytest.mark.asyncio
    async def test_disabled_by_default_still_edits_in_place(self):
        """``fresh_final_after_seconds=0`` preserves the legacy edit path."""
        adapter = _make_adapter()
        consumer = GatewayStreamConsumer(
            adapter=adapter,
            chat_id="chat",
            config=StreamConsumerConfig(fresh_final_after_seconds=0.0),
        )
        await consumer._send_or_edit("hello")
        # Pretend the preview has been visible for a long time.
        consumer._message_created_ts = 0.0  # far in the past
        await consumer._send_or_edit("hello world", finalize=True)
        # Should edit, not send a fresh message.
        assert adapter.send.call_count == 1  # only the initial send
        adapter.edit_message.assert_called_once()


    @pytest.mark.asyncio
    async def test_fresh_final_without_delete_support_is_best_effort(self):
        """Adapter lacking ``delete_message`` still gets the fresh send."""
        adapter = _make_adapter(supports_delete=False)
        adapter.send.side_effect = [
            SimpleNamespace(success=True, message_id="initial_preview"),
            SimpleNamespace(success=True, message_id="fresh_final"),
        ]
        consumer = GatewayStreamConsumer(
            adapter=adapter,
            chat_id="chat",
            config=StreamConsumerConfig(fresh_final_after_seconds=60.0),
        )
        await consumer._send_or_edit("hello")
        consumer._message_created_ts = 0.0
        await consumer._send_or_edit("hello world", finalize=True)
        assert adapter.send.call_count == 2
        adapter.edit_message.assert_not_called()
        # No delete attempt — just the fresh send.
        assert consumer._message_id == "fresh_final"


class TestSegmentBreakDoesNotMarkFinalSent:
    """Regression for #29346 — silent response loss after tool calls.

    When ``fresh_final_after_seconds > 0`` and a streamed *preamble* ("Let me
    search…") has aged past the threshold, finalizing it at a tool boundary
    used to route through ``_try_fresh_final``, which unconditionally set
    ``_final_response_sent = True`` even though this is a NON-final segment.
    The gateway (run.py:18128) then reads that flag as "final delivered" and
    suppresses the genuine final answer (which arrives on a later API call and
    does not re-stream), so the user gets nothing.

    The fix scopes the final-delivery flags to the turn-final segment and
    clears them at every tool/segment boundary, so a preamble can never mark
    the turn as delivered.
    """

    @staticmethod
    def _delivered_texts(adapter) -> list[str]:
        """Every text the adapter actually put on screen (sends + edits)."""
        texts = [c.kwargs.get("content", "") for c in adapter.send.call_args_list]
        texts += [c.kwargs.get("content", "") for c in adapter.edit_message.call_args_list]
        return texts

    @pytest.mark.asyncio
    async def test_preamble_fresh_final_at_tool_boundary_does_not_mark_final(self):
        """Real-aging reproduction (exercises the actual _should_send_fresh_final
        age gate, not a monkeypatch): a preamble ages past the threshold, then a
        tool boundary finalizes it via fresh-final.  The genuine final answer is
        produced on a later API call and is NOT streamed through this consumer
        (the #29346 repro), so the consumer must NOT believe the final was sent."""
        adapter = _make_adapter()
        consumer = GatewayStreamConsumer(
            adapter=adapter,
            chat_id="chat",
            config=StreamConsumerConfig(
                edit_interval=0.01, buffer_threshold=5, cursor=" ▉",
                fresh_final_after_seconds=0.001,  # tiny → real aging fires
            ),
        )
        consumer.on_delta("Let me search the web for that.")
        task = asyncio.create_task(consumer.run())
        await asyncio.sleep(0.05)  # preamble sent + aged well past 0.001s
        consumer.on_delta(None)  # tool boundary → segment-break fresh-final
        await asyncio.sleep(0.05)
        consumer.finish()
        await task

        # Fresh-final actually engaged (preamble preview + a fresh resend), yet
        # the turn is NOT marked delivered — no genuine final ever streamed.
        assert adapter.send.call_count >= 2
        assert consumer.final_response_sent is False
        assert consumer.final_content_delivered is False

    @pytest.mark.asyncio
    async def test_final_answer_after_preamble_is_delivered_exactly_once(self):
        """P0 user-visible contract: when the real final answer DOES stream in
        after the preamble + tool boundary, the user gets it exactly once AND
        the consumer marks it delivered (so the gateway correctly suppresses a
        redundant send)."""
        adapter = _make_adapter()
        consumer = GatewayStreamConsumer(
            adapter=adapter,
            chat_id="chat",
            config=StreamConsumerConfig(
                edit_interval=0.01, buffer_threshold=5, cursor=" ▉",
                fresh_final_after_seconds=0.001,
            ),
        )
        consumer.on_delta("Let me search the web for that.")
        task = asyncio.create_task(consumer.run())
        await asyncio.sleep(0.05)
        consumer.on_delta(None)  # tool boundary
        consumer.on_delta("The answer is 42.")  # genuine final answer streams
        await asyncio.sleep(0.05)
        consumer.finish()
        await task

        # The real final answer was delivered → suppression must engage.
        assert consumer.final_response_sent is True
        # And it reached the user exactly once (no duplicate fresh send).
        final_sends = [
            c for c in adapter.send.call_args_list
            if "answer is 42" in c.kwargs.get("content", "")
        ]
        assert len(final_sends) <= 1
        assert any("answer is 42" in t for t in self._delivered_texts(adapter))


class TestCancelledBestEffortDeliveryFinalizes:
    """Cancel-path best-effort delivery must go through the finalize path.

    The gateway cancels the consumer shortly after finish(). The
    CancelledError handler re-delivers the accumulated text; previously it
    did so with finalize=False, so REQUIRES_EDIT_FINALIZE platforms
    (Telegram) kept the plain streaming preview — the whole final reply
    rendered with raw markdown markers — while the success flags still
    suppressed the gateway's formatted re-send.
    """


    @pytest.mark.asyncio
    async def test_cancel_with_fresh_final_enabled_delivers_and_flags_via_handler(self):
        """With fresh_final_after_seconds enabled and an aged preview, the
        finalized cancel-path delivery is eligible for fresh-final
        (delete + fresh send). is_turn_final=False keeps _try_fresh_final
        from setting the flags itself; the cancel handler sets them after
        the successful delivery."""
        adapter = _make_adapter()
        adapter.REQUIRES_EDIT_FINALIZE = True
        adapter.send.side_effect = [
            SimpleNamespace(success=True, message_id="initial_preview"),
            SimpleNamespace(success=True, message_id="fresh_final"),
        ]
        consumer = GatewayStreamConsumer(
            adapter=adapter,
            chat_id="chat",
            config=StreamConsumerConfig(
                edit_interval=0.01, buffer_threshold=5, cursor=" ▉",
                fresh_final_after_seconds=0.001,
            ),
        )
        consumer.on_delta("Reply with **bold** and `code` markers.")
        task = asyncio.create_task(consumer.run())
        await asyncio.sleep(0.05)
        consumer._message_created_ts = 0.0  # force the preview stale
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

        # Fresh-final engaged: a second send replaced the stale preview.
        assert adapter.send.call_count == 2
        adapter.delete_message.assert_awaited_once_with("chat", "initial_preview")
        # Flags were set by the cancel handler after successful delivery.
        assert consumer.final_response_sent is True
        assert consumer.final_content_delivered is True


class TestGotDoneOverflowSplitNotRefinalized:
    """A got_done finalize edit that split-and-delivered across continuation
    messages must not be followed by the redundant requires-finalize edit.

    After a split, the consumer adopts the last continuation as the live
    message and the redundant finalize edit re-submits the FULL accumulated
    text against it; the adapter pre-flights that into another overflow
    split, editing chunk 1 over the continuation and re-sending the rest,
    so the user sees duplicated chunks. The finalize signal was already
    carried by the split edit itself.
    """

    def _consumer(self, adapter):
        # High interval/threshold so the only edit is the got_done finalize.
        return GatewayStreamConsumer(
            adapter=adapter,
            chat_id="chat",
            config=StreamConsumerConfig(
                edit_interval=10.0, buffer_threshold=10_000, cursor=" ▉",
            ),
        )


    @pytest.mark.asyncio
    async def test_non_split_finalize_edit_still_gets_explicit_refinalize(self):
        """The narrow fix must not regress the requires-finalize contract:
        a normal (non-split) got_done edit is still followed by the
        explicit finalize edit (#25010 semantics unchanged)."""
        adapter = _make_adapter()
        adapter.REQUIRES_EDIT_FINALIZE = True
        adapter.edit_message = AsyncMock(return_value=SimpleNamespace(
            success=True, message_id="initial_preview",
        ))
        consumer = self._consumer(adapter)
        consumer.on_delta("short final reply")
        task = asyncio.create_task(consumer.run())
        await asyncio.sleep(0.05)
        consumer.finish()
        await task

        finalize_edits = [
            c for c in adapter.edit_message.call_args_list
            if c.kwargs.get("finalize")
        ]
        assert len(finalize_edits) == 2
        assert consumer.final_response_sent is True


class TestFinalCleanupEditFloodControl:
    """Regression for duplicate final sends when the cursor-strip edit fails."""

    @pytest.mark.asyncio
    async def test_failed_final_cleanup_edit_marks_visible_content_delivered(self):
        adapter = _make_adapter()
        adapter.edit_message = AsyncMock(return_value=SimpleNamespace(
            success=False,
            error="Flood control exceeded. Retry in 12 seconds",
        ))
        consumer = GatewayStreamConsumer(
            adapter=adapter,
            chat_id="chat",
            config=StreamConsumerConfig(
                edit_interval=0.01, buffer_threshold=5, cursor=" ▉",
            ),
        )

        final_text = "The complete answer is already visible before cleanup."
        consumer.on_delta(final_text)
        task = asyncio.create_task(consumer.run())
        await asyncio.sleep(0.05)  # streaming preview lands with cursor
        assert consumer._last_sent_text == f"{final_text} ▉"

        consumer.finish()
        await task

        # The final cosmetic edit failed, so final_response_sent stays false;
        # the important signal is that content_delivered suppresses the
        # gateway's normal full final send and prevents a duplicate answer.
        assert consumer.final_response_sent is False
        assert consumer.final_content_delivered is True
        assert adapter.send.call_count == 1
        assert adapter.edit_message.call_count >= 1


class TestStreamConsumerConfigFreshFinalField:
    """The dataclass field must exist and default to 0 (disabled)."""

    def test_default_is_disabled(self):
        cfg = StreamConsumerConfig()
        assert cfg.fresh_final_after_seconds == 0.0


class TestStreamingConfigFreshFinalField:
    """The gateway-level StreamingConfig carries the setting."""


    def test_from_dict_respects_explicit_zero(self):
        from gateway.config import StreamingConfig
        cfg = StreamingConfig.from_dict({
            "enabled": True,
            "fresh_final_after_seconds": 0,
        })
        assert cfg.fresh_final_after_seconds == 0.0


class TestTelegramAdapterDeleteMessage:
    """Contract: Telegram adapter implements ``delete_message``."""

    def test_delete_message_method_exists(self):
        telegram = pytest.importorskip("plugins.platforms.telegram.adapter")
        import inspect
        cls = telegram.TelegramAdapter
        assert hasattr(cls, "delete_message"), (
            "TelegramAdapter.delete_message is required for the fresh-final "
            "cleanup path (openclaw/openclaw#72038 port)."
        )
        sig = inspect.signature(cls.delete_message)
        params = list(sig.parameters)
        assert params[:3] == ["self", "chat_id", "message_id"]

