"""Tests for native draft streaming in GatewayStreamConsumer.

Telegram Bot API 9.5 (March 2026) introduced sendMessageDraft for native
animated streaming previews in private chats.  This test suite covers the
consumer's transport-selection, fallback, and tool-boundary handling for
that path.

Adapter under test is a runtime subclass of BasePlatformAdapter that
overrides supports_draft_streaming + send_draft, since the consumer's
isinstance(BasePlatformAdapter) gate excludes plain MagicMocks.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.stream_consumer import (
    GatewayStreamConsumer,
    StreamConsumerConfig,
)


def _make_draft_capable_adapter(
    *, supports_draft: bool = True, draft_succeeds: bool = True,
):
    """Build a minimal BasePlatformAdapter subclass with draft support.

    The runtime subclass + cleared __abstractmethods__ pattern lets us
    construct an adapter without hauling in any platform's heavy state
    (Telegram bot, Discord client, etc.) while still satisfying the
    consumer's isinstance(BasePlatformAdapter) gate.
    """
    from gateway.platforms.base import BasePlatformAdapter, SendResult

    DraftCapableAdapter = type(
        "DraftCapableAdapter",
        (BasePlatformAdapter,),
        {"MAX_MESSAGE_LENGTH": 4096},
    )
    DraftCapableAdapter.__abstractmethods__ = frozenset()
    adapter = DraftCapableAdapter.__new__(DraftCapableAdapter)
    adapter._typing_paused = set()
    adapter._fatal_error_message = None

    # Track every send_draft call for assertions.
    adapter.draft_calls = []

    def _supports(chat_type=None, metadata=None):
        return bool(supports_draft) and (chat_type or "").lower() == "dm"
    adapter.supports_draft_streaming = _supports

    async def _send_draft(*, chat_id, draft_id, content, metadata=None):
        adapter.draft_calls.append({
            "chat_id": chat_id,
            "draft_id": draft_id,
            "content": content,
            "metadata": metadata,
        })
        if draft_succeeds:
            return SendResult(success=True, message_id=None)
        return SendResult(success=False, error="draft_rejected")
    adapter.send_draft = _send_draft

    # send / edit_message: count and return canned successes so the
    # consumer's first-send + finalize paths work when drafts fall back
    # or when delivering the final message.
    adapter.send = AsyncMock(
        return_value=SimpleNamespace(success=True, message_id="msg_real"),
    )
    adapter.edit_message = AsyncMock(
        return_value=SimpleNamespace(success=True),
    )
    return adapter


class TestDraftTransportSelection:
    """Verify _resolve_draft_streaming picks the right transport."""

    def test_default_transport_stays_on_edit(self):
        adapter = _make_draft_capable_adapter()
        consumer = GatewayStreamConsumer(adapter, "12345", StreamConsumerConfig(chat_type="dm"))
        assert consumer._resolve_draft_streaming() is False

    def test_auto_dm_with_draft_capable_adapter_picks_draft(self):
        adapter = _make_draft_capable_adapter()
        cfg = StreamConsumerConfig(transport="auto", chat_type="dm")
        consumer = GatewayStreamConsumer(adapter, "12345", cfg)
        assert consumer._resolve_draft_streaming() is True

    def test_auto_group_falls_back_to_edit(self):
        adapter = _make_draft_capable_adapter()
        cfg = StreamConsumerConfig(transport="auto", chat_type="group")
        consumer = GatewayStreamConsumer(adapter, "12345", cfg)
        assert consumer._resolve_draft_streaming() is False

    def test_explicit_edit_never_uses_drafts(self):
        adapter = _make_draft_capable_adapter()
        cfg = StreamConsumerConfig(transport="edit", chat_type="dm")
        consumer = GatewayStreamConsumer(adapter, "12345", cfg)
        assert consumer._resolve_draft_streaming() is False

    def test_explicit_draft_unsupported_falls_back(self):
        adapter = _make_draft_capable_adapter(supports_draft=False)
        cfg = StreamConsumerConfig(transport="draft", chat_type="dm")
        consumer = GatewayStreamConsumer(adapter, "12345", cfg)
        assert consumer._resolve_draft_streaming() is False

    def test_magicmock_adapter_falls_back_to_edit(self):
        """MagicMock adapters (used in many existing tests) must default to
        edit-based since their auto-attributes aren't real callables."""
        adapter = MagicMock()
        cfg = StreamConsumerConfig(transport="auto", chat_type="dm")
        consumer = GatewayStreamConsumer(adapter, "12345", cfg)
        assert consumer._resolve_draft_streaming() is False


class TestDraftStreamingHappyPath:
    """End-to-end: stream a few deltas in a DM, verify drafts animated and
    the final message was delivered as a real sendMessage."""

    @pytest.mark.asyncio
    async def test_dm_stream_animates_draft_then_finalizes_with_send(self):
        adapter = _make_draft_capable_adapter()
        cfg = StreamConsumerConfig(
            transport="auto", chat_type="dm",
            edit_interval=0.01, buffer_threshold=5, cursor="",
        )
        consumer = GatewayStreamConsumer(adapter, "12345", cfg)

        consumer.on_delta("Hello ")
        task = asyncio.create_task(consumer.run())
        await asyncio.sleep(0.05)
        consumer.on_delta("world!")
        await asyncio.sleep(0.05)
        consumer.finish()
        await task

        # At least one draft frame landed.
        assert len(adapter.draft_calls) >= 1, (
            "expected at least one send_draft frame"
        )
        # Final draft frame held the full accumulated text.
        assert adapter.draft_calls[-1]["content"] == "Hello world!"
        # All draft frames in this run shared a single draft_id (animation).
        draft_ids = {c["draft_id"] for c in adapter.draft_calls}
        assert len(draft_ids) == 1
        # Final answer was delivered as a regular sendMessage so the user
        # sees a real message in their history (drafts have no message_id).
        adapter.send.assert_awaited()
        # And the final send carried the complete reply.
        final_call = adapter.send.call_args
        sent_content = (
            final_call.kwargs.get("content")
            if "content" in final_call.kwargs
            else final_call.args[1] if len(final_call.args) > 1 else None
        )
        assert sent_content == "Hello world!"

    @pytest.mark.asyncio
    async def test_group_chat_skips_draft_path(self):
        adapter = _make_draft_capable_adapter()
        cfg = StreamConsumerConfig(
            transport="auto", chat_type="group",
            edit_interval=0.01, buffer_threshold=5, cursor="",
        )
        consumer = GatewayStreamConsumer(adapter, "67890", cfg)

        consumer.on_delta("Group message")
        task = asyncio.create_task(consumer.run())
        await asyncio.sleep(0.05)
        consumer.finish()
        await task

        # Group chats skip drafts entirely — no send_draft calls at all.
        assert adapter.draft_calls == []
        # Edit-based path delivered via send (first message).
        adapter.send.assert_awaited()


class TestDraftFallbackOnFailure:
    """When a draft frame fails, the consumer disables drafts for the rest
    of the response and continues via the edit-based path."""

    @pytest.mark.asyncio
    async def test_first_draft_failure_disables_drafts_for_run(self):
        adapter = _make_draft_capable_adapter(draft_succeeds=False)
        cfg = StreamConsumerConfig(
            transport="auto", chat_type="dm",
            edit_interval=0.01, buffer_threshold=5, cursor="",
        )
        consumer = GatewayStreamConsumer(adapter, "12345", cfg)

        consumer.on_delta("Hello ")
        task = asyncio.create_task(consumer.run())
        await asyncio.sleep(0.05)
        consumer.on_delta("world!")
        await asyncio.sleep(0.05)
        consumer.finish()
        await task

        # The consumer attempted draft, hit failure, disabled drafts.
        assert consumer._draft_failures >= 1
        assert consumer._use_draft_streaming is False
        # Final message delivered via the regular send path.
        adapter.send.assert_awaited()


class TestDraftIdLifecycle:
    """Each response gets its own draft_id (no animation collision across
    consecutive responses to the same chat)."""

    @pytest.mark.asyncio
    async def test_consecutive_responses_use_distinct_draft_ids(self):
        adapter = _make_draft_capable_adapter()
        cfg1 = StreamConsumerConfig(
            transport="auto", chat_type="dm",
            edit_interval=0.01, buffer_threshold=5, cursor="",
        )
        consumer1 = GatewayStreamConsumer(adapter, "12345", cfg1)
        consumer1.on_delta("First reply")
        task1 = asyncio.create_task(consumer1.run())
        await asyncio.sleep(0.05)
        consumer1.finish()
        await task1

        cfg2 = StreamConsumerConfig(
            transport="auto", chat_type="dm",
            edit_interval=0.01, buffer_threshold=5, cursor="",
        )
        consumer2 = GatewayStreamConsumer(adapter, "12345", cfg2)
        consumer2.on_delta("Second reply")
        task2 = asyncio.create_task(consumer2.run())
        await asyncio.sleep(0.05)
        consumer2.finish()
        await task2

        # Two responses → two distinct draft_ids.
        all_ids = {c["draft_id"] for c in adapter.draft_calls}
        assert len(all_ids) >= 2, (
            f"expected distinct draft_ids across responses; got {all_ids}"
        )
        # Every draft_id must be non-zero (Telegram's contract).
        assert all(did != 0 for did in all_ids)

    @pytest.mark.asyncio
    async def test_tool_boundary_bumps_draft_id(self):
        """After a segment break (tool boundary), the next text segment
        animates via a new draft_id so it appears below the tool-progress
        bubble rather than overwriting the prior segment's preview."""
        adapter = _make_draft_capable_adapter()
        cfg = StreamConsumerConfig(
            transport="auto", chat_type="dm",
            edit_interval=0.01, buffer_threshold=5, cursor="",
        )
        consumer = GatewayStreamConsumer(adapter, "12345", cfg)

        consumer.on_delta("Pre-tool ")
        task = asyncio.create_task(consumer.run())
        await asyncio.sleep(0.05)
        # Tool boundary
        consumer.on_segment_break()
        await asyncio.sleep(0.05)
        consumer.on_delta("Post-tool")
        await asyncio.sleep(0.05)
        consumer.finish()
        await task

        # Pre-tool and post-tool segments must use different draft_ids.
        draft_ids = [c["draft_id"] for c in adapter.draft_calls]
        if len(draft_ids) >= 2:
            # Find pre-tool and post-tool calls by content
            pre_ids = {
                c["draft_id"] for c in adapter.draft_calls
                if "Pre-tool" in c["content"] and "Post-tool" not in c["content"]
            }
            post_ids = {
                c["draft_id"] for c in adapter.draft_calls
                if "Post-tool" in c["content"]
            }
            if pre_ids and post_ids:
                assert pre_ids.isdisjoint(post_ids), (
                    f"pre-tool and post-tool segments must use distinct "
                    f"draft_ids; got pre={pre_ids} post={post_ids}"
                )


class TestAlreadySentInDraftMode:
    """Drafts must NOT mark _already_sent — that flag gates the gateway's
    fallback final-send path, which we still need to fire so the user gets
    a real message in their history (drafts have no message_id)."""

    @pytest.mark.asyncio
    async def test_drafts_do_not_set_already_sent_until_real_message(self):
        adapter = _make_draft_capable_adapter()
        cfg = StreamConsumerConfig(
            transport="auto", chat_type="dm",
            edit_interval=0.01, buffer_threshold=5, cursor="",
        )
        consumer = GatewayStreamConsumer(adapter, "12345", cfg)

        consumer.on_delta("Hello")
        # Drive the consumer for a bit but DON'T finish — only drafts have
        # been sent.
        task = asyncio.create_task(consumer.run())
        await asyncio.sleep(0.05)
        # At this point drafts may have fired but we haven't finalized.
        # _already_sent must still be False so a downstream fallback would
        # know it needs to deliver the final answer.
        if adapter.draft_calls:
            assert consumer._already_sent is False, (
                "drafts wrongly marked _already_sent — "
                "would suppress gateway fallback delivery"
            )

        consumer.finish()
        await task

        # After the regular sendMessage finalize, _already_sent is True.
        assert consumer._already_sent is True


def _make_fresh_final_adapter():
    """Build a non-draft adapter that prefers a fresh final send.

    Mirrors Telegram's rich-message contract: REQUIRES_EDIT_FINALIZE so the
    final tick is routed through even when the text is unchanged, and
    prefers_fresh_final_streaming() True so the consumer delivers the final
    answer via a *fresh* send + preview delete instead of an edit.

    ``send`` returns two distinct ids so the test can tell the preview
    (first send) from the fresh final (second send) and assert the preview
    is the one deleted.
    """
    from gateway.platforms.base import BasePlatformAdapter, SendResult

    FreshFinalAdapter = type(
        "FreshFinalAdapter",
        (BasePlatformAdapter,),
        {"MAX_MESSAGE_LENGTH": 4096, "REQUIRES_EDIT_FINALIZE": True},
    )
    FreshFinalAdapter.__abstractmethods__ = frozenset()
    adapter = FreshFinalAdapter.__new__(FreshFinalAdapter)
    adapter._typing_paused = set()
    adapter._fatal_error_message = None

    # Edit-based path only — no native drafts.
    adapter.supports_draft_streaming = lambda chat_type=None, metadata=None: False
    # Accepts the metadata kwarg the consumer passes; ignores it (like Telegram).
    adapter.prefers_fresh_final_streaming = lambda content, metadata=None: True

    adapter.send = AsyncMock(side_effect=[
        SendResult(success=True, message_id="preview1"),
        SendResult(success=True, message_id="final1"),
    ])
    adapter.edit_message = AsyncMock(return_value=SendResult(success=True))
    adapter.delete_message = AsyncMock(return_value=True)
    return adapter


class TestAdapterPrefersFreshFinal:
    """An adapter whose send path is richer than its edit path (e.g. Telegram
    rich messages) finalizes a streamed reply by sending a fresh final message
    and deleting the preview, instead of final-editing the preview."""

    @pytest.mark.asyncio
    async def test_edit_stream_finalizes_with_fresh_send_and_deletes_preview(self):
        adapter = _make_fresh_final_adapter()
        cfg = StreamConsumerConfig(
            transport="auto", chat_type="dm",
            edit_interval=0.01, buffer_threshold=5, cursor="",
            fresh_final_after_seconds=0.0,  # only the adapter hook drives fresh-final
        )
        consumer = GatewayStreamConsumer(adapter, "12345", cfg)

        consumer.on_delta("Full answer here")
        task = asyncio.create_task(consumer.run())
        # Let the first send land so a real preview message_id exists before
        # finalization — the fresh-final path only engages with a live preview.
        await asyncio.sleep(0.05)
        consumer.finish()
        await task

        # Two sends: the streaming preview, then the fresh final.
        assert adapter.send.await_count == 2
        first_content = adapter.send.call_args_list[0].kwargs.get("content")
        second_content = adapter.send.call_args_list[1].kwargs.get("content")
        # First update delivered the preview via adapter.send.
        assert first_content == "Full answer here"
        # Finalization re-sent the same completed content as a fresh message.
        assert second_content == "Full answer here"

        # The edit path must NOT be used to finalize a rich preview.
        adapter.edit_message.assert_not_called()

        # The stale preview is best-effort deleted (by its id, not the final's).
        adapter.delete_message.assert_awaited_once_with("12345", "preview1")

        assert consumer.final_response_sent is True


def _make_rich_capable_adapter(*, overflow_limit=32768, send_results=None):
    """Non-draft adapter that mimics Telegram rich messages: REQUIRES_EDIT_FINALIZE,
    prefers a fresh (rich) final send, and reports a 32,768 streaming overflow
    limit so the consumer doesn't pre-split a reply that fits one rich message.
    """
    from gateway.platforms.base import BasePlatformAdapter, SendResult

    RichAdapter = type(
        "RichCapableAdapter",
        (BasePlatformAdapter,),
        {"MAX_MESSAGE_LENGTH": 4096, "REQUIRES_EDIT_FINALIZE": True},
    )
    RichAdapter.__abstractmethods__ = frozenset()
    adapter = RichAdapter.__new__(RichAdapter)
    adapter._typing_paused = set()
    adapter._fatal_error_message = None
    adapter.supports_draft_streaming = lambda chat_type=None, metadata=None: False
    adapter.prefers_fresh_final_streaming = lambda content, metadata=None: True
    adapter.streaming_overflow_limit = lambda: overflow_limit
    adapter.send = AsyncMock(side_effect=send_results) if send_results else AsyncMock(
        return_value=SendResult(success=True, message_id="m1"),
    )
    adapter.edit_message = AsyncMock(return_value=SendResult(success=True))
    adapter.delete_message = AsyncMock(return_value=True)
    return adapter


class TestRichAwareOverflow:
    """Rich-capable adapters raise the consumer's overflow limit so a reply that
    fits one rich message isn't fragmented at the legacy 4,096 edit limit."""

    def test_raw_message_limit_uses_adapter_rich_cap(self):
        adapter = _make_rich_capable_adapter(overflow_limit=32768)
        consumer = GatewayStreamConsumer(adapter, "12345", StreamConsumerConfig())
        assert consumer._raw_message_limit() == 32768

    def test_raw_message_limit_falls_back_to_max_length(self):
        # Adapter whose hook returns None (default) keeps the legacy limit.
        adapter = _make_rich_capable_adapter()
        adapter.streaming_overflow_limit = lambda: None
        consumer = GatewayStreamConsumer(adapter, "12345", StreamConsumerConfig())
        assert consumer._raw_message_limit() == 4096

    def test_raw_message_limit_mock_adapter_is_safe(self):
        # MagicMock adapters (many existing tests) must not crash or wrongly
        # inflate the limit from a truthy auto-attribute.
        adapter = MagicMock()
        adapter.MAX_MESSAGE_LENGTH = 4096
        consumer = GatewayStreamConsumer(adapter, "12345", StreamConsumerConfig())
        assert consumer._raw_message_limit() == 4096

    @pytest.mark.asyncio
    async def test_long_rich_reply_not_split_and_final_is_whole(self):
        from gateway.platforms.base import SendResult

        long_text = "x" * 5000  # > 4096 legacy limit, < 32768 rich limit
        adapter = _make_rich_capable_adapter(send_results=[
            SendResult(success=True, message_id="preview1"),
            SendResult(success=True, message_id="final1"),
        ])
        cfg = StreamConsumerConfig(
            transport="auto", chat_type="dm",
            edit_interval=0.01, buffer_threshold=5, cursor="",
            fresh_final_after_seconds=0.0,
        )
        consumer = GatewayStreamConsumer(adapter, "12345", cfg)

        consumer.on_delta(long_text)
        task = asyncio.create_task(consumer.run())
        await asyncio.sleep(0.05)
        consumer.finish()
        await task

        # Exactly two whole sends: the preview and the fresh final — NOT split
        # into ~4096 chunks. Both carry the full 5,000-char reply.
        assert adapter.send.await_count == 2
        assert adapter.send.call_args_list[0].kwargs.get("content") == long_text
        assert adapter.send.call_args_list[1].kwargs.get("content") == long_text
        adapter.edit_message.assert_not_called()
        adapter.delete_message.assert_awaited_once_with("12345", "preview1")
        assert consumer.final_response_sent is True

    @pytest.mark.asyncio
    async def test_fresh_final_deletes_all_preview_fragments(self):
        from gateway.platforms.base import SendResult

        adapter = _make_rich_capable_adapter(send_results=[
            SendResult(success=True, message_id="final1"),
        ])
        consumer = GatewayStreamConsumer(adapter, "12345", StreamConsumerConfig())
        # Simulate a reply that was split across the edit limit while streaming:
        # three preview fragments, the last of which is the current message.
        consumer._message_id = "frag3"
        consumer._preview_message_ids = {"frag1", "frag2", "frag3"}

        ok = await consumer._try_fresh_final("the whole completed answer")

        assert ok is True
        # All three stale fragments deleted; the fresh final never deleted.
        deleted = {c.args[1] for c in adapter.delete_message.await_args_list}
        assert deleted == {"frag1", "frag2", "frag3"}
        assert "final1" not in deleted
        assert consumer._message_id == "final1"
        assert consumer._preview_message_ids == set()
        assert consumer.final_response_sent is True


class TestGatewayDeliveryHooks:
    @pytest.mark.asyncio
    async def test_send_message_emits_post_gateway_delivery_hook(self, monkeypatch):
        from gateway.platforms.base import SendResult

        calls = []
        monkeypatch.setattr(
            "hermes_cli.plugins.has_hook",
            lambda name: name == "post_gateway_delivery",
        )
        monkeypatch.setattr(
            "hermes_cli.plugins.invoke_hook",
            lambda name, **kwargs: calls.append((name, kwargs)) or [],
        )

        adapter = SimpleNamespace(platform="slack")

        async def _send(**kwargs):
            return SendResult(success=True, message_id="m-123")

        adapter.send = _send
        consumer = GatewayStreamConsumer(
            adapter,
            "C123",
            StreamConsumerConfig(chat_type="group"),
            metadata={"thread_id": "178.456"},
        )

        result = await consumer._send_message(
            chat_id="C123",
            content="hello delivery",
            metadata={"thread_id": "178.456"},
        )

        assert result.success is True
        assert calls
        name, payload = calls[0]
        assert name == "post_gateway_delivery"
        assert payload["platform"] == "slack"
        assert payload["chat_id"] == "C123"
        assert payload["thread_id"] == "178.456"
        assert payload["source"] == "slack:C123:178.456"
        assert payload["operation"] == "send"
        assert payload["status"] == "ok"
        assert payload["message_id"] == "m-123"
        assert payload["content_preview"] == "hello delivery"


class TestBaseAdapterDeliveryHooks:
    @pytest.mark.asyncio
    async def test_send_with_retry_emits_post_gateway_delivery_hook(self, monkeypatch):
        from gateway.platforms.base import BasePlatformAdapter, SendResult

        calls = []
        monkeypatch.setattr(
            "hermes_cli.plugins.has_hook",
            lambda name: name == "post_gateway_delivery",
        )
        monkeypatch.setattr(
            "hermes_cli.plugins.invoke_hook",
            lambda name, **kwargs: calls.append((name, kwargs)) or [],
        )

        Adapter = type(
            "DeliveryAdapter",
            (BasePlatformAdapter,),
            {"MAX_MESSAGE_LENGTH": 4096, "platform": SimpleNamespace(value="slack")},
        )
        Adapter.__abstractmethods__ = frozenset()
        adapter = Adapter.__new__(Adapter)
        adapter._typing_paused = set()
        adapter._fatal_error_message = None

        async def _send(**kwargs):
            return SendResult(success=True, message_id="m-retry")

        adapter.send = _send
        result = await adapter._send_with_retry(
            chat_id="C123",
            content="retry path response",
            metadata={"thread_id": "178.456"},
        )

        assert result.success is True
        assert calls
        name, payload = calls[0]
        assert name == "post_gateway_delivery"
        assert payload["platform"] == "slack"
        assert payload["chat_id"] == "C123"
        assert payload["thread_id"] == "178.456"
        assert payload["source"] == "slack:C123:178.456"
        assert payload["operation"] == "send"
        assert payload["status"] == "ok"
        assert payload["message_id"] == "m-retry"
        assert payload["content_preview"] == "retry path response"
        assert payload["transport"] == "send_with_retry"

    @pytest.mark.asyncio
    async def test_send_with_retry_emits_retry_success_hook(self, monkeypatch):
        from gateway.platforms.base import BasePlatformAdapter, SendResult

        calls = []
        monkeypatch.setattr(
            "hermes_cli.plugins.has_hook",
            lambda name: name == "post_gateway_delivery",
        )
        monkeypatch.setattr(
            "hermes_cli.plugins.invoke_hook",
            lambda name, **kwargs: calls.append((name, kwargs)) or [],
        )
        monkeypatch.setattr("gateway.platforms.base.asyncio.sleep", AsyncMock())

        Adapter = type(
            "RetryDeliveryAdapter",
            (BasePlatformAdapter,),
            {"MAX_MESSAGE_LENGTH": 4096, "platform": SimpleNamespace(value="slack")},
        )
        Adapter.__abstractmethods__ = frozenset()
        adapter = Adapter.__new__(Adapter)
        adapter._typing_paused = set()
        adapter._fatal_error_message = None
        results = iter([
            SendResult(success=False, error="temporary unavailable", retryable=True),
            SendResult(success=True, message_id="m-retry-2"),
        ])

        async def _send(**kwargs):
            return next(results)

        adapter.send = _send
        result = await adapter._send_with_retry(
            chat_id="C123",
            content="retry eventually succeeds",
            metadata={"thread_id": "178.456"},
            base_delay=0,
        )

        assert result.success is True
        operations = [payload["operation"] for _name, payload in calls]
        assert operations == ["send", "send_retry"]
        assert calls[-1][1]["status"] == "ok"
        assert calls[-1][1]["message_id"] == "m-retry-2"
