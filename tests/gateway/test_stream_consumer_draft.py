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
