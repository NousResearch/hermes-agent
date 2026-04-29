"""Tests for draft transport in GatewayStreamConsumer."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from gateway.platforms.base import BasePlatformAdapter, PlatformConfig, Platform, SendResult
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig


class StubAdapter(BasePlatformAdapter):
    """Minimal adapter for testing draft transport."""

    def __init__(self, *, draft_supported=False, draft_results=None):
        super().__init__(PlatformConfig(enabled=True, token="***"), Platform.TELEGRAM)
        self._draft_supported = draft_supported
        self._draft_results = list(draft_results or [])
        self.sent = []
        self.edited = []
        self.drafts = []
        # Hide send_draft_message if draft not supported
        if not draft_supported:
            self.send_draft_message = None

    async def connect(self):
        return True

    async def disconnect(self):
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append((chat_id, content, metadata))
        return SendResult(success=True, message_id="m1")

    async def edit_message(self, chat_id, message_id, content, *, finalize=False):
        self.edited.append((chat_id, message_id, content))
        return SendResult(success=True, message_id=message_id)

    async def send_typing(self, chat_id, metadata=None):
        return None

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}

    async def send_draft_message(self, chat_id, draft_id, content):
        if not self._draft_supported:
            raise NotImplementedError("draft not supported")
        # Strip cursor like the real Telegram adapter does
        draft_text = content
        if draft_text.endswith(" ▉"):
            draft_text = draft_text[:-2]
        self.drafts.append((chat_id, draft_id, draft_text))
        if self._draft_results:
            return SendResult(success=self._draft_results.pop(0))
        return SendResult(success=True)


class TestDraftTransportResolution:
    """Verify _resolve_draft_mode logic."""

    def _make_consumer(self, adapter, transport="auto", chat_type="dm"):
        cfg = StreamConsumerConfig(transport=transport, edit_interval=0.01, buffer_threshold=1, cursor=" ▉")
        return GatewayStreamConsumer(
            adapter=adapter,
            chat_id="123",
            config=cfg,
            metadata={"chat_type": chat_type},
        )

    def test_auto_dm_with_draft_support_uses_draft(self):
        adapter = StubAdapter(draft_supported=True)
        c = self._make_consumer(adapter, transport="auto", chat_type="dm")
        assert c._draft_mode is True

    def test_auto_group_chat_uses_edit(self):
        adapter = StubAdapter(draft_supported=True)
        c = self._make_consumer(adapter, transport="auto", chat_type="group")
        assert c._draft_mode is False

    def test_auto_no_draft_support_uses_edit(self):
        adapter = StubAdapter(draft_supported=False)
        c = self._make_consumer(adapter, transport="auto", chat_type="dm")
        assert c._draft_mode is False

    def test_force_draft_with_support(self):
        adapter = StubAdapter(draft_supported=True)
        c = self._make_consumer(adapter, transport="draft", chat_type="group")
        assert c._draft_mode is True

    def test_force_draft_without_support_falls_back(self):
        adapter = StubAdapter(draft_supported=False)
        c = self._make_consumer(adapter, transport="draft", chat_type="dm")
        assert c._draft_mode is False

    def test_force_edit_disables_draft(self):
        adapter = StubAdapter(draft_supported=True)
        c = self._make_consumer(adapter, transport="edit", chat_type="dm")
        assert c._draft_mode is False

    def test_off_disables_draft(self):
        adapter = StubAdapter(draft_supported=True)
        c = self._make_consumer(adapter, transport="off", chat_type="dm")
        assert c._draft_mode is False


class TestDraftStreamingEndToEnd:
    """Verify draft transport delivers tokens via send_draft_message."""

    async def _run_streaming(self, adapter, consumer, deltas, interval=0.06):
        """Run consumer in background, feed deltas progressively, then finish.

        Real streaming sends tokens one at a time; this simulates that by
        starting ``run()`` as a background task and feeding deltas with
        short delays so they arrive in separate batches.

        ``interval`` must exceed ``run()``'s internal ``sleep(0.05)`` so
        the consumer has time to drain the queue between deltas; default
        0.06 works.
        """
        task = asyncio.create_task(consumer.run())
        await asyncio.sleep(0.05)  # let run() finish its first empty-queue pass
        for d in deltas:
            consumer.on_delta(d)
            await asyncio.sleep(interval)
        consumer.finish()
        await task

    @pytest.mark.asyncio
    async def test_draft_transport_used_for_dm(self):
        adapter = StubAdapter(draft_supported=True)
        consumer = GatewayStreamConsumer(
            adapter=adapter,
            chat_id="123",
            config=StreamConsumerConfig(transport="auto", edit_interval=0.01, buffer_threshold=1, cursor=" ▉"),
            metadata={"chat_type": "dm"},
        )

        await self._run_streaming(adapter, consumer, ["Hel", "lo"])

        # Draft mode should use send_draft_message for intermediate tokens
        assert len(adapter.drafts) >= 1
        # Last draft text should be "Hello" (cursor stripped)
        last_draft_text = adapter.drafts[-1][2]
        assert last_draft_text == "Hello"
        # No intermediate real messages (only the final dismissal)
        assert len(adapter.sent) == 1
        assert adapter.sent[0][1] == "Hello"
        # Consumer sends real message to dismiss draft immediately
        assert consumer.final_response_sent is True

    @pytest.mark.asyncio
    async def test_edit_transport_used_for_group(self):
        adapter = StubAdapter(draft_supported=True)
        consumer = GatewayStreamConsumer(
            adapter=adapter,
            chat_id="456",
            config=StreamConsumerConfig(transport="auto", edit_interval=0.01, buffer_threshold=1, cursor=" ▉"),
            metadata={"chat_type": "group"},
        )

        consumer.on_delta("abc")
        consumer.finish()
        await consumer.run()

        # Edit mode: no drafts, uses send
        assert adapter.drafts == []
        assert consumer.already_sent is True
        assert len(adapter.sent) == 1

    @pytest.mark.asyncio
    async def test_draft_failure_skips_update_stays_in_draft_mode(self):
        """When a draft send fails, the consumer skips that update but stays
        in draft mode and sends the real message on stream end."""
        adapter = StubAdapter(draft_supported=True, draft_results=[False])
        consumer = GatewayStreamConsumer(
            adapter=adapter,
            chat_id="789",
            config=StreamConsumerConfig(transport="draft", edit_interval=0.01, buffer_threshold=1, cursor=" ▉"),
            metadata={"chat_type": "dm"},
        )

        await self._run_streaming(adapter, consumer, ["fallback"])

        # Draft was attempted (returned success=False) but still recorded
        assert len(adapter.drafts) >= 1
        # Consumer still sends real message on stream end (best-effort)
        assert len(adapter.sent) == 1
        assert consumer.final_response_sent is True
        # Consumer stays in draft mode (doesn't fall back to edit path)
        assert consumer._draft_mode is True
