"""Tests for Telegram native draft streaming fallback behavior."""

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig


class StubAdapter(BasePlatformAdapter):
    def __init__(self, *, draft_supported=False, draft_results=None):
        super().__init__(PlatformConfig(enabled=True, token="***"), Platform.TELEGRAM)
        self._draft_supported = draft_supported
        self.draft_results = list(draft_results or [])
        self.sent = []
        self.edited = []
        self.drafts = []

    async def connect(self):
        return True

    async def disconnect(self):
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append((chat_id, content, metadata))
        return SendResult(success=True, message_id="m1")

    async def edit_message(self, chat_id, message_id, content):
        self.edited.append((chat_id, message_id, content))
        return SendResult(success=True, message_id=message_id)

    async def send_typing(self, chat_id, metadata=None):
        return None

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}

    @property
    def supports_draft_streaming(self):
        return self._draft_supported

    async def send_draft(self, chat_id, draft_id, content, metadata=None):
        self.drafts.append((chat_id, draft_id, content, metadata))
        if self.draft_results:
            return self.draft_results.pop(0)
        return True


@pytest.mark.asyncio
async def test_draft_transport_used_for_dm_and_final_message_not_marked_sent():
    adapter = StubAdapter(draft_supported=True)
    consumer = GatewayStreamConsumer(
        adapter=adapter,
        chat_id="123",
        config=StreamConsumerConfig(transport="auto", edit_interval=0.01, buffer_threshold=1, cursor=" ▉"),
        metadata={"chat_type": "dm"},
    )

    consumer.on_delta("Hel")
    consumer.on_delta("lo")
    consumer.finish()
    await consumer.run()

    assert consumer.already_sent is False
    assert adapter.sent == []
    assert adapter.edited == []
    assert len(adapter.drafts) >= 1
    assert all(call[3] == {"chat_type": "dm"} for call in adapter.drafts)
    assert adapter.drafts[-1][2] == "Hello"


@pytest.mark.asyncio
async def test_group_chat_auto_mode_uses_edit_transport():
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

    assert consumer.already_sent is True
    assert adapter.drafts == []
    assert len(adapter.sent) == 1


@pytest.mark.asyncio
async def test_draft_failure_falls_back_to_edit_transport():
    adapter = StubAdapter(draft_supported=True, draft_results=[False])
    consumer = GatewayStreamConsumer(
        adapter=adapter,
        chat_id="789",
        config=StreamConsumerConfig(transport="draft", edit_interval=0.01, buffer_threshold=1, cursor=" ▉"),
        metadata={"chat_type": "dm"},
    )

    consumer.on_delta("fallback")
    consumer.finish()
    await consumer.run()

    assert adapter.drafts != []
    assert consumer.already_sent is True
    assert len(adapter.sent) == 1
    assert adapter.sent[0][1] == "fallback"
