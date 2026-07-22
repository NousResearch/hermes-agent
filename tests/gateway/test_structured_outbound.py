"""Platform-neutral contracts for structured outbound gateway messages."""

import asyncio
import inspect
from dataclasses import dataclass
from typing import Any, Dict, Optional
from unittest.mock import patch

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, OutboundMessage, SendResult
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig


@dataclass
class _DemoStructuredMessage(OutboundMessage):
    sections: tuple[str, ...]


class _CaptureAdapter(BasePlatformAdapter):
    MAX_MESSAGE_LENGTH = 64

    def __init__(self) -> None:
        super().__init__(PlatformConfig(), Platform.WEBHOOK)
        self.sent: list[str] = []

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        self.sent.append(content)
        return SendResult(success=True, message_id=f"message-{len(self.sent)}")

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {"name": "test", "type": "dm"}


class _StructuredCaptureAdapter(_CaptureAdapter):
    REQUIRES_COMPLETE_RESPONSE = True

    def __init__(self) -> None:
        super().__init__()
        self.prepared: list[str] = []
        self.structured_sent: list[_DemoStructuredMessage] = []
        self._results: list[SendResult] = []

    def prepare_outbound_message(self, content: str) -> _DemoStructuredMessage:
        self.prepared.append(content)
        return _DemoStructuredMessage(
            content=content,
            sections=tuple(content.split("\n\n")),
        )

    async def send_outbound_message(
        self,
        chat_id: str,
        message: OutboundMessage,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        assert isinstance(message, _DemoStructuredMessage)
        self.structured_sent.append(message)
        if self._results:
            return self._results.pop(0)
        return SendResult(success=True, message_id=f"structured-{len(self.structured_sent)}")


def test_default_adapter_keeps_plain_text_send_contract() -> None:
    async def exercise() -> None:
        adapter = _CaptureAdapter()

        result = await adapter._send_with_retry("chat", "ordinary text")

        assert result.success
        assert adapter.sent == ["ordinary text"]

    asyncio.run(exercise())


def test_base_send_contract_remains_plain_text() -> None:
    send_parameters = inspect.signature(BasePlatformAdapter.send).parameters

    assert send_parameters["content"].annotation is str


def test_retry_pipeline_uses_adapter_owned_outbound_representation() -> None:
    async def exercise() -> None:
        adapter = _StructuredCaptureAdapter()
        adapter._results = [
            SendResult(success=False, error="connection dropped", retryable=True),
            SendResult(success=True, message_id="structured-2"),
        ]

        with patch("gateway.platforms.base.random.uniform", return_value=0):
            result = await adapter._send_with_retry(
                "chat", "Title\n\nBody", max_retries=1, base_delay=0,
            )

        assert result.success
        assert adapter.prepared == ["Title\n\nBody"]
        assert len(adapter.structured_sent) == 2
        assert adapter.structured_sent[0] is adapter.structured_sent[1]
        outbound = adapter.structured_sent[0]
        assert isinstance(outbound, _DemoStructuredMessage)
        assert outbound.content == "Title\n\nBody"
        assert outbound.sections == ("Title", "Body")

    asyncio.run(exercise())


def test_complete_response_adapter_buffers_and_prepares_once() -> None:
    async def exercise() -> None:
        adapter = _StructuredCaptureAdapter()
        consumer = GatewayStreamConsumer(
            adapter,
            "chat",
            StreamConsumerConfig(
                edit_interval=0.01,
                buffer_threshold=1,
                cursor="",
            ),
        )
        task = asyncio.create_task(consumer.run())
        first = "A" * 600
        second = "B" * 600

        consumer.on_delta(first)
        await asyncio.sleep(0.05)
        assert adapter.sent == []

        consumer.on_delta(second)
        consumer.finish()
        await task

        assert adapter.sent == []
        assert adapter.prepared == [first + second]
        assert len(adapter.structured_sent) == 1
        outbound = adapter.structured_sent[0]
        assert isinstance(outbound, _DemoStructuredMessage)
        assert outbound.content == first + second

    asyncio.run(exercise())


def test_complete_response_adapter_discards_pre_tool_segment_until_final() -> None:
    async def exercise() -> None:
        adapter = _StructuredCaptureAdapter()
        consumer = GatewayStreamConsumer(
            adapter,
            "chat",
            StreamConsumerConfig(cursor=""),
        )

        consumer.on_delta("I will inspect the repository first.")
        consumer.on_segment_break()
        consumer.on_delta("The final answer is ready.")
        consumer.finish()

        await consumer.run()

        assert adapter.sent == []
        assert adapter.prepared == ["The final answer is ready."]
        assert [message.content for message in adapter.structured_sent] == [
            "The final answer is ready.",
        ]

    asyncio.run(exercise())


def test_complete_response_adapter_discards_commentary_until_final() -> None:
    async def exercise() -> None:
        adapter = _StructuredCaptureAdapter()
        consumer = GatewayStreamConsumer(
            adapter,
            "chat",
            StreamConsumerConfig(cursor=""),
        )

        consumer.on_delta("I will inspect the repository first.")
        consumer.on_commentary("I am checking the relevant files.")
        consumer.on_delta("The final answer is ready.")
        consumer.finish()

        await consumer.run()

        assert adapter.sent == []
        assert adapter.prepared == ["The final answer is ready."]
        assert [message.content for message in adapter.structured_sent] == [
            "The final answer is ready.",
        ]

    asyncio.run(exercise())
