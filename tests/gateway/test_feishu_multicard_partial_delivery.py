"""
Tests for Feishu multi-card partial delivery.
When a multi-card final response fails after at least one card has been
delivered, the adapter must return PARTIALLY_DELIVERED so the base layer
does not retry or fall back to a legacy text payload.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    FinalDeliveryState,
    MessageEvent,
    MessageType,
    ProcessingOutcome,
    SendResult,
    effective_delivery_state,
)
from gateway.session import SessionSource, build_session_key
from plugins.platforms.feishu.adapter import FeishuAdapter


class _FakeResponse:
    code = 0
    msg = "ok"
    data = SimpleNamespace(message_id="om_1")

    def success(self):
        return True


class _FakeFailure:
    code = 99992402
    msg = "field validation failed"
    data = None

    def success(self):
        return False


def _make_adapter():
    cfg = PlatformConfig(
        enabled=True,
        token="fake",
        typing_indicator=False,
        extra={"final_response_format": "card"},
    )
    adapter = FeishuAdapter(cfg)
    adapter._client = object()
    return adapter


def _partial_result():
    return SendResult(
        success=False,
        error="transient failure after first card",
        retryable=True,
        delivery_state=FinalDeliveryState.PARTIALLY_DELIVERED,
        raw_response={"delivered_cards": 1, "total_cards": 2, "failed_index": 1},
    )


class _BasePartialDeliveryAdapter(BasePlatformAdapter):
    """Concrete test adapter that exercises Base's real delivery orchestrator."""

    def __init__(self, *, rich_result=None, send_result=None):
        super().__init__(
            PlatformConfig(enabled=True, token="test"),
            Platform.FEISHU,
        )
        self.rich_result = rich_result
        self.send_result = send_result or SendResult(success=True, message_id="legacy")
        self.rich_calls = []
        self.send_calls = []
        self.processing_outcomes = []

    async def connect(self, *, is_reconnect: bool = False):
        return True

    async def disconnect(self):
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.send_calls.append(
            (
                chat_id,
                content,
                {"reply_to": reply_to, "metadata": metadata},
            )
        )
        return self.send_result

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "dm"}

    async def try_send_final_rich_response(self, **kwargs):
        self.rich_calls.append(kwargs)
        return self.rich_result

    async def on_processing_complete(self, event, outcome):
        self.processing_outcomes.append(outcome)


def _base_event():
    source = SessionSource(
        platform=Platform.FEISHU,
        chat_id="oc_base_partial",
        chat_type="dm",
    )
    return MessageEvent(
        text="render a multi-card response",
        message_type=MessageType.TEXT,
        source=source,
        message_id="om_inbound",
    )


@pytest.mark.asyncio
async def test_second_card_api_failure_returns_partial_not_success():
    """When send() fails on the 2nd card after 1st succeeded, return partial."""
    adapter = _make_adapter()
    calls = []

    async def fake_send(**kwargs):
        calls.append(kwargs)
        if kwargs["msg_type"] == "interactive" and len(calls) == 2:
            return _FakeFailure()
        return _FakeResponse()

    adapter._feishu_send_with_retry = fake_send

    # Generate enough text to produce at least 2 card payloads
    long_text = "\n\n".join(f"段 {i}: " + "x" * 500 for i in range(1, 101))

    result = await adapter.send("oc_1", long_text, metadata={"hermes_final_response": True})

    assert result.success is False
    assert result.delivery_state is FinalDeliveryState.PARTIALLY_DELIVERED
    assert result.raw_response["delivered_cards"] == 1


@pytest.mark.asyncio
async def test_second_card_exception_returns_partial_not_legacy():
    """When send() raises on the 2nd card after 1st succeeded, return partial."""
    adapter = _make_adapter()
    calls = []

    async def fake_send(**kwargs):
        calls.append(kwargs)
        if kwargs["msg_type"] == "interactive" and len(calls) == 2:
            raise RuntimeError("forced")
        return _FakeResponse()

    adapter._feishu_send_with_retry = fake_send

    long_text = "\n\n".join(f"段 {i}: " + "x" * 500 for i in range(1, 101))

    result = await adapter.send("oc_1", long_text, metadata={"hermes_final_response": True})

    assert result.success is False
    assert result.delivery_state is FinalDeliveryState.PARTIALLY_DELIVERED
    # No legacy text sends were attempted — only interactive (card) calls
    assert all(c["msg_type"] == "interactive" for c in calls)


@pytest.mark.asyncio
async def test_first_card_api_failure_still_falls_back():
    """When the very first card fails, no partial — safe to fall back."""
    adapter = _make_adapter()
    calls = []

    async def fake_send(**kwargs):
        calls.append(kwargs)
        return _FakeFailure()

    adapter._feishu_send_with_retry = fake_send

    long_text = "\n\n".join(f"段 {i}: " + "x" * 500 for i in range(1, 101))

    result = await adapter.send("oc_1", long_text, metadata={"hermes_final_response": True})

    # Should NOT be partial (0 cards delivered), should have fallen through
    # to legacy text send.
    assert result.delivery_state is not FinalDeliveryState.PARTIALLY_DELIVERED
    # At least one non-interactive call was made (legacy fallback)
    assert any(c["msg_type"] != "interactive" for c in calls)


@pytest.mark.asyncio
async def test_try_send_final_rich_second_card_failure_returns_partial():
    """try_send_final_rich_response returns partial on second card failure."""
    adapter = _make_adapter()
    calls = []

    async def fake_send(**kwargs):
        calls.append(kwargs)
        if kwargs["msg_type"] == "interactive" and len(calls) == 2:
            return _FakeFailure()
        return _FakeResponse()

    adapter._feishu_send_with_retry = fake_send

    # Produce enough blocks to trigger multi-card split
    long_text = "\n\n".join(f"段 {i}: " + "x" * 500 for i in range(1, 101))

    # We need a minimal image to enter the rich path
    adapter._download_remote_image = AsyncMock(return_value="/tmp/fake.png")
    adapter._upload_image_for_card = AsyncMock(return_value="img_key")

    result = await adapter.try_send_final_rich_response(
        chat_id="oc_1",
        original_response=long_text,
        text_content=long_text,
        images=[("http://example.com/img.png", "test")],
        media_files=[],
        local_files=[],
        force_document_attachments=False,
        reply_to=None,
        metadata={"hermes_final_response": True},
    )

    assert result is not None
    assert result.success is False
    assert result.delivery_state is FinalDeliveryState.PARTIALLY_DELIVERED
    assert result.raw_response["delivered_cards"] == 1


@pytest.mark.asyncio
async def test_base_send_with_retry_stops_after_partial_delivery(monkeypatch):
    """Base must not retry a logical response whose first card is visible."""
    partial = _partial_result()
    adapter = _BasePartialDeliveryAdapter(send_result=partial)
    sleep = AsyncMock()
    monkeypatch.setattr("gateway.platforms.base.asyncio.sleep", sleep)

    result = await adapter._send_with_retry(
        "oc_base_partial",
        "multi-card response",
        max_retries=2,
        base_delay=0,
    )

    assert result is partial
    assert len(adapter.send_calls) == 1
    sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_base_send_with_retry_normalizes_string_partial_state(monkeypatch):
    partial = SendResult(
        success=False,
        error="partial",
        retryable=True,
        delivery_state="partially_delivered",  # type: ignore[arg-type]
    )
    adapter = _BasePartialDeliveryAdapter(send_result=partial)
    sleep = AsyncMock()
    monkeypatch.setattr("gateway.platforms.base.asyncio.sleep", sleep)

    result = await adapter._send_with_retry(
        "oc_base_partial",
        "multi-card response",
        max_retries=2,
        base_delay=0,
    )

    assert result is partial
    assert result.delivery_state is FinalDeliveryState.PARTIALLY_DELIVERED
    assert len(adapter.send_calls) == 1
    sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_base_rich_partial_delivery_skips_legacy_send():
    """The real Base orchestrator must not duplicate a partially visible card."""
    adapter = _BasePartialDeliveryAdapter(rich_result=_partial_result())
    adapter._message_handler = AsyncMock(return_value="multi-card final response")
    event = _base_event()

    await adapter._process_message_background(event, build_session_key(event.source))

    assert len(adapter.rich_calls) == 1
    assert adapter.send_calls == []


@pytest.mark.asyncio
async def test_base_rich_partial_delivery_completes_as_failure():
    """Partial delivery suppresses fallback but remains a failed processing outcome."""
    adapter = _BasePartialDeliveryAdapter(rich_result=_partial_result())
    adapter._message_handler = AsyncMock(return_value="multi-card final response")
    event = _base_event()

    await adapter._process_message_background(event, build_session_key(event.source))

    assert adapter.processing_outcomes == [ProcessingOutcome.FAILURE]


@pytest.mark.asyncio
async def test_legacy_second_chunk_failure_returns_partial_and_stops():
    adapter = _make_adapter()
    adapter.config.extra["final_response_format"] = "legacy"
    adapter.MAX_MESSAGE_LENGTH = 8  # type: ignore[assignment]
    calls = []

    async def fake_send(**kwargs):
        calls.append(kwargs)
        return _FakeFailure() if len(calls) == 2 else _FakeResponse()

    adapter._feishu_send_with_retry = fake_send
    result = await adapter.send("oc_1", "abcdefghijklmnopqrstuvwxyz")

    assert result.success is False
    assert result.delivery_state is FinalDeliveryState.PARTIALLY_DELIVERED
    assert result.raw_response["delivered_chunks"] == 1
    assert result.raw_response["total_chunks"] > 1
    assert result.raw_response["failed_index"] == 1
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_legacy_exception_after_first_chunk_returns_partial():
    adapter = _make_adapter()
    adapter.config.extra["final_response_format"] = "legacy"
    adapter.MAX_MESSAGE_LENGTH = 8  # type: ignore[assignment]
    calls = []

    async def fake_send(**kwargs):
        calls.append(kwargs)
        if len(calls) == 2:
            raise RuntimeError("forced legacy failure")
        return _FakeResponse()

    adapter._feishu_send_with_retry = fake_send
    result = await adapter.send("oc_1", "abcdefghijklmnopqrstuvwxyz")

    assert result.success is False
    assert result.delivery_state is FinalDeliveryState.PARTIALLY_DELIVERED
    assert result.raw_response["delivered_chunks"] == 1
    assert result.raw_response["failed_index"] == 1
    assert len(calls) == 2
