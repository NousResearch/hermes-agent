import asyncio
import json
import logging

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)
from gateway.session import SessionSource, build_session_key


class DeliveryTimingAdapter(BasePlatformAdapter):
    def __init__(self, platform=Platform.SLACK):
        super().__init__(PlatformConfig(enabled=True, token="fake-token"), platform)
        self.sent = []
        self.delivered_kinds = []

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        await asyncio.sleep(0)
        self.sent.append({"chat_id": chat_id, "content": content})
        return SendResult(success=True, message_id="1")

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        return None

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}

    async def send_multiple_images(self, **kwargs):
        self.delivered_kinds.append("image")

    async def send_voice(self, **kwargs):
        self.delivered_kinds.append("voice")
        return SendResult(success=True, message_id="voice-1")

    async def play_tts(self, **kwargs):
        self.delivered_kinds.append("tts")
        return SendResult(success=True, message_id="tts-1")

    async def send_video(self, **kwargs):
        self.delivered_kinds.append("video")
        return SendResult(success=True, message_id="video-1")

    async def send_document(self, **kwargs):
        self.delivered_kinds.append("document")
        return SendResult(success=True, message_id="document-1")


async def _hold_typing(_chat_id, interval=2.0, metadata=None, stop_event=None):
    if stop_event is not None:
        await stop_event.wait()
    else:
        await asyncio.Event().wait()


def _event(platform=Platform.SLACK, *, message_type=MessageType.TEXT):
    return MessageEvent(
        text="hello",
        source=SessionSource(
            platform=platform,
            chat_id="C123",
            chat_type="group",
            user_id="U123",
        ),
        message_id="m1",
        message_type=message_type,
    )


def _delivery_payloads(caplog):
    return [
        json.loads(record.getMessage().split("request_budget.gateway_delivery.v1 ", 1)[1])
        for record in caplog.records
        if "request_budget.gateway_delivery.v1 " in record.getMessage()
    ]


@pytest.mark.asyncio
async def test_gateway_delivery_budget_logs_text_send(caplog):
    adapter = DeliveryTimingAdapter()
    adapter._keep_typing = _hold_typing
    adapter.set_message_handler(lambda _event: asyncio.sleep(0, result="ack"))
    event = _event()

    with caplog.at_level(logging.INFO, logger="gateway.platforms.base"):
        await adapter._process_message_background(event, build_session_key(event.source))

    payloads = _delivery_payloads(caplog)
    assert [payload["delivery_kind"] for payload in payloads] == ["text"]
    assert payloads[0]["response_chars"] == 3
    assert payloads[0]["delivery_succeeded"] is True


@pytest.mark.asyncio
async def test_gateway_delivery_budget_logs_remote_image_only_response(caplog):
    adapter = DeliveryTimingAdapter()
    adapter._keep_typing = _hold_typing
    adapter.set_message_handler(
        lambda _event: asyncio.sleep(
            0, result="![chart](https://example.com/chart.png)"
        )
    )
    event = _event()

    with caplog.at_level(logging.INFO, logger="gateway.platforms.base"):
        await adapter._process_message_background(event, build_session_key(event.source))

    assert adapter.delivered_kinds == ["image"]
    assert [payload["delivery_kind"] for payload in _delivery_payloads(caplog)] == [
        "image"
    ]


@pytest.mark.parametrize(
    ("suffix", "expected_kind"),
    [(".wav", "voice"), (".mp4", "video"), (".pdf", "document")],
)
@pytest.mark.asyncio
async def test_gateway_delivery_budget_logs_local_media_only_response(
    suffix, expected_kind, tmp_path, caplog
):
    media_path = tmp_path / f"artifact{suffix}"
    media_path.write_bytes(b"fixture")
    adapter = DeliveryTimingAdapter()
    adapter._keep_typing = _hold_typing
    adapter.set_message_handler(
        lambda _event: asyncio.sleep(0, result=f"MEDIA:{media_path}")
    )
    event = _event()

    with caplog.at_level(logging.INFO, logger="gateway.platforms.base"):
        await adapter._process_message_background(event, build_session_key(event.source))

    assert adapter.delivered_kinds == [expected_kind]
    assert [payload["delivery_kind"] for payload in _delivery_payloads(caplog)] == [
        expected_kind
    ]


@pytest.mark.asyncio
async def test_gateway_delivery_budget_logs_auto_tts(caplog, monkeypatch, tmp_path):
    import tools.tts_tool as tts_tool

    audio_path = tmp_path / "tts.wav"
    audio_path.write_bytes(b"fixture")
    monkeypatch.setattr(tts_tool, "check_tts_requirements", lambda: True)
    monkeypatch.setattr(
        tts_tool,
        "text_to_speech_tool",
        lambda **kwargs: json.dumps({"file_path": str(audio_path)}),
    )

    adapter = DeliveryTimingAdapter(Platform.TELEGRAM)
    adapter._keep_typing = _hold_typing
    adapter._should_auto_tts_for_chat = lambda _chat_id: True
    adapter.set_message_handler(lambda _event: asyncio.sleep(0, result="spoken"))
    event = _event(Platform.TELEGRAM, message_type=MessageType.VOICE)

    with caplog.at_level(logging.INFO, logger="gateway.platforms.base"):
        await adapter._process_message_background(event, build_session_key(event.source))

    assert adapter.delivered_kinds == ["tts"]
    assert [payload["delivery_kind"] for payload in _delivery_payloads(caplog)] == [
        "tts"
    ]
