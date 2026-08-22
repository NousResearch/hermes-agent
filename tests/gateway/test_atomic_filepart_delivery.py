"""Atomic text + attachment delivery for FilePart-capable adapters."""

import asyncio
from unittest.mock import AsyncMock

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    ProcessingOutcome,
    SendResult,
)
from gateway.session import SessionSource, build_session_key


class _AtomicAdapter(BasePlatformAdapter):
    @property
    def name(self) -> str:
        return "atomic-test"

    def __init__(self, *, atomic_success: bool = True):
        super().__init__(
            PlatformConfig(enabled=True, token="fake-token", typing_indicator=False),
            Platform.API_SERVER,
        )
        self.atomic_success = atomic_success
        self.atomic_calls: list[dict] = []
        self.text_calls: list[dict] = []
        self.document_calls: list[str] = []
        self.image_calls: list[tuple[str, str]] = []
        self.outcomes: list[ProcessingOutcome] = []

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.text_calls.append({"chat_id": chat_id, "content": content})
        return SendResult(success=True, message_id="text-1")

    async def send_message_with_files(
        self,
        chat_id,
        content,
        file_paths,
        file_urls=None,
        reply_to=None,
        metadata=None,
    ) -> SendResult:
        self.atomic_calls.append(
            {
                "chat_id": chat_id,
                "content": content,
                "file_paths": list(file_paths),
                "file_urls": list(file_urls or []),
                "reply_to": reply_to,
                "metadata": metadata,
            }
        )
        if self.atomic_success:
            return SendResult(success=True, message_id="atomic-1")
        return SendResult(success=False, error="atomic send failed")

    async def send_document(
        self,
        chat_id,
        file_path,
        caption=None,
        file_name=None,
        reply_to=None,
        metadata=None,
        **kwargs,
    ) -> SendResult:
        self.document_calls.append(str(file_path))
        return SendResult(success=True, message_id="doc-1")

    async def send_multiple_images(
        self,
        chat_id,
        images,
        metadata=None,
        human_delay=0.0,
    ) -> None:
        self.image_calls.extend(images)

    async def get_chat_info(self, chat_id: str) -> dict:
        return {"id": chat_id}

    async def on_processing_complete(self, event, outcome) -> None:
        self.outcomes.append(outcome)


def _event() -> MessageEvent:
    return MessageEvent(
        text="generate an image",
        message_type=MessageType.TEXT,
        source=SessionSource(platform=Platform.API_SERVER, chat_id="hugin", chat_type="dm"),
        message_id="customer-turn-1",
    )


def test_local_media_uses_one_atomic_text_and_file_send(tmp_path, monkeypatch):
    image = tmp_path / "generated.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    monkeypatch.setattr("gateway.platforms.base.MEDIA_DELIVERY_SAFE_ROOTS", (tmp_path,))

    adapter = _AtomicAdapter()
    adapter.set_message_handler(
        AsyncMock(return_value=f"Her er billedet.\nMEDIA:{image}")
    )
    event = _event()

    asyncio.run(
        adapter._process_message_background(event, build_session_key(event.source))
    )

    assert len(adapter.atomic_calls) == 1
    assert adapter.atomic_calls[0]["content"] == "Her er billedet."
    assert adapter.atomic_calls[0]["file_paths"] == [str(image.resolve())]
    assert adapter.atomic_calls[0]["file_urls"] == []
    assert adapter.text_calls == []
    assert adapter.document_calls == []
    assert adapter.image_calls == []
    assert adapter.outcomes == [ProcessingOutcome.SUCCESS]


def test_remote_media_url_uses_one_atomic_text_and_filepart_uri_send():
    url = "https://files-cdn.x.ai/generated/hugin-raven.png"
    adapter = _AtomicAdapter()
    adapter.set_message_handler(
        AsyncMock(return_value=f"Her er billedet. MEDIA:{url}")
    )
    event = _event()

    asyncio.run(
        adapter._process_message_background(event, build_session_key(event.source))
    )

    assert len(adapter.atomic_calls) == 1
    assert adapter.atomic_calls[0]["content"] == "Her er billedet."
    assert adapter.atomic_calls[0]["file_paths"] == []
    assert adapter.atomic_calls[0]["file_urls"] == [url]
    assert adapter.text_calls == []
    assert adapter.image_calls == []
    assert adapter.outcomes == [ProcessingOutcome.SUCCESS]


def test_uppercase_https_remote_media_stays_atomic():
    url = "HTTPS://files-cdn.x.ai/generated/hugin-raven.png"
    adapter = _AtomicAdapter()
    adapter.set_message_handler(
        AsyncMock(return_value=f"Her er billedet. MEDIA:{url}")
    )
    event = _event()

    asyncio.run(
        adapter._process_message_background(event, build_session_key(event.source))
    )

    assert len(adapter.atomic_calls) == 1
    assert adapter.atomic_calls[0]["file_urls"] == [url]
    assert adapter.text_calls == []
    assert adapter.image_calls == []


def test_non_image_remote_media_stays_visible():
    content = "Her er rapporten. MEDIA:https://files-cdn.x.ai/report.pdf"

    urls, cleaned = BasePlatformAdapter.extract_remote_media_urls(content)

    assert urls == []
    assert cleaned == content


def test_private_remote_media_url_stays_visible_and_is_not_delivered():
    content = "MEDIA:http://127.0.0.1/private/image.png"

    urls, cleaned = BasePlatformAdapter.extract_remote_media_urls(content)

    assert urls == []
    assert cleaned == content


def test_remote_media_directive_is_hidden_from_streaming_display():
    url = "https://files-cdn.x.ai/generated/hugin-raven.png"

    cleaned = BasePlatformAdapter.strip_media_directives_for_display(
        f"Her er billedet. MEDIA:{url}"
    )

    assert cleaned == "Her er billedet."


def test_remote_media_inside_serialized_tool_output_is_not_delivered():
    content = (
        '{"result":"Her er billedet. '
        'MEDIA:https://files-cdn.x.ai/generated/stale.png"}'
    )

    urls, cleaned = BasePlatformAdapter.extract_remote_media_urls(content)

    assert urls == []
    assert cleaned == content


def test_atomic_failure_does_not_emit_caption_or_split_attachment(tmp_path, monkeypatch):
    image = tmp_path / "generated.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    monkeypatch.setattr("gateway.platforms.base.MEDIA_DELIVERY_SAFE_ROOTS", (tmp_path,))

    adapter = _AtomicAdapter(atomic_success=False)
    adapter.set_message_handler(
        AsyncMock(return_value=f"Her er billedet.\nMEDIA:{image}")
    )
    event = _event()

    asyncio.run(
        adapter._process_message_background(event, build_session_key(event.source))
    )

    assert len(adapter.atomic_calls) == 1
    assert adapter.text_calls == []
    assert adapter.document_calls == []
    assert adapter.image_calls == []
    assert adapter.outcomes == [ProcessingOutcome.FAILURE]
