"""Media delivery must carry the turn's reply anchor.

Text sends get ``reply_to=_reply_anchor_for_event(event)``; both media paths —
``_send_image_batch`` → ``send_multiple_images`` and ``_deliver_media_attachments``
→ ``send_voice``/``send_video``/``send_document`` — dropped it. On a platform where a
send without reply semantics is a *different* message type, that loses the delivery:
QQ groups reject a topic-less send as a proactive message (40034105, no permission)
while a passive ``msg_id`` reply is accepted, so every image of a MEDIA: turn failed
even though the upload succeeded.

Contract: whatever anchor the turn resolved for text reaches the media senders too.
"""

import asyncio

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    SendResult,
)
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource, build_session_key


class _AnchorAdapter(BasePlatformAdapter):
    """Base-loop adapter (no media overrides) that records the anchors it is handed."""

    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="fake-token"), Platform.SIGNAL)
        self.image_anchors: list = []
        self.document_anchors: list = []

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        return SendResult(success=True, message_id="msg-1")

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        return None

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}

    async def send_image_file(self, chat_id, image_path, caption=None,
                              reply_to=None, metadata=None, **kwargs) -> SendResult:
        self.image_anchors.append(reply_to)
        return SendResult(success=True, message_id="img-1")

    async def send_document(self, chat_id, file_path, caption=None, file_name=None,
                            reply_to=None, metadata=None, **kwargs) -> SendResult:
        self.document_anchors.append(reply_to)
        return SendResult(success=True, message_id="doc-1")


async def _hold_typing(_chat_id, interval=2.0, metadata=None, stop_event=None):
    if stop_event is not None:
        await stop_event.wait()
    else:
        await asyncio.Event().wait()


def _allowed_file(tmp_path, monkeypatch, name: str):
    root = tmp_path / "media-cache"
    f = root / name
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(b"payload")
    monkeypatch.setattr("gateway.platforms.base.MEDIA_DELIVERY_SAFE_ROOTS", (root,))
    return f.resolve()


def _make_event() -> MessageEvent:
    return MessageEvent(
        text="send it",
        message_type=MessageType.TEXT,
        source=SessionSource(platform=Platform.SIGNAL, chat_id="111", chat_type="dm"),
        message_id="m1",
    )


def _adapter_with(handler) -> _AnchorAdapter:
    adapter = _AnchorAdapter()
    adapter._keep_typing = _hold_typing
    adapter.set_message_handler(handler)
    return adapter


@pytest.mark.asyncio
async def test_media_turn_carries_reply_anchor_to_image_sender(tmp_path, monkeypatch):
    """A MEDIA: image turn hands the sending image file the turn's reply anchor."""
    png = _allowed_file(tmp_path, monkeypatch, "photo.png")
    adapter = _adapter_with(lambda _event: _async_return(f"MEDIA:{png}"))
    event = _make_event()

    await adapter._process_message_background(event, build_session_key(event.source))

    assert adapter.image_anchors == ["m1"]


@pytest.mark.asyncio
async def test_media_turn_carries_reply_anchor_to_document_sender(tmp_path, monkeypatch):
    """Same anchor for the non-image media sibling path (``send_document``)."""
    doc = _allowed_file(tmp_path, monkeypatch, "notes.pdf")
    adapter = _adapter_with(lambda _event: _async_return(f"MEDIA:{doc}"))
    event = _make_event()

    await adapter._process_message_background(event, build_session_key(event.source))

    assert adapter.document_anchors == ["m1"]


async def _async_return(value):
    return value
