"""Guest reply images must reach the same attachment path as ordinary DMs."""
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from PIL import Image
from telegram import Message

from gateway.config import PlatformConfig
from plugins.platforms.telegram import adapter as telegram_mod


def _payload(media="photo", size=256):
    reply = {
        "message_id": 41, "date": 1700000000,
        "chat": {"id": 11, "type": "private"},
        "from": {"id": 99, "is_bot": False, "first_name": "Author"},
        "caption": "Image caption",
    }
    item = {"file_id": "photo-large", "file_unique_id": "unique-large", "file_size": size}
    if media == "photo":
        reply[media] = [
            {**item, "file_id": "photo-small", "width": 10, "height": 10},
            {**item, "width": 100, "height": 100},
        ]
    elif media == "document":
        reply[media] = {**item, "file_name": "picture.png", "mime_type": "image/png"}
    return {
        "message_id": 42, "date": 1700000001,
        "guest_query_id": "q-image",
        "chat": {"id": 11, "type": "private"},
        # Telegram's visible author may differ from the summoning caller.
        "from": {"id": 99, "is_bot": False, "first_name": "Author"},
        "guest_bot_caller_user": {"id": 42, "is_bot": False, "first_name": "Caller"},
        "text": "@hermes_test_bot describe this",
        "reply_to_message": reply,
    }


def _setup(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = telegram_mod.TelegramAdapter(PlatformConfig(
        enabled=True, extra={"guest_mode": True, "allow_from": ["42"]},
    ))
    adapter._bot = MagicMock()
    adapter._bot.defaults = None
    adapter._bot.username = "hermes_test_bot"
    image = BytesIO()
    Image.new("RGB", (16, 16), "red").save(image, format="PNG")
    data = image.getvalue()
    file = SimpleNamespace(
        file_path="photos/picture.png", download_as_bytearray=AsyncMock(return_value=bytearray(data)),
    )
    adapter._bot.get_file = AsyncMock(return_value=file)
    adapter.handle_message = AsyncMock()
    return adapter, file, data


def _update(adapter, raw, transport):
    if transport == "raw":
        return SimpleNamespace(update_id=7, guest_message=None, api_kwargs={"guest_message": raw})
    return SimpleNamespace(update_id=7, guest_message=Message.de_json(raw, adapter._bot), api_kwargs={})


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["raw", "sdk"])
@pytest.mark.parametrize("media", ["photo", "document"])
async def test_guest_reply_image_reaches_canonical_attachments(monkeypatch, tmp_path, transport, media):
    adapter, file, data = _setup(monkeypatch, tmp_path)
    await adapter._handle_guest_update(_update(adapter, _payload(media), transport), None)
    event = adapter.handle_message.await_args.args[0]
    assert len(event.media_urls) == 1
    assert event.media_types == ["image/png"]
    assert Path(event.media_urls[0]).read_bytes() == data
    assert event.reply_to_message_id == "41"
    assert event.reply_to_text == "Image caption"
    assert "describe this" in event.text
    assert "[Replied-to image" in event.text
    assert event.source.chat_id == "guest:q-image"
    assert event.source.user_id == "42"
    assert event.guest_mode_invocation is True
    assert event.source.session_key_override == event.session_key_override
    adapter._bot.get_file.assert_awaited_once()
    assert "photo-large" in str(adapter._bot.get_file.await_args)
    file.download_as_bytearray.assert_awaited_once()


@pytest.mark.asyncio
async def test_ordinary_reply_uses_canonical_attachments_without_overwriting(monkeypatch, tmp_path):
    adapter, _, data = _setup(monkeypatch, tmp_path)
    msg = Message.de_json(_payload(), adapter._bot)
    event = adapter._build_message_event(msg, telegram_mod.MessageType.TEXT)
    event.media_urls = ["/existing.png"]
    event.media_types = ["image/png"]
    await adapter._cache_replied_media(msg, event)
    assert event.media_urls[0] == "/existing.png"
    assert len(event.media_urls) == 2
    assert Path(event.media_urls[1]).read_bytes() == data
    assert event.media_types == ["image/png", "image/png"]


@pytest.mark.asyncio
@pytest.mark.parametrize("pairing", [False, True])
async def test_guest_unauthorized_caller_cannot_download_reply(monkeypatch, tmp_path, pairing):
    adapter, file, _ = _setup(monkeypatch, tmp_path)
    if pairing:
        adapter.config.extra["unauthorized_dm_behavior"] = "pair"
    raw = _payload()
    raw["from"]["id"] = 42  # Allowed visible author must not authorize another caller.
    raw["guest_bot_caller_user"]["id"] = 777
    await adapter._handle_guest_update(_update(adapter, raw, "raw"), None)
    adapter._bot.get_file.assert_not_awaited()
    file.download_as_bytearray.assert_not_awaited()
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [0, 21 * 1024 * 1024])
async def test_guest_rejected_reply_size_does_not_download(monkeypatch, tmp_path, size):
    adapter, file, _ = _setup(monkeypatch, tmp_path)
    await adapter._handle_guest_update(_update(adapter, _payload(size=size), "raw"), None)
    adapter._bot.get_file.assert_not_awaited()
    file.download_as_bytearray.assert_not_awaited()
    assert adapter.handle_message.await_args.args[0].media_urls == []


@pytest.mark.asyncio
async def test_guest_download_failure_preserves_text_turn(monkeypatch, tmp_path):
    adapter, _, _ = _setup(monkeypatch, tmp_path)
    adapter._bot.get_file.side_effect = RuntimeError("unavailable")
    await adapter._handle_guest_update(_update(adapter, _payload(), "raw"), None)
    adapter._bot.get_file.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.media_urls == []
    assert "describe this" in event.text


@pytest.mark.asyncio
@pytest.mark.parametrize("missing", ["file_name", "mime_type", "both"])
async def test_guest_raw_document_optional_fields(monkeypatch, tmp_path, missing):
    adapter, file, _ = _setup(monkeypatch, tmp_path)
    raw = _payload(media="document")
    doc = raw["reply_to_message"]["document"]
    for key in ("file_name", "mime_type"):
        if missing in (key, "both"):
            doc.pop(key)
    await adapter._handle_guest_update(_update(adapter, raw, "raw"), None)
    event = adapter.handle_message.await_args.args[0]
    assert len(event.media_urls) == 1
    assert event.media_types == ["image/png"]
    file.download_as_bytearray.assert_awaited_once()


@pytest.mark.asyncio
async def test_guest_text_reply_does_not_request_files(monkeypatch, tmp_path):
    adapter, _, _ = _setup(monkeypatch, tmp_path)
    raw = _payload(media="text")
    raw["reply_to_message"]["text"] = "Quoted text"
    await adapter._handle_guest_update(_update(adapter, raw, "raw"), None)
    adapter._bot.get_file.assert_not_awaited()
    event = adapter.handle_message.await_args.args[0]
    assert event.reply_to_text == "Quoted text"
    assert event.media_urls == []
