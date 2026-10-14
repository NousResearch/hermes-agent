from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gateway.platforms.event import MessageType
from plugins.platforms.telegram.adapter import TelegramAdapter


class _DummyApp:
    def __init__(self):
        self.handlers = []

    def add_handler(self, handler, group=0):
        self.handlers.append((group, handler))


class _Filter:
    def __init__(self, name):
        self.names = {name}

    def __or__(self, other):
        combined = _Filter("combined")
        combined.names = self.names | other.names
        return combined

    __and__ = __or__

    def __invert__(self):
        return self


def test_media_handler_registration_accepts_video_note():
    adapter = object.__new__(TelegramAdapter)
    app = _DummyApp()
    fake_filters = SimpleNamespace(
        TEXT=_Filter("text"),
        COMMAND=_Filter("command"),
        LOCATION=_Filter("location"),
        VENUE=_Filter("venue"),
        PHOTO=_Filter("photo"),
        VIDEO=_Filter("video"),
        VIDEO_NOTE=_Filter("video_note"),
        AUDIO=_Filter("audio"),
        VOICE=_Filter("voice"),
        Document=SimpleNamespace(ALL=_Filter("document")),
        Sticker=SimpleNamespace(ALL=_Filter("sticker")),
    )

    def handler(filter_value, callback):
        return SimpleNamespace(filter_value=filter_value, callback=callback)

    with patch.multiple(
        "plugins.platforms.telegram.adapter",
        filters=fake_filters,
        TelegramMessageHandler=handler,
        CallbackQueryHandler=lambda callback: SimpleNamespace(callback=callback),
        InlineQueryHandler=lambda callback: SimpleNamespace(callback=callback),
        TypeHandler=lambda update_type, callback: SimpleNamespace(callback=callback),
    ):
        adapter._register_handlers(app)

    media_filter_names = app.handlers[3][1].filter_value.names
    assert "video" in media_filter_names
    assert "video_note" in media_filter_names


@pytest.mark.asyncio
@pytest.mark.parametrize("attachment_attr", ["video", "video_note"])
async def test_inbound_video_types_use_existing_video_cache(attachment_attr):
    adapter = object.__new__(TelegramAdapter)
    adapter._is_user_authorized_from_message = lambda _message: True
    adapter._should_process_message = lambda _message: True
    adapter._telegram_media_size_allowed = lambda _source, _label: (True, None)
    adapter._apply_telegram_group_observe_attribution = lambda event: event

    handled = []

    async def handle_message(event):
        handled.append(event)

    adapter.handle_message = handle_message
    adapter._build_message_event = lambda _message, message_type, update_id=None: SimpleNamespace(
        text="",
        message_type=message_type,
        media_urls=[],
        media_types=[],
    )

    file_obj = SimpleNamespace(
        file_path="telegram-video.mp4",
        download_as_bytearray=AsyncMock(return_value=bytearray(b"mp4-data")),
    )
    attachment = SimpleNamespace(
        file_size=128,
        get_file=AsyncMock(return_value=file_obj),
    )
    fields = {
        "caption": None,
        "sticker": None,
        "photo": None,
        "voice": None,
        "audio": None,
        "video": None,
        "video_note": None,
        "document": None,
        "media_group_id": None,
    }
    fields[attachment_attr] = attachment
    msg = SimpleNamespace(**fields)
    update = SimpleNamespace(message=msg, update_id=303)

    with patch(
        "plugins.platforms.telegram.adapter.cache_video_from_bytes_async",
        new=AsyncMock(return_value="/cache/video.mp4"),
    ) as cache_video:
        await TelegramAdapter._handle_media_message(adapter, update, SimpleNamespace())

    assert len(handled) == 1
    event = handled[0]
    assert event.message_type == MessageType.VIDEO
    assert event.media_urls == ["/cache/video.mp4"]
    assert event.media_types == ["video/mp4"]
    attachment.get_file.assert_awaited_once_with()
    file_obj.download_as_bytearray.assert_awaited_once_with()
    cache_video.assert_awaited_once_with(b"mp4-data", ext=".mp4")
