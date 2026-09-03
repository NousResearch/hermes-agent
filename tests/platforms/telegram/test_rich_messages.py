from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageType
from plugins.platforms.telegram.adapter import (
    TelegramAdapter,
    TelegramMessageHandler,
    _RichMessageFilter,
)
from plugins.platforms.telegram.rich_messages import (
    _render_list,
    rich_message_to_markdown,
    rich_message_to_plaintext,
)


def _adapter():
    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter.config = PlatformConfig(enabled=True, token="fake-token", extra={})
    adapter._bot = SimpleNamespace(id=1, username="bot")
    adapter._ensure_forum_commands = AsyncMock()
    adapter._cache_replied_media = AsyncMock()
    adapter._clean_bot_trigger_text = lambda text: text
    adapter._apply_telegram_group_observe_attribution = lambda event: event
    adapter._should_process_message = lambda msg, is_command=False: True
    adapter._should_observe_unmentioned_group_message = lambda msg: False
    adapter._observe_unmentioned_group_message = lambda *args, **kwargs: None
    adapter._rich_inbound_handling = True
    adapter._pending_text_batches = {}
    adapter._pending_text_batch_tasks = {}
    adapter._forum_lock = AsyncMock()
    adapter._forum_command_registered = set()
    adapter._active_sessions = {}
    adapter._pending_messages = {}
    return adapter


def _message(
    *,
    text=None,
    rich_message=None,
    caption=None,
    photo=None,
    document=None,
    video=None,
    audio=None,
    voice=None,
    sticker=None,
):
    payload = {"rich_message": rich_message} if rich_message is not None else {}

    class Msg(SimpleNamespace):
        def to_dict(self):
            return payload

    return Msg(
        message_id=7,
        text=text,
        caption=caption,
        photo=photo,
        document=document,
        video=video,
        audio=audio,
        voice=voice,
        sticker=sticker,
        entities=[],
        caption_entities=[],
        message_thread_id=None,
        is_topic_message=False,
        chat=SimpleNamespace(id=-100, type="group", title="Test Group", is_forum=False),
        from_user=SimpleNamespace(id=123, full_name="User"),
        reply_to_message=None,
        date=None,
    )


def _update(msg):
    return SimpleNamespace(update_id=99, message=msg, effective_message=msg)


def test_rich_message_to_markdown_handles_representative_payload():
    rich = {
        "blocks": [
            {"type": "heading", "size": 2, "text": "Heading"},
            {"type": "paragraph", "text": ["Hello ", {"type": "bold", "text": "world"}]},
            {
                "type": "list",
                "items": [
                    {
                        "blocks": [{"type": "paragraph", "text": [{"type": "bold", "text": "Item"}]}],
                    },
                    {
                        "value": 2,
                        "blocks": [{"type": "paragraph", "text": "Second"}],
                    },
                ],
            },
            {
                "type": "blockquote",
                "blocks": [{"type": "paragraph", "text": "Quoted"}],
            },
            {"type": "pre", "language": "python", "text": "print('hi')"},
            {"type": "divider"},
            {"type": "details", "summary": "More", "blocks": [{"type": "paragraph", "text": "Inside"}]},
            {
                "type": "table",
                "caption": {"text": "Metrics"},
                "cells": [
                    [
                        {"text": {"type": "bold", "text": "Name"}, "is_header": True},
                        {"text": "Value", "is_header": True},
                    ],
                    [
                        {"text": "Latency"},
                        {"text": "42ms"},
                    ],
                ],
            },
        ]
    }

    md = rich_message_to_markdown(rich)

    assert "## Heading" in md
    assert "Hello **world**" in md
    assert "- **Item**" in md
    assert "2. Second" in md
    assert "> Quoted" in md
    assert "```python" in md
    assert "---" in md
    assert "**More**" in md
    assert "Metrics" in md
    assert "| **Name** | Value |" in md
    assert "| Latency | 42ms |" in md


@pytest.mark.parametrize(
    "rich_text, expected",
    [
        ({"type": "italic", "text": "ital"}, "*ital*"),
        ({"type": "underline", "text": "under"}, "__under__"),
        ({"type": "strikethrough", "text": "gone"}, "~~gone~~"),
        ({"type": "spoiler", "text": "secret"}, "||secret||"),
        ({"type": "marked", "text": "mark"}, "==mark=="),
        ({"type": "subscript", "text": "x"}, "~x~"),
        ({"type": "superscript", "text": "2"}, "^2^"),
        ({"type": "code", "text": "x=1"}, "`x=1`"),
        ({"type": "url", "text": "site", "url": "https://example.com"}, "[site](https://example.com)"),
        ({"type": "email_address", "text": "mail", "email_address": "a@example.com"}, "[mail](mailto:a@example.com)"),
        ({"type": "mention", "text": "bob"}, "@bob"),
        ({"type": "text_mention", "text": "Alice", "user": {"id": 1234}}, "[Alice](tg://user?id=1234)"),
        ({"type": "custom_emoji", "text": "smile", "custom_emoji_id": "e1"}, "smile"),
        ({"type": "mathematical_expression", "expression": "a+b"}, "$a+b$"),
        ({"type": "hashtag", "text": "#tag"}, "#tag"),
        ({"type": "anchor", "text": "hidden"}, "hidden"),
    ],
)
def test_rich_text_subtypes_render_to_sane_markdown(rich_text, expected):
    md = rich_message_to_markdown({"blocks": [{"type": "paragraph", "text": rich_text}]})
    assert expected in md


def test_unknown_types_fall_back_gracefully():
    md = rich_message_to_markdown({"blocks": [{"type": "mystery", "text": "fallback"}]})
    assert md == "fallback"
    assert rich_message_to_markdown(None) == ""
    assert rich_message_to_markdown({"blocks": None}) == ""


def test_plaintext_fallback_walks_nested_text():
    rich = {
        "blocks": [
            {"type": "paragraph", "text": "Top"},
            {
                "type": "details",
                "summary": "Ignored",
                "blocks": [{"type": "paragraph", "text": ["Nested ", {"type": "bold", "text": "text"}]}],
            },
        ]
    }
    assert rich_message_to_plaintext(rich) == "Top\nIgnored\nNested \ntext"


@pytest.mark.asyncio
async def test_rich_message_handler_builds_text_event_from_markdown():
    adapter = _adapter()
    captured = []
    adapter._enqueue_text_event = lambda event: captured.append(event)
    msg = _message(
        rich_message={
            "blocks": [
                {"type": "heading", "size": 1, "text": "Title"},
                {"type": "paragraph", "text": ["Body ", {"type": "bold", "text": "here"}]},
            ]
        }
    )

    await adapter._handle_rich_message(_update(msg), SimpleNamespace())

    assert len(captured) == 1
    assert captured[0].message_type == MessageType.TEXT
    assert captured[0].text == "# Title\nBody **here**"
    adapter._ensure_forum_commands.assert_awaited_once_with(msg)
    adapter._cache_replied_media.assert_awaited_once()


@pytest.mark.asyncio
async def test_rich_message_handler_respects_disable_flag():
    adapter = _adapter()
    adapter._rich_inbound_handling = False
    adapter._build_message_event = lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not build"))
    captured = []
    adapter._enqueue_text_event = lambda event: captured.append(event)
    msg = _message(
        rich_message={
            "blocks": [
                {"type": "paragraph", "text": "rich"},
            ]
        }
    )

    await adapter._handle_rich_message(_update(msg), SimpleNamespace())

    assert captured == []
    adapter._ensure_forum_commands.assert_not_awaited()
    adapter._cache_replied_media.assert_not_awaited()


@pytest.mark.asyncio
async def test_rich_message_handler_ignores_text_bearing_messages():
    adapter = _adapter()
    adapter._build_message_event = lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not build"))
    msg = _message(text="plain", rich_message={"blocks": [{"type": "paragraph", "text": "rich"}]})

    await adapter._handle_rich_message(_update(msg), SimpleNamespace())


@pytest.mark.asyncio
async def test_rich_message_handler_drops_unauthorized_users():
    adapter = _adapter()
    adapter._is_user_authorized_from_message = lambda msg: False
    adapter._build_message_event = lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not build"))
    captured = []
    adapter._enqueue_text_event = lambda event: captured.append(event)
    msg = _message(
        rich_message={
            "blocks": [
                {"type": "paragraph", "text": "rich"},
            ]
        }
    )

    await adapter._handle_rich_message(_update(msg), SimpleNamespace())

    assert captured == []
    adapter._ensure_forum_commands.assert_not_awaited()
    adapter._cache_replied_media.assert_not_awaited()


def test_rich_message_filter_matches_rich_only_messages():
    filt = _RichMessageFilter()
    assert filt.filter(_message(rich_message={"blocks": [{"type": "paragraph", "text": "x"}]}))
    assert not filt.filter(_message(text="hello", rich_message={"blocks": [{"type": "paragraph", "text": "x"}]}))


def test_rich_message_filter_returns_false_when_to_dict_raises():
    filt = _RichMessageFilter()

    class BrokenMessage(SimpleNamespace):
        text = None

        def to_dict(self):
            raise RuntimeError("boom")

    assert not filt.filter(BrokenMessage())


def test_rich_message_filter_falls_through_for_photo_with_rich_caption():
    filt = _RichMessageFilter()
    msg = _message(
        caption="rich caption",
        photo=[SimpleNamespace(file_id="p1")],
        rich_message={"blocks": [{"type": "paragraph", "text": "x"}]},
    )
    assert filt.filter(msg) is False


def test_rich_message_filter_falls_through_for_document_with_rich_caption():
    filt = _RichMessageFilter()
    msg = _message(
        caption="rich caption",
        document=[SimpleNamespace(file_id="d1")],
        rich_message={"blocks": [{"type": "paragraph", "text": "x"}]},
    )
    assert filt.filter(msg) is False


@pytest.mark.parametrize("attr", ["video", "audio", "voice", "sticker"])
def test_rich_message_filter_excludes_attachment_bearing_messages(attr):
    filt = _RichMessageFilter()
    msg = _message(
        caption="rich caption",
        rich_message={"blocks": [{"type": "paragraph", "text": "x"}]},
        **{attr: [SimpleNamespace(file_id="x")]},
    )
    assert filt.filter(msg) is False


@pytest.mark.parametrize("attr", ["photo", "document", "video", "audio", "voice", "sticker"])
def test_rich_message_filter_short_circuits_before_to_dict_for_media(attr):
    filt = _RichMessageFilter()
    msg = _message(**{attr: [SimpleNamespace(file_id="x")]})
    calls = []

    def recording_to_dict():
        calls.append(1)
        raise AssertionError("to_dict must not be called for media messages")

    msg.to_dict = recording_to_dict
    assert filt.filter(msg) is False
    assert calls == []


class _MediaAttrFilter:
    """Truthy-media-attr stub mirroring the adapter's media-handler filter."""

    def filter(self, message):
        return any(
            getattr(message, attr, None)
            for attr in ("photo", "document", "video", "audio", "voice", "sticker")
        )


class _DispatchApp:
    """Minimal PTB-like app: handlers claim updates in registration order."""

    def __init__(self):
        self.handlers = []

    def add_handler(self, handler, group=None):
        self.handlers.append((handler, group))

    async def dispatch(self, update):
        for handler, group in self.handlers:
            if group is not None:
                continue
            if handler.filters.filter(update.effective_message):
                await handler.callback(update, SimpleNamespace())
                return handler
        return None


@pytest.mark.asyncio
async def test_photo_with_rich_caption_reaches_media_path_not_rich_handler():
    rich_cb = AsyncMock()
    media_cb = AsyncMock()
    app = _DispatchApp()
    app.add_handler(TelegramMessageHandler(_RichMessageFilter(), rich_cb))
    app.add_handler(TelegramMessageHandler(_MediaAttrFilter(), media_cb))

    msg = _message(
        caption="rich caption",
        photo=[SimpleNamespace(file_id="p1")],
        rich_message={"blocks": [{"type": "paragraph", "text": "x"}]},
    )
    matched = await app.dispatch(_update(msg))

    assert matched is not None
    media_cb.assert_awaited_once()
    rich_cb.assert_not_awaited()


class _RecordingApp:
    def __init__(self):
        self.handlers = []

    def add_handler(self, handler, group=None):
        self.handlers.append((handler, group))


def _rich_handlers(app):
    return [
        handler
        for handler, _group in app.handlers
        if isinstance(getattr(handler, "filters", None), _RichMessageFilter)
    ]


def test_rich_handler_not_registered_when_inbound_disabled():
    adapter = TelegramAdapter(
        PlatformConfig(enabled=True, token="fake-token", extra={"rich_inbound_handling": False})
    )
    app = _RecordingApp()
    adapter._register_handlers(app)
    assert _rich_handlers(app) == []


def test_rich_handler_registered_when_inbound_enabled():
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="fake-token", extra={}))
    app = _RecordingApp()
    adapter._register_handlers(app)
    assert len(_rich_handlers(app)) == 1


def test_table_cell_escapes_pipes():
    rich = {
        "blocks": [
            {
                "type": "table",
                "cells": [
                    [{"text": "Name", "is_header": True}, {"text": "Value", "is_header": True}],
                    [{"text": "A|B"}, {"text": "ok"}],
                ],
            }
        ]
    }
    md = rich_message_to_markdown(rich)
    assert "| A\\|B | ok |" in md
    assert "A|B |" not in md


def test_list_prefixes_ordered_and_unordered():
    rich = {
        "blocks": [
            {
                "type": "list",
                "items": [
                    {"value": 2, "blocks": [{"type": "paragraph", "text": "x"}]},
                    {"blocks": [{"type": "paragraph", "text": "y"}]},
                ],
            }
        ]
    }
    md = rich_message_to_markdown(rich)
    assert "2. x" in md
    assert "- y" in md


def test_render_list_has_no_dead_ordered_branch():
    import inspect

    src = inspect.getsource(_render_list)
    assert "ordered = value is not None" not in src
    assert '"1. " if ordered' not in src
