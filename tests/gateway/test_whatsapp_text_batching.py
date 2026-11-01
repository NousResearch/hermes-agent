"""Text-debounce batching for the WhatsApp adapter (issue #35301).

WhatsApp delivers rapid multi-message bursts (forwarded batches, paste-splits)
individually.  Without debounce each fragment triggers a separate agent
invocation, wasting tokens and flooding the user with reply fragments.  This
mirrors the Telegram/WeCom/Feishu pattern.

Batch delays are read from ``config.extra`` (config.yaml), not env vars.

Caption-less media batching (issue #137832): a photo/document/video sent bare
is held for ``media_caption_wait_seconds`` so the instruction typed right after
it ("please print it") merges into one agent turn instead of interrupting the
turn the bare file spawned. Mirrors WeCom's attachment_text_merge window.
"""


import asyncio
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType
from plugins.platforms.whatsapp.adapter import WhatsAppAdapter
from gateway.session import SessionSource
from tests.waiting import async_wait_until


def _make_adapter(**extra):
    base = {"session_name": "test"}
    base.update(extra)
    return WhatsAppAdapter(PlatformConfig(enabled=True, extra=base))


def _event(text):
    src = SessionSource(
        platform=Platform.WHATSAPP,
        chat_id="chat123",
        chat_type="dm",
        user_id="user1",
        user_name="tester",
    )
    return MessageEvent(text=text, message_type=MessageType.TEXT, source=src)


def test_batch_delays_overridden_via_config_extra():
    adapter = _make_adapter(
        text_batch_delay_seconds="1.5",
        text_batch_split_delay_seconds=3,
    )
    assert adapter._text_batch_delay_seconds == 1.5
    assert adapter._text_batch_split_delay_seconds == 3.0


def test_invalid_config_value_falls_back_to_default():
    adapter = _make_adapter(
        text_batch_delay_seconds="garbage",
        text_batch_split_delay_seconds=-3,
    )
    default = _make_adapter()
    assert adapter._text_batch_delay_seconds == default._text_batch_delay_seconds
    assert adapter._text_batch_split_delay_seconds == default._text_batch_split_delay_seconds


def _media_event(msg_type=MessageType.DOCUMENT, text="", media_urls=None):
    src = SessionSource(
        platform=Platform.WHATSAPP,
        chat_id="chat123",
        chat_type="dm",
        user_id="user1",
        user_name="tester",
    )
    return MessageEvent(
        text=text,
        message_type=msg_type,
        source=src,
        media_urls=media_urls or ["/cache/doc_deadbeef.pdf"],
        media_types=["application/pdf"],
    )


def test_media_caption_wait_read_from_config_extra():
    assert _make_adapter()._media_caption_wait_seconds == 8.0
    tuned = _make_adapter(media_caption_wait_seconds="2.5")
    assert tuned._media_caption_wait_seconds == 2.5
    # ceiling clamps a misconfigured value; garbage falls back to the default
    assert _make_adapter(media_caption_wait_seconds=999)._media_caption_wait_seconds == 30.0
    assert _make_adapter(media_caption_wait_seconds="soon")._media_caption_wait_seconds == 8.0


def test_captionless_media_waits_the_caption_window():
    adapter = _make_adapter()
    assert adapter._text_batch_delay_for(_media_event()) == adapter._media_caption_wait_seconds
    # captioned media, voice notes and plain text keep the normal quiet period
    assert adapter._text_batch_delay_for(_media_event(text="print this")) == adapter._text_batch_delay_seconds
    assert adapter._text_batch_delay_for(_media_event(MessageType.VOICE)) == adapter._text_batch_delay_seconds
    assert adapter._text_batch_delay_for(_event("hello")) == adapter._text_batch_delay_seconds


def test_route_batchable_event_seam():
    adapter = _make_adapter()
    assert adapter._route_batchable_event(_event("hello")) is True
    assert adapter._route_batchable_event(_media_event()) is True  # caption-less document waits
    assert adapter._route_batchable_event(_media_event(MessageType.PHOTO)) is True
    assert adapter._route_batchable_event(_media_event(MessageType.VIDEO)) is True
    assert adapter._route_batchable_event(_media_event(MessageType.VOICE)) is False  # voice notes: never held
    assert adapter._route_batchable_event(_media_event(text="print this")) is False  # captioned: direct dispatch
    opted_out = _make_adapter(media_caption_wait_seconds=0)
    assert opted_out._route_batchable_event(_media_event()) is False  # 0 restores the old behavior


class TestCaptionlessMediaBatching:
    def _adapter(self, caption_wait=0.2):
        adapter = _make_adapter(media_caption_wait_seconds=caption_wait)
        adapter.handle_message = AsyncMock()
        return adapter

    @pytest.mark.asyncio
    async def test_file_then_text_is_one_dispatch(self):
        adapter = self._adapter()
        adapter._enqueue_text_event(_media_event())
        # still inside the caption window when the trailing text lands
        adapter._enqueue_text_event(_event("please print it"))

        await async_wait_until(lambda: adapter.handle_message.called, timeout=2)
        adapter.handle_message.assert_called_once()
        dispatched = adapter.handle_message.call_args[0][0]
        assert dispatched.media_urls == ["/cache/doc_deadbeef.pdf"]
        assert "please print it" in dispatched.text

    @pytest.mark.asyncio
    async def test_file_alone_dispatched_after_the_caption_window(self):
        adapter = self._adapter()
        adapter._enqueue_text_event(_media_event())

        await asyncio.sleep(0.05)  # health: allow HX013 -- the hold itself is the behaviour under test: nothing may dispatch inside the caption window
        adapter.handle_message.assert_not_called()

        await async_wait_until(lambda: adapter.handle_message.called, timeout=2)
        dispatched = adapter.handle_message.call_args[0][0]
        assert dispatched.media_urls == ["/cache/doc_deadbeef.pdf"]


