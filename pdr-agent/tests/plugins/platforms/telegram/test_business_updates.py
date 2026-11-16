"""Telegram Business updates are the owner's chats with third parties, never a DM to the bot.

PTB's ``Update.effective_message`` resolves ``business_message``, so without a guard core's text and
media handlers took an owner-written Business message (sender = the allowlisted owner, chat = the
contact) as an owner->bot DM and sent the agent's reply to the contact.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter

OWNER, CONTACT = 111, 4242


def _adapter() -> TelegramAdapter:
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="t", extra={"allow_from": [str(OWNER)]}))
    adapter.handle_message = AsyncMock()
    adapter.enqueued = []
    adapter._enqueue_text_event = adapter.enqueued.append
    return adapter


def _message(**fields):
    base = dict(message_id=7, chat=SimpleNamespace(id=CONTACT, type="private", first_name="Contact", title=None, is_forum=False),
                from_user=SimpleNamespace(id=OWNER, is_bot=False, username="owner", full_name="Owner"),
                text=None, caption=None, entities=[], caption_entities=[], message_thread_id=None,
                is_topic_message=False, reply_to_message=None, quote=None, date=None, forum_topic_created=None,
                media_group_id=None, photo=None, video=None, video_note=None, audio=None, voice=None,
                document=None, sticker=None, business_connection_id="bc-1", sender_chat=None)
    base.update(fields)
    return SimpleNamespace(**base)


def _update(msg, field="business_message"):
    fields = {k: None for k in ("message", "edited_message", "channel_post", "business_message",
                                "edited_business_message")}
    fields[field] = msg
    return SimpleNamespace(update_id=900, effective_message=msg, **fields)


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["business_message", "edited_business_message"])
async def test_business_text_from_the_owner_never_starts_a_turn(field):
    adapter = _adapter()
    adapter._is_user_authorized_from_message = lambda _m: True  # the owner IS allowlisted

    await adapter._handle_text_message(_update(_message(text="dinner tonight?"), field), SimpleNamespace())

    assert adapter.enqueued == []
    adapter.handle_message.assert_not_called()


@pytest.mark.asyncio
async def test_business_media_is_not_downloaded_or_dispatched():
    adapter = _adapter()
    adapter._is_user_authorized_from_message = lambda _m: True
    photo = SimpleNamespace(get_file=AsyncMock())

    await adapter._handle_media_message(_update(_message(photo=[photo], caption="look")), SimpleNamespace())

    photo.get_file.assert_not_awaited()
    adapter.handle_message.assert_not_called()


@pytest.mark.asyncio
async def test_a_real_owner_dm_still_reaches_the_agent():
    adapter = _adapter()
    adapter._is_user_authorized_from_message = lambda _m: True
    adapter._should_process_message = lambda _m, **_k: True
    dm = _message(chat=SimpleNamespace(id=OWNER, type="private", first_name="Owner", title=None, is_forum=False),
                  text="hello bot", business_connection_id=None)

    await adapter._handle_text_message(_update(dm, "message"), SimpleNamespace())

    assert [event.text for event in adapter.enqueued] == ["hello bot"]
