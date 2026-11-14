"""Every Telegram chat-type classifier routes through ``_normalize_chat_type`` (#74696).

Behaviour-preserving: each site keeps the answer it gave when it re-derived the rule inline.
``get_chat_info`` reports forum supergroups as ``forum``; an inbound event's ``chat_type`` stays
``group`` for any supergroup (the forum topic travels as ``thread_id``); a type that is not a known
group or channel is a DM at both.
"""

from types import SimpleNamespace

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.event import MessageType
from plugins.platforms.telegram.adapter import TelegramAdapter

# (chat.type, is_forum) -> (get_chat_info type, inbound event chat_type), as the inline code had it
CASES = {
    ("private", False): ("dm", "dm"),
    ("group", False): ("group", "group"),
    ("supergroup", False): ("group", "group"),
    ("supergroup", True): ("forum", "group"),
    ("channel", False): ("channel", "channel"),
    ("sender", False): ("dm", "dm"),  # unknown type
}


class _Bot:
    def __init__(self, chat):
        self._chat = chat

    async def get_chat(self, _chat_id):
        return self._chat


def _chat(kind, is_forum):
    return SimpleNamespace(id=-100123, type=kind, is_forum=is_forum, title="T", full_name="T", username=None)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind,is_forum", sorted(CASES))
async def test_chat_info_and_inbound_event_keep_their_chat_type(kind, is_forum):
    info_type, event_type = CASES[(kind, is_forum)]
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._bot = _Bot(_chat(kind, is_forum))
    assert (await adapter.get_chat_info("-100123"))["type"] == info_type

    message = SimpleNamespace(
        chat=_chat(kind, is_forum), from_user=SimpleNamespace(id=7, full_name="U", is_bot=False),
        message_id=5, message_thread_id=None, is_topic_message=False, text="hi", caption=None,
        entities=None, caption_entities=None, reply_to_message=None, date=None, forum_topic_created=None)
    event = adapter._build_message_event(message, MessageType.TEXT)
    assert event.source.chat_type == event_type
