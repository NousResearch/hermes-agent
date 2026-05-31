"""Per-topic system_prompt overlay (#5195) must apply to dm_topics AND group_topics.

The prompt is nested under the topic config (scoped by chat_id), so it surfaces
on event.channel_prompt for both DM topics and group forum topics.
"""

from gateway.platforms.base import MessageType
from tests.gateway.test_dm_topics import _make_adapter, _make_mock_message
from telegram.constants import ChatType as _ChatType


def test_dm_topic_system_prompt_sets_channel_prompt():
    adapter = _make_adapter([
        {
            "chat_id": 111,
            "topics": [
                {"name": "Research", "thread_id": 100, "system_prompt": "You are a researcher."},
            ],
        }
    ])
    adapter._dm_topics["111:Research"] = 100

    event = adapter._build_message_event(_make_mock_message(chat_id=111, thread_id=100), MessageType.TEXT)
    assert event.channel_prompt == "You are a researcher."


def test_group_topic_system_prompt_sets_channel_prompt():
    adapter = _make_adapter(group_topics_config=[
        {
            "chat_id": -1001234567890,
            "topics": [
                {"name": "Eng", "thread_id": 5, "system_prompt": "Speak like an engineer."},
            ],
        }
    ])
    msg = _make_mock_message(chat_id=-1001234567890, chat_type=_ChatType.SUPERGROUP, thread_id=5, is_topic_message=True, is_forum=True)
    event = adapter._build_message_event(msg, MessageType.TEXT)
    assert event.channel_prompt == "Speak like an engineer."


def test_dm_topic_without_prompt_is_none():
    adapter = _make_adapter([
        {"chat_id": 111, "topics": [{"name": "General", "thread_id": 200}]},
    ])
    adapter._dm_topics["111:General"] = 200
    event = adapter._build_message_event(_make_mock_message(chat_id=111, thread_id=200), MessageType.TEXT)
    assert event.channel_prompt is None
