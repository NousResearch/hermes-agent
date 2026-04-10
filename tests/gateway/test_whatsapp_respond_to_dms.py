"""Tests for the respond_to_dms flag on WhatsApp profiles.

When running multiple agent profiles on the same WhatsApp number (e.g. a
family agent that should only operate in specific group chats), setting
respond_to_dms=false in the platform config causes the adapter to ignore
all DM messages.
"""

from unittest.mock import AsyncMock

from gateway.config import Platform, PlatformConfig


def _make_adapter(**extra_overrides):
    from gateway.platforms.whatsapp import WhatsAppAdapter

    extra = {}
    extra.update(extra_overrides)

    adapter = object.__new__(WhatsAppAdapter)
    adapter.platform = Platform.WHATSAPP
    adapter.config = PlatformConfig(enabled=True, extra=extra)
    adapter._message_handler = AsyncMock()
    adapter._mention_patterns = adapter._compile_mention_patterns()
    return adapter


def _dm_message(body="hello"):
    return {
        "isGroup": False,
        "body": body,
        "chatId": "4912345678@s.whatsapp.net",
        "mentionedIds": [],
        "botIds": ["15551230000@s.whatsapp.net"],
        "quotedParticipant": "",
    }


def test_dms_accepted_by_default():
    """Without respond_to_dms set, DMs are processed normally."""
    adapter = _make_adapter()
    assert adapter._should_process_message(_dm_message()) is True


def test_dms_accepted_when_explicitly_true():
    """respond_to_dms=true should behave the same as the default."""
    adapter = _make_adapter(respond_to_dms=True)
    assert adapter._should_process_message(_dm_message()) is True


def test_dms_rejected_when_false():
    """respond_to_dms=false should cause DMs to be ignored."""
    adapter = _make_adapter(respond_to_dms=False)
    assert adapter._should_process_message(_dm_message()) is False


def test_groups_unaffected_by_respond_to_dms_false():
    """respond_to_dms only controls DMs — group messages should still
    be processed via the normal group gating logic."""
    adapter = _make_adapter(
        respond_to_dms=False,
        free_response_chats="120363001234567890@g.us",
    )
    group_msg = {
        "isGroup": True,
        "body": "hello",
        "chatId": "120363001234567890@g.us",
        "mentionedIds": [],
        "botIds": ["15551230000@s.whatsapp.net"],
        "quotedParticipant": "",
    }
    assert adapter._should_process_message(group_msg) is True
