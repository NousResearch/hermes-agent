import logging
from types import SimpleNamespace

import pytest

from gateway.config import Platform, PlatformConfig


PRIVATE_CHAT_ID = "SHOULD_NOT_LEAK_CHAT_ID"
PRIVATE_THREAD_ID = "424242"
PRIVATE_USERNAME = "SHOULD_NOT_LEAK_USERNAME"
PRIVATE_BODY = "SHOULD_NOT_LEAK_MESSAGE_BODY"
PRIVATE_FILE_ID = "SHOULD_NOT_LEAK_FILE_ID"
PRIVATE_URL = "SHOULD_NOT_LEAK_URL"
PRIVATE_PATH = "SHOULD_NOT_LEAK_PRIVATE_PATH"
PRIVATE_CONFIG_VALUE = "SHOULD_NOT_LEAK_PRIVATE_CONFIG_VALUE"


def _make_adapter(require_mention=True):
    from gateway.platforms.telegram import TelegramAdapter

    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter.config = PlatformConfig(
        enabled=True,
        token="***",
        extra={
            "require_mention": require_mention,
            "allowed_topics": [],
            "allowed_chats": [],
            "group_allowed_chats": [],
        },
    )
    adapter._bot = SimpleNamespace(id=999, username="hermes_bot")
    adapter._mention_patterns = []
    adapter._dm_topics_config = []
    adapter._dm_topic_chat_ids = set()
    adapter._is_callback_user_authorized = lambda user_id, **_kw: True
    return adapter


def _voice_message(*, text=PRIVATE_BODY, caption=PRIVATE_BODY, reply_to_bot=False):
    reply_to_message = None
    if reply_to_bot:
        reply_to_message = SimpleNamespace(
            from_user=SimpleNamespace(id=999),
            message_id=10,
            text="previous bot reply",
            caption=None,
        )
    return SimpleNamespace(
        message_id=42,
        text=text,
        caption=caption,
        entities=[],
        caption_entities=[],
        message_thread_id=PRIVATE_THREAD_ID,
        is_topic_message=True,
        chat=SimpleNamespace(id=PRIVATE_CHAT_ID, type="group", title="Private Group", is_forum=True),
        from_user=SimpleNamespace(id=111, full_name=PRIVATE_USERNAME, first_name="Test"),
        reply_to_message=reply_to_message,
        voice=SimpleNamespace(file_id=PRIVATE_FILE_ID, file_unique_id="unique", file_size=1234),
        audio=None,
        photo=None,
        video=None,
        document=None,
        sticker=None,
        file_path=PRIVATE_PATH,
        url=PRIVATE_URL,
        private_config=PRIVATE_CONFIG_VALUE,
        date=None,
    )


def test_voice_route_diagnostic_logs_redacted_drop_without_sensitive_values(caplog):
    adapter = _make_adapter(require_mention=True)
    message = _voice_message(reply_to_bot=False)

    with caplog.at_level(logging.INFO, logger="gateway.platforms.telegram"):
        accepted = adapter._should_process_message(message)
        adapter._log_voice_media_route_decision(message, accepted=accepted)

    assert accepted is False
    output = caplog.text
    assert "Telegram voice/media route diagnostic" in output
    assert "event_kind=voice" in output
    assert "route_decision=dropped" in output
    assert "drop_gate=route_gate" in output
    assert "stt_reached=no" in output
    assert "chat=<ID>" in output
    assert "thread=<ID>" in output

    for secret in (
        PRIVATE_CHAT_ID,
        PRIVATE_THREAD_ID,
        PRIVATE_USERNAME,
        PRIVATE_BODY,
        PRIVATE_FILE_ID,
        PRIVATE_URL,
        PRIVATE_PATH,
        PRIVATE_CONFIG_VALUE,
    ):
        assert secret not in output


def test_voice_route_diagnostic_logs_redacted_accept_without_sensitive_values(caplog):
    adapter = _make_adapter(require_mention=True)
    message = _voice_message(reply_to_bot=True)

    with caplog.at_level(logging.INFO, logger="gateway.platforms.telegram"):
        accepted = adapter._should_process_message(message)
        adapter._log_voice_media_route_decision(message, accepted=accepted)

    assert accepted is True
    output = caplog.text
    assert "event_kind=voice" in output
    assert "route_decision=accepted" in output
    assert "drop_gate=none" in output
    assert "stt_reached=no" in output

    for secret in (
        PRIVATE_CHAT_ID,
        PRIVATE_THREAD_ID,
        PRIVATE_USERNAME,
        PRIVATE_BODY,
        PRIVATE_FILE_ID,
        PRIVATE_URL,
        PRIVATE_PATH,
        PRIVATE_CONFIG_VALUE,
    ):
        assert secret not in output


@pytest.mark.asyncio
async def test_media_handler_emits_redacted_drop_diagnostic_before_cache_or_stt(caplog):
    adapter = _make_adapter(require_mention=True)

    def _do_not_observe(message):
        return False

    adapter._should_observe_unmentioned_group_message = _do_not_observe
    message = _voice_message(reply_to_bot=False)
    update = SimpleNamespace(message=message, update_id=100)

    with caplog.at_level(logging.INFO, logger="gateway.platforms.telegram"):
        await adapter._handle_media_message(update, SimpleNamespace())

    output = caplog.text
    assert "Telegram voice/media route diagnostic" in output
    assert "event_kind=voice" in output
    assert "route_decision=dropped" in output
    assert "drop_gate=route_gate" in output
    assert "stt_reached=no" in output
    assert "Cached user voice" not in output

    for secret in (
        PRIVATE_CHAT_ID,
        PRIVATE_THREAD_ID,
        PRIVATE_USERNAME,
        PRIVATE_BODY,
        PRIVATE_FILE_ID,
        PRIVATE_URL,
        PRIVATE_PATH,
        PRIVATE_CONFIG_VALUE,
    ):
        assert secret not in output


def test_ingress_diagnostic_logs_voice_before_media_handler_matching(caplog):
    adapter = _make_adapter(require_mention=True)
    message = _voice_message(reply_to_bot=True)
    update = SimpleNamespace(message=message, edited_message=None, channel_post=None, update_id=100)

    with caplog.at_level(logging.INFO, logger="gateway.platforms.telegram"):
        adapter._log_telegram_media_ingress_diagnostic(update)

    output = caplog.text
    assert "Telegram media ingress diagnostic" in output
    assert "event_kind=message" in output
    assert "message_kind=voice" in output
    assert "has_voice=yes" in output
    assert "has_audio=no" in output
    assert "has_media=yes" in output
    assert "media_handler_expected=yes" in output
    assert "chat=<ID>" in output
    assert "thread=<ID>" in output

    for secret in (
        PRIVATE_CHAT_ID,
        PRIVATE_THREAD_ID,
        PRIVATE_USERNAME,
        PRIVATE_BODY,
        PRIVATE_FILE_ID,
        PRIVATE_URL,
        PRIVATE_PATH,
        PRIVATE_CONFIG_VALUE,
    ):
        assert secret not in output


def test_ingress_diagnostic_ignores_plain_text_to_avoid_unrelated_log_noise(caplog):
    adapter = _make_adapter(require_mention=True)
    text_message = SimpleNamespace(
        message_id=43,
        text="plain group chatter",
        caption=None,
        entities=[],
        caption_entities=[],
        message_thread_id=None,
        chat=SimpleNamespace(id=PRIVATE_CHAT_ID, type="group", title="Private Group", is_forum=False),
        from_user=SimpleNamespace(id=111, full_name=PRIVATE_USERNAME, first_name="Test"),
        reply_to_message=None,
        voice=None,
        audio=None,
        photo=None,
        video=None,
        document=None,
        sticker=None,
        date=None,
    )
    update = SimpleNamespace(message=text_message, edited_message=None, channel_post=None, update_id=101)

    with caplog.at_level(logging.INFO, logger="gateway.platforms.telegram"):
        adapter._log_telegram_media_ingress_diagnostic(update)
        assert adapter._should_process_message(text_message) is False

    assert "Telegram media ingress diagnostic" not in caplog.text
    assert "Telegram voice/media route diagnostic" not in caplog.text
