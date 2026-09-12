import json
from types import SimpleNamespace

from gateway.config import Platform, PlatformConfig


def _make_json_adapter(allowed_chats):
    from plugins.platforms.telegram.adapter import TelegramAdapter

    extra = {
        "allowed_chats": allowed_chats,
        "allowed_topics": [],
        "group_allowed_chats": [],
    }
    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter.config = PlatformConfig(enabled=True, token="***", extra=extra)
    adapter._bot = SimpleNamespace(id=999, username="hermes_bot")
    return adapter


def _group_msg(chat_id=-100):
    return SimpleNamespace(
        message_id=42,
        text="hello",
        caption=None,
        entities=[],
        caption_entities=[],
        message_thread_id=None,
        chat=SimpleNamespace(id=chat_id, type="group", title="G", is_forum=False),
        from_user=SimpleNamespace(id=111, full_name="A B", first_name="A"),
        reply_to_message=None,
        date=None,
    )


def test_allowed_chats_json_string_parses_as_allowlist():
    adapter = _make_json_adapter('["-100","-200"]')
    assert adapter._telegram_allowed_chats() == {"-100", "-200"}


def test_allowed_chats_json_string_end_to_end_gating():
    adapter = _make_json_adapter(json.dumps(["-100"]))
    assert adapter._should_process_message(_group_msg(chat_id=-100)) is True
    assert adapter._should_process_message(_group_msg(chat_id=-300)) is False


def test_allowed_chats_comma_string_still_works():
    adapter = _make_json_adapter("-100,-200")
    assert adapter._telegram_allowed_chats() == {"-100", "-200"}


def test_allowed_chats_native_list_still_works():
    adapter = _make_json_adapter(["-100", "-200"])
    assert adapter._telegram_allowed_chats() == {"-100", "-200"}


def test_allowed_chats_malformed_json_falls_back_to_comma_split():
    adapter = _make_json_adapter('["-100", "-200')
    assert adapter._telegram_allowed_chats() == {'["-100"', '"-200'}


def test_ignored_threads_json_string_parses():
    from plugins.platforms.telegram.adapter import TelegramAdapter

    extra = {"ignored_threads": '["7", "9"]'}
    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter.config = PlatformConfig(enabled=True, token="***", extra=extra)
    adapter._bot = SimpleNamespace(id=999, username="hermes_bot")
    assert adapter._telegram_ignored_threads() == {7, 9}


def test_all_allowlist_keys_decode_json_string():
    adapter = _make_json_adapter('["-100"]')
    adapter.config.extra["group_allowed_chats"] = '["-300"]'
    adapter.config.extra["allowed_topics"] = '["5"]'
    adapter.config.extra["free_response_chats"] = '["-400"]'
    adapter.config.extra["free_response_topics"] = '["-100:3"]'
    assert adapter._telegram_group_allowed_chats() == {"-300"}
    assert adapter._telegram_allowed_topics() == {"5"}
    assert adapter._telegram_free_response_chats() == {"-400"}
    assert adapter._telegram_free_response_topics() == {"-100:3"}
