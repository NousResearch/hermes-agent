"""``hermes config set telegram.allowed_chats '["a","b"]'`` stores a JSON-encoded *string*; every
Telegram allowlist reader must decode it instead of comma-splitting the brackets onto the ids."""

from types import SimpleNamespace

from gateway.config import Platform, PlatformConfig


def _adapter(extra):
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter.config = PlatformConfig(enabled=True, token="***", extra=extra)
    adapter._bot = SimpleNamespace(id=999, username="hermes_bot")
    return adapter


def _group_msg(chat_id):
    return SimpleNamespace(
        message_id=42, text="hello", caption=None, entities=[], caption_entities=[],
        message_thread_id=None, reply_to_message=None, date=None,
        chat=SimpleNamespace(id=chat_id, type="group", title="G", is_forum=False),
        from_user=SimpleNamespace(id=111, full_name="A B", first_name="A"),
    )


def test_json_string_allowlists_decode_across_every_key():
    adapter = _adapter({
        "allowed_chats": '["-100","-200"]',
        "group_allowed_chats": '["-300"]',
        "allowed_topics": '["5"]',
        "free_response_chats": '["-400"]',
        "free_response_topics": '["-100:3"]',
        "ignored_threads": '["7", "9"]',
    })
    assert adapter._telegram_allowed_chats() == {"-100", "-200"}
    assert adapter._telegram_group_allowed_chats() == {"-300"}
    assert adapter._telegram_allowed_topics() == {"5"}
    assert adapter._telegram_free_response_chats() == {"-400"}
    assert adapter._telegram_free_response_topics() == {"-100:3"}
    assert adapter._telegram_ignored_threads() == {7, 9}
    # The user-visible symptom: a JSON-string allowlist dropped every group message.
    gated = _adapter({"allowed_chats": '["-100","-200"]'})
    assert gated._should_process_message(_group_msg(-100)) is True
    assert gated._should_process_message(_group_msg(-999)) is False


def test_comma_and_malformed_strings_keep_the_legacy_split():
    assert _adapter({"allowed_chats": "-100, -200"})._telegram_allowed_chats() == {"-100", "-200"}
    assert _adapter({"allowed_chats": ["-100", "-200"]})._telegram_allowed_chats() == {"-100", "-200"}
    assert _adapter({"allowed_chats": '["-100", "-200'})._telegram_allowed_chats() == {'["-100"', '"-200'}


def test_runner_side_allow_set_decodes_json_string(monkeypatch):
    """The runner's central gate reads the same env chain (``TELEGRAM_GROUP_ALLOWED_CHATS``
    via the YAML bridge) and must not comma-split the brackets onto the ids either."""
    from gateway.authz_mixin import _coerce_allow_set

    monkeypatch.setenv("TELEGRAM_GROUP_ALLOWED_CHATS", '["-100","-200"]')
    from gateway.platforms._shared import platform_gate_env

    assert _coerce_allow_set(platform_gate_env("TELEGRAM_GROUP_ALLOWED_CHATS")) == {"-100", "-200"}
    assert _coerce_allow_set("-100, -200") == {"-100", "-200"}
    assert _coerce_allow_set(["-100"]) == {"-100"}
