"""Allowlists stored as a JSON-list *string* are honoured by every platform gate (issue #76457).

Configs written by ``hermes config set KEY '["-100","-200"]'`` before the writer learned
to emit YAML lists hold the literal ``'["-100","-200"]'`` as a string. The Telegram gate
already decodes that shape (``gateway/platforms/_shared.py::decode_json_list_literal``);
the Discord / WhatsApp / DingTalk gates comma-split it into one bogus entry that matches
nothing, silently locking out every allowlisted chat or user.
"""

from gateway.config import Platform, PlatformConfig

BRACKET_LIST = '["-100", "-200"]'


def _discord(extra):
    from plugins.platforms.discord.adapter import DiscordAdapter

    adapter = object.__new__(DiscordAdapter)
    adapter.platform = Platform.DISCORD
    adapter.config = PlatformConfig(enabled=True, token="x", extra=dict(extra))
    adapter._gate_env_snapshot = None
    return adapter


def _whatsapp(extra):
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

    adapter = object.__new__(WhatsAppAdapter)
    adapter.platform = Platform.WHATSAPP
    adapter.config = PlatformConfig(enabled=True, extra=dict(extra))
    return adapter


def _dingtalk(extra):
    from plugins.platforms.dingtalk.adapter import DingTalkAdapter

    return DingTalkAdapter(PlatformConfig(enabled=True, extra=dict(extra)))


def test_bracket_list_string_is_decoded_by_every_platform_gate():
    assert _discord({"allowed_channels": BRACKET_LIST})._get_allowed_channels() == {"-100", "-200"}
    assert _discord({"free_response_channels": BRACKET_LIST})._discord_free_response_channels() == {"-100", "-200"}
    assert _whatsapp({"free_response_chats": BRACKET_LIST})._whatsapp_free_response_chats() == {"-100", "-200"}
    dingtalk = _dingtalk({"allowed_chats": BRACKET_LIST, "allowed_users": '["Alice"]'})
    assert dingtalk._dingtalk_allowed_chats() == {"-100", "-200"}
    assert dingtalk._allowed_users == {"alice"}


def test_plain_csv_and_yaml_lists_keep_their_meaning():
    assert _discord({"allowed_channels": "-100, -200"})._get_allowed_channels() == {"-100", "-200"}
    assert _discord({"allowed_channels": ["-100", "-200"]})._get_allowed_channels() == {"-100", "-200"}
    assert _whatsapp({"free_response_chats": "a,b"})._whatsapp_free_response_chats() == {"a", "b"}
    # Malformed JSON is not a list: it stays on the legacy comma-split path.
    assert _dingtalk({"allowed_chats": "[not-json"})._dingtalk_allowed_chats() == {"[not-json"}
