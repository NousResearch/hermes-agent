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


# ---------------------------------------------------------------------------
# 587bb105 covered the config-`extra`-backed gates above. These four remaining
# gates read a platform allowlist ENV VAR directly (not via `extra`) and were
# still on the raw comma split, so a JSON-list-shaped env value locked out
# every allowlisted user instead of just matching nothing.
# ---------------------------------------------------------------------------

def test_teams_card_action_gate_decodes_bracket_list_env(monkeypatch):
    from types import SimpleNamespace

    from plugins.platforms.teams.adapter import TeamsAdapter

    monkeypatch.setenv("TEAMS_ALLOWED_USERS", '["aad-111"]')
    monkeypatch.delenv("TEAMS_ALLOW_ALL_USERS", raising=False)
    clicker = SimpleNamespace(aad_object_id="aad-111", id="x")
    assert TeamsAdapter._card_action_denied(clicker) is None
    other = SimpleNamespace(aad_object_id="someone-else", id="y")
    assert TeamsAdapter._card_action_denied(other) is not None


def test_feishu_allowed_group_users_decodes_bracket_list_env(monkeypatch):
    from plugins.platforms.feishu.adapter import FeishuAdapter

    monkeypatch.setenv("FEISHU_ALLOWED_USERS", '["ou_111", "ou_222"]')
    settings = FeishuAdapter._load_settings({})
    assert settings.allowed_group_users == frozenset({"ou_111", "ou_222"})


def test_discord_component_auth_decodes_bracket_list_gateway_allowed_users(monkeypatch):
    from types import SimpleNamespace

    from plugins.platforms.discord.adapter import _component_check_auth

    monkeypatch.setenv("GATEWAY_ALLOWED_USERS", '["999"]')
    monkeypatch.delenv("DISCORD_ALLOW_ALL_USERS", raising=False)
    monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)
    interaction = SimpleNamespace(user=SimpleNamespace(id="999"))
    assert _component_check_auth(interaction, set(), None) is True
    other = SimpleNamespace(user=SimpleNamespace(id="000"))
    assert _component_check_auth(other, set(), None) is False


def test_telegram_env_allowlist_decision_decodes_bracket_list(monkeypatch):
    from plugins.platforms.telegram.adapter import TelegramAdapter

    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", '["888"]')
    assert TelegramAdapter._env_allowlist_decision("888") is True
    assert TelegramAdapter._env_allowlist_decision("000") is False


def test_slack_interactive_env_fallback_decodes_bracket_list(monkeypatch):
    from plugins.platforms.slack.adapter import SlackAdapter

    # object.__new__: no injected auth check and no runner handler, so the env-only fallback runs.
    adapter = object.__new__(SlackAdapter)
    for var in ("SLACK_ALLOW_ALL_USERS", "GATEWAY_ALLOW_ALL_USERS", "GATEWAY_ALLOWED_USERS"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("SLACK_ALLOWED_USERS", '["U111", "U222"]')
    assert adapter._is_interactive_user_authorized("U111", channel_id="C1") is True
    assert adapter._is_interactive_user_authorized("U999", channel_id="C1") is False

    monkeypatch.delenv("SLACK_ALLOWED_USERS")
    monkeypatch.setenv("GATEWAY_ALLOWED_USERS", '["U333"]')
    assert adapter._is_interactive_user_authorized("U333", channel_id="D1") is True

    monkeypatch.setenv("SLACK_ALLOWED_USERS", "U444, U555")  # plain CSV keeps its meaning
    assert adapter._is_interactive_user_authorized("U555", channel_id="C1") is True
    assert adapter._is_interactive_user_authorized("U999", channel_id="C1") is False
