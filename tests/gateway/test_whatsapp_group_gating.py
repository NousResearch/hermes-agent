import json
from unittest.mock import AsyncMock

from gateway.config import Platform, PlatformConfig, load_gateway_config


def _make_adapter(require_mention=None, mention_patterns=None, free_response_chats=None,
                  dm_policy=None, allow_from=None, group_policy=None, group_allow_from=None):
    from gateway.platforms.whatsapp import WhatsAppAdapter

    extra = {}
    if require_mention is not None:
        extra["require_mention"] = require_mention
    if mention_patterns is not None:
        extra["mention_patterns"] = mention_patterns
    if free_response_chats is not None:
        extra["free_response_chats"] = free_response_chats
    if dm_policy is not None:
        extra["dm_policy"] = dm_policy
    if allow_from is not None:
        extra["allow_from"] = allow_from
    if group_policy is not None:
        extra["group_policy"] = group_policy
    if group_allow_from is not None:
        extra["group_allow_from"] = group_allow_from

    adapter = object.__new__(WhatsAppAdapter)
    adapter.platform = Platform.WHATSAPP
    adapter.config = PlatformConfig(enabled=True, extra=extra)
    adapter._message_handler = AsyncMock()
    adapter._dm_policy = str(extra.get("dm_policy", "open")).strip().lower()
    adapter._allow_from = WhatsAppAdapter._coerce_allow_list(extra.get("allow_from"))
    adapter._group_policy = str(extra.get("group_policy", "open")).strip().lower()
    adapter._group_allow_from = WhatsAppAdapter._coerce_allow_list(extra.get("group_allow_from"))
    adapter._mention_patterns = adapter._compile_mention_patterns()
    adapter._free_response_chats = adapter._whatsapp_free_response_chats()
    return adapter


def _group_message(body="hello", **overrides):
    data = {
        "isGroup": True,
        "body": body,
        "chatId": "120363001234567890@g.us",
        "mentionedIds": [],
        "botIds": ["15551230000@s.whatsapp.net", "15551230000@lid"],
        "quotedParticipant": "",
    }
    data.update(overrides)
    return data


def _dm_message(body="hello", **overrides):
    data = {
        "isGroup": False,
        "body": body,
        "senderId": "6281234567890@s.whatsapp.net",
        "from": "6281234567890@s.whatsapp.net",
        "botIds": [],
        "mentionedIds": [],
    }
    data.update(overrides)
    return data


# --- Existing tests (unchanged logic, updated helper) ---

def test_group_messages_can_be_opened_via_config():
    adapter = _make_adapter(require_mention=False)

    assert adapter._should_process_message(_group_message("hello everyone")) is True


def test_group_messages_can_require_direct_trigger_via_config():
    adapter = _make_adapter(require_mention=True)

    assert adapter._should_process_message(_group_message("hello everyone")) is False
    assert adapter._should_process_message(
        _group_message(
            "hi there",
            mentionedIds=["15551230000@s.whatsapp.net"],
        )
    ) is True
    assert adapter._should_process_message(
        _group_message(
            "replying",
            quotedParticipant="15551230000@lid",
        )
    ) is True
    assert adapter._should_process_message(_group_message("/status")) is True


def test_regex_mention_patterns_allow_custom_wake_words():
    adapter = _make_adapter(require_mention=True, mention_patterns=[r"^\s*chompy\b"])

    assert adapter._should_process_message(_group_message("chompy status")) is True
    assert adapter._should_process_message(_group_message("   chompy help")) is True
    assert adapter._should_process_message(_group_message("hey chompy")) is False


def test_invalid_regex_patterns_are_ignored():
    adapter = _make_adapter(require_mention=True, mention_patterns=[r"(", r"^\s*chompy\b"])

    assert adapter._should_process_message(_group_message("chompy status")) is True
    assert adapter._should_process_message(_group_message("hello everyone")) is False


def test_config_bridges_whatsapp_group_settings(monkeypatch, tmp_path):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "whatsapp:\n"
        "  require_mention: true\n"
        "  mention_patterns:\n"
        "    - \"^\\\\s*chompy\\\\b\"\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("WHATSAPP_REQUIRE_MENTION", raising=False)
    monkeypatch.delenv("WHATSAPP_MENTION_PATTERNS", raising=False)

    config = load_gateway_config()

    assert config is not None
    assert config.platforms[Platform.WHATSAPP].extra["require_mention"] is True
    assert config.platforms[Platform.WHATSAPP].extra["mention_patterns"] == [r"^\s*chompy\b"]
    assert __import__("os").environ["WHATSAPP_REQUIRE_MENTION"] == "true"
    assert json.loads(__import__("os").environ["WHATSAPP_MENTION_PATTERNS"]) == [r"^\s*chompy\b"]


def test_free_response_chats_bypass_mention_gating():
    adapter = _make_adapter(
        require_mention=True,
        free_response_chats=["120363001234567890@g.us"],
    )

    assert adapter._should_process_message(_group_message("hello everyone")) is True


def test_free_response_chats_does_not_bypass_other_groups():
    adapter = _make_adapter(
        require_mention=True,
        free_response_chats=["999999999999@g.us"],
    )

    assert adapter._should_process_message(_group_message("hello everyone")) is False


def test_dm_passes_with_default_open_policy():
    adapter = _make_adapter(require_mention=True)

    dm = _dm_message("hello")
    assert adapter._should_process_message(dm) is True


def test_mention_stripping_removes_bot_phone_from_body():
    adapter = _make_adapter(require_mention=True)

    data = _group_message("@15551230000 what is the weather?")
    cleaned = adapter._clean_bot_mention_text(data["body"], data)
    assert "15551230000" not in cleaned
    assert "weather" in cleaned


def test_mention_stripping_preserves_body_when_no_mention():
    adapter = _make_adapter(require_mention=True)

    data = _group_message("just a normal message")
    cleaned = adapter._clean_bot_mention_text(data["body"], data)
    assert cleaned == "just a normal message"


# --- New dm_policy tests ---

def test_dm_policy_disabled_blocks_all_dms():
    adapter = _make_adapter(dm_policy="disabled")

    assert adapter._should_process_message(_dm_message("hello")) is False


def test_dm_policy_disabled_still_allows_groups():
    adapter = _make_adapter(dm_policy="disabled", require_mention=False)

    assert adapter._should_process_message(_group_message("hello")) is True


def test_dm_policy_allowlist_blocks_unlisted_sender():
    adapter = _make_adapter(dm_policy="allowlist", allow_from=["6289999999999@s.whatsapp.net"])

    assert adapter._should_process_message(_dm_message("hello")) is False


def test_dm_policy_allowlist_allows_listed_sender():
    adapter = _make_adapter(dm_policy="allowlist", allow_from=["6281234567890@s.whatsapp.net"])

    assert adapter._should_process_message(_dm_message("hello")) is True


def test_dm_policy_open_allows_all_dms():
    adapter = _make_adapter(dm_policy="open")

    assert adapter._should_process_message(_dm_message("hello")) is True


# --- New group_policy tests ---

def test_group_policy_disabled_blocks_all_groups():
    adapter = _make_adapter(group_policy="disabled", require_mention=False)

    assert adapter._should_process_message(_group_message("hello")) is False


def test_group_policy_disabled_still_allows_dms():
    adapter = _make_adapter(group_policy="disabled")

    assert adapter._should_process_message(_dm_message("hello")) is True


def test_group_policy_allowlist_blocks_unlisted_group():
    adapter = _make_adapter(group_policy="allowlist", group_allow_from=["999999999999@g.us"])

    assert adapter._should_process_message(_group_message("agus test")) is False


def test_group_policy_allowlist_allows_listed_group():
    adapter = _make_adapter(
        group_policy="allowlist",
        group_allow_from=["120363001234567890@g.us"],
        require_mention=True,
        mention_patterns=[r"^\s*(?:(?:@)?(?:agus|Augustus))\b"],
    )

    # Listed group — passes the allowlist gate, mention still required
    assert adapter._should_process_message(_group_message("hello")) is False
    assert adapter._should_process_message(_group_message("agus test")) is True


def test_group_policy_open_allows_all_groups():
    adapter = _make_adapter(group_policy="open", require_mention=True)

    # Open policy — all groups pass the gate (mention still needed)
    assert adapter._should_process_message(_group_message("hello")) is False
    assert adapter._should_process_message(_group_message("/status")) is True


# --- Config bridging tests ---

def test_config_bridges_whatsapp_dm_and_group_policy(monkeypatch, tmp_path):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "whatsapp:\n"
        "  dm_policy: disabled\n"
        "  group_policy: allowlist\n"
        "  group_allow_from:\n"
        "    - \"120363001234567890@g.us\"\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("WHATSAPP_DM_POLICY", raising=False)
    monkeypatch.delenv("WHATSAPP_GROUP_POLICY", raising=False)
    monkeypatch.delenv("WHATSAPP_GROUP_ALLOWED_USERS", raising=False)

    config = load_gateway_config()

    assert config is not None
    assert config.platforms[Platform.WHATSAPP].extra["dm_policy"] == "disabled"
    assert config.platforms[Platform.WHATSAPP].extra["group_policy"] == "allowlist"
    assert config.platforms[Platform.WHATSAPP].extra["group_allow_from"] == ["120363001234567890@g.us"]
    assert __import__("os").environ["WHATSAPP_DM_POLICY"] == "disabled"
    assert __import__("os").environ["WHATSAPP_GROUP_POLICY"] == "allowlist"
    assert __import__("os").environ["WHATSAPP_GROUP_ALLOWED_USERS"] == "120363001234567890@g.us"


def test_config_bridges_whatsapp_allow_from(monkeypatch, tmp_path):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "whatsapp:\n"
        "  dm_policy: allowlist\n"
        "  allow_from:\n"
        "    - \"6281234567890@s.whatsapp.net\"\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("WHATSAPP_DM_POLICY", raising=False)
    monkeypatch.delenv("WHATSAPP_ALLOWED_USERS", raising=False)

    config = load_gateway_config()

    assert config is not None
    assert config.platforms[Platform.WHATSAPP].extra["dm_policy"] == "allowlist"
    assert config.platforms[Platform.WHATSAPP].extra["allow_from"] == ["6281234567890@s.whatsapp.net"]
    assert __import__("os").environ["WHATSAPP_DM_POLICY"] == "allowlist"
    assert __import__("os").environ["WHATSAPP_ALLOWED_USERS"] == "6281234567890@s.whatsapp.net"


# --- Broadcast / status / newsletter pseudo-chats are always dropped ---


def test_status_broadcast_chats_are_always_dropped():
    """Felipe's gateway.log showed the agent replying to status@broadcast
    (a contact's WhatsApp Story update). These pseudo-chats aren't real
    conversations and the adapter must drop them regardless of dm_policy.
    """
    from gateway.platforms.whatsapp import WhatsAppAdapter

    # Even on the most permissive config — open DMs, no allowlist — Stories
    # and Channel posts must not reach the agent.
    adapter = _make_adapter(dm_policy="open")

    # Classic Story update — what Felipe was seeing in production.
    status_msg = _dm_message(
        body="[video received]",
        chatId="status@broadcast",
        senderId="34612345678@s.whatsapp.net",
    )
    assert adapter._should_process_message(status_msg) is False

    # Channel / Newsletter broadcast posts.
    newsletter_msg = _dm_message(
        body="check out our latest post",
        chatId="120363999999999999@newsletter",
        senderId="120363999999999999@newsletter",
    )
    assert adapter._should_process_message(newsletter_msg) is False


def test_broadcast_filter_runs_before_allowlist():
    """A status@broadcast message from an allowlisted sender still drops —
    we never want to reply to Stories, even from authorized contacts.
    """
    adapter = _make_adapter(
        dm_policy="allowlist",
        allow_from=["34612345678@s.whatsapp.net"],
    )

    msg = _dm_message(
        body="[image received]",
        chatId="status@broadcast",
        senderId="34612345678@s.whatsapp.net",
    )
    assert adapter._should_process_message(msg) is False


def test_real_dm_still_processed_after_broadcast_filter():
    """Sanity check: the broadcast filter doesn't accidentally drop real DMs."""
    adapter = _make_adapter(dm_policy="open")

    msg = _dm_message(
        body="hello",
        chatId="34612345678@s.whatsapp.net",
        senderId="34612345678@s.whatsapp.net",
    )
    assert adapter._should_process_message(msg) is True


def test_is_broadcast_chat_helper_recognizes_common_jids():
    from gateway.platforms.whatsapp import WhatsAppAdapter

    assert WhatsAppAdapter._is_broadcast_chat("status@broadcast") is True
    assert WhatsAppAdapter._is_broadcast_chat("STATUS@BROADCAST") is True
    assert WhatsAppAdapter._is_broadcast_chat("  status@broadcast  ") is True
    assert WhatsAppAdapter._is_broadcast_chat("120363999999999999@newsletter") is True
    assert WhatsAppAdapter._is_broadcast_chat("1234@broadcast") is True  # broadcast list
    # Real chats must not match.
    assert WhatsAppAdapter._is_broadcast_chat("34612345678@s.whatsapp.net") is False
    assert WhatsAppAdapter._is_broadcast_chat("120363001234567890@g.us") is False
    assert WhatsAppAdapter._is_broadcast_chat("") is False
    assert WhatsAppAdapter._is_broadcast_chat(None) is False  # type: ignore[arg-type]
