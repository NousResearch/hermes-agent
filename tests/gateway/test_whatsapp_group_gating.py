from typing import Optional

import json
import pytest
from unittest.mock import AsyncMock

from gateway.config import Platform, PlatformConfig, load_gateway_config


def _make_adapter(require_mention=None, mention_patterns=None, free_response_chats=None,
                  dm_policy=None, allow_from=None, group_policy=None, group_allow_from=None,
                  observe_unmentioned_group_messages=None,
                  reactions=None, reaction_allow_from=None,
                  reply_consider_reaction_senders=None,
                  reply_consider_allow_from=None):
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

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
    if observe_unmentioned_group_messages is not None:
        extra["observe_unmentioned_group_messages"] = observe_unmentioned_group_messages
    if reactions is not None:
        extra["reactions"] = reactions
    if reaction_allow_from is not None:
        extra["reaction_allow_from"] = reaction_allow_from
    if reply_consider_reaction_senders is not None:
        extra["reply_consider_reaction_senders"] = reply_consider_reaction_senders
    if reply_consider_allow_from is not None:
        extra["reply_consider_allow_from"] = reply_consider_allow_from

    adapter = object.__new__(WhatsAppAdapter)
    adapter.platform = Platform.WHATSAPP
    adapter.config = PlatformConfig(enabled=True, extra=extra)
    adapter._message_handler = AsyncMock()
    adapter._dm_policy = str(extra.get("dm_policy", "open")).strip().lower()
    adapter._allow_from = WhatsAppAdapter._coerce_allow_list(extra.get("allow_from"))
    adapter._group_policy = str(extra.get("group_policy", "open")).strip().lower()
    adapter._group_allow_from = WhatsAppAdapter._coerce_allow_list(extra.get("group_allow_from"))
    adapter._mention_patterns = adapter._compile_mention_patterns()
    adapter._reactions_enabled = adapter._coerce_bool_extra("reactions", False)
    adapter._reaction_emoji = str(extra.get("reaction_emoji") or "auto")
    adapter._reaction_allow_from = WhatsAppAdapter._coerce_allow_list(extra.get("reaction_allow_from"))
    adapter._reply_consider_allow_from = WhatsAppAdapter._coerce_allow_list(extra.get("reply_consider_allow_from") or extra.get("reaction_allow_from"))
    adapter._reaction_batch_delay_seconds = 4.0
    adapter._pending_reactions = {}
    adapter._pending_reaction_tasks = {}
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


def test_group_policy_allowlist_star_allows_all_groups():
    adapter = _make_adapter(
        group_policy="allowlist",
        group_allow_from="*",
        require_mention=True,
    )

    # ``*`` is useful for sandbox/testing configs: all groups pass the
    # allowlist gate, while mention/wake-word gating still controls replies.
    assert adapter._should_process_message(_group_message("hello")) is False
    assert adapter._should_process_message(_group_message("/status")) is True


def test_group_policy_open_allows_all_groups():
    adapter = _make_adapter(group_policy="open", require_mention=True)

    # Open policy — all groups pass the gate (mention still needed)
    assert adapter._should_process_message(_group_message("hello")) is False
    assert adapter._should_process_message(_group_message("/status")) is True


def test_observe_unmentioned_group_messages_with_star_allowlist():
    adapter = _make_adapter(
        require_mention=True,
        group_policy="allowlist",
        group_allow_from="*",
        observe_unmentioned_group_messages=True,
    )

    assert adapter._should_process_message(_group_message("ordinary group chatter")) is False
    assert adapter._should_observe_unmentioned_group_message(_group_message("ordinary group chatter")) is True
    assert adapter._should_observe_unmentioned_group_message(
        _group_message("@15551230000 please reply", mentionedIds=["15551230000@s.whatsapp.net"])
    ) is False


@pytest.mark.asyncio
async def test_observed_group_message_is_stored_without_dispatch():
    class FakeStore:
        def __init__(self):
            self.entries = []
            self.sources = []

        def get_or_create_session(self, source):
            self.sources.append(source)
            return type("Session", (), {"session_id": "session-1"})()

        def append_to_transcript(self, session_id, entry):
            self.entries.append((session_id, entry))

    adapter = _make_adapter(
        require_mention=True,
        group_policy="allowlist",
        group_allow_from="*",
        observe_unmentioned_group_messages=True,
    )
    store = FakeStore()
    adapter.set_session_store(store)

    event = await adapter._build_message_event(
        _group_message(
            "ordinary group chatter",
            senderId="5216640000000@s.whatsapp.net",
            senderName="Alice",
            chatName="Test Group",
            messageId="msg-1",
        )
    )

    assert event is None
    assert len(store.entries) == 1
    session_id, entry = store.entries[0]
    assert session_id == "session-1"
    assert entry["observed"] is True
    assert entry["message_id"] == "msg-1"
    assert "ordinary group chatter" in entry["content"]
    assert store.sources[0].user_id is None
    assert store.sources[0].chat_id == "120363001234567890@g.us"


# --- Config bridging tests ---

def test_config_bridges_whatsapp_dm_and_group_policy(monkeypatch, tmp_path):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "whatsapp:\n"
        "  dm_policy: disabled\n"
        "  group_policy: allowlist\n"
        "  observe_unmentioned_group_messages: true\n"
        "  group_allow_from:\n"
        "    - \"120363001234567890@g.us\"\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("WHATSAPP_DM_POLICY", raising=False)
    monkeypatch.delenv("WHATSAPP_GROUP_POLICY", raising=False)
    monkeypatch.delenv("WHATSAPP_OBSERVE_UNMENTIONED_GROUP_MESSAGES", raising=False)
    monkeypatch.delenv("WHATSAPP_GROUP_ALLOWED_USERS", raising=False)

    config = load_gateway_config()

    assert config is not None
    assert config.platforms[Platform.WHATSAPP].extra["dm_policy"] == "disabled"
    assert config.platforms[Platform.WHATSAPP].extra["group_policy"] == "allowlist"
    assert config.platforms[Platform.WHATSAPP].extra["observe_unmentioned_group_messages"] is True
    assert config.platforms[Platform.WHATSAPP].extra["group_allow_from"] == ["120363001234567890@g.us"]
    assert __import__("os").environ["WHATSAPP_DM_POLICY"] == "disabled"
    assert __import__("os").environ["WHATSAPP_GROUP_POLICY"] == "allowlist"
    assert __import__("os").environ["WHATSAPP_OBSERVE_UNMENTIONED_GROUP_MESSAGES"] == "true"
    assert __import__("os").environ["WHATSAPP_GROUP_ALLOWED_USERS"] == "120363001234567890@g.us"


def test_adapter_reads_group_allowlist_env_fallback(monkeypatch):
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

    monkeypatch.setenv("WHATSAPP_GROUP_POLICY", "allowlist")
    monkeypatch.setenv("WHATSAPP_GROUP_ALLOWED_USERS", "120363001234567890@g.us")
    monkeypatch.setenv("WHATSAPP_REQUIRE_MENTION", "false")

    adapter = WhatsAppAdapter(PlatformConfig(enabled=True, extra={}))

    assert adapter._group_policy == "allowlist"
    assert adapter._group_allow_from == {"120363001234567890@g.us"}
    assert adapter._should_process_message(_group_message("hello")) is True


def test_adapter_reads_group_allowlist_env_star(monkeypatch):
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

    monkeypatch.setenv("WHATSAPP_GROUP_POLICY", "allowlist")
    monkeypatch.setenv("WHATSAPP_GROUP_ALLOWED_USERS", "*")
    monkeypatch.setenv("WHATSAPP_REQUIRE_MENTION", "false")

    adapter = WhatsAppAdapter(PlatformConfig(enabled=True, extra={}))

    assert adapter._group_policy == "allowlist"
    assert adapter._group_allow_from == {"*"}
    assert adapter._should_process_message(_group_message("hello")) is True


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
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

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


def _reaction_event(chat_type="group", sender_id="5215551234567@s.whatsapp.net", message_id: Optional[str] = "ABC123"):
    from gateway.platforms.base import MessageEvent, MessageType
    from gateway.session import SessionSource

    source = SessionSource(
        platform=Platform.WHATSAPP,
        chat_id="120363001234567890@g.us" if chat_type == "group" else sender_id,
        chat_type=chat_type,
        user_id=sender_id,
    )
    return MessageEvent(
        text="hello",
        message_type=MessageType.TEXT,
        source=source,
        raw_message={"senderId": sender_id},
        message_id=message_id,
    )


def test_whatsapp_group_reactions_enabled_for_group_messages():
    adapter = _make_adapter(reactions=True)

    assert adapter._should_react_to_event(_reaction_event()) is True


def test_whatsapp_group_reactions_skip_dms_and_missing_message_ids():
    adapter = _make_adapter(reactions=True)

    assert adapter._should_react_to_event(_reaction_event(chat_type="dm")) is False
    assert adapter._should_react_to_event(_reaction_event(message_id=None)) is False


def test_whatsapp_group_reactions_respect_sender_allowlist():
    adapter = _make_adapter(
        reactions=True,
        reaction_allow_from=["5215551234567@s.whatsapp.net"],
    )

    assert adapter._should_react_to_event(
        _reaction_event(sender_id="5215551234567@s.whatsapp.net")
    ) is True
    assert adapter._should_react_to_event(
        _reaction_event(sender_id="5215559999999@s.whatsapp.net")
    ) is False


def test_whatsapp_group_reactions_happen_before_mention_gating_drop():
    adapter = _make_adapter(
        require_mention=True,
        reactions=True,
        reaction_allow_from=["5215551234567@s.whatsapp.net"],
        observe_unmentioned_group_messages=False,
        reply_consider_reaction_senders=False,
    )
    msg = _group_message(
        "ordinary group chatter without a Jack mention",
        messageId="MSG1",
        senderId="5215551234567@s.whatsapp.net",
        from_="5215551234567@s.whatsapp.net",
    )
    msg["from"] = msg.pop("from_")

    assert adapter._should_process_message(msg) is False
    assert adapter._should_observe_unmentioned_group_message(msg) is False
    assert adapter._should_react_to_message_data(msg) is True


def test_approved_reaction_sender_group_messages_are_considered_for_reply():
    adapter = _make_adapter(
        require_mention=True,
        reactions=True,
        reaction_allow_from=["5215551234567@s.whatsapp.net"],
    )
    msg = _group_message(
        "Battery 4140 installed in NOW 3150 temporarily",
        messageId="MSG1",
        senderId="5215551234567@s.whatsapp.net",
        from_="5215551234567@s.whatsapp.net",
    )
    msg["from"] = msg.pop("from_")

    assert adapter._should_react_to_message_data(msg) is True
    assert adapter._should_process_message(msg) is True
    assert adapter._should_observe_unmentioned_group_message(msg) is False


def test_non_approved_group_messages_stay_observe_only_when_unmentioned():
    adapter = _make_adapter(
        require_mention=True,
        reactions=True,
        reaction_allow_from=["5215551234567@s.whatsapp.net"],
        observe_unmentioned_group_messages=True,
    )
    msg = _group_message(
        "ordinary group chatter without a Jack mention",
        messageId="MSG1",
        senderId="5215559999999@s.whatsapp.net",
        from_="5215559999999@s.whatsapp.net",
    )
    msg["from"] = msg.pop("from_")

    assert adapter._should_react_to_message_data(msg) is False
    assert adapter._should_process_message(msg) is False
    assert adapter._should_observe_unmentioned_group_message(msg) is True


def test_react_to_all_group_members_but_reply_consider_only_approved_senders():
    adapter = _make_adapter(
        require_mention=True,
        reactions=True,
        reaction_allow_from=["*"],
        reply_consider_allow_from=["5215551234567@s.whatsapp.net"],
        observe_unmentioned_group_messages=True,
    )
    jacob_msg = _group_message(
        "Battery 4140 installed in NOW 3150 temporarily",
        messageId="MSG1",
        senderId="5215551234567@s.whatsapp.net",
        from_="5215551234567@s.whatsapp.net",
    )
    other_msg = _group_message(
        "ordinary group chatter without a Jack mention",
        messageId="MSG2",
        senderId="5215559999999@s.whatsapp.net",
        from_="5215559999999@s.whatsapp.net",
    )
    jacob_msg["from"] = jacob_msg.pop("from_")
    other_msg["from"] = other_msg.pop("from_")

    assert adapter._should_react_to_message_data(jacob_msg) is True
    assert adapter._should_process_message(jacob_msg) is True
    assert adapter._should_react_to_message_data(other_msg) is True
    assert adapter._should_process_message(other_msg) is False
    assert adapter._should_observe_unmentioned_group_message(other_msg) is True


def test_whatsapp_reaction_emoji_auto_matches_message_context():
    adapter = _make_adapter(reactions=True)

    assert adapter._reaction_emoji_for_message_data({"body": "Can you check this?"}) == "👀"
    assert adapter._reaction_emoji_for_message_data({"body": "NOW 3140 is stolen"}) == "🚨"
    assert adapter._reaction_emoji_for_message_data({"body": "payment proof sent"}) == "✅"
    assert adapter._reaction_emoji_for_message_data({"body": "gracias"}) == "🙏"
    assert adapter._reaction_emoji_for_message_data({"body": "FYI"}) == "👍"


def test_whatsapp_reaction_queue_keeps_only_latest_message_in_burst(monkeypatch):
    adapter = _make_adapter(reactions=True)
    created = []

    class FakeTask:
        def __init__(self):
            self.cancelled = False
        def done(self):
            return False
        def cancel(self):
            self.cancelled = True

    def fake_create_task(coro):
        coro.close()
        task = FakeTask()
        created.append(task)
        return task

    monkeypatch.setattr("plugins.platforms.whatsapp.adapter.asyncio.create_task", fake_create_task)
    adapter._queue_reaction_message_data(
        chat_id="120363001234567890@g.us",
        message_id="MSG1",
        sender_id="5215551234567@s.whatsapp.net",
        raw_message={"body": "first", "senderId": "5215551234567@s.whatsapp.net"},
    )
    adapter._queue_reaction_message_data(
        chat_id="120363001234567890@g.us",
        message_id="MSG2",
        sender_id="5215551234567@s.whatsapp.net",
        raw_message={"body": "Can you check this?", "senderId": "5215551234567@s.whatsapp.net"},
    )

    key = adapter._reaction_batch_key("120363001234567890@g.us", "5215551234567@s.whatsapp.net")
    assert created[0].cancelled is True
    assert adapter._pending_reactions[key]["message_id"] == "MSG2"
    assert adapter._pending_reactions[key]["emoji"] == "👀"
