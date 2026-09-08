import asyncio
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

from gateway.config import Platform, PlatformConfig, load_gateway_config


def _make_adapter(require_mention=None, mention_patterns=None, free_response_chats=None,
                  dm_policy=None, allow_from=None, group_policy=None, group_allow_from=None,
                  observe_unmentioned_group_messages=None,
                  share_observed_group_context=None,
                  observed_group_context_dm_allow_from=None,
                  observed_group_context_retention_days=None):
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
    for key, value in (
        ("observe_unmentioned_group_messages", observe_unmentioned_group_messages),
        ("share_observed_group_context", share_observed_group_context),
        ("observed_group_context_dm_allow_from", observed_group_context_dm_allow_from),
        ("observed_group_context_retention_days", observed_group_context_retention_days),
    ):
        if value is not None:
            extra[key] = value

    adapter = object.__new__(WhatsAppAdapter)
    adapter.platform = Platform.WHATSAPP
    adapter.config = PlatformConfig(enabled=True, extra=extra)
    adapter._message_handler = AsyncMock()
    adapter._dm_policy = str(extra.get("dm_policy", "pairing")).strip().lower()
    adapter._allow_from = WhatsAppAdapter._coerce_allow_list(extra.get("allow_from"))
    adapter._group_policy = str(extra.get("group_policy", "pairing")).strip().lower()
    adapter._group_allow_from = WhatsAppAdapter._coerce_allow_list(extra.get("group_allow_from"))
    adapter._mention_patterns = adapter._compile_mention_patterns()
    adapter._free_response_chats = adapter._whatsapp_free_response_chats()
    return adapter


class _FakeSessionStore:
    def __init__(self):
        self._ids, self.rows = {}, {}

    def get_or_create_session(self, source):
        key = (source.chat_id, source.user_id)
        session_id = self._ids.setdefault(key, f"session-{len(self._ids) + 1}")
        return SimpleNamespace(session_id=session_id)

    def append_to_transcript(self, session_id, entry):
        self.rows.setdefault(session_id, []).append(entry)

    def load_transcript(self, session_id):
        return list(self.rows.get(session_id, []))


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


def test_group_messages_can_require_direct_trigger_via_config():
    adapter = _make_adapter(require_mention=True, group_policy="open")

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
    adapter = _make_adapter(
        require_mention=True,
        mention_patterns=[r"^\s*chompy\b"],
        group_policy="open",
    )

    assert adapter._should_process_message(_group_message("chompy status")) is True
    assert adapter._should_process_message(_group_message("   chompy help")) is True
    assert adapter._should_process_message(_group_message("hey chompy")) is False


def test_invalid_regex_patterns_are_ignored():
    adapter = _make_adapter(
        require_mention=True,
        mention_patterns=[r"(", r"^\s*chompy\b"],
        group_policy="open",
    )

    assert adapter._should_process_message(_group_message("chompy status")) is True
    assert adapter._should_process_message(_group_message("hello everyone")) is False


def test_free_response_chats_bypass_mention_gating():
    adapter = _make_adapter(
        require_mention=True,
        free_response_chats=["120363001234567890@g.us"],
        group_policy="open",
    )

    assert adapter._should_process_message(_group_message("hello everyone")) is True


def test_free_response_chats_does_not_bypass_other_groups():
    adapter = _make_adapter(
        require_mention=True,
        free_response_chats=["999999999999@g.us"],
        group_policy="open",
    )

    assert adapter._should_process_message(_group_message("hello everyone")) is False


def test_mention_stripping_removes_bot_phone_from_body():
    adapter = _make_adapter(require_mention=True)

    data = _group_message("@15551230000 what is the weather?")
    cleaned = adapter._clean_bot_mention_text(data["body"], data)
    assert "15551230000" not in cleaned
    assert "weather" in cleaned


def test_unmentioned_group_messages_are_observed_without_dispatch():
    adapter = _make_adapter(require_mention=True, group_policy="open", observe_unmentioned_group_messages=True)
    adapter._session_store = _FakeSessionStore()

    assert asyncio.run(adapter._build_message_event(_group_message(
        "Friday works", senderId="alice@s.whatsapp.net", senderName="Alice"
    ))) is None
    rows = next(iter(adapter._session_store.rows.values()))
    assert rows[0]["observed"] is True
    assert rows[0]["content"] == "[Alice|alice@s.whatsapp.net]\nFriday works"


def test_group_mention_uses_observed_history_session():
    adapter = _make_adapter(require_mention=True, group_policy="open", observe_unmentioned_group_messages=True)
    event = asyncio.run(adapter._build_message_event(_group_message(
        "@15551230000 summarize", senderId="bob@s.whatsapp.net", senderName="Bob",
        mentionedIds=["15551230000@s.whatsapp.net"],
    )))

    assert event.source.user_id is None
    assert "observed WhatsApp group context" in event.channel_prompt


def test_cross_group_context_is_available_to_owner_dm_only():
    adapter = _make_adapter(
        require_mention=True, group_policy="open", observe_unmentioned_group_messages=True,
        share_observed_group_context=True, dm_policy="allowlist", allow_from=["owner@s.whatsapp.net", "other@s.whatsapp.net"],
        observed_group_context_dm_allow_from=["owner@s.whatsapp.net"],
    )
    adapter._session_store = _FakeSessionStore()
    asyncio.run(adapter._build_message_event(_group_message(
        "Meeting Friday", chatId="group-a@g.us", chatName="Group A", senderId="alice@s.whatsapp.net", senderName="Alice"
    )))

    owner = asyncio.run(adapter._build_message_event(_dm_message(
        "What happened?", chatId="owner@s.whatsapp.net", senderId="owner@s.whatsapp.net"
    )))
    other = asyncio.run(adapter._build_message_event(_dm_message(
        "What happened?", chatId="other@s.whatsapp.net", senderId="other@s.whatsapp.net"
    )))
    assert "Meeting Friday" in owner.metadata["gateway_turn_context"]
    assert "gateway_turn_context" not in other.metadata


def test_mentioned_group_receives_context_observed_in_another_group():
    adapter = _make_adapter(
        require_mention=True, group_policy="open", observe_unmentioned_group_messages=True,
        share_observed_group_context=True,
    )
    adapter._session_store = _FakeSessionStore()
    asyncio.run(adapter._build_message_event(_group_message(
        "Budget approved", chatId="group-a@g.us", chatName="Group A", senderId="alice@s.whatsapp.net", senderName="Alice"
    )))

    event = asyncio.run(adapter._build_message_event(_group_message(
        "@15551230000 what was approved?", chatId="group-b@g.us", chatName="Group B",
        senderId="bob@s.whatsapp.net", senderName="Bob", mentionedIds=["15551230000@s.whatsapp.net"],
    )))
    assert "Budget approved" in event.metadata["gateway_turn_context"]


def test_cross_group_context_excludes_expired_messages():
    adapter = _make_adapter(
        share_observed_group_context=True,
        observed_group_context_retention_days=30,
    )
    store = _FakeSessionStore()
    adapter._session_store = store
    source = adapter.build_source(chat_id="owner@s.whatsapp.net", chat_type="dm", user_id="owner@s.whatsapp.net")
    entry = store.get_or_create_session(adapter._whatsapp_cross_group_context_source(source))
    now = datetime.now(tz=timezone.utc)
    store.append_to_transcript(entry.session_id, {"role": "user", "content": "recent", "observed": True, "timestamp": now.isoformat()})
    store.append_to_transcript(entry.session_id, {"role": "user", "content": "expired", "observed": True, "timestamp": (now - timedelta(days=31)).isoformat()})

    context = adapter._load_whatsapp_cross_group_context(source)
    assert "recent" in context
    assert "expired" not in context


# --- New dm_policy tests ---


def test_dm_policy_disabled_still_allows_groups():
    adapter = _make_adapter(
        dm_policy="disabled",
        require_mention=False,
        group_policy="open",
    )

    assert adapter._should_process_message(_group_message("hello")) is True


# --- New group_policy tests ---


# --- Config bridging tests ---

def test_config_bridges_whatsapp_dm_and_group_policy(monkeypatch, tmp_path):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "whatsapp:\n"
        "  dm_policy: disabled\n"
        "  group_policy: allowlist\n"
        "  group_allow_from:\n"
        "    - \"120363001234567890@g.us\"\n"
        "  observe_unmentioned_group_messages: true\n"
        "  share_observed_group_context: true\n"
        "  observed_group_context_dm_allow_from:\n"
        "    - \"owner@s.whatsapp.net\"\n"
        "  observed_group_context_retention_days: 30\n",
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
    assert config.platforms[Platform.WHATSAPP].extra["observe_unmentioned_group_messages"] is True
    assert config.platforms[Platform.WHATSAPP].extra["share_observed_group_context"] is True
    assert config.platforms[Platform.WHATSAPP].extra["observed_group_context_dm_allow_from"] == ["owner@s.whatsapp.net"]
    assert config.platforms[Platform.WHATSAPP].extra["observed_group_context_retention_days"] == 30
    assert __import__("os").environ["WHATSAPP_DM_POLICY"] == "disabled"
    assert __import__("os").environ["WHATSAPP_GROUP_POLICY"] == "allowlist"
    assert __import__("os").environ["WHATSAPP_GROUP_ALLOWED_USERS"] == "120363001234567890@g.us"


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
