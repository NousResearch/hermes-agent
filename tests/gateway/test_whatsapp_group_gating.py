import asyncio
import json
import unittest.mock
from unittest.mock import AsyncMock, MagicMock

from gateway.config import Platform, PlatformConfig, load_gateway_config


def _make_adapter(require_mention=None, mention_patterns=None, free_response_chats=None,
                  dm_policy=None, allow_from=None, group_policy=None, group_allow_from=None):
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


def _group_message(body="hello", **overrides):
    data = {
        "isGroup": True,
        "body": body,
        "chatId": "120363001234567890@g.us",
        "senderId": "6281234567890@s.whatsapp.net",
        "senderName": "Tester",
        "messageId": "wamid.test1",
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




# ---------------------------------------------------------------------------
# observe_unmentioned_group_messages: store-but-don't-dispatch group context
# ---------------------------------------------------------------------------

def _make_observing_adapter(**kwargs):
    """Adapter with observe on, a named allowlisted group, and a MagicMock store."""
    adapter = _make_adapter(
        require_mention=True,
        group_policy="allowlist",
        group_allow_from=["120363001234567890@g.us"],
        **kwargs,
    )
    adapter.config.extra["observe_unmentioned_group_messages"] = True
    store = MagicMock()
    session_entry = MagicMock()
    session_entry.session_id = "sess1"
    store.get_or_create_session.return_value = session_entry
    adapter._session_store = store
    return adapter, store


def test_observe_disabled_by_default():
    adapter = _make_adapter(require_mention=True)
    assert adapter._whatsapp_observe_unmentioned_group_messages() is False


def test_unmentioned_group_message_observed_not_dispatched():
    adapter, store = _make_observing_adapter()
    msg = _group_message(body="just chatting")
    # The mention gate skips it, and the observe gate picks it up.
    assert adapter._should_process_message(msg) is False
    assert adapter._should_observe_unmentioned_group_message(msg) is True
    asyncio.run(adapter._observe_bridge_group_message(msg))
    store.get_or_create_session.assert_called_once()
    store.append_to_transcript.assert_called_once()
    entry = store.append_to_transcript.call_args.args[1]
    assert entry["observed"] is True
    assert entry["role"] == "user"
    assert "[6281234567890" in entry["content"] or "|6281234567890" in entry["content"]
    assert entry["content"].endswith("just chatting")


def test_mentioned_group_message_not_observed():
    """Mentioned messages dispatch — the observe gate must not swallow them."""
    adapter, store = _make_observing_adapter()
    msg = _group_message(body="@15551230000 hello", mentionedIds=["15551230000@s.whatsapp.net"])
    assert adapter._should_process_message(msg) is True
    assert adapter._should_observe_unmentioned_group_message(msg) is False
    event = asyncio.run(adapter._build_message_event(msg))
    assert event is not None  # dispatched
    store.append_to_transcript.assert_not_called()


def test_keyword_pattern_message_not_observed():
    """A mention-pattern keyword hit is a trigger, not observation."""
    adapter, store = _make_observing_adapter()
    adapter.config.extra["mention_patterns"] = ["hermes"]
    adapter._mention_patterns = adapter._compile_mention_patterns()
    msg = _group_message(body="hermes what do you think")
    assert adapter._should_process_message(msg) is True
    assert adapter._should_observe_unmentioned_group_message(msg) is False
    event = asyncio.run(adapter._build_message_event(msg))
    assert event is not None  # dispatched
    store.append_to_transcript.assert_not_called()


def test_observe_respects_group_allowlist():
    """Outside the observe allowlist (group_allow_from) → never observed."""
    adapter, store = _make_observing_adapter()
    msg = _group_message(body="hi", chatId="999999999@g.us")
    assert adapter._should_process_message(msg) is False
    assert adapter._should_observe_unmentioned_group_message(msg) is False
    event = asyncio.run(adapter._build_message_event(msg))
    assert event is None  # dropped entirely
    store.append_to_transcript.assert_not_called()


def test_observe_never_for_dms():
    adapter, store = _make_observing_adapter()
    msg = _dm_message(body="hi")
    assert adapter._should_observe_unmentioned_group_message(msg) is False


def test_observe_shared_chat_scoped_source():
    """Observed context uses a chat-scoped source (user_id stripped) so every
    group member's chatter lands in ONE shared session."""
    adapter, store = _make_observing_adapter()
    asyncio.run(adapter._observe_bridge_group_message(_group_message(senderId="628111@s.whatsapp.net", senderName="A")))
    asyncio.run(adapter._observe_bridge_group_message(_group_message(senderId="628222@s.whatsapp.net", senderName="B")))
    assert store.get_or_create_session.call_count == 2
    first = store.get_or_create_session.call_args_list[0].args[0]
    second = store.get_or_create_session.call_args_list[1].args[0]
    assert first.chat_id == second.chat_id == "120363001234567890@g.us"
    assert first.user_id is None and second.user_id is None


def test_trigger_attribution_and_safety_prompt():
    """A triggered message in an observing group gets [name|id] attribution,
    the shared chat-scoped source, and the observe safety prompt."""
    from dataclasses import replace
    adapter, _store = _make_observing_adapter()
    msg = _group_message(body="@15551230000 hello", mentionedIds=["15551230000@s.whatsapp.net"],
                         senderId="628999@s.whatsapp.net", senderName="Farel")
    event = asyncio.run(adapter._build_message_event(msg))
    assert event is not None
    adapted = adapter._apply_whatsapp_group_observe_attribution(event, msg)
    assert adapted.text.startswith("[Farel|628999@s.whatsapp.net]")
    assert adapted.source.user_id is None  # shared chat-scoped session
    assert "observed WhatsApp group context" in (adapted.channel_prompt or "")


def test_media_message_observed_records_kind_without_download():
    """No mediaUrls → the kind is recorded so the model knows media existed."""
    adapter, store = _make_observing_adapter()
    msg = _group_message(body="", mediaType="image", hasMedia=True)
    asyncio.run(adapter._observe_bridge_group_message(msg))
    entry = store.append_to_transcript.call_args.args[1]
    assert "[image]" in entry["content"]


def test_media_message_observed_with_cached_path_and_vision_note():
    """A downloaded image is referenced by cache path with the vision_analyze pointer."""
    import os
    from gateway.platforms.base import MessageType

    adapter, store = _make_observing_adapter()
    msg = _group_message(body="look at this", mediaType="image", hasMedia=True,
                         mediaUrls=["https://bridge.example/media/abc.jpg"])
    real_isfile = os.path.isfile
    with unittest.mock.patch("os.path.isfile", lambda p: real_isfile(p) or str(p).endswith(".jpg")), \
         unittest.mock.patch.object(adapter, "_collect_bridge_media",
                                    AsyncMock(return_value=(["/tmp/hermes-cache/img_abc.jpg"], ["image/jpeg"]))):
        asyncio.run(adapter._observe_bridge_group_message(msg))
    entry = store.append_to_transcript.call_args.args[1]
    assert "[image: /tmp/hermes-cache/img_abc.jpg]" in entry["content"]
    assert "[If you need a closer look, use vision_analyze with image_url: /tmp/hermes-cache/img_abc.jpg]" in entry["content"]
    assert entry["content"].startswith("[Tester|")  # attribution still leads
    assert "look at this" in entry["content"]


def test_media_download_failure_degrades_to_unavailable_note():
    """A failed download must never raise observation away: an unavailable note is recorded."""
    from gateway.platforms.base import MessageType

    adapter, store = _make_observing_adapter()
    msg = _group_message(body="", mediaType="image", hasMedia=True,
                         mediaUrls=["https://bridge.example/media/abc.jpg"])
    with unittest.mock.patch.object(adapter, "_collect_bridge_media",
                                    AsyncMock(return_value=(["/tmp/hermes-cache/img_missing.jpg"], ["image/jpeg"]))):
        asyncio.run(adapter._observe_bridge_group_message(msg))
    entry = store.append_to_transcript.call_args.args[1]
    assert "[image (unavailable: download failed)]" in entry["content"]


def test_observe_flag_without_mention_gating_keeps_normal_group_event_source():
    """Open groups (require_mention off) dispatch normally: no attribution, no shared
    source, no observe prompt — even with the observe flag on."""
    adapter = _make_adapter(require_mention=False, group_policy="allowlist",
                            group_allow_from=["120363001234567890@g.us"])
    adapter.config.extra["observe_unmentioned_group_messages"] = True
    msg = _group_message(body="free flowing chat", senderId="628999@s.whatsapp.net", senderName="Farel")
    event = asyncio.run(adapter._build_message_event(msg))
    assert event is not None
    adapted = adapter._apply_whatsapp_group_observe_attribution(event, msg)
    assert adapted is event  # untouched
    assert adapted.source.user_id == "628999@s.whatsapp.net"
    assert adapted.text == "free flowing chat"
    assert "observed WhatsApp group context" not in (adapted.channel_prompt or "")


def test_observe_flag_keeps_free_response_group_event_source():
    """Free-response chats dispatch per-user even in mention-gated mode: attribution
    must not apply there either."""
    adapter = _make_adapter(require_mention=True, group_policy="allowlist",
                            group_allow_from=["120363001234567890@g.us"],
                            free_response_chats=["120363001234567890@g.us"])
    adapter.config.extra["observe_unmentioned_group_messages"] = True
    msg = _group_message(body="free response chat", senderId="628999@s.whatsapp.net", senderName="Farel")
    event = asyncio.run(adapter._build_message_event(msg))
    assert event is not None
    adapted = adapter._apply_whatsapp_group_observe_attribution(event, msg)
    assert adapted is event  # untouched
    assert adapted.source.user_id == "628999@s.whatsapp.net"
    assert adapted.text == "free response chat"


def test_observe_attribution_not_applied_outside_observe_allowlist():
    """With no group allowlist (open policy) the observe allowlist is empty:
    triggered group turns keep per-user dispatch even with the flag on."""
    adapter = _make_adapter(require_mention=True, group_policy="open")
    adapter.config.extra["observe_unmentioned_group_messages"] = True
    msg = _group_message(body="@15551230000 hello", mentionedIds=["15551230000@s.whatsapp.net"],
                         senderId="628999@s.whatsapp.net", senderName="Farel")
    event = asyncio.run(adapter._build_message_event(msg))
    assert event is not None
    adapted = adapter._apply_whatsapp_group_observe_attribution(event, msg)
    assert adapted is event  # not an observed chat: normal per-user turn
    assert adapted.source.user_id == "628999@s.whatsapp.net"

