"""WhatsApp observe-unmentioned group context (Telegram observe-mode parity).

Covers the four review contracts for this feature:

- F-001: attributed trigger turns keep the caller identity (``user_id`` survives), so an
  already-allowed participant stays admitted without any extra chat-grant configuration.
- F-002: one effective observe scope (feature flag + explicit chat allowlist + mention-gated
  intake) governs BOTH collection and attribution; normal-processing and free-response chats
  are never observed nor re-sourced.
- F-003: text turns, observed chatter, commands and restored origins key the SAME shared
  session (``SessionSource.shared_session``) while commands keep their sender for access checks.
- F-004: observed-row timestamps persist through the real SessionDB writer without
  corrupt-timestamp warnings (datetime object, which ``coerce_epoch`` accepts).
"""

import asyncio
from datetime import datetime, timezone

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import _uses_telegram_observed_group_context as uses_observed_context  # noqa: E402
from gateway.session import SessionSource, build_session_key

GROUP_JID = "120363414190214470@g.us"
OTHER_JID = "120363433018457345@g.us"

OBSERVE_EXTRA = {
    "observe_unmentioned_group_messages": True,
    "observe_allowed_chats": f"{GROUP_JID},{OTHER_JID}",
    # Observe mode only applies to mention-gated intake (F-002 scope); the tests exercise
    # that posture explicitly.
    "require_mention": True,
}


def _adapter_with_extra(extra):
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter
    return WhatsAppAdapter(PlatformConfig(enabled=True, extra=extra))


def _group_data(body, *, chat=GROUP_JID, sender="447468411310", from_me=False, **kw):
    data = {"chatId": chat, "body": body, "isGroup": True, "fromMe": from_me,
            "senderId": sender, "senderName": "Tester", "messageId": "MSG1"}
    data.update(kw)
    return data


def _source(uid="447468411310", chat=GROUP_JID):
    return SessionSource(platform=Platform.WHATSAPP, chat_id=chat, chat_name="G",
                         chat_type="group", user_id=uid, user_name="Tester", message_id="MSG1")


class TestObserveGating:
    def test_feature_off_by_default(self):
        adapter = _adapter_with_extra({})
        assert adapter._whatsapp_observe_unmentioned_group_messages() is False
        assert adapter._whatsapp_observe_allowed_chats() == set()

    def test_plain_chatter_in_allowlisted_group_is_observed(self):
        adapter = _adapter_with_extra(OBSERVE_EXTRA)
        assert adapter._whatsapp_should_observe_unmentioned_group_message(_group_data("lunch later?")) is True

    def test_mention_is_never_observed(self, monkeypatch):
        adapter = _adapter_with_extra(OBSERVE_EXTRA)
        monkeypatch.setattr(type(adapter), "_message_is_reply_to_bot", lambda self, d: False)
        monkeypatch.setattr(type(adapter), "_message_mentions_bot", lambda self, d: True)
        monkeypatch.setattr(type(adapter), "_message_matches_mention_patterns", lambda self, d: False)
        assert adapter._whatsapp_should_observe_unmentioned_group_message(_group_data("hey @bot")) is False

    def test_reply_to_bot_is_never_observed(self, monkeypatch):
        adapter = _adapter_with_extra(OBSERVE_EXTRA)
        monkeypatch.setattr(type(adapter), "_message_is_reply_to_bot", lambda self, d: True)
        monkeypatch.setattr(type(adapter), "_message_mentions_bot", lambda self, d: False)
        monkeypatch.setattr(type(adapter), "_message_matches_mention_patterns", lambda self, d: False)
        assert adapter._whatsapp_should_observe_unmentioned_group_message(_group_data("thanks")) is False

    def test_wake_word_pattern_is_never_observed(self, monkeypatch):
        adapter = _adapter_with_extra(OBSERVE_EXTRA)
        monkeypatch.setattr(type(adapter), "_message_is_reply_to_bot", lambda self, d: False)
        monkeypatch.setattr(type(adapter), "_message_mentions_bot", lambda self, d: False)
        monkeypatch.setattr(type(adapter), "_message_matches_mention_patterns", lambda self, d: True)
        assert adapter._whatsapp_should_observe_unmentioned_group_message(_group_data("eva what time?")) is False

    def test_command_is_never_observed(self):
        adapter = _adapter_with_extra(OBSERVE_EXTRA)
        assert adapter._whatsapp_should_observe_unmentioned_group_message(_group_data("/new")) is False

    def test_non_allowlisted_chat_is_never_observed(self):
        adapter = _adapter_with_extra(OBSERVE_EXTRA)
        assert adapter._whatsapp_should_observe_unmentioned_group_message(
            _group_data("hi", chat="999999999@g.us")) is False

    def test_own_messages_and_dms_are_never_observed(self):
        adapter = _adapter_with_extra(OBSERVE_EXTRA)
        assert adapter._whatsapp_should_observe_unmentioned_group_message(
            _group_data("echo", from_me=True)) is False
        assert adapter._whatsapp_should_observe_unmentioned_group_message(
            _group_data("hi", chat="447468411310@s.whatsapp.net", isGroup=False)) is False

    def test_empty_observe_allowlist_observes_nothing(self):
        adapter = _adapter_with_extra({"observe_unmentioned_group_messages": True})
        assert adapter._whatsapp_should_observe_unmentioned_group_message(_group_data("hi")) is False


class TestEffectiveScopeExclusivity:
    """F-002: one effective scope governs collection AND attribution."""

    def test_require_mention_off_disables_observation(self):
        adapter = _adapter_with_extra(OBSERVE_EXTRA)
        monkey_free = {"free_response_chats": ""}
        adapter.config.extra = {**OBSERVE_EXTRA, **monkey_free, "require_mention": False}
        assert adapter._whatsapp_observe_scope_active(GROUP_JID) is False
        assert adapter._whatsapp_should_observe_unmentioned_group_message(_group_data("hi")) is False

    def test_free_response_chat_disables_observation_and_attribution(self):
        adapter = _adapter_with_extra({**OBSERVE_EXTRA, "free_response_chats": GROUP_JID})
        assert adapter._whatsapp_observe_scope_active(GROUP_JID) is False
        assert adapter._whatsapp_should_observe_unmentioned_group_message(_group_data("hi")) is False
        event = MessageEvent(text="hello", message_type=MessageType.TEXT, source=_source(),
                             raw_message=_group_data("hello"))
        out = adapter._apply_whatsapp_group_observe_attribution(event)
        assert out is event  # untouched: original principal/text preserved

    def test_attribution_noop_when_feature_disabled(self):
        adapter = _adapter_with_extra({})
        event = MessageEvent(text="hello", message_type=MessageType.TEXT, source=_source(),
                             raw_message=_group_data("hello"))
        out = adapter._apply_whatsapp_group_observe_attribution(event)
        assert out is event


class TestSharedSessionRouting:
    """F-001/F-003: the principal survives; routing is declared via ``shared_session``."""

    def _event(self, body="hello", uid="447468411310", msg_type=MessageType.TEXT):
        data = _group_data(body)
        return MessageEvent(text=body, message_type=msg_type, source=_source(uid),
                            raw_message=data, message_id="MSG1"), data

    def test_observer_keeps_principal_and_declares_shared_session(self, monkeypatch):
        adapter = _adapter_with_extra(OBSERVE_EXTRA)
        seen_sources, appended = [], []

        async def fake_build(data, *, _skip_policy_gate=False):
            return MessageEvent(text=data["body"], message_type=MessageType.TEXT,
                                source=_source(data.get("senderId")), raw_message=data,
                                message_id=data.get("messageId"))

        class FakeEntry:
            session_id = "sess_1"

        class FakeStore:
            def get_or_create_session(self, source):
                seen_sources.append(source)
                return FakeEntry()

            def append_to_transcript(self, sid, entry):
                appended.append((sid, entry))

        monkeypatch.setattr(type(adapter), "_build_message_event", staticmethod(fake_build))
        adapter._session_store = FakeStore()
        asyncio.run(adapter._observe_unmentioned_group_message(_group_data("lunch later?")))
        assert len(appended) == 1
        sid, entry = appended[0]
        # F-001: principal survives on the observed source
        assert seen_sources[0].user_id == "447468411310"
        assert seen_sources[0].shared_session is True
        # F-003: keys to the SHARED lane (no per-user suffix) even with the sender present
        assert build_session_key(seen_sources[0], group_sessions_per_user=True, profile="p") == \
            "agent:p:whatsapp:group:" + GROUP_JID
        assert entry["observed"] is True
        assert entry["content"].startswith("[Tester|447468411310]")

    def test_attribution_keeps_user_id_and_shares_session(self):
        adapter = _adapter_with_extra(OBSERVE_EXTRA)
        event, _ = self._event("@bot summarize this")
        out = adapter._apply_whatsapp_group_observe_attribution(event)
        # F-001: user_id survives admission
        assert out.source.user_id == "447468411310"
        assert out.source.shared_session is True
        assert "observed WhatsApp group context" in (out.channel_prompt or "")
        assert out.text.startswith("[Tester|447468411310]")

    def test_command_keeps_sender_and_shares_session(self):
        adapter = _adapter_with_extra(OBSERVE_EXTRA)
        event, _ = self._event("/new", uid="971555544440", msg_type=MessageType.COMMAND)
        out = adapter._apply_whatsapp_group_observe_attribution(event)
        assert out.source.user_id == "971555544440"
        assert out.source.shared_session is True
        assert "observed WhatsApp group context" in (out.channel_prompt or "")

    def test_text_command_observed_all_key_one_session(self):
        """F-003 invariant: addressed text, command, observed chatter and a restored command
        origin resolve to ONE session key at default ``group_sessions_per_user=True``."""
        adapter = _adapter_with_extra(OBSERVE_EXTRA)
        text_event, _ = self._event("@bot summarize this")
        cmd_event, _ = self._event("/new", uid="971555544440", msg_type=MessageType.COMMAND)
        observed_src = adapter._whatsapp_group_observe_shared_source(_source())
        text_out = adapter._apply_whatsapp_group_observe_attribution(text_event)
        cmd_out = adapter._apply_whatsapp_group_observe_attribution(cmd_event)
        restored_cmd = SessionSource.from_dict(cmd_out.source.to_dict())
        keys = {
            build_session_key(text_out.source, group_sessions_per_user=True, profile="p"),
            build_session_key(cmd_out.source, group_sessions_per_user=True, profile="p"),
            build_session_key(observed_src, group_sessions_per_user=True, profile="p"),
            build_session_key(restored_cmd, group_sessions_per_user=True, profile="p"),
        }
        assert len(keys) == 1
        assert keys.pop().endswith(f"whatsapp:group:{GROUP_JID}")
        # the command STILL carries its sender for access checks
        assert cmd_out.source.user_id == "971555544440"


class TestDurableTimestamp:
    """F-004: the observer's timestamp survives the real SessionDB writer."""

    def test_datetime_timestamp_round_trips_without_warning(self, tmp_path, monkeypatch):
        adapter = _adapter_with_extra(OBSERVE_EXTRA)
        frozen = datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
        monkeypatch.setattr("gateway.platforms.whatsapp_common.datetime", 
                            type("D", (), {"now": staticmethod(lambda tz=None: frozen)}))

        async def fake_build(data, *, _skip_policy_gate=False):
            return MessageEvent(text=data["body"], message_type=MessageType.TEXT,
                                source=_source(data.get("senderId")), raw_message=data,
                                message_id=data.get("messageId"))

        monkeypatch.setattr(type(adapter), "_build_message_event", staticmethod(fake_build))
        from agent import secret_scope as ss
        (tmp_path / ".env").write_text("", encoding="utf-8")
        ss.set_multiplex_active(True)
        tok = ss.set_secret_scope(ss.build_profile_secret_scope(tmp_path))
        try:
            from gateway.session import SessionStore
            from gateway.config import GatewayConfig
            store = SessionStore(tmp_path / "state", GatewayConfig())
            adapter._session_store = store
            asyncio.run(adapter._observe_unmentioned_group_message(_group_data("lunch later?")))
            session_entry = store._entries[next(iter(store._entries))]
            rows = store.load_transcript(session_entry.session_id)
            assert rows and rows[-1]["observed"] is True
            # exact durable readback, no fallback substitution
            assert rows[-1]["timestamp"] == pytest.approx(frozen.timestamp())
        finally:
            ss.reset_secret_scope(tok)
            ss.set_multiplex_active(False)


class TestHistoryBuilderMarker:
    def test_history_builder_uses_whatsapp_marker(self):
        assert uses_observed_context("You are handling a WhatsApp group chat message.\n- observed WhatsApp group context may be provided") is True
        assert uses_observed_context("observed Telegram group context ...") is True
        assert uses_observed_context(None) is False
        assert uses_observed_context("plain prompt") is False
