"""WhatsApp observe-unmentioned group context (Telegram observe-mode parity).

Two behaviors under test:

1. With ``observe_unmentioned_group_messages`` enabled and an explicit chat allowlist,
   group messages the ``require_mention`` gate skips are stored as ``observed=True``
   transcript rows in the group's SHARED session (no per-user split), mirroring
   ``telegram.observe_unmentioned_group_messages``. Triggered turns in the same chat are
   re-sourced to that shared session and tagged with the observed-context channel prompt
   so ``_build_gateway_agent_history`` splits the observed rows out of replayable history.

2. Regression for the user-less trigger turn: because attribution strips ``user_id``
   from the event source, gateway authorization must admit the turn via a chat-scoped
   group allowlist — ``_GROUP_CHAT_ENV`` gains ``WHATSAPP_GROUP_ALLOWED_CHATS`` (it only
   listed Telegram/QQBOT before). Without it, every triggered group turn is dropped by
   the no-user-id guard in ``_hm_admit_event`` while observed rows keep accumulating.
"""

import asyncio

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource
from gateway.run import _uses_telegram_observed_group_context as uses_observed_context  # noqa: E402

GROUP_JID = "120363414190214470@g.us"
OTHER_JID = "120363433018457345@g.us"

OBSERVE_EXTRA = {
    "observe_unmentioned_group_messages": True,
    "observe_allowed_chats": f"{GROUP_JID},{OTHER_JID}",
}


def _adapter_with_extra(extra):
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter
    return WhatsAppAdapter(PlatformConfig(enabled=True, extra=extra))


def _group_data(body, *, chat=GROUP_JID, sender="447468411310", from_me=False, **kw):
    data = {"chatId": chat, "body": body, "isGroup": True, "fromMe": from_me,
            "senderId": sender, "senderName": "Tester", "messageId": "MSG1"}
    data.update(kw)
    return data


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


class TestObserveStorageAndAttribution:
    def _event(self, body="hello", uid="447468411310"):
        data = _group_data(body)
        source = SessionSource(platform=Platform.WHATSAPP, chat_id=GROUP_JID, chat_name="G",
                               chat_type="group", user_id=uid, user_name="Tester", message_id="MSG1")
        return MessageEvent(text=body, message_type=MessageType.TEXT, source=source,
                            raw_message=data, message_id="MSG1"), source

    def test_observer_writes_observed_row_into_shared_session(self, monkeypatch):
        adapter = _adapter_with_extra(OBSERVE_EXTRA)
        appended, keys = [], []

        async def fake_build(data, *, _skip_policy_gate=False):
            _, src = self._event(data["body"])
            return MessageEvent(text=data["body"], message_type=MessageType.TEXT, source=src,
                                raw_message=data, message_id="MSG1")

        class FakeEntry:
            session_id = "sess_1"

        class FakeStore:
            def get_or_create_session(self, source):
                from gateway.session import build_session_key
                keys.append(build_session_key(source, group_sessions_per_user=True, profile="p"))
                return FakeEntry()

            def append_to_transcript(self, sid, entry):
                appended.append((sid, entry))

        monkeypatch.setattr(type(adapter), "_build_message_event", staticmethod(fake_build))
        adapter._session_store = FakeStore()
        asyncio.run(adapter._observe_unmentioned_group_message(_group_data("lunch later?")))
        assert len(appended) == 1
        sid, entry = appended[0]
        assert sid == "sess_1"
        assert entry["observed"] is True and entry["role"] == "user"
        assert entry["content"].startswith("[Tester|447468411310]")
        # Shared chat-scoped session: no per-user suffix (observed rows and later
        # attributed trigger turns must resolve to the SAME session key).
        assert keys and keys[0].endswith(f"whatsapp:group:{GROUP_JID}")

    def test_triggered_text_turn_is_attributed_and_shared(self):
        adapter = _adapter_with_extra(OBSERVE_EXTRA)
        event, _ = self._event("@bot summarize this")
        out = adapter._apply_whatsapp_group_observe_attribution(event)
        assert "observed WhatsApp group context" in (out.channel_prompt or "")
        assert out.source.user_id is None and out.source.chat_id == GROUP_JID
        assert out.text.startswith("[Tester|447468411310]")

    def test_triggered_command_keeps_sender_identity(self):
        adapter = _adapter_with_extra(OBSERVE_EXTRA)
        event, _ = self._event("/new", uid="971555544440")
        event = MessageEvent(text="/new", message_type=MessageType.COMMAND, source=event.source,
                             raw_message=event.raw_message)
        out = adapter._apply_whatsapp_group_observe_attribution(event)
        assert out.source.user_id == "971555544440"
        assert "observed WhatsApp group context" in (out.channel_prompt or "")

    def test_attribution_noop_when_feature_disabled(self):
        adapter = _adapter_with_extra({})
        event, source = self._event()
        out = adapter._apply_whatsapp_group_observe_attribution(event)
        assert out.source.user_id == source.user_id
        assert "observed WhatsApp group context" not in (out.channel_prompt or "")


class TestTriggeredTurnAuthorization:
    """Attributed trigger turns carry ``user_id=None``; the chat-scoped group allowlist must
    admit them (mirrors ``TELEGRAM_GROUP_ALLOWED_CHATS``)."""

    def test_whatsapp_registered_in_group_chat_env(self):
        from gateway.authz_mixin import _GROUP_CHAT_ENV
        assert _GROUP_CHAT_ENV.get(Platform.WHATSAPP) == "WHATSAPP_GROUP_ALLOWED_CHATS"

    def test_chat_scoped_grant_admits_userless_group_turn(self, monkeypatch):
        from gateway.authz_mixin import GatewayAuthorizationMixin
        source = SessionSource(platform=Platform.WHATSAPP, chat_id=GROUP_JID, chat_type="group",
                               user_id=None)
        mixin = GatewayAuthorizationMixin.__new__(GatewayAuthorizationMixin)
        monkeypatch.setattr("gateway.authz_mixin._auth_env",
                            lambda name: GROUP_JID if name == "WHATSAPP_GROUP_ALLOWED_CHATS" else "")
        monkeypatch.setattr(mixin, "_adapter_profile_for_source", lambda s: None, raising=False)
        monkeypatch.setattr(mixin, "_adapter_extra_for_source", lambda s: {}, raising=False)
        monkeypatch.setattr(mixin, "_adapter_flag", lambda *a, **k: False, raising=False)
        assert mixin._chat_scoped_grant(source, None, is_group=True, allow_adapter_delegation=True) is True

    def test_chat_scoped_grant_rejects_unlisted_chat(self, monkeypatch):
        from gateway.authz_mixin import GatewayAuthorizationMixin
        source = SessionSource(platform=Platform.WHATSAPP, chat_id="999999999@g.us", chat_type="group",
                               user_id=None)
        mixin = GatewayAuthorizationMixin.__new__(GatewayAuthorizationMixin)
        monkeypatch.setattr("gateway.authz_mixin._auth_env",
                            lambda name: GROUP_JID if name == "WHATSAPP_GROUP_ALLOWED_CHATS" else "")
        monkeypatch.setattr(mixin, "_adapter_profile_for_source", lambda s: None, raising=False)
        monkeypatch.setattr(mixin, "_adapter_extra_for_source", lambda s: {}, raising=False)
        monkeypatch.setattr(mixin, "_adapter_flag", lambda *a, **k: False, raising=False)
        assert mixin._chat_scoped_grant(source, None, is_group=True, allow_adapter_delegation=True) is False

    def test_history_builder_uses_whatsapp_marker(self):
        assert uses_observed_context("You are handling a WhatsApp group chat message.\n- observed WhatsApp group context may be provided") is True
        assert uses_observed_context("observed Telegram group context ...") is True
        assert uses_observed_context(None) is False
        assert uses_observed_context("plain prompt") is False
