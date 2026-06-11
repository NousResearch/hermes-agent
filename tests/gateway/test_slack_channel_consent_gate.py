"""Tests for the Slack channel consent gate (channel_consent_gate config).

Contract:
  - Gate disabled (default) → join posts no prompt, messages flow (stock).
  - Gate enabled → bot join marks the channel pending + posts the
    Activate/Decline Block Kit prompt; messages in pending/declined
    channels are dropped before any processing.
  - Activate click → approved, messages flow; Decline → stays dormant.
  - Re-invite to an approved channel does NOT reset consent.
  - DMs are never gated.
  - Untracked channels (joined before the gate existed) are never gated.
"""
import json

from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.slack import SlackAdapter
from gateway.platforms.slack_consent import ChannelConsentStore


# ---------------------------------------------------------------------------
# ChannelConsentStore
# ---------------------------------------------------------------------------


class TestChannelConsentStore:
    def test_untracked_channel_is_not_dormant(self, tmp_path):
        store = ChannelConsentStore(tmp_path / "consent.json")
        assert store.status("C1") is None
        assert store.is_dormant("C1") is False

    def test_pending_and_declined_are_dormant_approved_is_not(self, tmp_path):
        store = ChannelConsentStore(tmp_path / "consent.json")
        store.set("C1", "pending")
        assert store.is_dormant("C1") is True
        store.set("C1", "approved", by_user_id="U1", by_user_name="jake")
        assert store.is_dormant("C1") is False
        store.set("C1", "declined")
        assert store.is_dormant("C1") is True

    def test_persists_across_instances(self, tmp_path):
        path = tmp_path / "consent.json"
        ChannelConsentStore(path).set("C1", "approved", by_user_id="U1")
        store2 = ChannelConsentStore(path)
        assert store2.status("C1") == "approved"

    def test_corrupt_file_degrades_to_empty(self, tmp_path):
        path = tmp_path / "consent.json"
        path.write_text("{not json", encoding="utf-8")
        store = ChannelConsentStore(path)
        assert store.status("C1") is None
        # And writes still work afterwards
        store.set("C1", "pending")
        assert ChannelConsentStore(path).status("C1") == "pending"

    def test_invalid_states_in_file_are_dropped(self, tmp_path):
        path = tmp_path / "consent.json"
        path.write_text(
            json.dumps({"C1": {"status": "bogus"}, "C2": {"status": "pending"}}),
            encoding="utf-8",
        )
        store = ChannelConsentStore(path)
        assert store.status("C1") is None
        assert store.status("C2") == "pending"

    def test_set_rejects_invalid_status(self, tmp_path):
        store = ChannelConsentStore(tmp_path / "consent.json")
        with pytest.raises(ValueError):
            store.set("C1", "maybe")

    def test_forget_reverts_to_untracked(self, tmp_path):
        path = tmp_path / "consent.json"
        store = ChannelConsentStore(path)
        store.set("C1", "declined")
        store.forget("C1")
        assert store.is_dormant("C1") is False
        assert ChannelConsentStore(path).status("C1") is None


# ---------------------------------------------------------------------------
# SlackAdapter integration
# ---------------------------------------------------------------------------


def make_adapter(tmp_path, gate=True, cjn=None):
    config = PlatformConfig(
        enabled=True,
        token="***",
        channel_consent_gate=gate,
        channel_join_notification=cjn,
    )
    a = SlackAdapter(config)
    a._app = MagicMock()
    client = AsyncMock()
    a._app.client = client
    a._bot_user_id = "U_BOT"
    a._running = True
    a._consent_store = ChannelConsentStore(tmp_path / "consent.json")
    a.handle_message = AsyncMock()
    return a, client


def join_event(user="U_BOT", channel="C_NEW", inviter="U_HUMAN"):
    return {
        "type": "member_joined_channel",
        "user": user,
        "channel": channel,
        "team": "T1",
        "inviter": inviter,
    }


def consent_click(action_id, channel="C_NEW", user_id="U_HUMAN"):
    body = {
        "channel": {"id": channel},
        "user": {"id": user_id, "name": "jake"},
        "message": {"ts": "123.456"},
    }
    action = {"action_id": action_id, "value": channel}
    return body, action


@pytest.mark.asyncio
async def test_gate_disabled_join_posts_no_prompt(tmp_path):
    a, client = make_adapter(tmp_path, gate=False)
    await a._handle_member_joined_channel(join_event())
    client.chat_postMessage.assert_not_awaited()
    assert a._consent_store.status("C_NEW") is None


@pytest.mark.asyncio
async def test_join_marks_pending_and_posts_prompt(tmp_path):
    a, client = make_adapter(tmp_path)
    await a._handle_member_joined_channel(join_event())
    assert a._consent_store.status("C_NEW") == "pending"
    client.chat_postMessage.assert_awaited_once()
    kwargs = client.chat_postMessage.await_args.kwargs
    assert kwargs["channel"] == "C_NEW"
    action_ids = {
        el["action_id"]
        for b in kwargs["blocks"]
        if b["type"] == "actions"
        for el in b["elements"]
    }
    assert action_ids == {"hermes_consent_activate", "hermes_consent_decline"}


@pytest.mark.asyncio
async def test_other_member_join_does_not_trigger_gate(tmp_path):
    a, client = make_adapter(tmp_path)
    await a._handle_member_joined_channel(join_event(user="U_SOMEONE"))
    client.chat_postMessage.assert_not_awaited()
    assert a._consent_store.status("C_NEW") is None


@pytest.mark.asyncio
async def test_reinvite_to_approved_channel_keeps_approval(tmp_path):
    a, client = make_adapter(tmp_path)
    a._consent_store.set("C_NEW", "approved", by_user_id="U_HUMAN")
    await a._handle_member_joined_channel(join_event())
    client.chat_postMessage.assert_not_awaited()
    assert a._consent_store.status("C_NEW") == "approved"


@pytest.mark.asyncio
async def test_pending_channel_messages_are_dropped(tmp_path):
    a, _ = make_adapter(tmp_path)
    a._consent_store.set("C_NEW", "pending")
    await a._handle_slack_message(
        {
            "type": "message",
            "user": "U_HUMAN",
            "channel": "C_NEW",
            "channel_type": "channel",
            "team": "T1",
            "ts": "1.0",
            "text": "<@U_BOT> hello",
        }
    )
    a.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_untracked_channel_messages_flow(tmp_path):
    a, _ = make_adapter(tmp_path)
    await a._handle_slack_message(
        {
            "type": "message",
            "user": "U_HUMAN",
            "channel": "C_OLD",
            "channel_type": "channel",
            "team": "T1",
            "ts": "2.0",
            "text": "<@U_BOT> hello",
        }
    )
    a.handle_message.assert_awaited()


@pytest.mark.asyncio
async def test_dms_are_never_gated(tmp_path):
    a, _ = make_adapter(tmp_path)
    a._consent_store.set("D_DM", "pending")  # nonsensical, but must not gate
    await a._handle_slack_message(
        {
            "type": "message",
            "user": "U_HUMAN",
            "channel": "D_DM",
            "channel_type": "im",
            "team": "T1",
            "ts": "3.0",
            "text": "hello",
        }
    )
    a.handle_message.assert_awaited()


@pytest.mark.asyncio
async def test_activate_click_approves_and_unblocks(tmp_path):
    a, client = make_adapter(tmp_path)
    a._is_interactive_user_authorized = MagicMock(return_value=True)
    a._consent_store.set("C_NEW", "pending")

    body, action = consent_click("hermes_consent_activate")
    await a._handle_consent_action(AsyncMock(), body, action)

    assert a._consent_store.status("C_NEW") == "approved"
    assert a._is_channel_consent_blocked("C_NEW") is False
    client.chat_update.assert_awaited_once()  # prompt replaced, buttons gone


@pytest.mark.asyncio
async def test_decline_click_keeps_dormant(tmp_path):
    a, client = make_adapter(tmp_path)
    a._is_interactive_user_authorized = MagicMock(return_value=True)
    a._consent_store.set("C_NEW", "pending")

    body, action = consent_click("hermes_consent_decline")
    await a._handle_consent_action(AsyncMock(), body, action)

    assert a._consent_store.status("C_NEW") == "declined"
    assert a._is_channel_consent_blocked("C_NEW") is True


@pytest.mark.asyncio
async def test_unauthorized_click_is_ignored(tmp_path):
    a, client = make_adapter(tmp_path)
    a._is_interactive_user_authorized = MagicMock(return_value=False)
    a._consent_store.set("C_NEW", "pending")

    body, action = consent_click("hermes_consent_activate", user_id="U_RANDO")
    await a._handle_consent_action(AsyncMock(), body, action)

    assert a._consent_store.status("C_NEW") == "pending"
    client.chat_update.assert_not_awaited()


@pytest.mark.asyncio
async def test_join_notification_and_gate_both_fire(tmp_path):
    from gateway.platforms.base import SendResult

    a, client = make_adapter(
        tmp_path, cjn={"channel": "C_STATUS"}
    )
    a.send = AsyncMock(return_value=SendResult(success=True))
    await a._handle_member_joined_channel(join_event())
    # Status ping sent
    a.send.assert_awaited_once()
    assert a.send.await_args.args[0] == "C_STATUS"
    # Consent prompt posted into the joined channel
    client.chat_postMessage.assert_awaited_once()
    assert a._consent_store.status("C_NEW") == "pending"
