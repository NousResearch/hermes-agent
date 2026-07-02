"""Telegram Business owner-topic triage cards."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.telegram import adapter as telegram_adapter
from plugins.platforms.telegram.adapter import TelegramAdapter
from plugins.platforms.telegram.secretary import (
    SecretaryEvent,
    SecretaryMode,
    SecretaryPolicy,
    event_can_reply,
    render_owner_summary,
)


class _SentMessage:
    def __init__(self, message_id: int):
        self.message_id = message_id


def _make_adapter():
    config = PlatformConfig(
        enabled=True,
        token="test-token",
        extra={
            "secretary": {
                "enabled": True,
                "mode": "read_only",
                "display_name": "Business intake",
                "owner_chat_id": "1001",
                "topics": {"inbox": "11"},
            }
        },
    )
    adapter = TelegramAdapter(config)
    adapter._bot = SimpleNamespace(send_message=AsyncMock(return_value=_SentMessage(101)))
    return adapter


def test_secretary_policy_parses_owner_topic_without_outbound_surface():
    policy = SecretaryPolicy.from_config(
        {
            "enabled": True,
            "mode": "read_only",
            "owner_chat_id": "1001",
            "topics": {"inbox": "11"},
        }
    )

    assert policy.active is True
    assert policy.mode is SecretaryMode.READ_ONLY
    assert policy.reports_to_owner is True
    assert policy.topics["inbox"] == "11"
    assert not hasattr(policy, "allowed_outbound_aliases")
    assert not hasattr(policy, "can_send_after_approval")


def test_render_owner_summary_includes_required_triage_fields():
    event = SecretaryEvent(
        event_type="business_message",
        business_connection_id="bc_1",
        chat_id="2002",
        message_id="77",
        from_user_id="3003",
        from_username="alice",
        text="Can you send me the invoice?",
        rights={"can_reply": True},
    )

    card = render_owner_summary(event, assistant_name="Business intake")

    assert "Telegram Business triage" in card
    assert "Business intake" in card
    assert "Classification:" in card
    assert "inbound.text" in card
    assert "Summary:" in card
    assert "Suggested next action:" in card
    assert "no message was dispatched" in card
    assert "Source:" in card
    assert "rights.can_reply:</b> <code>true</code>" in card
    assert "business_connection_id" in card


@pytest.mark.asyncio
async def test_secretary_business_message_routes_one_read_only_owner_card(monkeypatch, tmp_path):
    # Tests run with a MagicMock telegram module in some files; force stable
    # primitive values for parse mode so kwargs are easy to inspect.
    monkeypatch.setattr(telegram_adapter.ParseMode, "HTML", "HTML", raising=False)

    adapter = _make_adapter()
    adapter._secretary_policy = SecretaryPolicy.from_config(
        {
            "enabled": True,
            "mode": "read_only",
            "display_name": "Business intake",
            "owner_chat_id": "1001",
            "topics": {"inbox": "11"},
            "audit_path": str(tmp_path / "business.jsonl"),
        }
    )
    adapter._secretary_audit_store = None
    connection_update = SimpleNamespace(
        update_id=1,
        business_connection=SimpleNamespace(
            id="bc_1",
            user=SimpleNamespace(id=3003, username="owner"),
            user_chat_id=1001,
            can_reply=True,
            is_enabled=True,
            rights=None,
        ),
    )
    message_update = SimpleNamespace(
        update_id=2,
        business_message=SimpleNamespace(
            business_connection_id="bc_1",
            chat=SimpleNamespace(id=2002, type="private"),
            from_user=SimpleNamespace(id=4004, username="alice", full_name="Alice"),
            message_id=77,
            text="Can you send me the invoice?",
            caption=None,
        ),
    )

    await adapter._handle_secretary_update(connection_update, None)
    await adapter._handle_secretary_update(message_update, None)

    assert adapter._bot.send_message.await_count == 2
    calls = adapter._bot.send_message.await_args_list
    assert calls[0].kwargs["chat_id"] == 1001
    assert calls[0].kwargs["message_thread_id"] == 11
    assert calls[1].kwargs["message_thread_id"] == 11
    assert "Telegram Business triage" in calls[1].kwargs["text"]
    assert "Can you send me the invoice?" in calls[1].kwargs["text"]
    assert "reply_markup" not in calls[1].kwargs


def test_event_can_reply_uses_known_connection_rights():
    event = SecretaryEvent(event_type="business_message", business_connection_id="bc_1")
    assert event_can_reply(event, {"bc_1": {"can_reply": True}}) is True
