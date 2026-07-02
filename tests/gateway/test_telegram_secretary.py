import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter, TelegramSecretaryUpdateHandler
from plugins.platforms.telegram.secretary import (
    SecretaryAuditStore,
    SecretaryEvent,
    SecretaryMode,
    SecretaryPolicy,
    normalize_secretary_update,
)

try:
    from telegram import Update
    from telegram.ext import ApplicationBuilder
except ImportError:  # pragma: no cover - optional dependency missing
    Update = None
    ApplicationBuilder = None

REAL_TELEGRAM_UPDATE = (
    isinstance(Update, type)
    and not isinstance(Update, Mock)
    and getattr(Update, "__module__", "").startswith("telegram.")
)


@pytest.mark.skipif(ApplicationBuilder is None, reason="real python-telegram-bot ApplicationBuilder not installed")
def test_secretary_update_handler_can_be_registered_in_ptb_application():
    async def callback(update, context):
        return None

    handler = TelegramSecretaryUpdateHandler(callback)
    app = ApplicationBuilder().token("123:fake").build()

    app.add_handler(handler)


def test_secretary_policy_is_disabled_and_generic_by_default():
    policy = SecretaryPolicy.from_config({})

    assert policy.active is False
    assert policy.mode is SecretaryMode.OFF
    assert policy.display_name == "Hermes"
    assert policy.owner_chat_id is None
    assert policy.allowed_business_connections == frozenset()
    assert not hasattr(policy, "allowed_outbound_chats")
    assert not hasattr(policy, "outbound_attribution")


def test_secretary_policy_parses_inbound_audit_only_config(tmp_path: Path):
    policy = SecretaryPolicy.from_config(
        {
            "enabled": True,
            "display_name": "Business intake",
            "mode": "read_only",
            "owner_chat_id": "1001",
            "topics": {"inbox": 11},
            "allowed_business_connections": ["bc-1"],
            "allowed_chats": ["2002"],
            "audit_path": str(tmp_path / "business.jsonl"),
        }
    )

    assert policy.active is True
    assert policy.mode is SecretaryMode.READ_ONLY
    assert policy.display_name == "Business intake"
    assert policy.reports_to_owner is True
    assert policy.topics == {"inbox": "11"}
    assert policy.allows_business_connection("bc-1") is True
    assert policy.allows_chat("2002") is True
    assert policy.allows_chat("999") is False
    assert policy.audit_path == tmp_path / "business.jsonl"


def test_secretary_audit_store_appends_jsonl(tmp_path: Path):
    audit_path = tmp_path / "audit" / "business.jsonl"
    store = SecretaryAuditStore(audit_path)

    store.append(
        SecretaryEvent(
            event_type="business_message",
            update_id=42,
            business_connection_id="bc-1",
            chat_id="100",
            message_id="7",
            text="hello",
            rights={"can_reply": False},
        )
    )

    row = json.loads(audit_path.read_text(encoding="utf-8"))
    assert row["event_type"] == "business_message"
    assert row["business_connection_id"] == "bc-1"
    assert row["rights"] == {"can_reply": False}


@pytest.mark.skipif(not REAL_TELEGRAM_UPDATE, reason="real python-telegram-bot Update not installed")
def test_secretary_normalizes_synthetic_ptb_business_message_payload():
    update = Update.de_json(
        {
            "update_id": 501,
            "business_message": {
                "message_id": 77,
                "date": 1700000000,
                "chat": {"id": 2002, "type": "private", "first_name": "Alice"},
                "from": {
                    "id": 4004,
                    "is_bot": False,
                    "first_name": "Alice",
                    "username": "alice",
                },
                "business_connection_id": "bc_1",
                "text": "Can you send me the invoice?",
            },
        },
        bot=None,
    )

    event = normalize_secretary_update(update)

    assert event is not None
    assert event.event_type == "business_message"
    assert event.update_id == 501
    assert event.business_connection_id == "bc_1"
    assert event.chat_id == "2002"
    assert event.message_id == "77"
    assert event.from_username == "alice"
    assert event.text == "Can you send me the invoice?"


@pytest.mark.skipif(not REAL_TELEGRAM_UPDATE, reason="real python-telegram-bot Update not installed")
def test_secretary_normalizes_synthetic_ptb_business_connection_and_deleted_payloads():
    connection_update = Update.de_json(
        {
            "update_id": 502,
            "business_connection": {
                "id": "bc_1",
                "user": {"id": 3003, "is_bot": False, "first_name": "Owner", "username": "owner"},
                "user_chat_id": 1001,
                "date": 1700000000,
                "can_reply": True,
                "is_enabled": True,
            },
        },
        bot=None,
    )
    deleted_update = Update.de_json(
        {
            "update_id": 503,
            "deleted_business_messages": {
                "business_connection_id": "bc_1",
                "chat": {"id": 2002, "type": "private", "first_name": "Alice"},
                "message_ids": [77, 78],
            },
        },
        bot=None,
    )

    connection_event = normalize_secretary_update(connection_update)
    deleted_event = normalize_secretary_update(deleted_update)

    assert connection_event is not None
    assert connection_event.event_type == "business_connection"
    assert connection_event.business_connection_id == "bc_1"
    assert connection_event.connection_user_id == "3003"
    assert connection_event.rights["can_reply"] is True
    assert deleted_event is not None
    assert deleted_event.event_type == "deleted_business_messages"
    assert deleted_event.business_connection_id == "bc_1"
    assert deleted_event.message_ids == ["77", "78"]


@pytest.mark.asyncio
async def test_secretary_business_message_is_audited_not_dispatched(tmp_path: Path):
    audit_path = tmp_path / "business.jsonl"
    adapter = TelegramAdapter(
        PlatformConfig(
            enabled=True,
            token="fake",
            extra={
                "secretary": {
                    "enabled": True,
                    "mode": "read_only",
                    "allowed_business_connections": ["bc-1"],
                    "audit_path": str(audit_path),
                }
            },
        )
    )
    dispatched = []

    async def fake_handle_message(event):
        dispatched.append(event)

    adapter.handle_message = fake_handle_message
    update = SimpleNamespace(
        update_id=99,
        business_message=SimpleNamespace(
            business_connection_id="bc-1",
            chat=SimpleNamespace(id=100, type="private"),
            from_user=SimpleNamespace(id=200, username="alice"),
            message_id=7,
            text="Please call me",
            caption=None,
        ),
    )

    await adapter._handle_secretary_update(update, SimpleNamespace())

    assert dispatched == []
    row = json.loads(audit_path.read_text(encoding="utf-8"))
    assert row["event_type"] == "business_message"
    assert row["text"] == "Please call me"
    assert row["ignored_reason"] is None


@pytest.mark.asyncio
async def test_secretary_disallowed_business_connection_is_audited_as_ignored(tmp_path: Path):
    audit_path = tmp_path / "business.jsonl"
    adapter = TelegramAdapter(
        PlatformConfig(
            enabled=True,
            token="fake",
            extra={
                "secretary": {
                    "enabled": True,
                    "mode": "audit_only",
                    "allowed_business_connections": ["bc-allowed"],
                    "audit_path": str(audit_path),
                }
            },
        )
    )
    update = SimpleNamespace(
        update_id=100,
        business_message=SimpleNamespace(
            business_connection_id="bc-blocked",
            chat=SimpleNamespace(id=100, type="private"),
            from_user=SimpleNamespace(id=200, username="alice"),
            message_id=7,
            text="Please call me",
            caption=None,
        ),
    )

    await adapter._handle_secretary_update(update, SimpleNamespace())

    row = json.loads(audit_path.read_text(encoding="utf-8"))
    assert row["ignored_reason"] == "business_connection_not_allowed"
