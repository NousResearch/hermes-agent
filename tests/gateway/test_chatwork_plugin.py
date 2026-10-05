"""Tests for the Chatwork platform adapter plugin."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
from unittest.mock import AsyncMock, MagicMock

from gateway.config import PlatformConfig
from gateway.platforms.base import MessageType
from tests.gateway._plugin_adapter_loader import load_plugin_adapter

_chatwork = load_plugin_adapter("chatwork")
ChatworkAdapter = _chatwork.ChatworkAdapter
verify_chatwork_signature = _chatwork.verify_chatwork_signature
strip_chatwork_mentions = _chatwork.strip_chatwork_mentions
_env_enablement = _chatwork._env_enablement
register = _chatwork.register


def _webhook_token(raw_key: bytes = b"webhook-secret") -> str:
    return base64.b64encode(raw_key).decode("ascii")


def _signature(body: bytes, raw_key: bytes = b"webhook-secret") -> str:
    digest = hmac.new(raw_key, body, hashlib.sha256).digest()
    return base64.b64encode(digest).decode("ascii")


def _adapter(monkeypatch) -> ChatworkAdapter:
    for name in (
        "CHATWORK_API_TOKEN",
        "CHATWORK_WEBHOOK_TOKEN",
        "CHATWORK_ALLOWED_ROOMS",
        "CHATWORK_HOST",
        "CHATWORK_PORT",
    ):
        monkeypatch.delenv(name, raising=False)
    return ChatworkAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "api_token": "api-token",
                "webhook_token": _webhook_token(),
            },
        )
    )


class TestSignature:
    def test_valid_signature(self):
        body = b'{"webhook_event_type":"message_created"}'
        assert verify_chatwork_signature(body, _signature(body), _webhook_token())

    def test_tampered_body_is_rejected(self):
        body = b"payload"
        assert not verify_chatwork_signature(body + b"x", _signature(body), _webhook_token())

    def test_invalid_base64_token_is_rejected(self):
        assert not verify_chatwork_signature(b"payload", "sig", "not-base64!")

    def test_missing_values_are_rejected(self):
        assert not verify_chatwork_signature(b"payload", "", _webhook_token())
        assert not verify_chatwork_signature(b"payload", "signature", "")


class TestTextNormalization:
    def test_removes_to_line_and_preserves_message(self):
        body = "[To:1234567] Hermes Bot\n調べてください"
        assert strip_chatwork_mentions(body) == "調べてください"

    def test_plain_text_is_unchanged(self):
        assert strip_chatwork_mentions("hello") == "hello"

    def test_inline_mention_preserves_message_text(self):
        assert strip_chatwork_mentions("[To:1484814]おかずはなんですか？") == "おかずはなんですか？"


class TestInbound:
    def test_dispatches_message_event(self, monkeypatch):
        adapter = _adapter(monkeypatch)
        adapter._account_id = "999"
        adapter.get_chat_info = AsyncMock(return_value={"name": "開発", "type": "group"})
        adapter.handle_message = AsyncMock()
        payload = {
            "webhook_event_type": "mention_to_me",
            "webhook_event": {
                "room_id": 100,
                "from_account_id": 200,
                "message_id": "300",
                "body": "[To:999] Hermes\nこんにちは",
            },
        }

        asyncio.run(adapter._dispatch_webhook(payload))

        adapter.handle_message.assert_awaited_once()
        event = adapter.handle_message.await_args.args[0]
        assert event.text == "こんにちは"
        assert event.message_type is MessageType.TEXT
        assert event.source.chat_id == "100"
        assert event.source.user_id == "200"
        assert event.message_id == "300"
        assert adapter._reply_context["300"] == "200"

    def test_ignores_self_message(self, monkeypatch):
        adapter = _adapter(monkeypatch)
        adapter._account_id = "200"
        adapter.handle_message = AsyncMock()
        payload = {
            "webhook_event_type": "message_created",
            "webhook_event": {
                "room_id": 100,
                "account_id": 200,
                "message_id": "300",
                "body": "self echo",
            },
        }
        asyncio.run(adapter._dispatch_webhook(payload))
        adapter.handle_message.assert_not_awaited()

    def test_message_created_uses_account_id(self, monkeypatch):
        adapter = _adapter(monkeypatch)
        adapter._account_id = "999"
        adapter.get_chat_info = AsyncMock(return_value={"name": "全体", "type": "group"})
        adapter.handle_message = AsyncMock()
        payload = {
            "webhook_event_type": "message_created",
            "webhook_event": {
                "room_id": 100,
                "account_id": 201,
                "message_id": "301",
                "body": "room message",
            },
        }

        asyncio.run(adapter._dispatch_webhook(payload))

        event = adapter.handle_message.await_args.args[0]
        assert event.source.user_id == "201"
        assert event.text == "room message"

    def test_ignores_non_allowed_room(self, monkeypatch):
        adapter = _adapter(monkeypatch)
        adapter.allowed_rooms = {"101"}
        adapter.handle_message = AsyncMock()
        payload = {
            "webhook_event_type": "message_created",
            "webhook_event": {
                "room_id": 100,
                "account_id": 200,
                "message_id": "300",
                "body": "not allowed",
            },
        }
        asyncio.run(adapter._dispatch_webhook(payload))
        adapter.handle_message.assert_not_awaited()


class TestWebhookHandler:
    def test_query_signature_is_accepted_and_dispatch_is_backgrounded(self, monkeypatch):
        adapter = _adapter(monkeypatch)
        body = b'{"webhook_event_type":"message_created","webhook_event":{}}'
        blocker = asyncio.Event()

        async def slow_dispatch(_payload):
            await blocker.wait()

        adapter._dispatch_webhook = AsyncMock(side_effect=slow_dispatch)
        request = MagicMock()
        request.read = AsyncMock(return_value=body)
        request.headers = {}
        request.query = {"chatwork_webhook_signature": _signature(body)}

        async def scenario():
            response = await adapter._handle_webhook(request)
            assert response.status == 200
            assert len(adapter._webhook_tasks) == 1
            await asyncio.sleep(0)
            adapter._dispatch_webhook.assert_awaited_once()
            await adapter.disconnect()

        asyncio.run(scenario())


class TestSend:
    def test_reply_tag_is_added_for_known_trigger(self, monkeypatch):
        adapter = _adapter(monkeypatch)
        adapter._session = MagicMock()
        adapter._reply_context["300"] = "200"
        adapter._post_message = AsyncMock(
            return_value=_chatwork.SendResult(success=True, message_id="301")
        )

        result = asyncio.run(adapter.send("100", "回答です", reply_to="300"))

        assert result.success
        adapter._post_message.assert_awaited_once_with(
            "100", "[rp aid=200 to=100-300]\n回答です"
        )

    def test_flat_send_when_trigger_is_unknown(self, monkeypatch):
        adapter = _adapter(monkeypatch)
        adapter._session = MagicMock()
        adapter._post_message = AsyncMock(
            return_value=_chatwork.SendResult(success=True, message_id="301")
        )
        asyncio.run(adapter.send("100", "通知", reply_to="missing"))
        adapter._post_message.assert_awaited_once_with("100", "通知")


class TestConfigAndRegistration:
    def test_env_enablement_seeds_home_channel(self, monkeypatch):
        monkeypatch.setenv("CHATWORK_API_TOKEN", "api")
        monkeypatch.setenv("CHATWORK_WEBHOOK_TOKEN", "webhook")
        monkeypatch.setenv("CHATWORK_HOME_CHANNEL", "123")
        seeded = _env_enablement()
        assert seeded["home_channel"]["chat_id"] == "123"

    def test_register_metadata(self):
        ctx = MagicMock()
        register(ctx)
        kwargs = ctx.register_platform.call_args.kwargs
        assert kwargs["name"] == "chatwork"
        assert kwargs["allowed_users_env"] == "CHATWORK_ALLOWED_USERS"
        assert kwargs["cron_deliver_env_var"] == "CHATWORK_HOME_CHANNEL"
        assert kwargs["standalone_sender_fn"] is not None
