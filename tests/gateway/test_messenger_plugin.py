"""Focused contract tests for the native Facebook Messenger plugin."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
from unittest.mock import AsyncMock, MagicMock

from tests.gateway._plugin_adapter_loader import load_plugin_adapter


_messenger = load_plugin_adapter("messenger")
MessengerAdapter = _messenger.MessengerAdapter
ProcessedCommentLedger = _messenger.ProcessedCommentLedger
PRIVATE_REPLY_OPENING = _messenger.PRIVATE_REPLY_OPENING


def _config(tmp_path, **extra):
    from gateway.config import PlatformConfig

    values = {
        "page_id": "page-1",
        "page_access_token": "page-token",
        "app_secret": "app-secret",
        "verify_token": "verify-me",
        "receipt_path": str(tmp_path / "receipts.json"),
    }
    values.update(extra)
    return PlatformConfig(enabled=True, extra=values)


def test_webhook_verify_success_and_failure():
    body = b'{"object":"page"}'
    signature = "sha256=" + hmac.new(b"app-secret", body, hashlib.sha256).hexdigest()

    assert _messenger.verify_messenger_signature(body, signature, "app-secret")
    assert not _messenger.verify_messenger_signature(body, signature, "wrong-secret")
    assert _messenger.verify_webhook_challenge("subscribe", "verify-me", "verify-me")
    assert not _messenger.verify_webhook_challenge("subscribe", "wrong", "verify-me")


def test_webhook_rejects_bad_signature_and_oversized_body(tmp_path):
    adapter = MessengerAdapter(_config(tmp_path))

    class Request:
        headers = {"X-Hub-Signature-256": "sha256=bad"}
        query = {}

        async def read(self):
            return b"{}"

    response = asyncio.run(adapter._handle_webhook(Request()))
    assert response.status == 401

    class OversizedRequest(Request):
        headers = {"Content-Length": str(_messenger.WEBHOOK_BODY_MAX_BYTES + 1)}

    response = asyncio.run(adapter._handle_webhook(OversizedRequest()))
    assert response.status == 413


def test_comment_trigger_is_normalized_to_whole_words_and_deduplicated(tmp_path):
    assert _messenger.comment_matches_trigger("  HOTEL!!! ", "hotel")
    assert not _messenger.comment_matches_trigger("hotelier", "hotel")

    adapter = MessengerAdapter(_config(tmp_path))
    adapter._send_private_comment_reply = AsyncMock(return_value=True)
    comment = {"item": "comment", "comment_id": "comment-1", "message": "HOTEL"}

    asyncio.run(adapter._handle_comment(comment))
    asyncio.run(adapter._handle_comment(comment))

    adapter._send_private_comment_reply.assert_awaited_once_with("comment-1")
    assert adapter._ledger.contains("comment-1")


def test_comment_receipt_is_written_only_after_meta_accepts(tmp_path):
    adapter = MessengerAdapter(_config(tmp_path))
    adapter._send_private_comment_reply = AsyncMock(side_effect=[False, True])
    comment = {"item": "comment", "comment_id": "comment-2", "message": "hotel"}

    asyncio.run(adapter._handle_comment(comment))
    assert not adapter._ledger.contains("comment-2")
    asyncio.run(adapter._handle_comment(comment))
    assert adapter._ledger.contains("comment-2")
    assert adapter._send_private_comment_reply.await_count == 2


def test_dm_and_postback_normalize_to_gateway_events(tmp_path):
    adapter = MessengerAdapter(_config(tmp_path))
    adapter._fetch_psid_history = AsyncMock(return_value=["previous message"])
    adapter.handle_message = AsyncMock()

    asyncio.run(
        adapter._handle_messaging_event(
            {"sender": {"id": "psid-1"}, "message": {"mid": "mid-1", "text": "hello"}}
        )
    )
    event = adapter.handle_message.await_args.args[0]
    assert event.text == "hello"
    assert event.source.chat_id == "psid-1"
    assert event.channel_context == "previous message"

    adapter.handle_message.reset_mock()
    asyncio.run(
        adapter._handle_messaging_event(
            {
                "sender": {"id": "psid-1"},
                "postback": {"title": "Get answer", "payload": "answer-now"},
            }
        )
    )
    event = adapter.handle_message.await_args.args[0]
    assert event.text == "Get answer"
    assert event.metadata["messenger_postback"]["payload"] == "answer-now"


def test_graph_send_uses_response_payload(tmp_path):
    adapter = MessengerAdapter(_config(tmp_path))
    adapter._http_client = MagicMock()
    response = MagicMock(status_code=200)
    response.json.return_value = {"message_id": "mid-out"}
    adapter._http_client.post = AsyncMock(return_value=response)

    result = asyncio.run(adapter.send("psid-1", "Hola"))

    assert result.success
    kwargs = adapter._http_client.post.await_args.kwargs
    assert kwargs["json"] == {
        "recipient": {"id": "psid-1"},
        "message": {"text": "Hola"},
        "messaging_type": "RESPONSE",
    }
    assert "/page-1/messages" in adapter._http_client.post.await_args.args[0]


def test_comment_private_reply_uses_fixed_opening(tmp_path):
    adapter = MessengerAdapter(_config(tmp_path))
    adapter._http_client = MagicMock()
    response = MagicMock(status_code=200)
    response.json.return_value = {"message_id": "private-1"}
    adapter._http_client.post = AsyncMock(return_value=response)

    assert asyncio.run(adapter._send_private_comment_reply("comment-1"))
    payload = adapter._http_client.post.await_args.kwargs["json"]
    assert payload == {
        "recipient": {"comment_id": "comment-1"},
        "message": {"text": PRIVATE_REPLY_OPENING},
    }


def test_lookback_values_are_bounded(tmp_path):
    adapter = MessengerAdapter(
        _config(tmp_path, lookback_days=99, max_comments=99999)
    )
    assert adapter.lookback_days == _messenger.MAX_LOOKBACK_DAYS
    assert adapter.max_comments == _messenger.MAX_LOOKBACK_COMMENTS


def test_register_advertises_messenger_credentials():
    class Context:
        def __init__(self):
            self.kwargs = None

        def register_platform(self, **kwargs):
            self.kwargs = kwargs

    context = Context()
    _messenger.register(context)
    assert set(context.kwargs["required_env"]) == {
        "MESSENGER_PAGE_ID",
        "MESSENGER_PAGE_ACCESS_TOKEN",
        "MESSENGER_APP_SECRET",
        "MESSENGER_VERIFY_TOKEN",
    }
