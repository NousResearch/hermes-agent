"""Tests for the WhatsApp Cloud API adapter (Phase 2).

Covers the outbound Graph API send path and the inbound verify-token
handshake. The webhook POST path is currently a stub (Phase 3 will add
signature verification + dispatch); we just confirm it accepts a body
and returns 200 here.

All tests are fixture-driven — no live network. httpx is patched so the
adapter never reaches graph.facebook.com, and the aiohttp server is
exercised with synthetic ``Request`` objects.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_adapter(**overrides):
    """Build a WhatsAppCloudAdapter with test attributes (bypass __init__).

    Mirrors the pattern in tests/gateway/test_whatsapp_*.py.
    """
    from gateway.platforms.whatsapp_cloud import WhatsAppCloudAdapter

    adapter = WhatsAppCloudAdapter.__new__(WhatsAppCloudAdapter)
    adapter.platform = Platform.WHATSAPP_CLOUD
    adapter.config = MagicMock()
    adapter.config.extra = {}

    # Cloud-API-specific attributes
    adapter._phone_number_id = overrides.pop("phone_number_id", "1234567890")
    adapter._access_token = overrides.pop("access_token", "test-token")
    adapter._app_id = overrides.pop("app_id", "")
    adapter._app_secret = overrides.pop("app_secret", "")
    adapter._waba_id = overrides.pop("waba_id", "")
    adapter._verify_token = overrides.pop("verify_token", "")
    adapter._webhook_host = "127.0.0.1"
    adapter._webhook_port = 8090
    adapter._webhook_path = "/whatsapp/webhook"
    adapter._health_path = "/health"
    adapter._api_version = overrides.pop("api_version", "v20.0")
    adapter._runner = None
    adapter._http_client = None

    # Behavior-mixin contract
    adapter._reply_prefix = None
    adapter._dm_policy = "open"
    adapter._allow_from = set()
    adapter._group_policy = "open"
    adapter._group_allow_from = set()
    adapter._mention_patterns = []

    # Webhook dispatch state (Phase 3)
    from collections import OrderedDict
    adapter._seen_wamids = OrderedDict()
    adapter._duplicate_count = 0
    adapter._accepted_count = 0
    adapter._rejected_signature_count = 0

    # Phase 4 state — one-shot warnings.
    adapter._warned_no_ffmpeg = False

    # Phase 10 state — per-chat latest inbound wamid (for typing/read).
    adapter._last_inbound_wamid_by_chat = {}

    # Phase 9 state — interactive-button correlation dicts.
    adapter._clarify_state = {}
    adapter._exec_approval_state = {}
    adapter._slash_confirm_state = {}

    # BasePlatformAdapter contract — minimum to keep send/lifecycle happy
    adapter._running = True
    adapter._message_handler = None
    adapter._fatal_error_code = None
    adapter._fatal_error_message = None
    adapter._fatal_error_retryable = True
    adapter._fatal_error_handler = None
    adapter._active_sessions = {}
    adapter._pending_messages = {}
    adapter._background_tasks = set()
    adapter._auto_tts_disabled_chats = set()

    # Apply any leftover overrides directly
    for key, value in overrides.items():
        setattr(adapter, key, value)
    return adapter


def _mock_httpx_response(status_code: int, json_body: dict):
    """Build an httpx-Response-like mock the adapter's ``send`` will accept."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json = MagicMock(return_value=json_body)
    resp.text = json.dumps(json_body)
    return resp


# ---------------------------------------------------------------------------
# Outbound send via Graph API
# ---------------------------------------------------------------------------

class TestSendText:
    """Outbound text-message path."""

    @pytest.mark.asyncio
    async def test_send_builds_correct_url(self):
        adapter = _make_adapter(phone_number_id="9999", api_version="v20.0")
        adapter._http_client = MagicMock()
        adapter._http_client.post = AsyncMock(
            return_value=_mock_httpx_response(
                200, {"messages": [{"id": "wamid.abc"}]}
            )
        )

        await adapter.send("15551234567", "hello")

        called_url = adapter._http_client.post.call_args.args[0]
        assert called_url == "https://graph.facebook.com/v20.0/9999/messages"

    @pytest.mark.asyncio
    async def test_send_includes_bearer_auth(self):
        adapter = _make_adapter(access_token="my-secret-token")
        adapter._http_client = MagicMock()
        adapter._http_client.post = AsyncMock(
            return_value=_mock_httpx_response(
                200, {"messages": [{"id": "wamid.abc"}]}
            )
        )

        await adapter.send("15551234567", "hi")

        headers = adapter._http_client.post.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer my-secret-token"
        assert headers["Content-Type"] == "application/json"

    @pytest.mark.asyncio
    async def test_send_payload_shape(self):
        adapter = _make_adapter()
        adapter._http_client = MagicMock()
        adapter._http_client.post = AsyncMock(
            return_value=_mock_httpx_response(
                200, {"messages": [{"id": "wamid.abc"}]}
            )
        )

        await adapter.send("15551234567", "hello world")

        payload = adapter._http_client.post.call_args.kwargs["json"]
        assert payload["messaging_product"] == "whatsapp"
        assert payload["recipient_type"] == "individual"
        assert payload["to"] == "15551234567"
        assert payload["type"] == "text"
        assert payload["text"]["body"] == "hello world"
        assert payload["text"]["preview_url"] is True

    @pytest.mark.asyncio
    async def test_send_returns_wamid(self):
        adapter = _make_adapter()
        adapter._http_client = MagicMock()
        adapter._http_client.post = AsyncMock(
            return_value=_mock_httpx_response(
                200, {"messages": [{"id": "wamid.HBgL...="}]}
            )
        )

        result = await adapter.send("15551234567", "hi")

        assert result.success is True
        assert result.message_id == "wamid.HBgL...="

    @pytest.mark.asyncio
    async def test_send_applies_markdown_conversion(self):
        """Mixin's format_message should run before send."""
        adapter = _make_adapter()
        adapter._http_client = MagicMock()
        adapter._http_client.post = AsyncMock(
            return_value=_mock_httpx_response(
                200, {"messages": [{"id": "wamid.x"}]}
            )
        )

        await adapter.send("15551234567", "**bold** text")

        payload = adapter._http_client.post.call_args.kwargs["json"]
        assert payload["text"]["body"] == "*bold* text"

    @pytest.mark.asyncio
    async def test_send_reply_to_attaches_context_first_chunk_only(self):
        adapter = _make_adapter()
        adapter._http_client = MagicMock()
        adapter._http_client.post = AsyncMock(
            return_value=_mock_httpx_response(
                200, {"messages": [{"id": "wamid.x"}]}
            )
        )

        await adapter.send("15551234567", "short reply", reply_to="wamid.original")

        payload = adapter._http_client.post.call_args.kwargs["json"]
        assert payload["context"] == {"message_id": "wamid.original"}

    @pytest.mark.asyncio
    async def test_send_long_message_chunked(self):
        """Messages over the chunk limit are split into multiple POSTs."""
        adapter = _make_adapter()
        adapter._http_client = MagicMock()
        adapter._http_client.post = AsyncMock(
            return_value=_mock_httpx_response(
                200, {"messages": [{"id": "wamid.x"}]}
            )
        )

        # MAX_MESSAGE_LENGTH = 4096 from the mixin. 8500 chars forces 2+ chunks.
        long_text = "a" * 8500
        await adapter.send("15551234567", long_text)

        # At least 2 POST calls
        assert adapter._http_client.post.call_count >= 2
        # Second call should NOT have context (only first chunk gets reply_to)
        first_call = adapter._http_client.post.call_args_list[0]
        second_call = adapter._http_client.post.call_args_list[1]
        # No reply_to passed → no context anywhere, but verify structure anyway
        assert "context" not in second_call.kwargs["json"]

    @pytest.mark.asyncio
    async def test_send_graph_error_returns_failure(self):
        adapter = _make_adapter()
        adapter._http_client = MagicMock()
        adapter._http_client.post = AsyncMock(
            return_value=_mock_httpx_response(
                400,
                {
                    "error": {
                        "message": "Invalid parameter",
                        "type": "OAuthException",
                        "code": 100,
                        "fbtrace_id": "abc",
                    }
                },
            )
        )

        result = await adapter.send("15551234567", "hi")

        assert result.success is False
        assert "graph error 100" in result.error
        assert "Invalid parameter" in result.error

    @pytest.mark.asyncio
    async def test_send_empty_content_no_request(self):
        adapter = _make_adapter()
        adapter._http_client = MagicMock()
        adapter._http_client.post = AsyncMock()

        result = await adapter.send("15551234567", "")
        assert result.success is True
        assert result.message_id is None
        adapter._http_client.post.assert_not_called()

        result = await adapter.send("15551234567", "   \n  ")
        assert result.success is True
        adapter._http_client.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_send_not_connected_returns_failure(self):
        adapter = _make_adapter()
        adapter._http_client = None

        result = await adapter.send("15551234567", "hi")
        assert result.success is False
        assert "Not connected" in result.error

    @pytest.mark.asyncio
    async def test_send_network_exception_returns_failure(self):
        adapter = _make_adapter()
        adapter._http_client = MagicMock()
        adapter._http_client.post = AsyncMock(side_effect=RuntimeError("boom"))

        result = await adapter.send("15551234567", "hi")
        assert result.success is False
        assert "boom" in result.error


# ---------------------------------------------------------------------------
# Inbound webhook verify (GET) handshake
# ---------------------------------------------------------------------------

def _verify_request(query: dict):
    """Build a minimal aiohttp.web.Request stub for verify tests."""
    request = MagicMock()
    request.query = query
    return request


class TestWebhookVerify:
    """GET <webhook>?hub.mode=...&hub.verify_token=...&hub.challenge=..."""

    @pytest.mark.asyncio
    async def test_verify_echoes_challenge_on_match(self):
        adapter = _make_adapter(verify_token="shared-secret-123")
        request = _verify_request({
            "hub.mode": "subscribe",
            "hub.verify_token": "shared-secret-123",
            "hub.challenge": "abc-12345",
        })

        response = await adapter._handle_verify(request)

        assert response.status == 200
        assert response.text == "abc-12345"
        assert response.content_type == "text/plain"

    @pytest.mark.asyncio
    async def test_verify_rejects_token_mismatch(self):
        adapter = _make_adapter(verify_token="shared-secret-123")
        request = _verify_request({
            "hub.mode": "subscribe",
            "hub.verify_token": "wrong-token",
            "hub.challenge": "abc-12345",
        })

        response = await adapter._handle_verify(request)

        assert response.status == 403

    @pytest.mark.asyncio
    async def test_verify_rejects_wrong_mode(self):
        adapter = _make_adapter(verify_token="shared-secret-123")
        request = _verify_request({
            "hub.mode": "unsubscribe",
            "hub.verify_token": "shared-secret-123",
            "hub.challenge": "abc-12345",
        })

        response = await adapter._handle_verify(request)

        assert response.status == 400

    @pytest.mark.asyncio
    async def test_verify_rejects_missing_challenge(self):
        adapter = _make_adapter(verify_token="shared-secret-123")
        request = _verify_request({
            "hub.mode": "subscribe",
            "hub.verify_token": "shared-secret-123",
        })

        response = await adapter._handle_verify(request)

        assert response.status == 400

    @pytest.mark.asyncio
    async def test_verify_refuses_when_token_unconfigured(self):
        """An empty verify_token must NOT match an empty incoming token —
        otherwise an attacker who guesses the misconfiguration could
        subscribe their own webhook URL.
        """
        adapter = _make_adapter(verify_token="")
        request = _verify_request({
            "hub.mode": "subscribe",
            "hub.verify_token": "",
            "hub.challenge": "abc",
        })

        response = await adapter._handle_verify(request)

        assert response.status == 503  # service refuses to perform handshake


# ---------------------------------------------------------------------------
# Inbound webhook POST — signature verification + dispatch (Phase 3)
# ---------------------------------------------------------------------------

import hashlib
import hmac as _hmac_lib


def _sign(secret: str, body: bytes) -> str:
    """Compute the X-Hub-Signature-256 header value Meta would send."""
    digest = _hmac_lib.new(
        secret.encode("utf-8"), body, hashlib.sha256
    ).hexdigest()
    return f"sha256={digest}"


def _post_request(body: bytes, headers: dict | None = None):
    """Build a minimal aiohttp.web.Request stub for POST tests."""
    request = MagicMock()
    request.read = AsyncMock(return_value=body)
    request.headers = headers or {}
    return request


# A realistic Meta inbound text-message payload, modelled on the
# get-started docs sample.
_SAMPLE_INBOUND_TEXT_PAYLOAD = {
    "object": "whatsapp_business_account",
    "entry": [
        {
            "id": "215589313241560883",
            "changes": [
                {
                    "field": "messages",
                    "value": {
                        "messaging_product": "whatsapp",
                        "metadata": {
                            "display_phone_number": "15551797781",
                            "phone_number_id": "7794189252778687",
                        },
                        "contacts": [
                            {
                                "profile": {"name": "Jessica Laverdetman"},
                                "wa_id": "13557825698",
                            }
                        ],
                        "messages": [
                            {
                                "from": "13557825698",
                                "id": "wamid.HBgLMTM1NTc4MjU2OTgVAGHAYWYET688aASGNTI1QzZFQjhEMDk2QQA=",
                                "timestamp": "1758254144",
                                "text": {"body": "Hi!"},
                                "type": "text",
                            }
                        ],
                    },
                }
            ],
        }
    ],
}


class TestWebhookSignature:
    """X-Hub-Signature-256 HMAC verification."""

    @pytest.mark.asyncio
    async def test_valid_signature_accepted(self):
        adapter = _make_adapter(app_secret="signing-key-123")
        # Patch the dispatcher to a no-op so we don't depend on
        # MessageEvent construction here (covered separately).
        adapter._dispatch_payload = AsyncMock()
        body = b'{"object":"whatsapp_business_account","entry":[]}'
        request = _post_request(body, {"X-Hub-Signature-256": _sign("signing-key-123", body)})

        response = await adapter._handle_webhook(request)

        assert response.status == 200
        adapter._dispatch_payload.assert_called_once()

    @pytest.mark.asyncio
    async def test_tampered_body_rejected(self):
        adapter = _make_adapter(app_secret="signing-key-123")
        adapter._dispatch_payload = AsyncMock()
        original = b'{"object":"whatsapp_business_account"}'
        tampered = b'{"object":"evil_payload"}'
        sig_for_original = _sign("signing-key-123", original)
        request = _post_request(tampered, {"X-Hub-Signature-256": sig_for_original})

        response = await adapter._handle_webhook(request)

        assert response.status == 401
        adapter._dispatch_payload.assert_not_called()
        assert adapter._rejected_signature_count == 1

    @pytest.mark.asyncio
    async def test_missing_signature_header_rejected(self):
        adapter = _make_adapter(app_secret="signing-key-123")
        adapter._dispatch_payload = AsyncMock()
        body = b'{"object":"whatsapp_business_account"}'
        request = _post_request(body, {})

        response = await adapter._handle_webhook(request)

        assert response.status == 401
        adapter._dispatch_payload.assert_not_called()

    @pytest.mark.asyncio
    async def test_wrong_signature_format_rejected(self):
        adapter = _make_adapter(app_secret="signing-key-123")
        adapter._dispatch_payload = AsyncMock()
        body = b"{}"
        # Missing the required ``sha256=`` prefix
        request = _post_request(body, {"X-Hub-Signature-256": "deadbeef"})

        response = await adapter._handle_webhook(request)
        assert response.status == 401

    @pytest.mark.asyncio
    async def test_unconfigured_app_secret_refuses_503(self):
        """Don't quietly accept webhooks when we can't authenticate them."""
        adapter = _make_adapter(app_secret="")
        adapter._dispatch_payload = AsyncMock()
        body = b'{"object":"whatsapp_business_account"}'
        request = _post_request(body, {"X-Hub-Signature-256": "sha256=deadbeef"})

        response = await adapter._handle_webhook(request)

        assert response.status == 503
        adapter._dispatch_payload.assert_not_called()

    @pytest.mark.asyncio
    async def test_signature_uses_constant_time_compare(self):
        """Smoke-test: equivalent signatures with case differences both pass."""
        adapter = _make_adapter(app_secret="key")
        adapter._dispatch_payload = AsyncMock()
        body = b'{"object":"whatsapp_business_account","entry":[]}'
        proper = _sign("key", body)
        # Capitalize hex — hmac.compare_digest is case-sensitive but our
        # implementation lowercases both sides so case differences in the
        # incoming header don't accidentally fail valid signatures.
        upper = proper.upper().replace("SHA256=", "sha256=")
        request = _post_request(body, {"X-Hub-Signature-256": upper})

        response = await adapter._handle_webhook(request)
        assert response.status == 200

    @pytest.mark.asyncio
    async def test_oversize_body_rejected_before_signature(self):
        """3MB cap per Meta — refuse without computing HMAC over giant junk."""
        adapter = _make_adapter(app_secret="key")
        adapter._dispatch_payload = AsyncMock()
        body = b"x" * (4 * 1024 * 1024)
        request = _post_request(body, {"X-Hub-Signature-256": "sha256=ignored"})

        response = await adapter._handle_webhook(request)
        assert response.status == 413
        adapter._dispatch_payload.assert_not_called()

    @pytest.mark.asyncio
    async def test_unreadable_body_rejected(self):
        adapter = _make_adapter(app_secret="key")
        request = MagicMock()
        request.read = AsyncMock(side_effect=RuntimeError("read failed"))
        request.headers = {}

        response = await adapter._handle_webhook(request)
        assert response.status == 400


class TestWebhookReplay:
    """wamid dedup — Meta retries failed deliveries up to 7 days."""

    @pytest.mark.asyncio
    async def test_duplicate_wamid_not_redispatched(self):
        adapter = _make_adapter(app_secret="key")
        adapter.handle_message = AsyncMock()
        body = json.dumps(_SAMPLE_INBOUND_TEXT_PAYLOAD).encode("utf-8")
        sig = _sign("key", body)

        # First delivery
        await adapter._handle_webhook(_post_request(body, {"X-Hub-Signature-256": sig}))
        # Second delivery (same payload, valid signature, same wamid)
        await adapter._handle_webhook(_post_request(body, {"X-Hub-Signature-256": sig}))

        # handle_message fires once, even though the webhook fired twice
        assert adapter.handle_message.call_count == 1
        assert adapter._duplicate_count == 1
        assert adapter._accepted_count == 1

    def test_dedup_cache_evicts_oldest(self):
        from gateway.platforms.whatsapp_cloud import WAMID_DEDUP_CACHE_SIZE
        adapter = _make_adapter()
        # Fill the cache plus 5 extra
        for i in range(WAMID_DEDUP_CACHE_SIZE + 5):
            assert adapter._dedup_wamid(f"wamid_{i}") is True
        assert len(adapter._seen_wamids) == WAMID_DEDUP_CACHE_SIZE
        # The first 5 should have been evicted
        assert "wamid_0" not in adapter._seen_wamids
        assert "wamid_4" not in adapter._seen_wamids
        assert "wamid_5" in adapter._seen_wamids
        assert f"wamid_{WAMID_DEDUP_CACHE_SIZE + 4}" in adapter._seen_wamids

    def test_dedup_no_wamid_lets_through(self):
        """Defensive — Meta should always populate ``id``, but we don't
        want to silently drop messages if it's missing."""
        adapter = _make_adapter()
        assert adapter._dedup_wamid("") is True
        assert adapter._dedup_wamid("") is True  # both pass


class TestWebhookDispatch:
    """End-to-end dispatch from a verified payload to handle_message."""

    @pytest.mark.asyncio
    async def test_text_message_dispatched_with_event_shape(self):
        adapter = _make_adapter(app_secret="key")
        captured = []

        async def _capture(event):
            captured.append(event)

        adapter.handle_message = _capture
        body = json.dumps(_SAMPLE_INBOUND_TEXT_PAYLOAD).encode("utf-8")
        sig = _sign("key", body)
        request = _post_request(body, {"X-Hub-Signature-256": sig})

        response = await adapter._handle_webhook(request)

        assert response.status == 200
        assert len(captured) == 1
        event = captured[0]
        assert event.text == "Hi!"
        assert event.message_id == (
            "wamid.HBgLMTM1NTc4MjU2OTgVAGHAYWYET688aASGNTI1QzZFQjhEMDk2QQA="
        )
        assert event.source.platform == Platform.WHATSAPP_CLOUD
        assert event.source.chat_id == "13557825698"
        assert event.source.user_name == "Jessica Laverdetman"
        assert event.source.chat_type == "dm"

    @pytest.mark.asyncio
    async def test_dispatch_filters_via_mixin_gating(self):
        adapter = _make_adapter(app_secret="key")
        adapter._dm_policy = "disabled"  # block all DMs
        adapter.handle_message = AsyncMock()
        body = json.dumps(_SAMPLE_INBOUND_TEXT_PAYLOAD).encode("utf-8")
        sig = _sign("key", body)

        response = await adapter._handle_webhook(
            _post_request(body, {"X-Hub-Signature-256": sig})
        )

        assert response.status == 200
        adapter.handle_message.assert_not_called()
        # Gated messages don't increment the accepted counter
        assert adapter._accepted_count == 0

    @pytest.mark.asyncio
    async def test_dispatch_handler_exception_does_not_crash(self):
        """If the agent dispatch raises, we still return 200 to Meta so
        retries don't multiply the bug into a 7-day storm."""
        adapter = _make_adapter(app_secret="key")
        adapter.handle_message = AsyncMock(side_effect=RuntimeError("boom"))
        body = json.dumps(_SAMPLE_INBOUND_TEXT_PAYLOAD).encode("utf-8")
        sig = _sign("key", body)

        response = await adapter._handle_webhook(
            _post_request(body, {"X-Hub-Signature-256": sig})
        )
        assert response.status == 200

    @pytest.mark.asyncio
    async def test_dispatch_ignores_non_message_field(self):
        """``field: 'statuses'`` etc. should not produce MessageEvents."""
        adapter = _make_adapter(app_secret="key")
        adapter.handle_message = AsyncMock()
        payload = {
            "object": "whatsapp_business_account",
            "entry": [
                {
                    "id": "x",
                    "changes": [
                        {
                            "field": "account_alerts",
                            "value": {"some": "alert"},
                        }
                    ],
                }
            ],
        }
        body = json.dumps(payload).encode("utf-8")
        sig = _sign("key", body)

        response = await adapter._handle_webhook(
            _post_request(body, {"X-Hub-Signature-256": sig})
        )
        assert response.status == 200
        adapter.handle_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_dispatch_ignores_non_waba_object(self):
        adapter = _make_adapter(app_secret="key")
        adapter.handle_message = AsyncMock()
        payload = {"object": "page", "entry": []}
        body = json.dumps(payload).encode("utf-8")
        sig = _sign("key", body)

        response = await adapter._handle_webhook(
            _post_request(body, {"X-Hub-Signature-256": sig})
        )
        assert response.status == 200
        adapter.handle_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_dispatch_handles_button_reply(self):
        adapter = _make_adapter(app_secret="key")
        captured = []

        async def _capture(event):
            captured.append(event)

        adapter.handle_message = _capture
        payload = {
            "object": "whatsapp_business_account",
            "entry": [
                {
                    "id": "x",
                    "changes": [
                        {
                            "field": "messages",
                            "value": {
                                "messaging_product": "whatsapp",
                                "metadata": {"phone_number_id": "1"},
                                "contacts": [
                                    {"profile": {"name": "U"}, "wa_id": "1555"}
                                ],
                                "messages": [
                                    {
                                        "from": "1555",
                                        "id": "wamid.button1",
                                        "timestamp": "0",
                                        "type": "interactive",
                                        "interactive": {
                                            "type": "button_reply",
                                            "button_reply": {
                                                "id": "yes",
                                                "title": "Yes please",
                                            },
                                        },
                                    }
                                ],
                            },
                        }
                    ],
                }
            ],
        }
        body = json.dumps(payload).encode("utf-8")
        sig = _sign("key", body)

        response = await adapter._handle_webhook(
            _post_request(body, {"X-Hub-Signature-256": sig})
        )
        assert response.status == 200
        assert len(captured) == 1
        assert captured[0].text == "Yes please"

    @pytest.mark.asyncio
    async def test_dispatch_propagates_reply_to(self):
        """``context.id`` on inbound = user replied to one of our messages."""
        adapter = _make_adapter(app_secret="key")
        captured = []

        async def _capture(event):
            captured.append(event)

        adapter.handle_message = _capture

        payload_with_ctx = json.loads(
            json.dumps(_SAMPLE_INBOUND_TEXT_PAYLOAD)
        )  # deep copy
        msg = payload_with_ctx["entry"][0]["changes"][0]["value"]["messages"][0]
        msg["context"] = {"id": "wamid.our_outbound", "from": "15551797781"}
        body = json.dumps(payload_with_ctx).encode("utf-8")
        sig = _sign("key", body)

        await adapter._handle_webhook(
            _post_request(body, {"X-Hub-Signature-256": sig})
        )
        assert len(captured) == 1
        assert captured[0].reply_to_message_id == "wamid.our_outbound"

    @pytest.mark.asyncio
    async def test_invalid_json_after_signature_returns_400(self):
        """Pathological case: signature passes but body isn't JSON."""
        adapter = _make_adapter(app_secret="key")
        body = b"not-json"
        sig = _sign("key", body)
        response = await adapter._handle_webhook(
            _post_request(body, {"X-Hub-Signature-256": sig})
        )
        assert response.status == 400


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------

class TestHealth:
    @pytest.mark.asyncio
    async def test_health_reports_config_visibility(self):
        adapter = _make_adapter(
            phone_number_id="555",
            verify_token="secret",
            app_secret="signing-key",
        )
        request = MagicMock()

        response = await adapter._handle_health(request)

        # web.json_response stores the dict on .text as JSON
        body = json.loads(response.text)
        assert body["status"] == "ok"
        assert body["platform"] == "whatsapp_cloud"
        assert body["phone_number_id"] == "555"
        assert body["verify_token_configured"] is True
        assert body["app_secret_configured"] is True
        assert body["accepted"] == 0
        assert body["duplicates"] == 0
        assert body["rejected_signature"] == 0
        # ffmpeg_present is True/False depending on the test host;
        # just verify the key is exposed.
        assert "ffmpeg_present" in body
        assert isinstance(body["ffmpeg_present"], bool)

    @pytest.mark.asyncio
    async def test_health_flags_missing_secrets(self):
        adapter = _make_adapter(verify_token="", app_secret="")
        request = MagicMock()

        response = await adapter._handle_health(request)
        body = json.loads(response.text)
        assert body["verify_token_configured"] is False
        assert body["app_secret_configured"] is False


# ---------------------------------------------------------------------------
# Mixin contract — gating still works on the cloud adapter
# ---------------------------------------------------------------------------

class TestMixinInherited:
    """Sanity-check: the Cloud adapter inherits the same gating behavior
    as the Baileys adapter via WhatsAppBehaviorMixin.
    """

    def test_format_message_converts_markdown(self):
        adapter = _make_adapter()
        assert adapter.format_message("**bold**") == "*bold*"
        assert adapter.format_message("# Title") == "*Title*"

    def test_should_process_message_dm_open(self):
        adapter = _make_adapter()
        adapter._dm_policy = "open"
        assert adapter._should_process_message({
            "chatId": "15551234567@c.us",
            "senderId": "15551234567@c.us",
            "isGroup": False,
            "body": "hi",
        }) is True

    def test_should_process_message_dm_disabled(self):
        adapter = _make_adapter()
        adapter._dm_policy = "disabled"
        assert adapter._should_process_message({
            "chatId": "15551234567@c.us",
            "senderId": "15551234567@c.us",
            "isGroup": False,
            "body": "hi",
        }) is False

    def test_broadcast_chats_filtered(self):
        adapter = _make_adapter()
        assert adapter._should_process_message({
            "chatId": "status@broadcast",
            "isGroup": False,
            "body": "x",
        }) is False


# ---------------------------------------------------------------------------
# Outbound media — link mode + upload mode (Phase 4)
# ---------------------------------------------------------------------------

import os as _os
import tempfile as _tempfile
from unittest.mock import patch as _patch


def _mock_upload_response(media_id: str = "media_abc123"):
    """Graph /media POST response shape."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json = MagicMock(return_value={"id": media_id})
    resp.text = json.dumps({"id": media_id})
    return resp


def _mock_message_response(wamid: str = "wamid.outbound1"):
    """Graph /messages POST response shape."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json = MagicMock(return_value={"messages": [{"id": wamid}]})
    resp.text = json.dumps({"messages": [{"id": wamid}]})
    return resp


def _tmpfile(suffix: str = ".jpg", content: bytes = b"\xff\xd8\xff\xe0") -> str:
    """Write a small temp file and return its path. Caller cleans up."""
    fd, path = _tempfile.mkstemp(suffix=suffix)
    with _os.fdopen(fd, "wb") as fh:
        fh.write(content)
    return path


class TestSendImage:
    """send_image — public URL takes the link path; local file uploads first."""

    @pytest.mark.asyncio
    async def test_send_image_link_mode_skips_upload(self):
        adapter = _make_adapter()
        adapter._http_client = MagicMock()
        adapter._http_client.post = AsyncMock(return_value=_mock_message_response())

        result = await adapter.send_image("15551234567", "https://cdn.example.com/cat.jpg")

        assert result.success is True
        # Exactly one POST — straight to /messages, no /media upload
        assert adapter._http_client.post.call_count == 1
        url = adapter._http_client.post.call_args.args[0]
        assert url.endswith("/messages")
        payload = adapter._http_client.post.call_args.kwargs["json"]
        assert payload["type"] == "image"
        assert payload["image"] == {"link": "https://cdn.example.com/cat.jpg"}

    @pytest.mark.asyncio
    async def test_send_image_local_path_uploads_then_sends(self):
        adapter = _make_adapter()
        adapter._http_client = MagicMock()
        adapter._http_client.post = AsyncMock(side_effect=[
            _mock_upload_response("media_uploaded_id"),
            _mock_message_response(),
        ])
        path = _tmpfile(".jpg")
        try:
            result = await adapter.send_image_file("15551234567", path)
            assert result.success is True
            assert adapter._http_client.post.call_count == 2

            upload_url = adapter._http_client.post.call_args_list[0].args[0]
            send_url = adapter._http_client.post.call_args_list[1].args[0]
            assert upload_url.endswith("/media")
            assert send_url.endswith("/messages")

            send_payload = adapter._http_client.post.call_args_list[1].kwargs["json"]
            assert send_payload["image"] == {"id": "media_uploaded_id"}
        finally:
            _os.unlink(path)

    @pytest.mark.asyncio
    async def test_send_image_caption_attached(self):
        adapter = _make_adapter()
        adapter._http_client = MagicMock()
        adapter._http_client.post = AsyncMock(return_value=_mock_message_response())

        await adapter.send_image(
            "15551234567", "https://cdn.example.com/cat.jpg", caption="cute cat"
        )
        payload = adapter._http_client.post.call_args.kwargs["json"]
        assert payload["image"]["caption"] == "cute cat"

    @pytest.mark.asyncio
    async def test_send_image_oversize_rejected_locally(self):
        """Don't round-trip to Graph just to be told the file's too big."""
        adapter = _make_adapter()
        adapter._http_client = MagicMock()
        adapter._http_client.post = AsyncMock()
        # 6MB > 5MB image cap
        path = _tmpfile(".jpg", content=b"x" * (6 * 1024 * 1024))
        try:
            result = await adapter.send_image_file("15551234567", path)
            assert result.success is False
            assert "5242880" in result.error or "cap is" in result.error
            # Never even POSTed
            adapter._http_client.post.assert_not_called()
        finally:
            _os.unlink(path)

    @pytest.mark.asyncio
    async def test_send_image_missing_local_file_returns_failure(self):
        adapter = _make_adapter()
        adapter._http_client = MagicMock()
        adapter._http_client.post = AsyncMock()

        result = await adapter.send_image_file(
            "15551234567", "/nonexistent/path/foo.jpg"
        )
        assert result.success is False
        assert "File not found" in result.error
        adapter._http_client.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_send_image_upload_failure_returns_failure(self):
        adapter = _make_adapter()
        # First call (upload) fails with a Graph error
        upload_fail = MagicMock()
        upload_fail.status_code = 400
        upload_fail.json = MagicMock(return_value={
            "error": {"code": 100, "message": "Bad media"}
        })
        upload_fail.text = '{"error":{"code":100,"message":"Bad media"}}'
        adapter._http_client = MagicMock()
        adapter._http_client.post = AsyncMock(return_value=upload_fail)

        path = _tmpfile(".jpg")
        try:
            result = await adapter.send_image_file("15551234567", path)
            assert result.success is False
            assert "graph error 100" in result.error
            # Only the upload call — never reached /messages
            assert adapter._http_client.post.call_count == 1
        finally:
            _os.unlink(path)


class TestSendVideo:
    @pytest.mark.asyncio
    async def test_send_video_link_mode(self):
        adapter = _make_adapter()
        adapter._http_client = MagicMock()
        adapter._http_client.post = AsyncMock(return_value=_mock_message_response())

        await adapter.send_video("15551234567", "https://cdn.example.com/v.mp4", caption="clip")
        payload = adapter._http_client.post.call_args.kwargs["json"]
        assert payload["type"] == "video"
        assert payload["video"]["link"] == "https://cdn.example.com/v.mp4"
        assert payload["video"]["caption"] == "clip"


class TestSendMethodsAcceptBaseClassKwargs:
    """Regression: every send_* method must absorb ``metadata=`` (and any
    other future kwargs) without raising TypeError.

    base.BasePlatformAdapter.send_multiple_images and friends pass
    ``metadata=...`` to send_image; if a subclass forgets ``**kwargs``,
    the agent crashes mid-send_multiple_images instead of just sending
    the image. This test guards against that for every Cloud send_*
    surface.
    """

    @pytest.mark.asyncio
    async def test_send_image_accepts_metadata(self):
        adapter = _make_adapter()
        adapter._http_client = MagicMock()
        adapter._http_client.post = AsyncMock(return_value=_mock_message_response())
        # Should not raise TypeError.
        result = await adapter.send_image(
            "15551234567", "https://cdn.example.com/x.jpg",
            metadata={"trace_id": "abc"},
        )
        assert result.success is True

    @pytest.mark.asyncio
    async def test_send_image_file_accepts_metadata(self):
        adapter = _make_adapter()
        adapter._http_client = MagicMock()
        adapter._http_client.post = AsyncMock(side_effect=[
            _mock_upload_response(),
            _mock_message_response(),
        ])
        path = _tmpfile(".jpg")
        try:
            result = await adapter.send_image_file(
                "15551234567", path, metadata={"x": 1},
            )
            assert result.success is True
        finally:
            _os.unlink(path)

    @pytest.mark.asyncio
    async def test_send_video_accepts_metadata(self):
        adapter = _make_adapter()
        adapter._http_client = MagicMock()
        adapter._http_client.post = AsyncMock(return_value=_mock_message_response())
        result = await adapter.send_video(
            "15551234567", "https://cdn.example.com/v.mp4",
            metadata={"x": 1},
        )
        assert result.success is True

    @pytest.mark.asyncio
    async def test_send_voice_accepts_metadata(self):
        adapter = _make_adapter()
        adapter._http_client = MagicMock()
        adapter._http_client.post = AsyncMock(return_value=_mock_message_response())
        result = await adapter.send_voice(
            "15551234567", "https://cdn.example.com/a.ogg",
            metadata={"x": 1},
        )
        assert result.success is True

    @pytest.mark.asyncio
    async def test_send_document_accepts_metadata(self):
        adapter = _make_adapter()
        adapter._http_client = MagicMock()
        adapter._http_client.post = AsyncMock(side_effect=[
            _mock_upload_response(),
            _mock_message_response(),
        ])
        path = _tmpfile(".pdf", content=b"%PDF")
        try:
            result = await adapter.send_document(
                "15551234567", path, metadata={"x": 1},
            )
            assert result.success is True
        finally:
            _os.unlink(path)


class TestSendDocument:
    @pytest.mark.asyncio
    async def test_send_document_filename_attached(self):
        adapter = _make_adapter()
        adapter._http_client = MagicMock()
        adapter._http_client.post = AsyncMock(side_effect=[
            _mock_upload_response("doc_id"),
            _mock_message_response(),
        ])
        path = _tmpfile(".pdf", content=b"%PDF-1.4 ...")
        try:
            await adapter.send_document(
                "15551234567", path, caption="Q3 report",
                file_name="report.pdf",
            )
            send_payload = adapter._http_client.post.call_args_list[1].kwargs["json"]
            assert send_payload["type"] == "document"
            assert send_payload["document"]["id"] == "doc_id"
            assert send_payload["document"]["caption"] == "Q3 report"
            assert send_payload["document"]["filename"] == "report.pdf"
        finally:
            _os.unlink(path)


class TestSendVoice:
    """Local non-Opus audio converts to Opus; fallback sends the original."""

    @pytest.mark.parametrize("suffix", [".ogg", ".opus"])
    @pytest.mark.asyncio
    async def test_send_voice_native_opus_uploads_without_conversion(self, suffix):
        adapter = _make_adapter()
        adapter._http_client = MagicMock()
        adapter._http_client.post = AsyncMock(side_effect=[
            _mock_upload_response("voice_id"),
            _mock_message_response(),
        ])
        adapter._convert_to_opus = AsyncMock()

        path = _tmpfile(suffix, content=b"OggS")
        try:
            result = await adapter.send_voice("15551234567", path)
            assert result.success is True
            adapter._convert_to_opus.assert_not_awaited()

            upload_files = adapter._http_client.post.call_args_list[0].kwargs["files"]
            assert upload_files["file"][2] == "audio/ogg; codecs=opus"
            assert upload_files["type"] == (None, "audio/ogg; codecs=opus")

            send_payload = adapter._http_client.post.call_args_list[1].kwargs["json"]
            assert send_payload["type"] == "audio"
            assert send_payload["audio"]["id"] == "voice_id"
        finally:
            _os.unlink(path)

    @pytest.mark.asyncio
    async def test_send_voice_no_ffmpeg_falls_back_to_mp3(self):
        adapter = _make_adapter()
        adapter._http_client = MagicMock()
        adapter._http_client.post = AsyncMock(side_effect=[
            _mock_upload_response("audio_id"),
            _mock_message_response(),
        ])
        # Simulate ffmpeg absent — adapter._convert_to_opus returns None
        adapter._convert_to_opus = AsyncMock(return_value=None)

        path = _tmpfile(".mp3", content=b"ID3\x04\x00\x00\x00\x00")
        try:
            result = await adapter.send_voice("15551234567", path)
            assert result.success is True
            # Adapter still uploaded + sent the MP3 as audio
            assert adapter._http_client.post.call_count == 2
            send_payload = adapter._http_client.post.call_args_list[1].kwargs["json"]
            assert send_payload["type"] == "audio"
            assert send_payload["audio"]["id"] == "audio_id"
        finally:
            _os.unlink(path)

    @pytest.mark.asyncio
    async def test_send_voice_ffmpeg_present_uses_opus(self):
        adapter = _make_adapter()
        adapter._http_client = MagicMock()
        adapter._http_client.post = AsyncMock(side_effect=[
            _mock_upload_response("voice_id"),
            _mock_message_response(),
        ])
        # Pretend ffmpeg conversion succeeded by returning a fake opus path.
        opus_path = _tmpfile(".ogg", content=b"OggS")
        adapter._convert_to_opus = AsyncMock(return_value=opus_path)

        mp3_path = _tmpfile(".mp3", content=b"ID3")
        try:
            result = await adapter.send_voice("15551234567", mp3_path)
            assert result.success is True
            # Conversion was invoked with the original MP3
            uploaded_path = adapter._convert_to_opus.call_args.args[0]
            assert uploaded_path == mp3_path
            send_payload = adapter._http_client.post.call_args_list[1].kwargs["json"]
            assert send_payload["type"] == "audio"
        finally:
            _os.unlink(mp3_path)
            if _os.path.exists(opus_path):
                _os.unlink(opus_path)

    @pytest.mark.asyncio
    async def test_send_voice_wav_uses_opus_conversion(self):
        adapter = _make_adapter()
        adapter._http_client = MagicMock()
        adapter._http_client.post = AsyncMock(side_effect=[
            _mock_upload_response("voice_id"),
            _mock_message_response(),
        ])
        # Pretend ffmpeg conversion succeeded by returning a fake Opus path.
        opus_path = _tmpfile(".ogg", content=b"OggS")
        adapter._convert_to_opus = AsyncMock(return_value=opus_path)

        wav_path = _tmpfile(".wav", content=b"RIFF\x24\x00\x00\x00WAVE")
        try:
            result = await adapter.send_voice("15551234567", wav_path)
            assert result.success is True
            adapter._convert_to_opus.assert_awaited_once_with(wav_path)
            upload_files = adapter._http_client.post.call_args_list[0].kwargs["files"]
            assert upload_files["file"][0] == _os.path.basename(opus_path)
            assert upload_files["file"][2] == "audio/ogg; codecs=opus"
            send_payload = adapter._http_client.post.call_args_list[1].kwargs["json"]
            assert send_payload["type"] == "audio"
        finally:
            _os.unlink(wav_path)
            if _os.path.exists(opus_path):
                _os.unlink(opus_path)

    @pytest.mark.asyncio
    async def test_send_voice_existing_ogg_uploads_as_opus_without_conversion(self):
        adapter = _make_adapter()
        adapter._http_client = MagicMock()
        adapter._http_client.post = AsyncMock(side_effect=[
            _mock_upload_response("voice_id"),
            _mock_message_response(),
        ])
        adapter._convert_to_opus = AsyncMock()

        ogg_path = _tmpfile(".ogg", content=b"OggS")
        try:
            result = await adapter.send_voice("15551234567", ogg_path)
            assert result.success is True
            adapter._convert_to_opus.assert_not_awaited()
            upload_files = adapter._http_client.post.call_args_list[0].kwargs["files"]
            assert upload_files["file"][0] == _os.path.basename(ogg_path)
            assert upload_files["file"][2] == "audio/ogg; codecs=opus"
        finally:
            _os.unlink(ogg_path)

    @pytest.mark.asyncio
    async def test_warn_once_no_ffmpeg_actually_only_warns_once(self):
        adapter = _make_adapter()
        adapter._warned_no_ffmpeg = False
        adapter._warn_once_no_ffmpeg()
        assert adapter._warned_no_ffmpeg is True
        # Second call: no-op (we just verify no exception + flag stays True)
        adapter._warn_once_no_ffmpeg()
        assert adapter._warned_no_ffmpeg is True

    @pytest.mark.asyncio
    async def test_convert_to_opus_forces_whatsapp_voice_note_shape(self):
        from pathlib import Path

        from gateway.platforms import whatsapp_cloud as wac

        class _Proc:
            returncode = 0

            async def communicate(self):
                return b"", b""

        adapter = _make_adapter()
        source_path = _tmpfile(".wav", content=b"RIFF....WAVE")
        captured_args = None

        async def fake_create_subprocess_exec(*args, **_kwargs):
            nonlocal captured_args
            captured_args = args
            Path(str(args[-1])).write_bytes(b"OggS")
            return _Proc()

        try:
            with _patch.object(wac, "_FFMPEG_PATH", "ffmpeg"), _patch(
                "gateway.platforms.whatsapp_cloud.asyncio.create_subprocess_exec",
                new=fake_create_subprocess_exec,
            ):
                opus_path = await adapter._convert_to_opus(source_path)

            assert opus_path is not None
            args = captured_args
            assert args is not None
            assert args[0] == "ffmpeg"
            assert args[3] == source_path
            assert args[args.index("-ac") + 1] == "1"
            assert args[args.index("-ar") + 1] == "48000"
            assert args[args.index("-c:a") + 1] == "libopus"
            assert args[args.index("-b:a") + 1] == "32k"
            assert args[args.index("-vbr") + 1] == "on"
            assert args[args.index("-application") + 1] == "voip"
            assert args[-1] == opus_path
        finally:
            _os.unlink(source_path)
            if "opus_path" in locals() and opus_path and _os.path.exists(opus_path):
                _os.unlink(opus_path)


# ---------------------------------------------------------------------------
# Inbound media — Graph two-step download (Phase 4)
# ---------------------------------------------------------------------------

class TestDownloadMedia:
    """Two-step Graph media download: meta -> temp URL -> bytes."""

    @pytest.mark.asyncio
    async def test_two_step_download_writes_cache_file(self, tmp_path):
        from gateway.platforms import whatsapp_cloud as wac

        adapter = _make_adapter()
        adapter._http_client = MagicMock()

        # Step 1 — metadata returns temp URL + mime
        meta_resp = MagicMock(status_code=200)
        meta_resp.json = MagicMock(return_value={
            "url": "https://lookaside.fbsbx.com/whatsapp/m/...",
            "mime_type": "image/jpeg",
            "sha256": "abc",
            "file_size": 12345,
            "id": "media_xyz",
            "messaging_product": "whatsapp",
        })
        # Step 2 — bytes
        blob_resp = MagicMock(status_code=200, content=b"\xff\xd8\xff\xe0jpegdata")

        adapter._http_client.get = AsyncMock(side_effect=[meta_resp, blob_resp])

        with _patch.object(wac, "_INBOUND_MEDIA_CACHE", tmp_path):
            local_path, mime = await adapter._download_media_to_cache("media_xyz")

        assert mime == "image/jpeg"
        assert local_path is not None
        assert _os.path.exists(local_path)
        assert _os.path.basename(local_path).startswith("media_xyz")
        assert _os.path.basename(local_path).endswith(".jpg")
        with open(local_path, "rb") as fh:
            assert fh.read() == b"\xff\xd8\xff\xe0jpegdata"

    @pytest.mark.asyncio
    async def test_metadata_failure_returns_none(self):
        adapter = _make_adapter()
        adapter._http_client = MagicMock()
        meta_fail = MagicMock(status_code=404)
        meta_fail.json = MagicMock(return_value={"error": {"code": 100}})
        adapter._http_client.get = AsyncMock(return_value=meta_fail)

        local_path, mime = await adapter._download_media_to_cache("missing")
        assert local_path is None and mime is None

    @pytest.mark.asyncio
    async def test_bytes_failure_returns_none(self, tmp_path):
        from gateway.platforms import whatsapp_cloud as wac

        adapter = _make_adapter()
        adapter._http_client = MagicMock()
        meta_resp = MagicMock(status_code=200)
        meta_resp.json = MagicMock(return_value={
            "url": "https://lookaside.fbsbx.com/...",
            "mime_type": "image/jpeg",
        })
        blob_fail = MagicMock(status_code=403, content=b"")
        adapter._http_client.get = AsyncMock(side_effect=[meta_resp, blob_fail])

        with _patch.object(wac, "_INBOUND_MEDIA_CACHE", tmp_path):
            local_path, mime = await adapter._download_media_to_cache("x")
        assert local_path is None

    @pytest.mark.asyncio
    async def test_metadata_includes_auth_header(self):
        adapter = _make_adapter(access_token="bearer-tok")
        adapter._http_client = MagicMock()
        adapter._http_client.get = AsyncMock(return_value=MagicMock(status_code=500))
        await adapter._download_media_to_cache("x")
        headers = adapter._http_client.get.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer bearer-tok"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("mime,expected_ext", [
        # Regression for the ".oga vs .ogg" voice-note bug — Python's
        # mimetypes module returns the RFC-correct .oga which downstream
        # STT pipelines reject.
        ("audio/ogg", ".ogg"),
        ("audio/ogg; codecs=opus", ".ogg"),
        ("audio/x-opus+ogg", ".ogg"),
        ("audio/opus", ".ogg"),
        # iOS voice memos arrive as audio/mp4 — must become .m4a, not .mp4.
        ("audio/mp4", ".m4a"),
        ("audio/x-m4a", ".m4a"),
        # JPEG should never land as .jpe (legacy IANA).
        ("image/jpeg", ".jpg"),
    ])
    async def test_extension_overrides_for_real_world_mimes(self, tmp_path, mime, expected_ext):
        from gateway.platforms import whatsapp_cloud as wac

        adapter = _make_adapter()
        adapter._http_client = MagicMock()
        meta_resp = MagicMock(status_code=200)
        meta_resp.json = MagicMock(return_value={
            "url": "https://lookaside.fbsbx.com/test",
            "mime_type": mime,
        })
        blob_resp = MagicMock(status_code=200, content=b"x")
        adapter._http_client.get = AsyncMock(side_effect=[meta_resp, blob_resp])

        with _patch.object(wac, "_INBOUND_MEDIA_CACHE", tmp_path):
            local_path, _ = await adapter._download_media_to_cache("media_x")

        assert local_path is not None
        assert local_path.endswith(expected_ext), (
            f"mime {mime!r} should map to {expected_ext} but got {local_path}"
        )


class TestInboundMediaDispatch:
    """End-to-end: webhook with image_id -> adapter downloads -> MessageEvent.media_urls populated."""

    @pytest.mark.asyncio
    async def test_inbound_image_populates_media_urls(self, tmp_path):
        from gateway.platforms import whatsapp_cloud as wac

        adapter = _make_adapter(app_secret="key")
        captured: list = []

        async def _capture(event):
            captured.append(event)

        adapter.handle_message = _capture

        # Mock the two-step Graph download
        meta_resp = MagicMock(status_code=200)
        meta_resp.json = MagicMock(return_value={
            "url": "https://lookaside.fbsbx.com/whatsapp/m/abc",
            "mime_type": "image/jpeg",
        })
        blob_resp = MagicMock(status_code=200, content=b"\xff\xd8\xff\xe0fake_jpeg")
        adapter._http_client = MagicMock()
        adapter._http_client.get = AsyncMock(side_effect=[meta_resp, blob_resp])

        # Build an inbound image webhook payload
        payload = {
            "object": "whatsapp_business_account",
            "entry": [{
                "id": "x",
                "changes": [{
                    "field": "messages",
                    "value": {
                        "messaging_product": "whatsapp",
                        "metadata": {"phone_number_id": "1"},
                        "contacts": [{"profile": {"name": "U"}, "wa_id": "1555"}],
                        "messages": [{
                            "from": "1555",
                            "id": "wamid.img1",
                            "timestamp": "0",
                            "type": "image",
                            "image": {
                                "id": "media_image_abc",
                                "mime_type": "image/jpeg",
                                "sha256": "...",
                                "caption": "look at this",
                            },
                        }],
                    },
                }],
            }],
        }
        body = json.dumps(payload).encode("utf-8")
        sig = _sign("key", body)

        with _patch.object(wac, "_INBOUND_MEDIA_CACHE", tmp_path):
            response = await adapter._handle_webhook(
                _post_request(body, {"X-Hub-Signature-256": sig})
            )

        assert response.status == 200
        assert len(captured) == 1
        event = captured[0]
        # Caption became the body
        assert event.text == "look at this"
        # Cached file path populated
        assert len(event.media_urls) == 1
        assert _os.path.exists(event.media_urls[0])
        assert event.media_types[0] == "image/jpeg"
        from gateway.platforms.base import MessageType
        assert event.message_type == MessageType.PHOTO

    @pytest.mark.asyncio
    async def test_inbound_text_document_injected_into_body(self, tmp_path):
        """A .txt document should have its content prepended to the body."""
        from gateway.platforms import whatsapp_cloud as wac

        adapter = _make_adapter(app_secret="key")
        captured: list = []

        async def _capture(event):
            captured.append(event)

        adapter.handle_message = _capture

        text_content = b"hello\nthis is the file\n"
        meta_resp = MagicMock(status_code=200)
        meta_resp.json = MagicMock(return_value={
            "url": "https://lookaside.fbsbx.com/whatsapp/m/doc",
            "mime_type": "text/plain",
        })
        blob_resp = MagicMock(status_code=200, content=text_content)
        adapter._http_client = MagicMock()
        adapter._http_client.get = AsyncMock(side_effect=[meta_resp, blob_resp])

        payload = {
            "object": "whatsapp_business_account",
            "entry": [{
                "id": "x",
                "changes": [{
                    "field": "messages",
                    "value": {
                        "messaging_product": "whatsapp",
                        "metadata": {"phone_number_id": "1"},
                        "contacts": [{"profile": {"name": "U"}, "wa_id": "1555"}],
                        "messages": [{
                            "from": "1555",
                            "id": "wamid.doc1",
                            "timestamp": "0",
                            "type": "document",
                            "document": {
                                "id": "media_doc_abc",
                                "mime_type": "text/plain",
                                "filename": "notes.txt",
                            },
                        }],
                    },
                }],
            }],
        }
        body = json.dumps(payload).encode("utf-8")
        sig = _sign("key", body)

        with _patch.object(wac, "_INBOUND_MEDIA_CACHE", tmp_path):
            await adapter._handle_webhook(
                _post_request(body, {"X-Hub-Signature-256": sig})
            )

        assert len(captured) == 1
        event = captured[0]
        assert "hello\nthis is the file" in event.text
        assert "[Content of" in event.text
        # File still available in media_urls for the agent's other tools
        assert len(event.media_urls) == 1

    @pytest.mark.asyncio
    async def test_inbound_image_download_failure_still_dispatches(self, tmp_path):
        """If the binary fetch fails we still want the agent to see the
        message metadata + caption — better than silently dropping."""
        from gateway.platforms import whatsapp_cloud as wac

        adapter = _make_adapter(app_secret="key")
        captured: list = []

        async def _capture(event):
            captured.append(event)

        adapter.handle_message = _capture
        adapter._http_client = MagicMock()
        # Metadata fetch fails
        adapter._http_client.get = AsyncMock(return_value=MagicMock(status_code=500))

        payload = {
            "object": "whatsapp_business_account",
            "entry": [{
                "id": "x",
                "changes": [{
                    "field": "messages",
                    "value": {
                        "messaging_product": "whatsapp",
                        "metadata": {"phone_number_id": "1"},
                        "contacts": [{"profile": {"name": "U"}, "wa_id": "1555"}],
                        "messages": [{
                            "from": "1555",
                            "id": "wamid.bad_img",
                            "timestamp": "0",
                            "type": "image",
                            "image": {"id": "borked", "mime_type": "image/jpeg"},
                        }],
                    },
                }],
            }],
        }
        body = json.dumps(payload).encode("utf-8")
        sig = _sign("key", body)

        with _patch.object(wac, "_INBOUND_MEDIA_CACHE", tmp_path):
            response = await adapter._handle_webhook(
                _post_request(body, {"X-Hub-Signature-256": sig})
            )

        assert response.status == 200
        assert len(captured) == 1
        # Agent gets the event, just with empty media_urls
        assert captured[0].media_urls == []


# ---------------------------------------------------------------------------
# Group-shaped message guard
# ---------------------------------------------------------------------------

class TestGroupMessageGuard:
    """Cloud API group support is deferred to v2 (Meta capability-tier
    gated, different payload shape than DMs). If Meta delivers a
    group-shaped message — identifiable by a populated ``chat`` field
    on the message object — the adapter should refuse cleanly rather
    than silently treating the sender's wa_id as the chat_id (which
    would route the bot's reply back to the sender as a DM, not the
    group)."""

    @pytest.mark.asyncio
    async def test_group_shaped_message_dropped_with_warning(self, caplog):
        adapter = _make_adapter()
        adapter.handle_message = AsyncMock()
        raw = {
            "from": "15551234567",
            "id": "wamid.group1",
            "timestamp": "0",
            "type": "text",
            "text": {"body": "hi from a group"},
            "chat": "120363012345678901@g.us",  # presence of `chat` = group
        }
        with caplog.at_level("WARNING"):
            event = await adapter._build_message_event_from_cloud(
                raw, {"15551234567": "Alice"}, {}
            )
        assert event is None
        # Warning surfaced so the operator knows group messages are being dropped
        assert any(
            "group-shaped" in rec.message
            for rec in caplog.records
        )
        # Defensive: handler not invoked
        adapter.handle_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_normal_dm_still_dispatches(self):
        """Sanity: the guard is keyed on `chat`, not just `from`. Normal
        DMs (which only have `from`, no `chat`) must still dispatch."""
        adapter = _make_adapter()
        raw = {
            "from": "15551234567",
            "id": "wamid.dm1",
            "timestamp": "0",
            "type": "text",
            "text": {"body": "hi from a DM"},
            # NO `chat` field — this is a DM
        }
        event = await adapter._build_message_event_from_cloud(
            raw, {"15551234567": "Alice"}, {}
        )
        assert event is not None
        assert event.text == "hi from a DM"
        assert event.source.chat_id == "15551234567"


# =========================================================================
# Phase 9 — Interactive button messages (clarify / approval / slash-confirm)
# =========================================================================
#
# These tests cover the four hooks the gateway uses for richer UX on
# platforms that support interactive buttons:
#   - send_clarify         (mid-conversation multi-choice question)
#   - send_exec_approval   (dangerous-command Y/N gate)
#   - send_slash_confirm   (3-button slash-command preview)
#   - _dispatch_interactive_reply (inbound side: route button taps to
#                                  the right resolver)
# Telegram and Discord have the same hooks; we mirror their callback-id
# format (cl:, appr:, sc:) so the gateway's existing degrade-to-text
# fallback works transparently.


class TestSendClarifyButtons:
    """``send_clarify`` outbound — picks button vs list mode by choice count."""

    @pytest.mark.asyncio
    async def test_three_choices_uses_button_mode(self):
        """1–3 choices → interactive.type=button (inline pills)."""
        adapter = _make_adapter()
        adapter._http_client = MagicMock()
        adapter._http_client.post = AsyncMock(
            return_value=_mock_httpx_response(200, {"messages": [{"id": "wamid.q1"}]})
        )

        result = await adapter.send_clarify(
            chat_id="15551234567",
            question="Pick one",
            choices=["Alpha", "Bravo", "Charlie"],
            clarify_id="abc123",
            session_key="sess-1",
        )

        assert result.success
        payload = adapter._http_client.post.call_args.kwargs["json"]
        assert payload["type"] == "interactive"
        assert payload["interactive"]["type"] == "button"
        buttons = payload["interactive"]["action"]["buttons"]
        assert len(buttons) == 3
        assert [b["reply"]["title"] for b in buttons] == ["1", "2", "3"]
        assert buttons[0]["reply"]["id"] == "cl:abc123:0"
        assert buttons[2]["reply"]["id"] == "cl:abc123:2"
        body_text = payload["interactive"]["body"]["text"]
        assert "Alpha" in body_text and "Bravo" in body_text and "Charlie" in body_text
        assert adapter._clarify_state["abc123"] == "sess-1"

    @pytest.mark.asyncio
    async def test_four_choices_promoted_to_list_mode(self):
        """4+ choices → interactive.type=list (sheet with rows)."""
        adapter = _make_adapter()
        adapter._http_client = MagicMock()
        adapter._http_client.post = AsyncMock(
            return_value=_mock_httpx_response(200, {"messages": [{"id": "wamid.q2"}]})
        )

        result = await adapter.send_clarify(
            chat_id="15551234567",
            question="Pick one",
            choices=["A", "B", "C", "D"],
            clarify_id="q2",
            session_key="sess-2",
        )

        assert result.success
        payload = adapter._http_client.post.call_args.kwargs["json"]
        assert payload["interactive"]["type"] == "list"
        rows = payload["interactive"]["action"]["sections"][0]["rows"]
        assert len(rows) == 5  # 4 choices + 1 "Other"
        assert rows[0]["id"] == "cl:q2:0"
        assert rows[3]["id"] == "cl:q2:3"
        assert rows[4]["id"] == "cl:q2:other"
        assert "Other" in rows[4]["title"]

    @pytest.mark.asyncio
    async def test_open_ended_falls_back_to_plain_text(self):
        """No choices → plain text send, no interactive payload."""
        adapter = _make_adapter()
        adapter._http_client = MagicMock()
        adapter._http_client.post = AsyncMock(
            return_value=_mock_httpx_response(200, {"messages": [{"id": "wamid.q3"}]})
        )

        result = await adapter.send_clarify(
            chat_id="15551234567",
            question="What's your name?",
            choices=None,
            clarify_id="q3",
            session_key="sess-3",
        )

        assert result.success
        payload = adapter._http_client.post.call_args.kwargs["json"]
        assert payload["type"] == "text"
        assert "What's your name?" in payload["text"]["body"]
        # Open-ended state is NOT stored on the adapter — the gateway's
        # text-intercept handles open-ended resolution (mirrors Telegram).
        assert "q3" not in adapter._clarify_state

    @pytest.mark.asyncio
    async def test_send_failure_does_not_register_state(self):
        """If Meta rejects the send, don't leave dangling state behind."""
        adapter = _make_adapter()
        adapter._http_client = MagicMock()
        adapter._http_client.post = AsyncMock(
            return_value=_mock_httpx_response(
                400, {"error": {"code": 100, "message": "bad payload"}}
            )
        )

        result = await adapter.send_clarify(
            chat_id="15551234567",
            question="hi",
            choices=["yes", "no"],
            clarify_id="dead",
            session_key="sess-x",
        )

        assert not result.success
        assert "dead" not in adapter._clarify_state


class TestSendExecApprovalButtons:
    """``send_exec_approval`` outbound — 2-button Approve/Deny gate."""

    @pytest.mark.asyncio
    async def test_approval_renders_two_buttons(self):
        adapter = _make_adapter()
        adapter._http_client = MagicMock()
        adapter._http_client.post = AsyncMock(
            return_value=_mock_httpx_response(200, {"messages": [{"id": "wamid.a1"}]})
        )

        result = await adapter.send_exec_approval(
            chat_id="15551234567",
            command="rm -rf /tmp/foo",
            session_key="sess-app-1",
            description="cleanup script",
        )

        assert result.success
        payload = adapter._http_client.post.call_args.kwargs["json"]
        assert payload["interactive"]["type"] == "button"
        buttons = payload["interactive"]["action"]["buttons"]
        assert len(buttons) == 2
        assert "Approve" in buttons[0]["reply"]["title"]
        assert "Deny" in buttons[1]["reply"]["title"]
        approve_id = buttons[0]["reply"]["id"]
        deny_id = buttons[1]["reply"]["id"]
        assert approve_id.startswith("appr:") and approve_id.endswith(":approve")
        assert deny_id.startswith("appr:") and deny_id.endswith(":deny")
        approval_id = approve_id.split(":")[1]
        assert deny_id.split(":")[1] == approval_id
        body = payload["interactive"]["body"]["text"]
        assert "rm -rf /tmp/foo" in body
        assert "cleanup script" in body
        assert adapter._exec_approval_state[approval_id] == "sess-app-1"

    @pytest.mark.asyncio
    async def test_long_command_is_truncated(self):
        """Body must stay under WhatsApp's 1024-char interactive cap."""
        adapter = _make_adapter()
        adapter._http_client = MagicMock()
        adapter._http_client.post = AsyncMock(
            return_value=_mock_httpx_response(200, {"messages": [{"id": "x"}]})
        )

        huge = "echo " + ("x" * 5000)
        result = await adapter.send_exec_approval(
            chat_id="15551234567",
            command=huge,
            session_key="sess-x",
        )
        assert result.success
        payload = adapter._http_client.post.call_args.kwargs["json"]
        assert len(payload["interactive"]["body"]["text"]) <= 1024


class TestSendSlashConfirmButtons:
    """``send_slash_confirm`` outbound — 3-button Once/Always/Cancel."""

    @pytest.mark.asyncio
    async def test_three_buttons_with_ids(self):
        adapter = _make_adapter()
        adapter._http_client = MagicMock()
        adapter._http_client.post = AsyncMock(
            return_value=_mock_httpx_response(200, {"messages": [{"id": "wamid.s1"}]})
        )

        result = await adapter.send_slash_confirm(
            chat_id="15551234567",
            title="Reload MCP",
            message="This will restart all MCP servers.",
            session_key="sess-sc-1",
            confirm_id="cf-9",
        )

        assert result.success
        payload = adapter._http_client.post.call_args.kwargs["json"]
        assert payload["interactive"]["type"] == "button"
        buttons = payload["interactive"]["action"]["buttons"]
        ids = [b["reply"]["id"] for b in buttons]
        assert ids == ["sc:once:cf-9", "sc:always:cf-9", "sc:cancel:cf-9"]
        assert adapter._slash_confirm_state["cf-9"] == "sess-sc-1"


class TestDispatchInteractiveReplyClarify:
    """Inbound side: button-tap → clarify resolver."""

    @pytest.mark.asyncio
    async def test_clarify_tap_resolves_and_pops_state(self, monkeypatch):
        adapter = _make_adapter()
        adapter._clarify_state["q1"] = "sess-1"

        captured = {}

        def fake_resolve(clarify_id, response):
            captured["clarify_id"] = clarify_id
            captured["response"] = response
            return True

        monkeypatch.setattr(
            "tools.clarify_gateway.resolve_gateway_clarify", fake_resolve
        )

        raw = {
            "from": "15551234567",
            "type": "interactive",
            "interactive": {
                "type": "button_reply",
                "button_reply": {"id": "cl:q1:2", "title": "3"},
            },
        }
        handled = await adapter._dispatch_interactive_reply(raw, {})

        assert handled is True
        assert captured == {"clarify_id": "q1", "response": "3"}
        assert "q1" not in adapter._clarify_state

    @pytest.mark.asyncio
    async def test_clarify_other_button_keeps_state_and_prompts(self, monkeypatch):
        """Picking 'Other' should NOT resolve — it should flip the
        clarify entry into text-capture mode (via mark_awaiting_text)
        AND keep the state mapping so the gateway's text-intercept can
        resolve the next typed message. Without the flip,
        ``get_pending_for_session`` wouldn't return the entry and the
        user's next message would collide with the still-blocked agent
        thread, producing an "Interrupting current task" loop."""
        adapter = _make_adapter()
        adapter._clarify_state["q1"] = "sess-1"
        adapter._http_client = MagicMock()
        adapter._http_client.post = AsyncMock(
            return_value=_mock_httpx_response(200, {"messages": [{"id": "x"}]})
        )

        flipped_ids = []
        monkeypatch.setattr(
            "tools.clarify_gateway.mark_awaiting_text",
            lambda cid: flipped_ids.append(cid) or True,
        )

        raw = {
            "from": "15551234567",
            "type": "interactive",
            "interactive": {
                "type": "list_reply",
                "list_reply": {"id": "cl:q1:other", "title": "Other"},
            },
        }
        handled = await adapter._dispatch_interactive_reply(raw, {})

        assert handled is True
        # State stays so text-intercept can resolve the next message
        assert adapter._clarify_state.get("q1") == "sess-1"
        # mark_awaiting_text was called with the right clarify_id
        assert flipped_ids == ["q1"]
        # Follow-up "type your answer" prompt was sent
        adapter._http_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_clarify_other_with_no_entry_falls_back(self, monkeypatch):
        """If the underlying clarify entry vanished (timed out, /new,
        gateway restart) between the prompt and the tap,
        ``mark_awaiting_text`` returns False — drop the stale adapter
        state and fall through to text dispatch."""
        adapter = _make_adapter()
        adapter._clarify_state["q1"] = "sess-1"
        monkeypatch.setattr(
            "tools.clarify_gateway.mark_awaiting_text",
            lambda cid: False,  # entry missing on the gateway side
        )

        raw = {
            "from": "15551234567",
            "type": "interactive",
            "interactive": {
                "type": "list_reply",
                "list_reply": {"id": "cl:q1:other", "title": "Other"},
            },
        }
        handled = await adapter._dispatch_interactive_reply(raw, {})
        assert handled is False
        # Adapter state was already popped before the gateway check; we
        # leave it popped on the missing-entry path so a real follow-up
        # text doesn't try to resolve a ghost.
        assert "q1" not in adapter._clarify_state

    @pytest.mark.asyncio
    async def test_stale_clarify_tap_falls_back_to_text(self):
        """No state entry → return False so caller treats it as text."""
        adapter = _make_adapter()  # _clarify_state is empty

        raw = {
            "from": "15551234567",
            "type": "interactive",
            "interactive": {
                "type": "button_reply",
                "button_reply": {"id": "cl:ghost:0", "title": "1"},
            },
        }
        handled = await adapter._dispatch_interactive_reply(raw, {})
        assert handled is False

    @pytest.mark.asyncio
    async def test_clarify_resolver_no_waiter_falls_back(self, monkeypatch):
        """Resolver returns False (e.g. agent timed out) → caller falls
        back to text dispatch."""
        adapter = _make_adapter()
        adapter._clarify_state["q1"] = "sess-1"
        monkeypatch.setattr(
            "tools.clarify_gateway.resolve_gateway_clarify",
            lambda cid, r: False,
        )

        raw = {
            "from": "15551234567",
            "type": "interactive",
            "interactive": {
                "type": "button_reply",
                "button_reply": {"id": "cl:q1:0", "title": "1"},
            },
        }
        handled = await adapter._dispatch_interactive_reply(raw, {})
        assert handled is False


class TestDispatchInteractiveReplyApproval:
    """Inbound side: approval-tap → resolve_gateway_approval."""

    @pytest.mark.asyncio
    async def test_approve_tap_calls_resolver_and_confirms(self, monkeypatch):
        adapter = _make_adapter()
        adapter._exec_approval_state["app1"] = "sess-app-1"
        adapter._http_client = MagicMock()
        adapter._http_client.post = AsyncMock(
            return_value=_mock_httpx_response(200, {"messages": [{"id": "x"}]})
        )

        calls = []
        monkeypatch.setattr(
            "tools.approval.resolve_gateway_approval",
            lambda session_key, choice: calls.append((session_key, choice)) or 1,
        )

        raw = {
            "from": "15551234567",
            "type": "interactive",
            "interactive": {
                "type": "button_reply",
                "button_reply": {"id": "appr:app1:approve", "title": "Approve"},
            },
        }
        handled = await adapter._dispatch_interactive_reply(raw, {})

        assert handled is True
        assert calls == [("sess-app-1", "approve")]
        assert "app1" not in adapter._exec_approval_state
        confirm_payload = adapter._http_client.post.call_args.kwargs["json"]
        assert confirm_payload["type"] == "text"
        assert "Approved" in confirm_payload["text"]["body"]

    @pytest.mark.asyncio
    async def test_deny_tap_passes_deny_choice(self, monkeypatch):
        adapter = _make_adapter()
        adapter._exec_approval_state["app2"] = "sess-app-2"
        adapter._http_client = MagicMock()
        adapter._http_client.post = AsyncMock(
            return_value=_mock_httpx_response(200, {"messages": [{"id": "x"}]})
        )

        choices_seen = []
        monkeypatch.setattr(
            "tools.approval.resolve_gateway_approval",
            lambda session_key, choice: choices_seen.append(choice) or 1,
        )

        raw = {
            "from": "15551234567",
            "type": "interactive",
            "interactive": {
                "type": "button_reply",
                "button_reply": {"id": "appr:app2:deny", "title": "Deny"},
            },
        }
        await adapter._dispatch_interactive_reply(raw, {})

        assert choices_seen == ["deny"]
        confirm_payload = adapter._http_client.post.call_args.kwargs["json"]
        assert "Denied" in confirm_payload["text"]["body"]


class TestDispatchInteractiveReplySlashConfirm:
    """Inbound side: slash-confirm-tap → tools.slash_confirm.resolve."""

    @pytest.mark.asyncio
    async def test_once_tap_calls_resolver(self, monkeypatch):
        adapter = _make_adapter()
        adapter._slash_confirm_state["cf-9"] = "sess-sc-1"
        adapter._http_client = MagicMock()
        adapter._http_client.post = AsyncMock(
            return_value=_mock_httpx_response(200, {"messages": [{"id": "x"}]})
        )

        captured = {}

        async def fake_resolve(session_key, confirm_id, choice):
            captured.update(
                session_key=session_key, confirm_id=confirm_id, choice=choice
            )
            return "MCP reloaded."

        import tools.slash_confirm as _sc
        monkeypatch.setattr(_sc, "resolve", fake_resolve)

        raw = {
            "from": "15551234567",
            "type": "interactive",
            "interactive": {
                "type": "button_reply",
                "button_reply": {"id": "sc:once:cf-9", "title": "Approve Once"},
            },
        }
        handled = await adapter._dispatch_interactive_reply(raw, {})

        assert handled is True
        assert captured == {
            "session_key": "sess-sc-1",
            "confirm_id": "cf-9",
            "choice": "once",
        }
        reply_payload = adapter._http_client.post.call_args.kwargs["json"]
        assert "MCP reloaded" in reply_payload["text"]["body"]


class TestInteractiveReplyEndToEnd:
    """Integration: `_build_message_event_from_cloud` must SHORT-CIRCUIT
    on a recognized interactive reply and NOT also produce a fresh
    conversation turn (which would double-fire the agent)."""

    @pytest.mark.asyncio
    async def test_recognized_tap_returns_none_no_text_dispatch(self, monkeypatch):
        adapter = _make_adapter()
        adapter._clarify_state["q1"] = "sess-1"
        monkeypatch.setattr(
            "tools.clarify_gateway.resolve_gateway_clarify",
            lambda cid, r: True,
        )

        raw = {
            "from": "15551234567",
            "id": "wamid.tap1",
            "type": "interactive",
            "interactive": {
                "type": "button_reply",
                "button_reply": {"id": "cl:q1:0", "title": "1"},
            },
        }
        event = await adapter._build_message_event_from_cloud(
            raw, {"15551234567": "Alice"}, {}
        )
        # The tap resolved the clarify; no MessageEvent dispatched so the
        # agent thread that was waiting on clarify is unblocked exactly
        # once, not once + a new turn for the tap.
        assert event is None

    @pytest.mark.asyncio
    async def test_unrecognized_tap_falls_through_to_text(self):
        """Button taps from unrelated plugin adapters (or stale taps)
        should be treated as plain text input — this preserves the
        graceful-degrade path the gateway already relies on."""
        adapter = _make_adapter()
        raw = {
            "from": "15551234567",
            "id": "wamid.tap2",
            "type": "interactive",
            "interactive": {
                "type": "button_reply",
                "button_reply": {"id": "unknown:foo", "title": "Hello"},
            },
        }
        event = await adapter._build_message_event_from_cloud(
            raw, {"15551234567": "Alice"}, {}
        )
        # Falls through to text dispatch — the button title becomes the
        # user message body so the agent at least sees what they tapped.
        assert event is not None
        assert event.text == "Hello"


# =========================================================================
# Phase 10 — Typing indicator + mark-as-read
# =========================================================================
#
# Meta couples the read receipt and typing indicator into a single POST
# to the messages endpoint. We refresh _last_inbound_wamid_by_chat on
# every accepted inbound message so the gateway can call send_typing()
# without threading event.message_id through the base contract.


class TestInboundWamidCache:
    """Cache hygiene: refreshes on accepted inbound, skipped on filtered."""

    @pytest.mark.asyncio
    async def test_accepted_message_populates_cache(self):
        adapter = _make_adapter()
        raw = {
            "from": "15551234567",
            "id": "wamid.AAA",
            "type": "text",
            "text": {"body": "hi"},
        }
        event = await adapter._build_message_event_from_cloud(
            raw, {"15551234567": "Alice"}, {}
        )
        assert event is not None
        assert adapter._last_inbound_wamid_by_chat["15551234567"] == "wamid.AAA"

    @pytest.mark.asyncio
    async def test_subsequent_messages_overwrite_cache(self):
        """Cache holds the LATEST inbound, not the first — typing indicator
        must attach to the most recent message in the conversation."""
        adapter = _make_adapter()
        for wamid in ("wamid.first", "wamid.second", "wamid.third"):
            await adapter._build_message_event_from_cloud(
                {
                    "from": "15551234567",
                    "id": wamid,
                    "type": "text",
                    "text": {"body": "msg"},
                },
                {"15551234567": "Alice"},
                {},
            )
        assert adapter._last_inbound_wamid_by_chat["15551234567"] == "wamid.third"

    @pytest.mark.asyncio
    async def test_filtered_message_does_not_pollute_cache(self):
        """Group-shaped messages get dropped before the cache write —
        we don't want typing indicators triggered by inbound traffic the
        agent never sees."""
        adapter = _make_adapter()
        raw = {
            "from": "15551234567",
            "id": "wamid.BBB",
            "type": "text",
            "text": {"body": "hi from group"},
            "chat": "120363012345678901@g.us",  # group marker
        }
        event = await adapter._build_message_event_from_cloud(
            raw, {"15551234567": "Alice"}, {}
        )
        assert event is None  # group guard rejected it
        # Cache stays empty
        assert "15551234567" not in adapter._last_inbound_wamid_by_chat


class TestSendTyping:
    """``send_typing`` outbound — combined read receipt + indicator."""

    @pytest.mark.asyncio
    async def test_send_typing_posts_correct_payload(self):
        adapter = _make_adapter()
        adapter._last_inbound_wamid_by_chat["15551234567"] = "wamid.LATEST"
        adapter._http_client = MagicMock()
        adapter._http_client.post = AsyncMock(
            return_value=_mock_httpx_response(200, {"success": True})
        )

        await adapter.send_typing("15551234567")

        adapter._http_client.post.assert_called_once()
        payload = adapter._http_client.post.call_args.kwargs["json"]
        # Meta's combined endpoint shape
        assert payload["messaging_product"] == "whatsapp"
        assert payload["status"] == "read"
        assert payload["message_id"] == "wamid.LATEST"
        assert payload["typing_indicator"] == {"type": "text"}

    @pytest.mark.asyncio
    async def test_send_typing_uses_latest_cached_wamid(self):
        """If multiple messages have arrived, the indicator must attach
        to the LATEST one (mirrors Meta's documented behavior — the
        typing indicator only renders against the most recent message
        in the conversation)."""
        adapter = _make_adapter()
        adapter._last_inbound_wamid_by_chat["15551234567"] = "wamid.OLD"
        adapter._last_inbound_wamid_by_chat["15551234567"] = "wamid.NEW"
        adapter._http_client = MagicMock()
        adapter._http_client.post = AsyncMock(
            return_value=_mock_httpx_response(200, {"success": True})
        )

        await adapter.send_typing("15551234567")
        payload = adapter._http_client.post.call_args.kwargs["json"]
        assert payload["message_id"] == "wamid.NEW"

    @pytest.mark.asyncio
    async def test_send_typing_no_cached_wamid_is_noop(self):
        """No inbound message yet for this chat (or cache cleared on
        gateway restart) → skip silently. Don't fail, don't log noisily.
        The next inbound message will repopulate the cache."""
        adapter = _make_adapter()
        # _last_inbound_wamid_by_chat is empty
        adapter._http_client = MagicMock()
        adapter._http_client.post = AsyncMock(
            return_value=_mock_httpx_response(200, {"success": True})
        )

        await adapter.send_typing("15551234567")
        # No HTTP call at all
        adapter._http_client.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_send_typing_swallows_network_errors(self):
        """Any HTTP exception must NOT propagate — typing is best-effort
        UX polish and must never block the agent's main reply path.
        Verified by the absence of a raise."""
        adapter = _make_adapter()
        adapter._last_inbound_wamid_by_chat["15551234567"] = "wamid.X"
        adapter._http_client = MagicMock()
        adapter._http_client.post = AsyncMock(
            side_effect=RuntimeError("connection refused")
        )

        # Should NOT raise
        await adapter.send_typing("15551234567")

    @pytest.mark.asyncio
    async def test_send_typing_stale_message_logged_at_info(self, caplog):
        """Graph error 131009 = wamid > 30 days old. Common after a
        long-quiet conversation — log at INFO so it doesn't pollute
        WARNING-level monitoring dashboards."""
        adapter = _make_adapter()
        adapter._last_inbound_wamid_by_chat["15551234567"] = "wamid.OLD"
        adapter._http_client = MagicMock()
        adapter._http_client.post = AsyncMock(
            return_value=_mock_httpx_response(
                400, {"error": {"code": 131009, "message": "Parameter value is not valid"}}
            )
        )

        with caplog.at_level("INFO"):
            await adapter.send_typing("15551234567")

        assert any(
            "older than 30 days" in rec.message
            for rec in caplog.records
        )

    @pytest.mark.asyncio
    async def test_send_typing_no_http_client_is_noop(self):
        """If the adapter isn't connected yet, send_typing must be a
        silent no-op — matches the rest of the adapter's "best-effort
        when not running" pattern."""
        adapter = _make_adapter()
        adapter._http_client = None
        adapter._last_inbound_wamid_by_chat["15551234567"] = "wamid.X"
        # Should NOT raise
        await adapter.send_typing("15551234567")

    @pytest.mark.asyncio
    async def test_send_typing_includes_bearer_auth(self):
        """Same auth shape as the rest of the Graph API surface — bearer
        token in the Authorization header."""
        adapter = _make_adapter(access_token="my-test-token")
        adapter._last_inbound_wamid_by_chat["15551234567"] = "wamid.X"
        adapter._http_client = MagicMock()
        adapter._http_client.post = AsyncMock(
            return_value=_mock_httpx_response(200, {"success": True})
        )

        await adapter.send_typing("15551234567")
        headers = adapter._http_client.post.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer my-test-token"


# ---------------------------------------------------------------------------
# Allowlist normalization + env decoupling (salvage follow-up)
# ---------------------------------------------------------------------------

class TestAllowlistNormalization:
    def test_normalize_allow_ids_strips_jid_suffix_and_punctuation(self):
        from gateway.platforms.whatsapp_cloud import WhatsAppCloudAdapter

        ids = {"15551234567@s.whatsapp.net", "+1 (555) 765-4321", "15550000000"}
        normalized = WhatsAppCloudAdapter._normalize_allow_ids(ids)
        assert normalized == {"15551234567", "15557654321", "15550000000"}

    def test_dm_allowlist_matches_bare_wa_id_against_jid_entry(self):
        """A Baileys-style JID in the allowlist must match the Cloud API's
        bare wa_id sender — users share allowlists between both adapters."""
        from gateway.platforms.whatsapp_cloud import WhatsAppCloudAdapter

        adapter = _make_adapter()
        adapter._dm_policy = "allowlist"
        adapter._allow_from = WhatsAppCloudAdapter._normalize_allow_ids(
            {"15551234567@s.whatsapp.net"}
        )
        assert adapter._is_dm_allowed("15551234567") is True
        assert adapter._is_dm_allowed("19998887777") is False

    def test_cloud_env_overrides_take_precedence(self, monkeypatch):
        """WHATSAPP_CLOUD_DM_POLICY wins over the shared WHATSAPP_DM_POLICY
        so both adapters can run in parallel with independent policies."""
        from gateway.platforms.whatsapp_cloud import WhatsAppCloudAdapter

        monkeypatch.setenv("WHATSAPP_DM_POLICY", "allowlist")
        monkeypatch.setenv("WHATSAPP_CLOUD_DM_POLICY", "open")
        monkeypatch.setenv("WHATSAPP_CLOUD_ALLOW_FROM", "+1 555 123 4567")

        config = MagicMock()
        config.extra = {
            "phone_number_id": "123",
            "access_token": "tok",
        }
        adapter = WhatsAppCloudAdapter(config)
        assert adapter._dm_policy == "open"
        assert adapter._allow_from == {"15551234567"}


class TestBoundedInteractiveState:
    def test_bounded_put_evicts_oldest(self):
        from collections import OrderedDict

        from gateway.platforms.whatsapp_cloud import (
            INTERACTIVE_STATE_CACHE_SIZE,
            WhatsAppCloudAdapter,
        )

        cache: OrderedDict = OrderedDict()
        for i in range(INTERACTIVE_STATE_CACHE_SIZE + 10):
            WhatsAppCloudAdapter._bounded_put(cache, f"id-{i}", "sess")
        assert len(cache) == INTERACTIVE_STATE_CACHE_SIZE
        assert "id-0" not in cache
        assert f"id-{INTERACTIVE_STATE_CACHE_SIZE + 9}" in cache


class TestMediaIdValidation:
    @pytest.mark.asyncio
    async def test_traversal_media_id_refused(self):
        adapter = _make_adapter()
        adapter._http_client = MagicMock()  # would be used if not refused
        path, mime = await adapter._download_media_to_cache("../../etc/passwd")
        assert path is None and mime is None
        adapter._http_client.get.assert_not_called()
