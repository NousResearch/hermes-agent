"""Tests for the shared httpx.Limits helper that all long-lived platform
adapters use to tighten their keep-alive pool.

Context: #18451 — on macOS behind Cloudflare Warp, httpx's default
keepalive_expiry=5s let idle CLOSE_WAIT sockets accumulate across
multiple long-lived gateway adapters (QQ Bot, Feishu, WeCom, DingTalk,
Signal, BlueBubbles, WeCom-callback) until the process hit the default
256 fd limit.  These tests just verify the helper returns sensibly
tuned limits and respects env-var overrides; the actual fd-pressure
behaviour is only observable at runtime under load.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.platforms.base import SendResult


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    monkeypatch.delenv("HERMES_GATEWAY_HTTPX_KEEPALIVE_EXPIRY", raising=False)
    monkeypatch.delenv("HERMES_GATEWAY_HTTPX_MAX_KEEPALIVE", raising=False)


def test_env_override_rejects_garbage(monkeypatch):
    """Malformed env values fall back to defaults rather than raising."""
    monkeypatch.setenv("HERMES_GATEWAY_HTTPX_KEEPALIVE_EXPIRY", "not-a-number")
    monkeypatch.setenv("HERMES_GATEWAY_HTTPX_MAX_KEEPALIVE", "-3")
    from gateway.platforms._http_client_limits import platform_httpx_limits
    limits = platform_httpx_limits()
    # Non-positive / non-numeric → fell back to defaults (not the override values)
    assert limits.keepalive_expiry is not None and limits.keepalive_expiry > 0
    assert limits.max_keepalive_connections is not None
    assert limits.max_keepalive_connections > 0


@pytest.fixture
def whatsapp_adapter():
    from gateway.config import PlatformConfig
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

    adapter = WhatsAppAdapter(PlatformConfig(enabled=True, extra={"session_name": "test"}))
    adapter._running = True
    adapter._check_managed_bridge_exit = AsyncMock(return_value=False)
    adapter._http_session = MagicMock()
    return adapter


class _FakeBridgeResponse:
    """Async context manager mimicking aiohttp's response CM for _bridge_req: a plain
    status/text/json double used to drive send()/edit_message() through their real
    control flow without a live bridge process."""

    def __init__(self, status: int, text: str = "", json_body=None):
        self.status = status
        self._text = text
        self._json = json_body if json_body is not None else {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    async def text(self):
        return self._text

    async def json(self):
        return self._json


class TestWhatsappSendPathDegraded:
    """WhatsApp's send()/edit_message() previously returned the bridge's raw failure text
    (e.g. "Not connected to WhatsApp") or a bare aiohttp exception string on a transient
    bridge disconnect (WhatsApp's own socket flapping mid-reconnect — see #63277). That
    string didn't match BasePlatformAdapter._RETRYABLE_ERROR_PATTERNS, so _send_with_retry
    never retried it and the delivery ledger never redelivered it on reconnect
    (send_path_degraded is the one error string the ledger's runtime-redelivery sweep
    explicitly recognizes).

    Telegram avoids this by translating its OWN connection-health signal into the literal
    "send_path_degraded" string with retryable=True before a send is even attempted. This
    mirrors that translation for WhatsApp's bridge failures instead of inventing a bespoke
    WhatsApp-only retry path.
    """

    @pytest.mark.asyncio
    async def test_send_maps_bridge_503_to_send_path_degraded(self, whatsapp_adapter):
        whatsapp_adapter._bridge_req = MagicMock(
            return_value=_FakeBridgeResponse(503, text="Not connected to WhatsApp")
        )

        result = await whatsapp_adapter.send("15551234567", "hello")

        assert not result.success
        assert result.error == "send_path_degraded"
        assert result.retryable is True

    @pytest.mark.asyncio
    async def test_send_maps_transport_exception_to_send_path_degraded(self, whatsapp_adapter):
        def _raise(*a, **k):
            raise ConnectionError("Cannot connect to host 127.0.0.1:3000")

        whatsapp_adapter._bridge_req = _raise

        result = await whatsapp_adapter.send("15551234567", "hello")

        assert not result.success
        assert result.error == "send_path_degraded"
        assert result.retryable is True

    @pytest.mark.asyncio
    async def test_send_does_not_reclassify_a_real_rejection(self, whatsapp_adapter):
        """A non-connectivity rejection (bad chatId, WhatsApp itself refused the payload)
        must NOT be turned into send_path_degraded — retrying it would just be rejected
        again, and the ledger would spend redelivery attempts on an unfixable error."""
        whatsapp_adapter._bridge_req = MagicMock(
            return_value=_FakeBridgeResponse(400, text="chatId and message are required")
        )

        result = await whatsapp_adapter.send("15551234567", "hello")

        assert not result.success
        assert result.error == "chatId and message are required"
        assert result.retryable is False

    @pytest.mark.asyncio
    async def test_edit_message_maps_bridge_disconnect_to_send_path_degraded(self, whatsapp_adapter):
        whatsapp_adapter._bridge_req = MagicMock(
            return_value=_FakeBridgeResponse(503, text="Not connected to WhatsApp")
        )

        result = await whatsapp_adapter.edit_message("15551234567", "msg-1", "hello")

        assert not result.success
        assert result.error == "send_path_degraded"
        assert result.retryable is True

    @pytest.mark.asyncio
    async def test_send_path_degraded_triggers_generic_retry_and_recovers(self, whatsapp_adapter):
        """End-to-end through the REAL base-adapter retry loop (not just the classifier):
        a first send hits send_path_degraded, _send_with_retry recognizes it as retryable
        and retries; the second attempt succeeds. This is the exact mechanism Telegram
        already gets — proving WhatsApp now participates in it too."""
        attempts = {"n": 0}

        async def _send(chat_id, content, reply_to=None, metadata=None):
            attempts["n"] += 1
            if attempts["n"] == 1:
                return SendResult(success=False, error="send_path_degraded", retryable=True)
            return SendResult(success=True, message_id="abc123")

        whatsapp_adapter.send = _send

        result = await whatsapp_adapter._send_with_retry(
            chat_id="15551234567", content="hello", max_retries=2, base_delay=0.01
        )

        assert result.success
        assert result.message_id == "abc123"
        assert attempts["n"] == 2, "the base adapter's retry loop did not retry a send_path_degraded failure"
