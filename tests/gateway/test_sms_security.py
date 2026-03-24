import asyncio
import base64
import hashlib
import hmac
import os
from urllib.parse import parse_qs
from unittest.mock import AsyncMock, patch

import pytest

from gateway.config import Platform, PlatformConfig


class DummyRequest:
    def __init__(self, *, body: bytes, headers=None, remote="127.0.0.1"):
        self._body = body
        self.headers = headers or {}
        self.remote = remote

    async def read(self):
        return self._body


def _twilio_signature(url: str, body: bytes, token: str) -> str:
    form = parse_qs(body.decode("utf-8"), keep_blank_values=True)
    payload = url
    for key in sorted(form):
        for value in form[key]:
            payload += key + value
    digest = hmac.new(token.encode("utf-8"), payload.encode("utf-8"), hashlib.sha1).digest()
    return base64.b64encode(digest).decode("ascii")


@pytest.mark.asyncio
async def test_sms_adapter_rejects_missing_signature():
    from gateway.platforms.sms import SmsAdapter

    env = {
        "TWILIO_ACCOUNT_SID": "ACtest",
        "TWILIO_AUTH_TOKEN": "token_abc",
        "TWILIO_PHONE_NUMBER": "+15550001111",
    }
    with patch.dict(os.environ, env, clear=False):
        adapter = SmsAdapter(PlatformConfig(enabled=True, api_key="token_abc"))
        adapter.handle_message = AsyncMock()
        request = DummyRequest(body=b"From=%2B15551234567&Body=hello&MessageSid=SM123")

        response = await adapter._handle_webhook(request)

    assert response.status == 403
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_sms_adapter_accepts_valid_signature_and_dispatches_event():
    from gateway.platforms.sms import SmsAdapter

    token = "token_abc"
    public_url = "https://sms.example.test/webhooks/twilio"
    body = b"From=%2B15551234567&To=%2B15550001111&Body=hello&MessageSid=SM123"
    signature = _twilio_signature(public_url, body, token)

    env = {
        "TWILIO_ACCOUNT_SID": "ACtest",
        "TWILIO_AUTH_TOKEN": token,
        "TWILIO_PHONE_NUMBER": "+15550001111",
    }
    with patch.dict(os.environ, env, clear=False):
        adapter = SmsAdapter(PlatformConfig(enabled=True, api_key=token))
        adapter.handle_message = AsyncMock()
        request = DummyRequest(
            body=body,
            headers={
                "X-Twilio-Signature": signature,
                "X-Twilio-Original-Url": public_url,
            },
        )

        response = await adapter._handle_webhook(request)
        await asyncio.sleep(0)

    assert response.status == 200
    adapter.handle_message.assert_awaited_once()


def test_sms_adapter_defaults_to_loopback_binding():
    from gateway.platforms.sms import SmsAdapter

    env = {
        "TWILIO_ACCOUNT_SID": "ACtest",
        "TWILIO_AUTH_TOKEN": "token_abc",
        "TWILIO_PHONE_NUMBER": "+15550001111",
    }
    with patch.dict(os.environ, env, clear=False):
        adapter = SmsAdapter(PlatformConfig(enabled=True, api_key="token_abc"))

    assert adapter._bind_host == "127.0.0.1"


def test_sms_platform_default_toolset_maps_to_hermes_sms():
    from gateway.run import _default_platform_toolset_map

    mapping = _default_platform_toolset_map()
    assert mapping[Platform.SMS] == "hermes-sms"


def test_hermes_sms_toolset_is_least_privilege():
    from toolsets import get_toolset

    ts = get_toolset("hermes-sms")
    assert ts is not None
    tools = set(ts["tools"])

    assert "terminal" not in tools
    assert "process" not in tools
    assert "write_file" not in tools
    assert "patch" not in tools
    assert "browser_navigate" not in tools
    assert "execute_code" not in tools
    assert "delegate_task" not in tools
    assert "cronjob" not in tools
    assert "send_message" not in tools
    assert "ha_call_service" not in tools


# ── Invalid signature ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_sms_adapter_rejects_invalid_signature():
    """A request with a wrong signature must be rejected with 403."""
    from gateway.platforms.sms import SmsAdapter

    token = "real_token_abc"
    public_url = "https://sms.example.test/webhooks/twilio"
    body = b"From=%2B15551234567&Body=hello&MessageSid=SM456"

    env = {
        "TWILIO_ACCOUNT_SID": "ACtest",
        "TWILIO_AUTH_TOKEN": token,
        "TWILIO_PHONE_NUMBER": "+155****1111",
    }
    with patch.dict(os.environ, env, clear=False):
        adapter = SmsAdapter(PlatformConfig(enabled=True, api_key=token))
        adapter.handle_message = AsyncMock()
        request = DummyRequest(
            body=body,
            headers={
                "X-Twilio-Signature": "totally_bogus_signature",
                "X-Twilio-Original-Url": public_url,
            },
        )

        response = await adapter._handle_webhook(request)

    assert response.status == 403
    adapter.handle_message.assert_not_awaited()


# ── Spoofed sender blocked ────────────────────────────────────────

@pytest.mark.asyncio
async def test_sms_adapter_ignores_echo_from_own_number():
    """Messages from the adapter's own number (echo) must be silently dropped."""
    from gateway.platforms.sms import SmsAdapter

    token = "token_echo"
    own_number = "+155****1111"
    public_url = "https://sms.example.test/webhooks/twilio"
    # From = own number (URL-encoded +)
    body = b"From=%2B15550001111&To=%2B15559999999&Body=echo&MessageSid=SM789"
    signature = _twilio_signature(public_url, body, token)

    env = {
        "TWILIO_ACCOUNT_SID": "ACtest",
        "TWILIO_AUTH_TOKEN": token,
        "TWILIO_PHONE_NUMBER": own_number,
    }
    with patch.dict(os.environ, env, clear=False):
        adapter = SmsAdapter(PlatformConfig(enabled=True, api_key=token))
        adapter.handle_message = AsyncMock()
        request = DummyRequest(
            body=body,
            headers={
                "X-Twilio-Signature": signature,
                "X-Twilio-Original-Url": public_url,
            },
        )

        response = await adapter._handle_webhook(request)

    assert response.status == 200
    adapter.handle_message.assert_not_awaited()


# ── Replay suppression (duplicate MessageSid) ─────────────────────

@pytest.mark.asyncio
async def test_sms_adapter_ignores_replayed_webhook():
    """A second webhook with the same MessageSid must be silently dropped."""
    from gateway.platforms.sms import SmsAdapter

    token = "token_replay"
    public_url = "https://sms.example.test/webhooks/twilio"
    body = b"From=%2B15551234567&To=%2B15550001111&Body=hello&MessageSid=SMdupe1"
    signature = _twilio_signature(public_url, body, token)

    env = {
        "TWILIO_ACCOUNT_SID": "ACtest",
        "TWILIO_AUTH_TOKEN": token,
        "TWILIO_PHONE_NUMBER": "+155****1111",
    }
    with patch.dict(os.environ, env, clear=False):
        adapter = SmsAdapter(PlatformConfig(enabled=True, api_key=token))
        adapter.handle_message = AsyncMock()
        request_factory = lambda: DummyRequest(
            body=body,
            headers={
                "X-Twilio-Signature": signature,
                "X-Twilio-Original-Url": public_url,
            },
        )

        # First request should be accepted
        resp1 = await adapter._handle_webhook(request_factory())
        await asyncio.sleep(0)
        assert resp1.status == 200
        assert adapter.handle_message.await_count == 1

        # Second request with same MessageSid should be silently dropped
        resp2 = await adapter._handle_webhook(request_factory())
        assert resp2.status == 200
        assert adapter.handle_message.await_count == 1  # NOT incremented


# ── Rate limiting ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_sms_adapter_rate_limits_excessive_requests():
    """Exceeding the rate limit must return 429."""
    from gateway.platforms.sms import SmsAdapter

    token = "token_ratelimit"
    public_url = "https://sms.example.test/webhooks/twilio"

    env = {
        "TWILIO_ACCOUNT_SID": "ACtest",
        "TWILIO_AUTH_TOKEN": token,
        "TWILIO_PHONE_NUMBER": "+155****1111",
        "SMS_WEBHOOK_MAX_REQUESTS_PER_WINDOW": "3",
        "SMS_WEBHOOK_WINDOW_SECONDS": "60",
    }
    with patch.dict(os.environ, env, clear=False):
        adapter = SmsAdapter(PlatformConfig(enabled=True, api_key=token))
        adapter.handle_message = AsyncMock()

        for i in range(5):
            sid = f"SMrl{i:03d}"
            body = f"From=%2B15551234567&To=%2B15550001111&Body=msg{i}&MessageSid={sid}".encode()
            signature = _twilio_signature(public_url, body, token)
            request = DummyRequest(
                body=body,
                headers={
                    "X-Twilio-Signature": signature,
                    "X-Twilio-Original-Url": public_url,
                },
            )
            resp = await adapter._handle_webhook(request)
            await asyncio.sleep(0)

            if i < 3:
                assert resp.status == 200, f"Request {i} should succeed"
            else:
                assert resp.status == 429, f"Request {i} should be rate-limited"
