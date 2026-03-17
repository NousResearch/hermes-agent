import base64
import hmac
import os

import pytest


def _twilio_sig(auth_token: str, url: str, params: dict[str, str]) -> str:
    payload = url + "".join(f"{k}{params[k]}" for k in sorted(params.keys()))
    mac = hmac.new(auth_token.encode("utf-8"), payload.encode("utf-8"), digestmod="sha1")
    return base64.b64encode(mac.digest()).decode("ascii")


@pytest.mark.asyncio
async def test_sms_webhook_rejects_missing_signature(aiohttp_client, monkeypatch):
    from aiohttp import web
    from gateway.config import PlatformConfig
    from gateway.platforms.sms import SmsAdapter

    monkeypatch.setenv("TWILIO_ACCOUNT_SID", "AC123")
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "secret")
    monkeypatch.setenv("TWILIO_PHONE_NUMBER", "+15550001111")
    monkeypatch.setenv("SMS_VALIDATE_TWILIO_SIGNATURE", "true")
    monkeypatch.setenv("SMS_WEBHOOK_PUBLIC_URL", "https://example.com/webhooks/twilio")

    adapter = SmsAdapter(PlatformConfig(name="sms", token="", extra={}))

    app = web.Application()
    app.router.add_post("/webhooks/twilio", adapter._handle_webhook)
    client = await aiohttp_client(app)

    resp = await client.post(
        "/webhooks/twilio",
        data={"From": "+15551234567", "To": "+15550001111", "Body": "hi", "MessageSid": "SM1"},
    )
    assert resp.status == 403


@pytest.mark.asyncio
async def test_sms_webhook_accepts_valid_signature(aiohttp_client, monkeypatch):
    from aiohttp import web
    from gateway.config import PlatformConfig
    from gateway.platforms.sms import SmsAdapter

    monkeypatch.setenv("TWILIO_ACCOUNT_SID", "AC123")
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "secret")
    monkeypatch.setenv("TWILIO_PHONE_NUMBER", "+15550001111")
    monkeypatch.setenv("SMS_VALIDATE_TWILIO_SIGNATURE", "true")
    monkeypatch.setenv("SMS_WEBHOOK_PUBLIC_URL", "https://example.com/webhooks/twilio")

    adapter = SmsAdapter(PlatformConfig(name="sms", token="", extra={}))

    app = web.Application()
    app.router.add_post("/webhooks/twilio", adapter._handle_webhook)
    client = await aiohttp_client(app)

    params = {"Body": "hi", "From": "+15551234567", "MessageSid": "SM1", "To": "+15550001111"}
    url = "https://example.com/webhooks/twilio"
    sig = _twilio_sig("secret", url, params)

    resp = await client.post(
        "/webhooks/twilio",
        data=params,
        headers={"X-Twilio-Signature": sig},
    )
    assert resp.status in (200, 204)


@pytest.mark.asyncio
async def test_sms_webhook_allows_disabled_validation(aiohttp_client, monkeypatch):
    from aiohttp import web
    from gateway.config import PlatformConfig
    from gateway.platforms.sms import SmsAdapter

    monkeypatch.setenv("TWILIO_ACCOUNT_SID", "AC123")
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "secret")
    monkeypatch.setenv("TWILIO_PHONE_NUMBER", "+15550001111")
    monkeypatch.setenv("SMS_VALIDATE_TWILIO_SIGNATURE", "false")

    adapter = SmsAdapter(PlatformConfig(name="sms", token="", extra={}))

    app = web.Application()
    app.router.add_post("/webhooks/twilio", adapter._handle_webhook)
    client = await aiohttp_client(app)

    resp = await client.post(
        "/webhooks/twilio",
        data={"From": "+15551234567", "To": "+15550001111", "Body": "hi", "MessageSid": "SM1"},
    )
    assert resp.status in (200, 204)

