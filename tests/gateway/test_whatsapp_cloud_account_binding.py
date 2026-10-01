"""WhatsApp Cloud webhook account/phone binding regression tests.

All Meta identifiers and signatures are synthetic; no network is used.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.whatsapp_cloud import WhatsAppCloudAdapter


_SHARED_APP_SECRET = "a" * 32
_WABA_A = "100000000000001"
_PHONE_A = "200000000000001"
_WABA_B = "100000000000002"
_PHONE_B = "200000000000002"
_SENDER = "199900000000001"


class _RequestContent:
    def __init__(self, body: bytes):
        self._body = body

    async def readexactly(self, size: int) -> bytes:
        if len(self._body) < size:
            raise asyncio.IncompleteReadError(self._body, size)
        return self._body[:size]


def _request(body: bytes):
    request = MagicMock()
    request.content = _RequestContent(body)
    signature = hmac.new(_SHARED_APP_SECRET.encode(), body, hashlib.sha256).hexdigest()
    request.headers = {"X-Hub-Signature-256": f"sha256={signature}"}
    return request


def _config(waba_id: str, phone_number_id: str) -> PlatformConfig:
    return PlatformConfig(
        enabled=True,
        extra={
            "waba_id": waba_id,
            "phone_number_id": phone_number_id,
            "access_token": "EAA" + "f" * 100,
            "app_secret": _SHARED_APP_SECRET,
        },
    )


def _adapter(config: PlatformConfig) -> WhatsAppCloudAdapter:
    adapter = WhatsAppCloudAdapter(config)
    adapter.handle_message = AsyncMock()
    adapter._download_media_to_cache = AsyncMock(
        return_value=("/synthetic/inbound.jpg", "image/jpeg")
    )
    return adapter


def _payload(waba_id: str, phone_number_id: str, *, media: bool) -> dict:
    message = {
        "from": _SENDER,
        "id": f"wamid.synthetic.{waba_id}.{phone_number_id}",
        "timestamp": "0",
        "type": "image" if media else "text",
    }
    if media:
        message["image"] = {
            "id": "synthetic-media-id",
            "mime_type": "image/jpeg",
        }
    else:
        message["text"] = {"body": "synthetic message"}
    return {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "id": waba_id,
                "changes": [
                    {
                        "field": "messages",
                        "value": {
                            "messaging_product": "whatsapp",
                            "metadata": {
                                "display_phone_number": "+0 000 000 0000",
                                "phone_number_id": phone_number_id,
                            },
                            "contacts": [
                                {"profile": {"name": "Synthetic User"}, "wa_id": _SENDER}
                            ],
                            "messages": [message],
                        },
                    }
                ],
            }
        ],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("webhook_waba_id", "webhook_phone_number_id"),
    [
        (_WABA_B, _PHONE_A),
        (_WABA_A, _PHONE_B),
        (_WABA_B, _PHONE_B),
    ],
    ids=["foreign-waba", "foreign-phone", "foreign-waba-and-phone"],
)
async def test_signed_foreign_identity_is_dropped_before_media_or_dispatch(
    monkeypatch, webhook_waba_id, webhook_phone_number_id
):
    monkeypatch.setenv("WHATSAPP_CLOUD_ALLOW_ALL_USERS", "true")
    profile_a = _adapter(_config(_WABA_A, _PHONE_A))
    profile_b = _adapter(_config(_WABA_B, _PHONE_B))
    body = json.dumps(_payload(webhook_waba_id, webhook_phone_number_id, media=True)).encode()
    request = _request(body)

    assert profile_a._verify_signature(body, request.headers["X-Hub-Signature-256"])
    assert profile_b._verify_signature(body, request.headers["X-Hub-Signature-256"])

    response = await profile_a._handle_webhook(request)

    assert response.status == 200
    profile_a._download_media_to_cache.assert_not_awaited()
    profile_a.handle_message.assert_not_awaited()
    assert profile_a._accepted_count == 0


@pytest.mark.asyncio
async def test_identity_change_requires_explicit_adapter_reconfiguration(monkeypatch):
    monkeypatch.setenv("WHATSAPP_CLOUD_ALLOW_ALL_USERS", "true")
    original_config = _config(_WABA_A, _PHONE_A)
    original = _adapter(original_config)
    original_config.extra.update(waba_id=_WABA_B, phone_number_id=_PHONE_B)
    body = json.dumps(_payload(_WABA_B, _PHONE_B, media=False)).encode()

    response = await original._handle_webhook(_request(body))

    assert response.status == 200
    original.handle_message.assert_not_awaited()

    reconfigured = _adapter(_config(_WABA_B, _PHONE_B))
    response = await reconfigured._handle_webhook(_request(body))

    assert response.status == 200
    reconfigured.handle_message.assert_awaited_once()
    assert reconfigured.handle_message.await_args.args[0].text == "synthetic message"
