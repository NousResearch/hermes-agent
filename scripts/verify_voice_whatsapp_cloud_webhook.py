#!/usr/bin/env python3
"""Verify WhatsApp Cloud webhook ingress locally without live Meta calls."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
from pathlib import Path
import sys
import tempfile
from typing import Any

from aiohttp import ClientSession

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gateway.config import PlatformConfig
from gateway.platforms import whatsapp_cloud as whatsapp_cloud_module
from gateway.platforms.whatsapp_cloud import WhatsAppCloudAdapter


PHONE_NUMBER_ID = "7794189252778687"
VERIFY_TOKEN = "local-verify-token-value"
APP_SECRET = "0123456789abcdef0123456789abcdef"
WEBHOOK_PATH = "/whatsapp/webhook"
VERIFY_CHALLENGE = "hermes-local-cloud-webhook-smoke"
VOICE_MEDIA_ID = "media_voice_note_abc"
VOICE_BYTES = b"OggS\x00\x02voice-note"


class VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def bound_port(adapter: WhatsAppCloudAdapter) -> int:
    runner = getattr(adapter, "_runner", None)
    sites = list(getattr(runner, "sites", []) or [])
    for site in sites:
        server = getattr(site, "_server", None)
        sockets = getattr(server, "sockets", None) or []
        for sock in sockets:
            return int(sock.getsockname()[1])
    raise VerificationError("could not determine local webhook port")


def status_payload() -> dict[str, Any]:
    return {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "id": "local-cloud-webhook-smoke",
                "changes": [
                    {
                        "field": "messages",
                        "value": {
                            "messaging_product": "whatsapp",
                            "metadata": {
                                "display_phone_number": "15555550100",
                                "phone_number_id": PHONE_NUMBER_ID,
                            },
                            "statuses": [
                                {
                                    "id": "wamid.local-cloud-webhook-smoke",
                                    "status": "delivered",
                                    "timestamp": "1760000000",
                                    "recipient_id": "15555550101",
                                }
                            ],
                        },
                    }
                ],
            }
        ],
    }


def voice_payload() -> dict[str, Any]:
    return {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "id": "local-cloud-webhook-smoke",
                "changes": [
                    {
                        "field": "messages",
                        "value": {
                            "messaging_product": "whatsapp",
                            "metadata": {
                                "display_phone_number": "15555550100",
                                "phone_number_id": PHONE_NUMBER_ID,
                            },
                            "contacts": [
                                {
                                    "profile": {"name": "Hermes Voice Smoke"},
                                    "wa_id": "15555550101",
                                }
                            ],
                            "messages": [
                                {
                                    "from": "15555550101",
                                    "id": "wamid.local-cloud-voice-note",
                                    "timestamp": "1760000001",
                                    "type": "audio",
                                    "audio": {
                                        "id": VOICE_MEDIA_ID,
                                        "mime_type": "audio/ogg; codecs=opus",
                                        "sha256": "synthetic",
                                        "voice": True,
                                    },
                                }
                            ],
                        },
                    }
                ],
            }
        ],
    }


def signature_header(body: bytes) -> str:
    digest = hmac.new(APP_SECRET.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


class FakeResponse:
    def __init__(
        self,
        *,
        status_code: int,
        body: dict[str, Any] | None = None,
        content: bytes = b"",
    ) -> None:
        self.status_code = status_code
        self._body = body or {}
        self.content = content

    def json(self) -> dict[str, Any]:
        return self._body


class FakeMediaClient:
    def __init__(self, *, media_id: str, content: bytes) -> None:
        self.media_id = media_id
        self.content = content
        self.get_urls: list[str] = []
        self.closed = False

    async def get(self, url: str, **_kwargs: Any) -> FakeResponse:
        self.get_urls.append(url)
        if url.endswith(f"/{self.media_id}"):
            return FakeResponse(
                status_code=200,
                body={
                    "url": "https://lookaside.fbsbx.com/whatsapp/m/local-voice",
                    "mime_type": "audio/ogg; codecs=opus",
                    "id": self.media_id,
                },
            )
        if url == "https://lookaside.fbsbx.com/whatsapp/m/local-voice":
            return FakeResponse(status_code=200, content=self.content)
        return FakeResponse(status_code=404, body={"error": "not found"})

    async def aclose(self) -> None:
        self.closed = True


async def signed_post(
    *,
    session: ClientSession,
    webhook_url: str,
    payload: dict[str, Any],
) -> int:
    body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    async with session.post(
        webhook_url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "X-Hub-Signature-256": signature_header(body),
        },
    ) as response:
        status = response.status
        await response.text()
    return status


async def verify() -> dict[str, Any]:
    config = PlatformConfig(
        enabled=True,
        extra={
            "phone_number_id": PHONE_NUMBER_ID,
            "access_token": "EAA" + ("x" * 120),
            "app_secret": APP_SECRET,
            "verify_token": VERIFY_TOKEN,
            "webhook_host": "127.0.0.1",
            "webhook_port": 0,
            "webhook_path": WEBHOOK_PATH,
        },
    )
    adapter = WhatsAppCloudAdapter(config)
    dispatched_messages: list[Any] = []

    async def capture_message(event: Any) -> None:
        dispatched_messages.append(event)

    adapter.handle_message = capture_message

    original_cache = whatsapp_cloud_module._INBOUND_MEDIA_CACHE
    connected = await adapter.connect()
    require(connected is True, "WhatsApp Cloud adapter did not connect")
    try:
        port = bound_port(adapter)
        base_url = f"http://127.0.0.1:{port}"
        webhook_url = base_url + WEBHOOK_PATH
        with tempfile.TemporaryDirectory(prefix="hermes-cloud-webhook-media.") as tmp:
            whatsapp_cloud_module._INBOUND_MEDIA_CACHE = Path(tmp)
            async with ClientSession() as session:
                async with session.get(base_url + "/health") as response:
                    health = await response.json()
                require(health.get("status") == "ok", f"health was not ok: {health}")
                require(
                    health.get("verify_token_configured") is True,
                    "health did not report verify token configured",
                )
                require(
                    health.get("app_secret_configured") is True,
                    "health did not report app secret configured",
                )

                async with session.get(
                    webhook_url,
                    params={
                        "hub.mode": "subscribe",
                        "hub.verify_token": VERIFY_TOKEN,
                        "hub.challenge": VERIFY_CHALLENGE,
                    },
                ) as response:
                    verify_body = await response.text()
                    verify_status = response.status
                require(verify_status == 200, f"verify handshake status {verify_status}")
                require(
                    verify_body == VERIFY_CHALLENGE,
                    "verify handshake did not echo challenge",
                )

                post_status = await signed_post(
                    session=session,
                    webhook_url=webhook_url,
                    payload=status_payload(),
                )
                require(post_status == 200, f"signed POST status {post_status}")

                async with session.get(base_url + "/health") as response:
                    final_health = await response.json()
                require(
                    final_health.get("rejected_signature") == 0,
                    "signed POST unexpectedly incremented rejected_signature",
                )
                require(
                    final_health.get("accepted") == 0,
                    "status-only POST unexpectedly dispatched a message",
                )
                require(
                    not dispatched_messages,
                    "status-only POST unexpectedly reached handle_message",
                )
                status_dispatched_messages = len(dispatched_messages)

                existing_http_client = adapter._http_client
                if existing_http_client is not None:
                    await existing_http_client.aclose()
                media_client = FakeMediaClient(
                    media_id=VOICE_MEDIA_ID,
                    content=VOICE_BYTES,
                )
                adapter._http_client = media_client
                voice_status = await signed_post(
                    session=session,
                    webhook_url=webhook_url,
                    payload=voice_payload(),
                )
                require(voice_status == 200, f"voice webhook status {voice_status}")
                require(len(dispatched_messages) == 1, "voice webhook did not dispatch")
                voice_dispatched_messages = len(dispatched_messages)
                voice_event = dispatched_messages[0]
                require(
                    getattr(voice_event, "media_types", []) == [
                        "audio/ogg; codecs=opus"
                    ],
                    "voice event media type mismatch",
                )
                media_urls = list(getattr(voice_event, "media_urls", []) or [])
                require(len(media_urls) == 1, "voice event missing cached media URL")
                media_path = Path(media_urls[0])
                require(media_path.suffix == ".ogg", "voice media did not cache as .ogg")
                require(media_path.read_bytes() == VOICE_BYTES, "voice media bytes changed")
                require(
                    any(url.endswith(f"/{VOICE_MEDIA_ID}") for url in media_client.get_urls),
                    "media metadata endpoint was not requested",
                )
        whatsapp_cloud_module._INBOUND_MEDIA_CACHE = original_cache

        return {
            "success": True,
            "checks": {
                "health": {
                    "status": health.get("status"),
                    "platform": health.get("platform"),
                    "verify_token_configured": (
                        health.get("verify_token_configured") is True
                    ),
                    "app_secret_configured": health.get("app_secret_configured")
                    is True,
                },
                "verify_handshake": {
                    "status": verify_status,
                    "challenge_echoed": True,
                },
                "signed_post": {
                    "status": post_status,
                    "payload": "status_delivery_receipt",
                    "signature_accepted": True,
                    "dispatched_messages": status_dispatched_messages,
                },
                "voice_note": {
                    "status": voice_status,
                    "media_type": voice_event.media_types[0],
                    "cached_extension": media_path.suffix,
                    "cached_bytes": len(VOICE_BYTES),
                    "dispatched_messages": voice_dispatched_messages,
                },
            },
        }
    finally:
        whatsapp_cloud_module._INBOUND_MEDIA_CACHE = original_cache
        await adapter.disconnect()


def main() -> int:
    try:
        result = asyncio.run(verify())
    except VerificationError as exc:
        print(json.dumps({"success": False, "error": str(exc)}, indent=2))
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
