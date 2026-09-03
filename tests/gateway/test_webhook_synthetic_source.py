import asyncio
import base64
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import _reply_anchor_for_event, _thread_metadata_for_source
from gateway.platforms.webhook import WebhookAdapter


class _FakeRequest:
    def __init__(self, route_name: str, body: dict, headers: dict | None = None):
        self.match_info = {"route_name": route_name}
        self._body = json.dumps(body).encode("utf-8")
        self.headers = headers or {}
        self.content_length = len(self._body)

    async def read(self):
        return self._body


class _CapturingAdapter:
    def __init__(self):
        self.events = []
        self.config = SimpleNamespace(enabled=True)
        self._running = True
        self._message_handler = AsyncMock()

    async def handle_message(self, event):
        self.events.append(event)
        await self._message_handler(event)


def _app(adapter: WebhookAdapter) -> web.Application:
    app = web.Application(client_max_size=adapter._max_body_bytes)
    app.router.add_post("/webhooks/{route_name}", adapter._handle_webhook)
    return app


@pytest.mark.asyncio
async def test_synthetic_telegram_webhook_source_does_not_use_delivery_id_as_reply_anchor():
    config = PlatformConfig(
        enabled=True,
        extra={
            "host": "127.0.0.1",
            "routes": {
                "siri": {
                    "secret": "INSECURE_NO_AUTH",
                    "prompt": "{text}",
                    "source_platform": "telegram",
                    "source_chat_id": "1000000001",
                    "source_chat_type": "dm",
                    "source_user_id": "1000000001",
                    "source_user_name": "User",
                }
            },
        },
    )
    adapter = WebhookAdapter(config)
    telegram_adapter = _CapturingAdapter()
    setattr(adapter, "gateway_runner", SimpleNamespace(adapters={Platform.TELEGRAM: telegram_adapter}))

    response = await adapter._handle_webhook(
        _FakeRequest(
            "siri",
            {"text": "Give me the Burning Man itinerary"},
            headers={"X-Request-ID": "1781840547354"},
        )
    )
    assert response.status == 202

    # Let the task created by _handle_webhook run.
    await asyncio.sleep(0)

    assert len(telegram_adapter.events) == 1
    event = telegram_adapter.events[0]
    assert event.source.platform == Platform.TELEGRAM
    assert event.source.chat_id == "1000000001"
    assert event.source.message_id is None
    assert event.message_id is None
    assert _reply_anchor_for_event(event) is None

    metadata = _thread_metadata_for_source(event.source)
    assert metadata is None


@pytest.mark.asyncio
async def test_synthetic_telegram_webhook_audio_payload_routes_as_voice(monkeypatch, tmp_path):
    cached_audio = tmp_path / "shortcut_audio.m4a"
    m4a_bytes = b"\x00\x00\x00\x18ftypM4A fake-m4a-bytes"

    def fake_cache_audio_from_bytes(data: bytes, ext: str = ".ogg") -> str:
        assert data == m4a_bytes
        assert ext == ".m4a"
        cached_audio.write_bytes(data)
        return str(cached_audio)

    monkeypatch.setattr(
        "gateway.platforms.webhook.cache_audio_from_bytes",
        fake_cache_audio_from_bytes,
    )

    config = PlatformConfig(
        enabled=True,
        extra={
            "host": "127.0.0.1",
            "routes": {
                "siri": {
                    "secret": "INSECURE_NO_AUTH",
                    "prompt": "[Siri relay / {route}]\n\n{message}",
                    "source_platform": "telegram",
                    "source_chat_id": "1000000001",
                    "source_chat_type": "dm",
                    "source_user_id": "1000000001",
                    "source_user_name": "User",
                }
            },
        },
    )
    adapter = WebhookAdapter(config)
    telegram_adapter = _CapturingAdapter()
    setattr(adapter, "gateway_runner", SimpleNamespace(adapters={Platform.TELEGRAM: telegram_adapter}))

    response = await adapter._handle_webhook(
        _FakeRequest(
            "siri",
            {
                "event_type": "siri",
                "message": "",
                "audio_base64": base64.b64encode(m4a_bytes).decode("ascii"),
                "audio_mime_type": "audio/mp4",
                "audio_filename": "Shortcut Recording.m4a",
            },
            headers={"X-Request-ID": "shortcut-audio-1"},
        )
    )
    assert response.status == 202

    await asyncio.sleep(0)

    assert len(telegram_adapter.events) == 1
    event = telegram_adapter.events[0]
    assert event.source.platform == Platform.TELEGRAM
    assert event.source.chat_id == "1000000001"
    assert event.source.message_id is None
    assert event.message_id is None
    assert _reply_anchor_for_event(event) is None
    assert event.message_type.value == "voice"
    assert event.media_urls == [str(cached_audio)]
    assert event.media_types == ["audio/mp4"]


@pytest.mark.asyncio
async def test_synthetic_telegram_webhook_image_payload_routes_as_photo(monkeypatch, tmp_path):
    cached_image = tmp_path / "shortcut_screenshot.png"
    png_bytes = b"\x89PNG\r\n\x1a\n" + b"fake-png-body"

    def fake_cache_image_from_bytes(data: bytes, ext: str = ".jpg") -> str:
        assert data == png_bytes
        assert ext == ".png"
        cached_image.write_bytes(data)
        return str(cached_image)

    monkeypatch.setattr(
        "gateway.platforms.webhook.cache_image_from_bytes",
        fake_cache_image_from_bytes,
    )

    config = PlatformConfig(
        enabled=True,
        extra={
            "host": "127.0.0.1",
            "routes": {
                "siri-screen-phone": {
                    "secret": "INSECURE_NO_AUTH",
                    "prompt": "[Siri screen relay / phone]\n\n{message}",
                    "source_platform": "telegram",
                    "source_chat_id": "1000000001",
                    "source_chat_type": "dm",
                    "source_user_id": "1000000001",
                    "source_user_name": "User",
                }
            },
        },
    )
    adapter = WebhookAdapter(config)
    telegram_adapter = _CapturingAdapter()
    setattr(adapter, "gateway_runner", SimpleNamespace(adapters={Platform.TELEGRAM: telegram_adapter}))

    response = await adapter._handle_webhook(
        _FakeRequest(
            "siri-screen-phone",
            {
                "event_type": "siri_screen_phone",
                "message": "Act on this screenshot as Todd should.",
                "screenshot_base64": base64.b64encode(png_bytes).decode("ascii"),
                "screenshot_mime_type": "image/png",
                "screenshot_filename": "Screenshot Todd.png",
            },
            headers={"X-Request-ID": "shortcut-screen-1"},
        )
    )
    assert response.status == 202

    await asyncio.sleep(0)

    assert len(telegram_adapter.events) == 1
    event = telegram_adapter.events[0]
    assert event.source.platform == Platform.TELEGRAM
    assert event.source.chat_id == "1000000001"
    assert event.source.message_id is None
    assert event.message_id is None
    assert _reply_anchor_for_event(event) is None
    assert event.message_type.value == "photo"
    assert event.media_urls == [str(cached_image)]
    assert event.media_types == ["image/png"]


@pytest.mark.asyncio
async def test_real_aiohttp_image_payload_is_cached_and_malformed_base64_is_rejected(
    monkeypatch, tmp_path
):
    """The HTTP boundary dispatches valid image media and never caches bad base64."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = PlatformConfig(
        enabled=True,
        extra={
            "host": "127.0.0.1",
            "routes": {
                "siri": {
                    "secret": "INSECURE_NO_AUTH",
                    "prompt": "{message}",
                    "source_platform": "telegram",
                    "source_chat_id": "1000000001",
                    "source_user_id": "1000000001",
                }
            },
        },
    )
    adapter = WebhookAdapter(config)
    telegram_adapter = _CapturingAdapter()
    adapter.gateway_runner = SimpleNamespace(adapters={Platform.TELEGRAM: telegram_adapter})
    png_bytes = b"\x89PNG\r\n\x1a\nvalid-test-image"

    async with TestClient(TestServer(_app(adapter))) as client:
        bad = await client.post(
            "/webhooks/siri",
            json={"message": "bad", "image_base64": "not valid base64!"},
        )
        assert bad.status == 400
        assert not telegram_adapter.events

        accepted = await client.post(
            "/webhooks/siri",
            json={
                "message": "inspect this",
                "image_base64": base64.b64encode(png_bytes).decode(),
                "image_mime_type": "image/png",
            },
            headers={"X-Request-ID": "valid-image-1"},
        )
        assert accepted.status == 202

    await asyncio.sleep(0)
    assert len(telegram_adapter.events) == 1
    cached_path = telegram_adapter.events[0].media_urls[0]
    try:
        with open(cached_path, "rb") as cached:
            assert cached.read() == png_bytes
    finally:
        # The production cache uses gateway TTL cleanup; remove this test file
        # explicitly so the integration fixture leaves no temp artifact behind.
        import os
        os.unlink(cached_path)


@pytest.mark.asyncio
async def test_synthetic_source_uses_only_route_identity_and_native_handler():
    """Authenticated payload identity cannot impersonate a different native chat."""
    config = PlatformConfig(
        enabled=True,
        extra={"host": "127.0.0.1", "routes": {"relay": {
            "secret": "INSECURE_NO_AUTH", "prompt": "{message}",
            "source_platform": "telegram", "source_chat_id": "configured-chat",
            "source_user_id": "configured-user", "source_thread_id": "configured-thread",
        }}},
    )
    adapter = WebhookAdapter(config)
    target = _CapturingAdapter()
    adapter.gateway_runner = SimpleNamespace(adapters={Platform.TELEGRAM: target})

    response = await adapter._handle_webhook(_FakeRequest("relay", {
        "message": "hello", "source_chat_id": "spoofed-chat", "source_user_id": "spoofed-user",
        "source_thread_id": "spoofed-thread", "source_platform": "discord",
    }, headers={"X-Request-ID": "identity-1"}))
    assert response.status == 202
    await asyncio.sleep(0)
    target._message_handler.assert_awaited_once()
    event = target.events[0]
    assert (event.source.platform, event.source.chat_id, event.source.user_id, event.source.thread_id) == (
        Platform.TELEGRAM, "configured-chat", "configured-user", "configured-thread"
    )


@pytest.mark.asyncio
async def test_synthetic_source_rejects_unavailable_and_recursive_targets():
    base_route = {"secret": "INSECURE_NO_AUTH", "prompt": "x", "source_chat_id": "c", "source_user_id": "u"}
    for platform in ("telegram", "webhook", "api_server"):
        adapter = WebhookAdapter(PlatformConfig(enabled=True, extra={"host": "127.0.0.1", "routes": {
            "relay": {**base_route, "source_platform": platform},
        }}))
        adapter.gateway_runner = SimpleNamespace(adapters={})
        response = await adapter._handle_webhook(_FakeRequest("relay", {"message": "x"}))
        assert response.status in {500, 503}


@pytest.mark.asyncio
async def test_media_signature_mismatch_is_rejected_before_idempotency_or_cache(monkeypatch):
    adapter = WebhookAdapter(PlatformConfig(enabled=True, extra={"host": "127.0.0.1", "routes": {
        "relay": {"secret": "INSECURE_NO_AUTH", "prompt": "x"},
    }}))
    cache = AsyncMock()
    monkeypatch.setattr("gateway.platforms.webhook.cache_image_from_bytes", cache)
    response = await adapter._handle_webhook(_FakeRequest("relay", {
        "image_base64": base64.b64encode(b"OggS not an image").decode(), "image_mime_type": "image/png",
    }, headers={"X-Request-ID": "bad-signature"}))
    assert response.status == 400
    assert "bad-signature" not in adapter._seen_deliveries
    cache.assert_not_called()


@pytest.mark.asyncio
async def test_duplicate_media_delivery_caches_once_and_cache_failure_can_retry(monkeypatch, tmp_path):
    adapter = WebhookAdapter(PlatformConfig(enabled=True, extra={"host": "127.0.0.1", "routes": {
        "relay": {"secret": "INSECURE_NO_AUTH", "prompt": "x"},
    }}))
    calls = 0
    cache_path = tmp_path / "image.png"

    def cache_image(data, ext):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("temporary disk failure")
        cache_path.write_bytes(data)
        return str(cache_path)

    monkeypatch.setattr("gateway.platforms.webhook.cache_image_from_bytes", cache_image)
    adapter.handle_message = AsyncMock()
    payload = {"image_base64": base64.b64encode(b"\x89PNG\r\n\x1a\nimage").decode(), "image_mime_type": "image/png"}
    first = await adapter._handle_webhook(_FakeRequest("relay", payload, {"X-Request-ID": "retry-cache"}))
    second = await adapter._handle_webhook(_FakeRequest("relay", payload, {"X-Request-ID": "retry-cache"}))
    duplicate = await adapter._handle_webhook(_FakeRequest("relay", payload, {"X-Request-ID": "retry-cache"}))
    assert (first.status, second.status, duplicate.status) == (503, 202, 200)
    assert calls == 2
    assert sum(
        chat_id == "webhook:relay:retry-cache"
        for _created_at, chat_id in adapter._delivery_info_order
    ) == 1
    await asyncio.sleep(0)
    adapter.handle_message.assert_awaited_once()
    assert cache_path.exists()
