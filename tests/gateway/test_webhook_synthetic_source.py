import asyncio
import base64
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import Platform, PlatformConfig
from gateway.platforms.event import MessageType
from gateway.platforms.webhook import WebhookAdapter


def _make_adapter(routes):
    return WebhookAdapter(PlatformConfig(enabled=True, extra={"routes": routes}))


def _create_app(adapter):
    app = web.Application()
    app.router.add_post("/webhooks/{route_name}", adapter._handle_webhook)
    return app


def _configure_source_adapter(adapter, target):
    adapter.gateway_runner = SimpleNamespace(
        _authorization_adapter=lambda platform, profile: target if platform is Platform.TELEGRAM else None,
    )


@pytest.mark.asyncio
async def test_synthetic_source_route_interprets_audio_and_dispatches_native_event(tmp_path, monkeypatch):
    adapter = _make_adapter({"phone": {
        "secret": "INSECURE_NO_AUTH", "prompt": "{message}",
        "source_platform": "telegram", "source_chat_id": "42", "source_user_id": "42",
    }})
    received = []

    class Target:
        _running = True

        async def handle_message(self, event):
            received.append(event)

    _configure_source_adapter(adapter, Target())
    cached = tmp_path / "voice.ogg"
    monkeypatch.setattr("gateway.platforms.webhook.cache_audio_from_bytes", lambda *_args, **_kwargs: str(cached))
    payload = {"message": "hello", "audio_base64": base64.b64encode(b"audio").decode(), "audio_mime_type": "audio/ogg"}

    async with TestClient(TestServer(_create_app(adapter))) as client:
        response = await client.post("/webhooks/phone", json=payload, headers={"X-GitHub-Delivery": "delivery-audio"})
        assert response.status == 202

    await asyncio.gather(*adapter._background_tasks)
    assert len(received) == 1
    event = received[0]
    assert event.source.platform is Platform.TELEGRAM
    assert event.source.chat_id == "42"
    assert event.source.user_id == "42"
    assert event.message_type is MessageType.VOICE
    assert event.media_urls == [str(cached)]
    assert event.media_types == ["audio/ogg"]
    assert event.message_id is None


@pytest.mark.asyncio
async def test_thread_creation_failure_removes_cache_and_delivery_marker_then_retry_dispatches(tmp_path, monkeypatch):
    adapter = _make_adapter({"phone": {
        "secret": "INSECURE_NO_AUTH", "prompt": "{message}",
        "source_platform": "telegram", "source_chat_id": "42", "source_user_id": "42", "source_new_thread": True,
    }})
    received = []

    class Target:
        _running = True
        thread_attempts = 0

        async def create_handoff_thread(self, chat_id, name):
            self.thread_attempts += 1
            return None if self.thread_attempts == 1 else "thread-42"

        async def handle_message(self, event):
            received.append(event)

    target = Target()
    _configure_source_adapter(adapter, target)
    cached_paths = []

    def cache_audio(*_args, **_kwargs):
        path = tmp_path / f"voice-{len(cached_paths)}.ogg"
        path.write_bytes(b"cached audio")
        cached_paths.append(path)
        return str(path)

    monkeypatch.setattr("gateway.platforms.webhook.cache_audio_from_bytes", cache_audio)
    payload = {"message": "hello", "audio_base64": base64.b64encode(b"audio").decode()}
    headers = {"X-GitHub-Delivery": "delivery-retry"}

    async with TestClient(TestServer(_create_app(adapter))) as client:
        failed = await client.post("/webhooks/phone", json=payload, headers=headers)
        assert failed.status == 503
        assert "delivery-retry" not in adapter._seen_deliveries
        assert cached_paths[0].exists() is False
        assert adapter._background_tasks == set()
        assert received == []

        retried = await client.post("/webhooks/phone", json=payload, headers=headers)
        assert retried.status == 202

    await asyncio.gather(*adapter._background_tasks)
    assert target.thread_attempts == 2
    assert len(received) == 1
    assert received[0].source.thread_id == "thread-42"
    assert received[0].message_type is MessageType.VOICE
    assert received[0].media_types == ["audio/ogg"]


@pytest.mark.asyncio
async def test_ordinary_route_does_not_cache_synthetic_media_payload(tmp_path, monkeypatch):
    adapter = _make_adapter({"ordinary": {"secret": "INSECURE_NO_AUTH", "prompt": "{message}"}})
    adapter._spawn_agent_run = MagicMock()
    cache_audio = MagicMock(return_value=str(tmp_path / "should-not-exist.ogg"))
    monkeypatch.setattr("gateway.platforms.webhook.cache_audio_from_bytes", cache_audio)
    payload = {"message": "ordinary", "audio_base64": base64.b64encode(b"audio").decode()}

    async with TestClient(TestServer(_create_app(adapter))) as client:
        response = await client.post("/webhooks/ordinary", json=payload, headers={"X-GitHub-Delivery": "ordinary-audio"})
        assert response.status == 202

    cache_audio.assert_not_called()
    adapter._spawn_agent_run.assert_called_once()
