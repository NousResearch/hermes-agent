import asyncio
import base64
import io
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import Platform, PlatformConfig
from gateway.platforms.event import MessageType
from gateway.platforms.webhook import WebhookAdapter
from gateway.relay.adapter import RelayAdapter
from gateway.relay.descriptor import CONTRACT_VERSION, CapabilityDescriptor
from gateway.session import SessionSource
from tests.gateway.relay.stub_connector import StubConnector


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
@pytest.mark.parametrize("kind,mime,suffix,message_type", [
    ("audio", "audio/wav", ".wav", MessageType.VOICE),
    ("image", "image/gif", ".gif", MessageType.PHOTO),
])
async def test_synthetic_source_route_interprets_audio_and_dispatches_native_event(
        tmp_path, monkeypatch, kind, mime, suffix, message_type):
    adapter = _make_adapter({"phone": {
        "secret": "INSECURE_NO_AUTH", "prompt": "{message}",
        "source_platform": "telegram", "source_chat_id": "42", "source_user_id": "42",
        "source_new_thread": True,
    }})
    descriptor = CapabilityDescriptor(
        contract_version=CONTRACT_VERSION, platform="telegram", label="Telegram",
        max_message_length=4096, supports_draft_streaming=False, supports_edit=True,
        supports_threads=True, markdown_dialect="telegram", len_unit="chars",
        supported_ops=("thread_create",),
    )
    transport = StubConnector(descriptor)
    transport.send_outbound = AsyncMock(return_value={"success": True, "thread_id": "77"})
    target = RelayAdapter(PlatformConfig(), descriptor, transport=transport)
    target._running = True
    target._message_handler = AsyncMock()
    # Exercise real native ingress and session-key selection; stop before launching an agent.
    target._start_session_processing = MagicMock(return_value=True)
    _configure_source_adapter(adapter, target)
    monkeypatch.setattr("gateway.platforms.base.get_audio_cache_dir", lambda: tmp_path)
    monkeypatch.setattr("gateway.platforms.base.get_image_cache_dir", lambda: tmp_path)
    if kind == "audio":
        stream = io.BytesIO()
        with wave.open(stream, "wb") as wav:
            wav.setparams((1, 2, 8000, 0, "NONE", "not compressed"))
            wav.writeframes(b"\x00\x00" * 8)
        media = stream.getvalue()
    else:
        media = base64.b64decode("R0lGODlhAQABAIAAAAAAAP///ywAAAAAAQABAAACAUwAOw==")
    payload = {"message": "hello", f"{kind}_base64": base64.b64encode(media).decode(), f"{kind}_mime_type": mime}

    async with TestClient(TestServer(_create_app(adapter))) as client:
        response = await client.post("/webhooks/phone", json=payload, headers={"X-GitHub-Delivery": "delivery-audio"})
        assert response.status == 202

    await asyncio.gather(*adapter._background_tasks)
    target._start_session_processing.assert_called_once()
    event, session_key = target._start_session_processing.call_args.args
    transport.send_outbound.assert_awaited_once()
    assert transport.send_outbound.call_args.args[0]["op"] == "thread_create"
    assert event.source.thread_id == "77"
    native_source = SessionSource(platform=Platform.TELEGRAM, chat_id="42", user_id="42", chat_type="dm", thread_id="77")
    assert session_key == target._source_session_key(native_source)
    assert session_key != target._source_session_key(SessionSource(
        platform=Platform.TELEGRAM, chat_id="42", user_id="42", chat_type="dm"))
    assert event.source.platform is Platform.TELEGRAM
    assert event.source.chat_id == "42"
    assert event.source.user_id == "42"
    assert event.message_type is message_type
    assert len(event.media_urls) == 1
    cached = Path(event.media_urls[0])
    assert cached.parent == tmp_path
    assert cached.suffix == suffix
    assert cached.read_bytes() == media
    assert event.media_types == [mime]
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
    payload = {"message": "ordinary", "audio_base64": "not-base64", "image_base64": {"application": "data"}}

    async with TestClient(TestServer(_create_app(adapter))) as client:
        response = await client.post("/webhooks/ordinary", json=payload, headers={"X-GitHub-Delivery": "ordinary-audio"})
        assert response.status == 202

    cache_audio.assert_not_called()
    adapter._spawn_agent_run.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("override,error", [
    ({"cron_job": "job"}, "cron_job"),
    ({"deliver_only": True, "deliver": "telegram"}, "deliver_only"),
    ({"coalesce": {"key": "{message}"}}, "coalesce"),
    ({"source_platform": "webhook"}, "Invalid configured source platform"),
    ({"source_platform": "api_server"}, "Invalid configured source platform"),
    ({"source_platform": "not-a-platform"}, "Invalid configured source platform"),
    ({"source_user_id": ""}, "source_user_id"),
])
async def test_invalid_source_routes_reject_at_startup_and_http_before_side_effects(override, error):
    route = {"secret": "INSECURE_NO_AUTH", "source_platform": "telegram",
             "source_chat_id": "42", "source_user_id": "42", **override}
    adapter = _make_adapter({"phone": route})
    adapter._host = "127.0.0.1"
    with pytest.raises(ValueError, match=error):
        adapter._validate_route("phone", route)
    adapter._find_adapter = MagicMock()
    async with TestClient(TestServer(_create_app(adapter))) as client:
        response = await client.post("/webhooks/phone", json={"message": "hello"})
        assert response.status == 500
        assert error in (await response.json())["error"]
    adapter._find_adapter.assert_not_called()
    assert not adapter._seen_deliveries
    assert not adapter._background_tasks


@pytest.mark.asyncio
@pytest.mark.parametrize("running,kind,mime,status", [
    (False, "audio", "audio/ogg", 503),
    (True, "audio", "audio/unknown", 400),
    (True, "image", "image/unknown", 400),
])
async def test_unavailable_source_or_unsupported_media_never_caches_or_dispatches(monkeypatch, running, kind, mime, status):
    adapter = _make_adapter({"phone": {"secret": "INSECURE_NO_AUTH", "source_platform": "telegram",
                                      "source_chat_id": "42", "source_user_id": "42"}})
    target = SimpleNamespace(_running=running, handle_message=AsyncMock())
    _configure_source_adapter(adapter, target)
    audio_cache, image_cache = MagicMock(), MagicMock()
    monkeypatch.setattr("gateway.platforms.webhook.cache_audio_from_bytes", audio_cache)
    monkeypatch.setattr("gateway.platforms.webhook.cache_image_from_bytes", image_cache)
    payload = {f"{kind}_base64": base64.b64encode(b"data").decode(), f"{kind}_mime_type": mime}
    async with TestClient(TestServer(_create_app(adapter))) as client:
        response = await client.post("/webhooks/phone", json=payload)
        assert response.status == status
    audio_cache.assert_not_called()
    image_cache.assert_not_called()
    target.handle_message.assert_not_called()
    assert not adapter._seen_deliveries
    assert not adapter._background_tasks
