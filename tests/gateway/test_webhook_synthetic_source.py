import asyncio
from types import SimpleNamespace

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.webhook import WebhookAdapter


@pytest.mark.asyncio
async def test_dispatch_uses_configured_native_source_adapter():
    adapter = WebhookAdapter(PlatformConfig(enabled=True, extra={"routes": {}}))
    received = []

    class Target:
        async def handle_message(self, event):
            received.append(event)

    target = Target()
    response = await adapter._dispatch_agent_run(
        SimpleNamespace(method="POST"), {
            "source_chat_id": "42", "source_chat_type": "dm", "source_user_id": "42",
            "source_chat_name": "Geo", "source_user_name": "Geo",
        }, "phone", None, {}, "hello", "siri_audio", "delivery-1", 1.0,
        source_platform=Platform.TELEGRAM, target_adapter=target,
        media_urls=["/tmp/voice.m4a"], media_types=["audio/mp4"])
    await asyncio.gather(*adapter._background_tasks)
    assert response.status == 202
    assert len(received) == 1
    event = received[0]
    assert event.source.platform is Platform.TELEGRAM
    assert event.source.chat_id == "42"
    assert event.source.user_id == "42"
    assert event.media_urls == ["/tmp/voice.m4a"]
    assert event.message_id is None


@pytest.mark.asyncio
async def test_new_thread_failure_is_retryable_and_not_dispatched():
    adapter = WebhookAdapter(PlatformConfig(enabled=True, extra={"routes": {}}))

    class Target:
        async def create_handoff_thread(self, chat_id, name):
            return None

        async def handle_message(self, event):
            raise AssertionError("must not dispatch")

    response = await adapter._dispatch_agent_run(
        SimpleNamespace(method="POST"), {
            "source_chat_id": "42", "source_user_id": "42", "source_new_thread": True,
        }, "phone", None, {}, "hello", "siri_audio", "delivery-2", 2.0,
        source_platform=Platform.TELEGRAM, target_adapter=Target())
    assert response.status == 503