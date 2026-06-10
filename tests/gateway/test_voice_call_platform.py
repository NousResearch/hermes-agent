"""Gateway-level integration tests for the voice_call platform.

Exercises the real adapter lifecycle (connect → runtime → webhook server →
disconnect) and the full turn loop: inbound caller speech becomes a
MessageEvent with the right SessionSource, and the "agent reply" delivered
through adapter.send() is spoken on the live call.
"""

import asyncio
from unittest.mock import MagicMock

import pytest

pytest.importorskip("aiohttp")

from plugins.platforms.voice_call import runtime as runtime_mod
from plugins.platforms.voice_call.adapter import VoiceCallAdapter
from plugins.platforms.voice_call.events import CallState, EventType, NormalizedEvent


def _platform_config(extra):
    config = MagicMock()
    config.extra = extra
    config.enabled = True
    config.token = None
    config.api_key = None
    config.home_channel = None
    config.reply_to_mode = "first"
    return config


def _mock_extra(**overrides):
    extra = {
        "provider": "mock",
        "from_number": "+15555550000",
        "inbound_policy": "allowlist",
        "allow_from": ["+15555550009"],
        "serve": {"port": 0},
        "timeouts": {"silence_s": 0},
        "outbound": {"default_mode": "conversation"},
    }
    extra.update(overrides)
    return extra


@pytest.fixture(autouse=True)
def _clean_runtime():
    runtime_mod._runtime = None
    runtime_mod._runtime_lock = None
    yield
    runtime_mod._runtime = None
    runtime_mod._runtime_lock = None


async def _connect(extra=None):
    adapter = VoiceCallAdapter(_platform_config(extra or _mock_extra()))
    assert await adapter.connect() is True
    return adapter


@pytest.mark.asyncio
async def test_connect_starts_runtime_disconnect_stops_it():
    adapter = await _connect()
    assert adapter.is_connected
    runtime = runtime_mod.get_runtime()
    assert runtime is not None
    assert runtime.webhook_server.bound_port
    await adapter.disconnect()
    assert runtime_mod.get_runtime() is None
    assert not adapter.is_connected
    # Idempotent disconnect.
    await adapter.disconnect()


@pytest.mark.asyncio
async def test_connect_with_invalid_config_sets_fatal_error():
    adapter = VoiceCallAdapter(
        _platform_config({"provider": "telnyx"})  # no creds, no public URL
    )
    assert await adapter.connect() is False
    assert adapter.has_fatal_error


@pytest.mark.asyncio
async def test_inbound_speech_becomes_message_event_per_phone():
    adapter = await _connect()
    received = []
    runtime = runtime_mod.get_runtime()

    async def fake_handle_message(event):
        received.append(event)

    adapter.handle_message = fake_handle_message
    try:
        manager = runtime.manager
        await manager.process_event(
            NormalizedEvent(
                type=EventType.CALL_INITIATED, provider="mock",
                provider_call_id="prov-1", direction="inbound",
                from_number="+15555550009", to_number="+15555550000",
            )
        )
        await manager.process_event(
            NormalizedEvent(
                type=EventType.CALL_ANSWERED, provider="mock",
                provider_call_id="prov-1",
            )
        )
        await manager.process_event(
            NormalizedEvent(
                type=EventType.CALL_SPEECH, provider="mock",
                provider_call_id="prov-1", text="what's on my calendar?",
            )
        )
        await asyncio.sleep(0.05)
        assert len(received) == 1
        event = received[0]
        assert event.text == "what's on my calendar?"
        assert event.source.chat_id == "+15555550009"
        assert event.source.user_id == "+15555550009"
        assert event.source.thread_id is None  # per-phone scope
        assert event.source.chat_type == "dm"
    finally:
        await adapter.disconnect()


@pytest.mark.asyncio
async def test_inbound_speech_per_call_scope_sets_thread_id():
    adapter = await _connect(_mock_extra(session_scope="per-call"))
    received = []

    async def fake_handle_message(event):
        received.append(event)

    adapter.handle_message = fake_handle_message
    runtime = runtime_mod.get_runtime()
    try:
        manager = runtime.manager
        await manager.process_event(
            NormalizedEvent(
                type=EventType.CALL_INITIATED, provider="mock",
                provider_call_id="prov-2", direction="inbound",
                from_number="+15555550009", to_number="+15555550000",
            )
        )
        record = manager.call_for_chat("+15555550009")
        await manager.process_event(
            NormalizedEvent(type=EventType.CALL_ANSWERED, provider="mock",
                            provider_call_id="prov-2")
        )
        await manager.process_event(
            NormalizedEvent(type=EventType.CALL_SPEECH, provider="mock",
                            provider_call_id="prov-2", text="hello")
        )
        await asyncio.sleep(0.05)
        assert len(received) == 1
        assert received[0].source.thread_id == record.call_id
        assert record.session_key == f"+15555550009:{record.call_id}"
    finally:
        await adapter.disconnect()


@pytest.mark.asyncio
async def test_send_speaks_agent_reply_on_live_call():
    """The reply half of the turn loop: gateway → send() → carrier TTS."""
    adapter = await _connect()
    runtime = runtime_mod.get_runtime()
    try:
        record = await runtime.manager.initiate_call("+15555550001", message="hi")
        deadline = asyncio.get_running_loop().time() + 1.0
        while (
            record.state != CallState.LISTENING
            and asyncio.get_running_loop().time() < deadline
        ):
            await asyncio.sleep(0.01)

        result = await adapter.send(
            "+15555550001", "Your meeting is at **3pm** — see https://cal.example"
        )
        assert result.success is True
        assert result.message_id == record.call_id
        spoken = runtime.provider.spoken[-1][1]
        assert "**" not in spoken and "https://" not in spoken
        assert "3pm" in spoken
        # Spoken reply lands in the transcript as a bot turn.
        assert record.transcript[-1].speaker == "bot"
    finally:
        await adapter.disconnect()


@pytest.mark.asyncio
async def test_send_fails_cleanly_without_live_call():
    adapter = await _connect()
    try:
        result = await adapter.send("+19998887777", "anyone?")
        assert result.success is False
        assert "no active voice call" in result.error
    finally:
        await adapter.disconnect()


@pytest.mark.asyncio
async def test_gateway_shutdown_releases_webhook_port():
    adapter = await _connect()
    port = runtime_mod.get_runtime().webhook_server.bound_port
    await adapter.disconnect()

    # Rebinding the same fixed port proves release.
    adapter2 = await _connect(_mock_extra(serve={"port": port}))
    assert runtime_mod.get_runtime().webhook_server.bound_port == port
    await adapter2.disconnect()
