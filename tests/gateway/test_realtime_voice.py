"""Tests for the OpenAI Realtime voice WebSocket session."""

from __future__ import annotations

import asyncio
import base64
from types import SimpleNamespace
from typing import Any

import aiohttp
import pytest

from gateway.realtime_voice import RealtimeVoiceConfig, RealtimeVoiceSession


class FakeWebSocket:
    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self._events: asyncio.Queue[Any] = asyncio.Queue()
        self.closed = False

    async def send_json(self, payload: dict[str, Any]) -> None:
        self.sent.append(payload)

    async def close(self) -> None:
        self.closed = True
        await self._events.put(None)

    def feed_json(self, payload: dict[str, Any]) -> None:
        self._events.put_nowait(
            SimpleNamespace(
                type=aiohttp.WSMsgType.TEXT,
                data=__import__("json").dumps(payload),
            )
        )

    def __aiter__(self) -> "FakeWebSocket":
        return self

    async def __anext__(self) -> Any:
        item = await self._events.get()
        if item is None:
            raise StopAsyncIteration
        return item


@pytest.fixture
def fake_ws() -> FakeWebSocket:
    return FakeWebSocket()


@pytest.fixture
def session_factory(fake_ws: FakeWebSocket):
    async def _connect(*_args: Any, **_kwargs: Any) -> FakeWebSocket:
        return fake_ws

    def _make(**kwargs: Any) -> RealtimeVoiceSession:
        config = RealtimeVoiceConfig(
            api_key="test-key",
            instructions="Be concise.",
            **kwargs.pop("config_overrides", {}),
        )
        return RealtimeVoiceSession(
            config=config,
            audio_sink=kwargs.pop("audio_sink", lambda _pcm: None),
            websocket_factory=_connect,
            **kwargs,
        )

    return _make


@pytest.mark.asyncio
async def test_start_sends_session_update(session_factory, fake_ws: FakeWebSocket):
    session = session_factory()

    await session.start()
    await session.stop()

    assert fake_ws.sent[0]["type"] == "session.update"
    session_payload = fake_ws.sent[0]["session"]
    assert session_payload["model"] == "gpt-realtime-2"
    assert session_payload["audio"]["output"]["voice"] == "marin"
    assert session_payload["audio"]["input"]["format"] == {
        "type": "audio/pcm",
        "rate": 24000,
    }
    assert session_payload["audio"]["output"]["format"] == {
        "type": "audio/pcm",
        "rate": 24000,
    }


@pytest.mark.asyncio
async def test_append_discord_pcm_sends_base64_audio_append(session_factory, fake_ws: FakeWebSocket):
    session = session_factory()
    discord_silence_frame = b"\x00" * 3840

    await session.start()
    await session.append_discord_pcm(user_id=42, pcm_48k_stereo=discord_silence_frame)
    await session.stop()

    append_event = fake_ws.sent[1]
    assert append_event["type"] == "input_audio_buffer.append"
    assert isinstance(append_event["audio"], str)
    assert base64.b64decode(append_event["audio"]) == b"\x00" * 960


@pytest.mark.asyncio
async def test_audio_delta_is_forwarded_to_sink(session_factory, fake_ws: FakeWebSocket):
    received: list[bytes] = []
    session = session_factory(audio_sink=received.append)

    await session.start()
    fake_ws.feed_json({
        "type": "response.output_audio.delta",
        "delta": base64.b64encode(b"audio").decode("ascii"),
    })
    await asyncio.sleep(0)
    await session.stop()

    assert received == [b"audio"]


@pytest.mark.asyncio
async def test_legacy_audio_delta_name_is_forwarded_to_sink(session_factory, fake_ws: FakeWebSocket):
    received: list[bytes] = []
    session = session_factory(audio_sink=received.append)

    await session.start()
    fake_ws.feed_json({
        "type": "response.audio.delta",
        "delta": base64.b64encode(b"legacy").decode("ascii"),
    })
    await asyncio.sleep(0)
    await session.stop()

    assert received == [b"legacy"]


@pytest.mark.asyncio
async def test_speech_started_clears_sink_when_supported(session_factory, fake_ws: FakeWebSocket):
    class Sink:
        def __init__(self) -> None:
            self.cleared = False

        def __call__(self, _pcm: bytes) -> None:
            pass

        def clear(self) -> None:
            self.cleared = True

    sink = Sink()
    session = session_factory(audio_sink=sink)

    await session.start()
    fake_ws.feed_json({"type": "input_audio_buffer.speech_started"})
    await asyncio.sleep(0)
    await session.stop()

    assert sink.cleared is True


@pytest.mark.asyncio
async def test_function_call_executes_tool_and_returns_output_when_enabled(
    session_factory,
    fake_ws: FakeWebSocket,
):
    calls: list[tuple[str, dict[str, Any]]] = []

    async def execute_tool(name: str, arguments: dict[str, Any]) -> str:
        calls.append((name, arguments))
        return "tool output"

    session = session_factory(
        config_overrides={"tools_enabled": True},
        tool_executor=execute_tool,
    )

    await session.start()
    fake_ws.feed_json({
        "type": "response.output_item.done",
        "item": {
            "type": "function_call",
            "name": "lookup",
            "arguments": '{"query": "status"}',
            "call_id": "call_1",
        },
    })
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    await session.stop()

    assert calls == [("lookup", {"query": "status"})]
    assert fake_ws.sent[1] == {
        "type": "conversation.item.create",
        "item": {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "tool output",
        },
    }
    assert fake_ws.sent[2] == {"type": "response.create"}
