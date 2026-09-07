"""Gateway invariants for session-aware native voice-note routing."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.run_turn_runner import TurnRunner
from gateway.session import SessionSource, build_session_key
from gateway.turn_context import TurnContext


def _source() -> SessionSource:
    return SessionSource(platform=Platform.TELEGRAM, chat_id="audio-chat", chat_type="dm")


def _bare_gateway() -> GatewayRunner:
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(stt_enabled=True)
    runner.adapters = {}
    runner._model = "google/gemini-test"
    runner._base_url = None
    return runner


def test_gateway_decision_honors_config_precedence_and_endpoint_denial(monkeypatch):
    import agent.audio_routing as audio_routing

    runner = _bare_gateway()
    monkeypatch.setattr(audio_routing, "supported_input_modalities", lambda *_: set())

    assert runner._decide_audio_input_mode(
        provider="openrouter",
        model="unknown",
        user_config={"gateway": {"audio_mode": "native"}},
    ) == "native"
    assert runner._decide_audio_input_mode(
        provider="openrouter",
        model="unknown",
        user_config={"audio_mode": "stt", "gateway": {"audio_mode": "native"}},
    ) == "stt"
    assert runner._decide_audio_input_mode(
        provider="meta",
        model="audio-model",
        user_config={"gateway": {"audio_mode": "native"}},
    ) == "stt"


@pytest.mark.asyncio
async def test_voice_note_is_staged_once_without_invoking_stt(tmp_path):
    runner = _bare_gateway()
    runner._decide_audio_input_mode = lambda **_: "native"
    runner._enrich_message_with_transcription = lambda *_args: (_ for _ in ()).throw(
        AssertionError("native voice routing must not invoke STT")
    )
    source = _source()
    voice = tmp_path / "voice.ogg"
    voice.write_bytes(b"OggSvoice-bytes")
    event = MessageEvent(
        text="",
        message_type=MessageType.VOICE,
        source=source,
        media_urls=[str(voice)],
        media_types=["audio/ogg"],
    )

    prepared = await runner._prepare_inbound_message_text(event=event, source=source, history=[])

    session_key = build_session_key(source)
    assert prepared == ""
    assert runner._consume_pending_native_audio_attachments(session_key) == [
        {"path": str(voice), "mime_type": "audio/ogg"}
    ]
    assert runner._consume_pending_native_audio_attachments(session_key) == []


@pytest.mark.asyncio
async def test_construction_fallback_reuses_stt_and_echoes_each_transcript_once():
    runner = _bare_gateway()
    adapter = SimpleNamespace(send=AsyncMock())
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._enrich_message_with_transcription = AsyncMock(
        return_value=("[transcripts]", ["one", "two"]),
    )
    runner._thread_metadata_for_source = lambda *_args: {"thread_id": "topic"}

    fallback = await runner._transcribe_native_audio_fallback(
        ["one.ogg", "two.ogg"],
        source=_source(),
        reply_to_message_id="reply",
    )

    assert fallback == "[transcripts]"
    runner._enrich_message_with_transcription.assert_awaited_once_with(
        "", ["one.ogg", "two.ogg"],
    )
    assert adapter.send.await_count == 2


def test_mixed_media_keeps_valid_parts_and_persists_no_audio_base64(tmp_path):
    image = tmp_path / "photo.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"pixels")
    voice = tmp_path / "voice.mp3"
    raw_audio = b"ID3native-voice"
    voice.write_bytes(raw_audio)
    missing = tmp_path / "missing.ogg"

    class BufferRunner:
        def __init__(self):
            self.images = [str(image)]
            self.audio = [
                {"path": str(voice), "mime_type": "audio/mpeg"},
                {"path": str(missing), "mime_type": "audio/ogg"},
            ]

        def _consume_pending_native_image_paths(self, _session_key):
            paths, self.images = self.images, []
            return paths

        def _consume_pending_native_audio_attachments(self, _session_key):
            attachments, self.audio = self.audio, []
            return attachments

        async def _transcribe_native_audio_fallback(self, *_args, **_kwargs):
            raise AssertionError("the sync bridge is replaced in this test")

    runner = BufferRunner()
    ctx = TurnContext(message="look and listen", session_key="session", source=_source())
    turn = TurnRunner(runner, ctx)
    fallback_calls = []
    turn._transcribe_native_audio_fallback_sync = lambda paths: (
        fallback_calls.append(list(paths)) or "[transcript: recovered clip]"
    )

    captured = {}

    class Agent:
        provider = "openai"
        requested_provider = "openai"

        def run_conversation(self, message, **kwargs):
            captured.update(message=message, kwargs=kwargs)
            return {"final_response": "done"}

    result = turn._run_conversation_with_approval(Agent(), [], None, None, None)

    assert result == {"final_response": "done"}
    message = captured["message"]
    assert {part.get("type") for part in message} >= {"text", "image_url", "input_audio"}
    assert fallback_calls == [[str(missing)]]
    persisted = captured["kwargs"]["persist_user_message"]
    assert persisted == (
        "[transcript: recovered clip]\n\n[Voice message attached natively]\n\nlook and listen"
    )
    base64_payload = next(
        part["input_audio"]["data"] for part in message if part.get("type") == "input_audio"
    )
    assert base64_payload
    assert base64_payload not in persisted
    assert runner.images == []
    assert runner.audio == []
