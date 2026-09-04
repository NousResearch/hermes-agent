"""Tests for gateway runtime audio input routing and session buffer handling."""

import pytest
from unittest.mock import AsyncMock

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource, build_session_key


def _make_runner() -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="fake")}
    )
    runner.adapters = {}
    runner._pending_native_image_paths_by_session = {}
    runner._pending_native_audio_paths_by_session = {}
    runner._session_model_overrides = {}
    runner._session_reasoning_overrides = {}
    return runner


def _source(chat_id: str = "273403055") -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id=chat_id,
        chat_type="private",
        user_id="42",
        user_name=f"user-{chat_id}",
    )


def _voice_event(source: SessionSource, path: str = "/tmp/voice.oga") -> MessageEvent:
    return MessageEvent(
        text="",
        message_type=MessageType.VOICE,
        source=source,
        media_urls=[path],
        media_types=["audio/ogg"],
    )


@pytest.mark.asyncio
async def test_native_audio_routing_buffers_paths_when_model_supports_audio(monkeypatch):
    runner = _make_runner()
    source = _source()
    event = _voice_event(source, "/tmp/voice_note.oga")

    runner._decide_audio_input_mode = lambda **_: "native"

    # Mock enrich with transcription to verify it is NOT called
    mock_stt = AsyncMock(return_value=("enriched", []))
    runner._enrich_message_with_transcription = mock_stt

    text = await runner._prepare_inbound_message_text(
        event=event,
        source=source,
        history=[],
    )

    # In native mode, STT is bypassed
    assert mock_stt.await_count == 0
    # Pending audio paths are buffered for the session
    session_key = build_session_key(source)
    buffered = runner._consume_pending_native_audio_paths(session_key)
    assert buffered == ["/tmp/voice_note.oga"]


@pytest.mark.asyncio
async def test_stt_audio_routing_calls_transcription_when_model_is_text_only(monkeypatch):
    runner = _make_runner()
    source = _source()
    event = _voice_event(source, "/tmp/voice_note.oga")

    runner._decide_audio_input_mode = lambda **_: "stt"

    mock_stt = AsyncMock(return_value=("[User spoke: Hello world]", ["Hello world"]))
    runner._enrich_message_with_transcription = mock_stt
    runner._should_echo_stt_transcripts = lambda: False

    text = await runner._prepare_inbound_message_text(
        event=event,
        source=source,
        history=[],
    )

    assert mock_stt.await_count == 1
    assert "[User spoke: Hello world]" in text
    session_key = build_session_key(source)
    buffered = runner._consume_pending_native_audio_paths(session_key)
    assert buffered == []


@pytest.mark.asyncio
async def test_native_audio_buffer_isolated_per_session():
    runner = _make_runner()
    runner._decide_audio_input_mode = lambda **_: "native"

    source_a = _source("chat-a")
    source_b = _source("chat-b")

    await runner._prepare_inbound_message_text(
        event=_voice_event(source_a, "/tmp/voice_a.oga"),
        source=source_a,
        history=[],
    )
    await runner._prepare_inbound_message_text(
        event=_voice_event(source_b, "/tmp/voice_b.oga"),
        source=source_b,
        history=[],
    )

    session_key_a = build_session_key(source_a)
    session_key_b = build_session_key(source_b)

    assert runner._consume_pending_native_audio_paths(session_key_a) == ["/tmp/voice_a.oga"]
    assert runner._consume_pending_native_audio_paths(session_key_b) == ["/tmp/voice_b.oga"]
    # Consumed once, second consume is empty
    assert runner._consume_pending_native_audio_paths(session_key_a) == []
    assert runner._consume_pending_native_audio_paths(session_key_b) == []


def test_decide_audio_input_mode_integration(monkeypatch):
    runner = _make_runner()
    cfg = {
        "agent": {"audio_input_mode": "auto"},
        "model": {"provider": "gemini", "default": "gemini-2.0-flash"},
    }

    monkeypatch.setattr(
        runner,
        "_resolve_session_agent_runtime",
        lambda **_: ("gemini-2.0-flash", {"provider": "gemini"}),
    )

    assert runner._decide_audio_input_mode(
        source=_source(),
        user_config=cfg,
    ) == "native"


def test_multimodal_parts_turn_building_with_image_and_audio(tmp_path):
    img_file = tmp_path / "test.png"
    img_file.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4")
    aud_file = tmp_path / "voice.oga"
    aud_file.write_bytes(b"OggS\x00\x02\x00\x00\x00\x00\x00\x00fakeopus")

    from agent.image_routing import build_native_content_parts
    from agent.audio_routing import build_native_audio_content_parts

    img_parts, img_skipped = build_native_content_parts(
        "Look and listen",
        [str(img_file)],
    )
    aud_parts, aud_skipped = build_native_audio_content_parts(
        "",
        [str(aud_file)],
    )

    combined_parts = []
    text_pieces = []
    for p in img_parts:
        if p.get("type") == "text" and p.get("text"):
            text_pieces.append(p["text"])
    for p in aud_parts:
        if p.get("type") == "text" and p.get("text"):
            text_pieces.append(p["text"])
    if text_pieces:
        combined_parts.append({"type": "text", "text": "\n\n".join(text_pieces)})
    for p in img_parts:
        if p.get("type") != "text":
            combined_parts.append(p)
    for p in aud_parts:
        if p.get("type") != "text":
            combined_parts.append(p)

    assert any(p.get("type") == "image_url" for p in combined_parts)
    assert any(p.get("type") == "input_audio" for p in combined_parts)
    assert any(p.get("type") == "text" and "[Image attached at:" in p.get("text", "") and "[Audio attached at:" in p.get("text", "") for p in combined_parts)

