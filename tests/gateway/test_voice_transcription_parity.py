from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource, build_session_key


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="123",
        chat_type="dm",
        user_id="u1",
        user_name="Pol",
    )


def _make_voice_event(*, text: str = "", media_path: str = "/tmp/sample.ogg") -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=MessageType.VOICE,
        source=_make_source(),
        media_urls=[media_path],
        media_types=["audio/ogg"],
        message_id="m1",
    )


@pytest.mark.asyncio
async def test_prepare_event_for_control_flow_promotes_voice_transcript_when_no_text(monkeypatch):
    from gateway.run import GatewayRunner

    monkeypatch.setattr(
        "tools.transcription_tools.transcribe_audio",
        lambda path, model=None: {"success": True, "transcript": "buy oat milk", "provider": "local"},
    )

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(stt_enabled=True)
    event = _make_voice_event()

    prepared = await runner._prepare_event_for_control_flow(event)

    assert prepared.text == "buy oat milk"
    assert prepared.transcription_text == "buy oat milk"
    assert prepared.transcription_backend == "local"
    assert prepared.transcription_origin == "voice"


@pytest.mark.asyncio
async def test_handle_message_interrupt_uses_transcribed_voice_text(monkeypatch):
    from gateway.run import GatewayRunner

    monkeypatch.setattr(
        "tools.transcription_tools.transcribe_audio",
        lambda path, model=None: {"success": True, "transcript": "stop now", "provider": "local"},
    )

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(stt_enabled=True)
    runner._is_user_authorized = MagicMock(return_value=True)
    running_agent = MagicMock()
    event = _make_voice_event()
    session_key = build_session_key(event.source)
    runner._running_agents = {session_key: running_agent}
    runner._pending_messages = {}

    result = await runner._handle_message(event)

    assert result is None
    running_agent.interrupt.assert_called_once_with("stop now")
    assert runner._pending_messages[session_key] == "stop now"
    assert event.text == "stop now"


@pytest.mark.asyncio
async def test_handle_message_routes_transcribed_voice_into_command_parsing(monkeypatch):
    from gateway.run import GatewayRunner

    monkeypatch.setattr(
        "tools.transcription_tools.transcribe_audio",
        lambda path, model=None: {"success": True, "transcript": "/status", "provider": "local"},
    )

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(stt_enabled=True)
    runner._is_user_authorized = MagicMock(return_value=True)
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock())
    runner._handle_status_command = AsyncMock(return_value="status ok")

    event = _make_voice_event()

    result = await runner._handle_message(event)

    assert result == "status ok"
    runner._handle_status_command.assert_awaited_once_with(event)
    assert event.text == "/status"


def test_build_agent_message_text_keeps_transcript_natural_but_marks_stt_uncertainty():
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    event = _make_voice_event(text="buy oat milk")
    event.transcription_text = "buy oat milk"
    event.transcription_backend = "local"
    event.transcription_origin = "voice"

    message_text = runner._build_agent_message_text(event)

    assert message_text.startswith("buy oat milk")
    assert "transcribed from a voice message" in message_text
    assert "minor transcription errors" in message_text
    assert "Here's what they said" not in message_text
    assert "The user sent a voice message" not in message_text
