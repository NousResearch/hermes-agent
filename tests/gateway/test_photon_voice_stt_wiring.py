"""Regression: photon iMessage voice notes route through the central STT engine.

Photon (iMessage) inbound voice notes are dispatched as ``MessageEvent`` with
``message_type == MessageType.VOICE`` and a ``.caf`` media path — the same
shape Telegram/Discord voice messages use. They must reach the gateway's
central inbound STT pipeline (``transcription_tools.transcribe_audio``) and
have the transcript prepended to the agent-visible text, exactly like every
other channel.

This guards against the photon path silently bypassing the central engine
(the bug that left inbound iMessage voice notes un-transcribed on sealed-venv
installs, where faster-whisper lives in the durable ``HERMES_LAZY_INSTALL_TARGET``
volume and was never importable by the running interpreter — fixed in
``tools/lazy_deps`` by binding that target onto ``sys.path`` at import).

Mirrors the harness in ``test_telegram_audio_vs_voice.py``; photon uses the
dynamic ``Platform("photon")`` member and a ``.caf`` clip path.
"""

from unittest.mock import patch

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource


def _make_runner(stt_enabled: bool = True) -> "GatewayRunner":  # type: ignore[name-defined]
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(stt_enabled=stt_enabled)
    runner.adapters = {}
    runner._model = "test-model"
    runner._base_url = ""
    runner._has_setup_skill = lambda: False
    return runner


def _photon_voice_event(path: str = "/tmp/photon_voice.caf") -> MessageEvent:
    """Photon iMessage inbound voice note: VOICE type, .caf media path."""
    return MessageEvent(
        text="",
        message_type=MessageType.VOICE,
        source=SessionSource(
            platform=Platform("photon"), chat_id="iMessage:+15551234567", chat_type="dm"
        ),
        media_urls=[path],
        media_types=["audio/x-caf"],
    )


@pytest.mark.asyncio
async def test_photon_voice_note_reaches_central_stt():
    """A photon VOICE event must be sent to transcribe_audio and get a transcript."""
    runner = _make_runner(stt_enabled=True)
    source = SessionSource(
        platform=Platform("photon"), chat_id="iMessage:+15551234567", chat_type="dm"
    )
    event = _photon_voice_event("/tmp/photon_voice.caf")

    with patch(
        "tools.transcription_tools.transcribe_audio",
        return_value={"success": True, "transcript": "hey can you send that file", "provider": "whisper"},
    ) as mock_transcribe:
        result = await runner._prepare_inbound_message_text(
            event=event,
            source=source,
            history=[],
        )

    # Central STT engine was invoked for the photon clip.
    mock_transcribe.assert_called_once_with("/tmp/photon_voice.caf", None, "gateway")
    # Transcript lands in the agent-visible text.
    assert "hey can you send that file" in result


@pytest.mark.asyncio
async def test_photon_voice_with_caption_prepends_transcript():
    """Voice + caption: transcript is prepended, caption preserved."""
    runner = _make_runner(stt_enabled=True)
    source = SessionSource(
        platform=Platform("photon"), chat_id="iMessage:+15551234567", chat_type="dm"
    )
    event = MessageEvent(
        text="fyi",
        message_type=MessageType.VOICE,
        source=source,
        media_urls=["/tmp/photon_voice2.caf"],
        media_types=["audio/x-caf"],
    )

    with patch(
        "tools.transcription_tools.transcribe_audio",
        return_value={"success": True, "transcript": "call me later", "provider": "whisper"},
    ):
        result = await runner._prepare_inbound_message_text(
            event=event,
            source=source,
            history=[],
        )

    assert "call me later" in result
    assert "fyi" in result


@pytest.mark.asyncio
async def test_photon_voice_stt_disabled_yields_path_note_not_transcript():
    """With STT disabled, photon voice must NOT reach transcribe_audio."""
    runner = _make_runner(stt_enabled=False)
    source = SessionSource(
        platform=Platform("photon"), chat_id="iMessage:+15551234567", chat_type="dm"
    )
    event = _photon_voice_event("/tmp/photon_voice.caf")

    with patch(
        "tools.transcription_tools.transcribe_audio",
        side_effect=AssertionError("transcribe_audio must not be called when STT disabled"),
    ), patch(
        "tools.credential_files.to_agent_visible_cache_path",
        side_effect=lambda p: p,
    ):
        result = await runner._prepare_inbound_message_text(
            event=event,
            source=source,
            history=[],
        )

    # Graceful fallback: a voice-message path note (no transcript), and the
    # central STT engine was never invoked.
    assert "voice message" in result.lower()
    assert "could not be transcribed" not in result.lower()
