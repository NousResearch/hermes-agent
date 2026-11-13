"""A voice note transcribed while queued keeps a reachable audio path once its turn is routed.

The pending-STT cache (``_gateway_pending_stt_text``) is built before the routed turn moves the
recording into the routed profile's cache (``rehome_inbound_media``), and it replaces ``event.text``
for that turn, so its ``[User sent audio: <path>]`` marker must follow the move.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import hermes_yaml as yaml

from agent import secret_scope as ss
from hermes_agent.gateway.config import GatewayConfig
from hermes_agent.gateway.platforms.event import MessageEvent, MessageType
from hermes_agent.gateway.run import GatewayRunner, _profile_runtime_scope
from hermes_agent.gateway.run_inbound_media import rehome_inbound_media


@pytest.fixture
def two_homes(tmp_path, monkeypatch):
    a = tmp_path / ".hermes"
    b = a / "profiles" / "b"
    b.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(a))
    for home in (a, b):
        (home / "config.yaml").write_text(yaml.safe_dump({"terminal": {"backend": "local"}}), encoding="utf-8")
    ss.set_multiplex_active(True)
    yield a, b
    ss.set_multiplex_active(False)


@pytest.mark.asyncio
async def test_queued_voice_transcript_marker_follows_the_rehomed_recording(two_homes):
    a, b = two_homes
    clip = a / "cache" / "audio" / "audio_0123.ogg"
    clip.parent.mkdir(parents=True)
    clip.write_bytes(b"OggS voice")
    event = MessageEvent(text="", message_type=MessageType.VOICE, source=None,
                         media_urls=[str(clip)], media_types=["audio/ogg"])
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(stt_enabled=True)
    with patch("tools.transcription_tools.transcribe_audio",
               return_value={"success": True, "transcript": "two cartons", "provider": "local"}):
        await runner._transcribe_pending_audio_event_once(event)  # queued: still on the launch home
    assert str(clip) in event._gateway_pending_stt_text

    with _profile_runtime_scope(b):
        rehome_inbound_media(event)

    moved = b / "cache" / "audio" / "audio_0123.ogg"
    assert moved.is_file() and not clip.exists()
    assert event._gateway_pending_stt_text == f'"two cartons"\n[User sent audio: {moved}]'
