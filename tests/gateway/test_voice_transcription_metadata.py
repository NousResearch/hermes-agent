"""Gateway voice transcription metadata tests."""

import asyncio
import sys
import types
from types import SimpleNamespace


def test_transcribed_voice_message_includes_cached_audio_path(monkeypatch):
    from gateway.run import GatewayRunner

    fake_tools = types.ModuleType("tools")
    fake_transcription_tools = types.ModuleType("tools.transcription_tools")

    def fake_transcribe_audio(path):
        return {"success": True, "transcript": "hello from audio"}

    fake_transcription_tools.transcribe_audio = fake_transcribe_audio
    monkeypatch.setitem(sys.modules, "tools", fake_tools)
    monkeypatch.setitem(sys.modules, "tools.transcription_tools", fake_transcription_tools)

    runner = object.__new__(GatewayRunner)
    runner.config = SimpleNamespace(stt_enabled=True)

    result = asyncio.run(
        GatewayRunner._enrich_message_with_transcription(
            runner,
            "caption text",
            ["/tmp/voice-note.ogg"],
        )
    )

    assert "Audio file saved at: /tmp/voice-note.ogg." in result
    assert 'Here\'s what they said: "hello from audio"' in result
    assert result.endswith("caption text")
