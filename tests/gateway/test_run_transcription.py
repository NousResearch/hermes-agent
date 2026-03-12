"""Tests for gateway transcription message enrichment."""

import asyncio
import importlib.util
import sys
import types
from pathlib import Path

from gateway.run import GatewayRunner

MODULE_PATH = Path(__file__).resolve().parents[2] / "tools" / "transcription_tools.py"
SPEC = importlib.util.spec_from_file_location("test_transcription_tools_module", MODULE_PATH)
tt = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(tt)


def _install_fake_tools_package(monkeypatch):
    tools_pkg = types.ModuleType("tools")
    tools_pkg.transcription_tools = tt
    monkeypatch.setitem(sys.modules, "tools", tools_pkg)
    monkeypatch.setitem(sys.modules, "tools.transcription_tools", tt)


class TestGatewayTranscriptionFormatting:
    def test_successful_transcription_uses_lightweight_note(self, monkeypatch):
        _install_fake_tools_package(monkeypatch)

        def fake_transcribe_audio(_path):
            return {"success": True, "transcript": "hello from audio"}

        monkeypatch.setattr(tt, "transcribe_audio", fake_transcribe_audio)

        runner = GatewayRunner.__new__(GatewayRunner)
        result = asyncio.run(
            runner._enrich_message_with_transcription(
                "caption text",
                ["/tmp/audio.ogg"],
            )
        )

        assert "// Transcript of the user's audio message:" in result
        assert "hello from audio" in result
        assert result.endswith("caption text")

    def test_failed_transcription_uses_single_failure_comment(self, monkeypatch):
        _install_fake_tools_package(monkeypatch)

        def fake_transcribe_audio(_path):
            return {"success": False, "error": "whisper.cpp failed"}

        monkeypatch.setattr(tt, "transcribe_audio", fake_transcribe_audio)

        runner = GatewayRunner.__new__(GatewayRunner)
        result = asyncio.run(
            runner._enrich_message_with_transcription(
                "",
                ["/tmp/audio.ogg"],
            )
        )

        assert result == "// The user sent an audio message that couldn't get transcribed"
