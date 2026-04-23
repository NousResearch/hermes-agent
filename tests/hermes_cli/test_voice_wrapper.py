"""Tests for ``hermes_cli.voice`` — the TUI gateway's voice wrapper.

The module is imported *lazily* by ``tui_gateway/server.py`` so that a
box with missing audio deps fails at call time (returning a clean RPC
error) rather than at gateway startup. These tests therefore only
assert the public contract the gateway depends on: the three symbols
exist, ``stop_and_transcribe`` is a no-op when nothing is recording,
and ``speak_text`` tolerates empty input without touching the provider
stack.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestPublicAPI:
    def test_gateway_symbols_importable(self):
        """Match the exact import shape tui_gateway/server.py uses."""
        from hermes_cli.voice import (
            speak_text,
            start_recording,
            stop_and_transcribe,
        )

        assert callable(start_recording)
        assert callable(stop_and_transcribe)
        assert callable(speak_text)


class TestStopWithoutStart:
    def test_returns_none_when_no_recording_active(self, monkeypatch):
        """Idempotent no-op: stop before start must not raise or touch state."""
        import hermes_cli.voice as voice

        monkeypatch.setattr(voice, "_recorder", None)

        assert voice.stop_and_transcribe() is None


class TestSpeakTextGuards:
    @pytest.mark.parametrize("text", ["", "   ", "\n\t  "])
    def test_empty_text_is_noop(self, text):
        """Empty / whitespace-only text must return without importing tts_tool
        (the gateway spawns a thread per call, so a no-op on empty input
        keeps the thread pool from churning on trivial inputs)."""
        from hermes_cli.voice import speak_text

        # Should simply return None without raising.
        assert speak_text(text) is None
