"""Regression: the CLI voice-mode TTS gate must not lock out providers without a chunked
streamer (Piper, edge).

``_speak_streaming`` gated on ``resolve_streaming_provider(cfg) is None`` → whole-file
synthesis. But Piper/edge have no chunked *streamer* by design (no chunked-PCM API), while
``stream_tts_to_speaker``'s own dispatcher already speaks them per-sentence via
``_SyncSentencePipeline``. The gate therefore sent exactly those providers down the
whole-file path — the entire reply synthesized before the first byte played. The fix gates
on provider runnability (``check_tts_requirements``) and lets the dispatcher choose.
"""

from __future__ import annotations

import queue

import pytest


@pytest.fixture
def piper_config(monkeypatch):
    """A resolved provider with NO registered chunked streamer (the Piper shape)."""
    import tools.tts_streaming as tts_streaming
    import tools.tts_tool as tts_tool

    cfg = {"provider": "piper", "piper": {"voice": "en_GB-vctk-medium"}}
    monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: dict(cfg))
    return cfg


def test_piper_has_no_chunked_streamer_but_speaks_per_sentence(piper_config):
    """The premise of the bug: resolve_streaming_provider(piper) is None, yet the speaker
    dispatcher's sync fallback exists for exactly that case."""
    from tools.tts_streaming import resolve_streaming_provider

    assert resolve_streaming_provider(piper_config) is None


def test_speak_streaming_routes_provider_without_streamer(piper_config, monkeypatch):
    """A runnable provider with no chunked streamer must still route through the
    per-sentence pipeline (audio starts on sentence one), not fall back to whole-file."""
    import tools.tts_tool as tts_tool
    import tools.tts_tool_speaker as tts_tool_speaker
    from hermes_cli.voice import _speak_streaming

    seen: dict = {}

    def fake_stream(text_queue, stop_event, done_event):
        item = text_queue.get()
        done = text_queue.get()
        seen["text"] = item
        seen["sentinel"] = done
        seen["queue"] = text_queue
        done_event.set()

    monkeypatch.setattr(tts_tool, "check_tts_requirements", lambda: True)
    monkeypatch.setattr(tts_tool_speaker, "stream_tts_to_speaker", fake_stream)

    assert _speak_streaming("First sentence. Second sentence.", None) is True
    assert seen["text"] == "First sentence. Second sentence."
    assert seen["sentinel"] is None  # end-of-text sentinel delivered


def test_speak_streaming_defers_when_provider_cannot_run(monkeypatch):
    """An unrunnable provider still defers to the caller's whole-file path (unchanged)."""
    import tools.tts_tool as tts_tool
    import tools.tts_tool_speaker as tts_tool_speaker
    from hermes_cli.voice import _speak_streaming

    monkeypatch.setattr(tts_tool, "check_tts_requirements", lambda: False)

    def boom(*a, **k):  # pragma: no cover - must not be reached
        raise AssertionError("must not stream when the provider cannot run")

    monkeypatch.setattr(tts_tool_speaker, "stream_tts_to_speaker", boom)

    assert _speak_streaming("hello", None) is False
