"""#105315 — Telegram voice messages cached but never auto-transcribed.

Local repro with fakes only: no Telegram, no real audio, no STT model. The gateway's
automatic transcription runs as one blocking call off-loaded with
``asyncio.to_thread``; when that call never returns (the reporter's faster-whisper /
CUDA hang) the inbound turn — and the platform typing indicator — stays alive
forever, because:

  * the "transcribing" marker was logged at DEBUG only (default INFO logs show
    nothing after "Cached user voice"), and
  * no timeout bounded the call, so every failure branch that would have produced
    the neutral ``[voice message could not be transcribed automatically; ...]``
    note was never reached.
"""

import asyncio
import logging
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gateway.run import GatewayRunner  # noqa: E402


VOICE = "/tmp/hermes-cache/audio/audio_3f20dbc49b73.ogg"


def _runner():
    runner = object.__new__(GatewayRunner)
    runner.config = SimpleNamespace(stt_enabled=True, stt_echo_transcripts=True)
    return runner


@pytest.mark.asyncio
async def test_hanging_stt_call_is_bounded_and_marked(monkeypatch, caplog):
    """A wedged STT call must not stall the turn: bounded, logged, and marked."""
    runner = _runner()
    entered = threading.Event()
    release = threading.Event()
    fallback_calls = []

    def hanging_transcribe(path, model=None, source=None):
        entered.set()
        # The gateway cannot kill the worker thread; bounded so the test process can exit.
        release.wait(15)
        return {"success": True, "transcript": "ghost", "provider": "fake"}

    def fallback(path, model=None):
        fallback_calls.append(path)
        return {"success": False, "error": "no local backend"}

    monkeypatch.setattr(
        "gateway.run_inbound._auto_stt_timeout_seconds", lambda: 0.3, raising=False
    )

    with patch("tools.transcription_tools.transcribe_audio", hanging_transcribe), patch(
        "tools.transcription_tools.transcribe_audio_local_fallback", fallback
    ):
        try:
            with caplog.at_level(logging.INFO, logger="gateway.run"):
                enriched, transcripts = await asyncio.wait_for(
                    runner._enrich_message_with_transcription("", [VOICE]), timeout=8
                )
        finally:
            release.set()

    assert entered.is_set(), "the fake STT call was never entered"
    assert transcripts == []
    # The agent still gets a turn: a neutral marker instead of an endless "typing...".
    assert "could not be transcribed automatically" in enriched
    # A timed-out backend is abandoned, not stacked behind a second (also wedged) backend.
    assert fallback_calls == []
    assert any(
        record.levelno >= logging.ERROR and "did not return within" in record.getMessage()
        for record in caplog.records
    ), [record.getMessage() for record in caplog.records]


@pytest.mark.asyncio
async def test_auto_transcription_start_is_logged_at_info(caplog):
    """Operators must be able to tell 'STT is wedged' from 'STT never ran'."""

    def ok_transcribe(path, model=None, source=None):
        return {"success": True, "transcript": "مرحبا", "provider": "fake"}

    with patch("tools.transcription_tools.transcribe_audio", ok_transcribe):
        with caplog.at_level(logging.INFO, logger="gateway.run"):
            enriched, transcripts = await asyncio.wait_for(
                _runner()._enrich_message_with_transcription("", [VOICE]), timeout=8
            )

    assert transcripts == ["مرحبا"]
    assert enriched == '"مرحبا"'
    assert any(
        record.levelno == logging.INFO
        and "Auto-transcribing inbound voice" in record.getMessage()
        and VOICE in record.getMessage()
        for record in caplog.records
    ), [record.getMessage() for record in caplog.records]


@pytest.mark.asyncio
async def test_raising_provider_still_recovers_with_local_stt(caplog):
    """A provider that *raises* must not skip the local recovery path it was designed to have."""

    def raising_transcribe(path, model=None, source=None):
        raise RuntimeError("cuda lib load failed")

    def local_ok(path, model=None):
        return {"success": True, "transcript": "hello from local", "provider": "local"}

    with patch("tools.transcription_tools.transcribe_audio", raising_transcribe), patch(
        "tools.transcription_tools.transcribe_audio_local_fallback", local_ok
    ):
        with caplog.at_level(logging.INFO, logger="gateway.run"):
            enriched, transcripts = await asyncio.wait_for(
                _runner()._enrich_message_with_transcription("", [VOICE]), timeout=8
            )

    assert transcripts == ["hello from local"]
    assert enriched == '"hello from local"'
    assert not any("could not be transcribed automatically" in record.getMessage() for record in caplog.records)


def test_timeout_seconds_reads_config_with_a_safe_default(monkeypatch):
    """``stt.timeout_seconds`` tunes the bound; a missing/unparsable/zero value keeps the default."""
    from gateway import run_inbound

    monkeypatch.setattr("tools.transcription_tools._load_stt_config", lambda: {"timeout_seconds": 42})
    assert run_inbound._auto_stt_timeout_seconds() == 42.0

    monkeypatch.setattr("tools.transcription_tools._load_stt_config", lambda: {"timeout_seconds": "nope"})
    assert run_inbound._auto_stt_timeout_seconds() == run_inbound._AUTO_STT_TIMEOUT_SECONDS

    monkeypatch.setattr("tools.transcription_tools._load_stt_config", lambda: {"timeout_seconds": 0})
    assert run_inbound._auto_stt_timeout_seconds() == run_inbound._AUTO_STT_TIMEOUT_SECONDS

    monkeypatch.setattr("tools.transcription_tools._load_stt_config", lambda: {})
    assert run_inbound._auto_stt_timeout_seconds() == run_inbound._AUTO_STT_TIMEOUT_SECONDS
