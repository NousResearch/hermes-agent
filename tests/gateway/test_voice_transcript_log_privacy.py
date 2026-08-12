"""Privacy regressions for Discord voice transcript handling in the gateway."""

from types import SimpleNamespace

import pytest

from gateway.config import Platform
from gateway.run import GatewayRunner


@pytest.mark.asyncio
async def test_duplicate_voice_transcript_log_contains_metadata_only(caplog):
    runner = object.__new__(GatewayRunner)
    runner.adapters = {
        Platform.DISCORD: SimpleNamespace(
            _voice_text_channels={111: 222},
            _voice_sources={},
        )
    }
    runner._recent_voice_transcripts = {}
    runner._is_user_authorized = lambda _source: True
    transcript = "duplicate private voice transcript canary"
    caplog.set_level("INFO", logger="gateway.run")

    assert runner._is_duplicate_voice_transcript(111, 42, transcript) is False
    await runner._handle_voice_channel_input(111, 42, transcript)

    assert "Suppressing duplicate voice transcript guild=111 user=42" in caplog.text
    assert f"transcript_chars={len(transcript)}" in caplog.text
    assert transcript not in caplog.text
