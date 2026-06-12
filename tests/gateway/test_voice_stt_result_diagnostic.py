import logging
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gateway.run import GatewayRunner


PRIVATE_AUDIO_PATH = "/private/cache/SHOULD_NOT_LEAK_AUDIO.ogg"
PRIVATE_TRANSCRIPT = "SHOULD_NOT_LEAK_TRANSCRIPT_BODY"


def _runner(*, stt_enabled=True):
    runner = object.__new__(GatewayRunner)
    setattr(runner, "config", SimpleNamespace(stt_enabled=stt_enabled))
    setattr(runner, "_has_setup_skill", lambda: False)
    return runner


@pytest.mark.asyncio
async def test_stt_success_empty_logs_redacted_diagnostic_and_no_useful_transcript(caplog):
    runner = _runner()

    with patch(
        "tools.transcription_tools.transcribe_audio",
        return_value={"success": True, "transcript": "   "},
    ):
        with caplog.at_level(logging.INFO, logger="gateway.run"):
            enriched, transcripts = await runner._enrich_message_with_transcription(
                "",
                [PRIVATE_AUDIO_PATH],
            )

    assert transcripts == []
    assert "speech-to-text did not produce usable transcript text" in enriched
    output = caplog.text
    assert "Telegram voice STT result diagnostic" in output
    assert "stt_attempted=yes" in output
    assert "stt_provider_available=yes" in output
    assert "stt_result_class=success_empty" in output
    assert "transcript_non_empty=no" in output
    assert "transcript_length_bucket=0" in output
    assert PRIVATE_AUDIO_PATH not in output
    assert PRIVATE_TRANSCRIPT not in output


@pytest.mark.asyncio
async def test_stt_success_punctuation_placeholder_treated_as_empty(caplog):
    runner = _runner()

    with patch(
        "tools.transcription_tools.transcribe_audio",
        return_value={"success": True, "transcript": "... ... ..."},
    ):
        with caplog.at_level(logging.INFO, logger="gateway.run"):
            enriched, transcripts = await runner._enrich_message_with_transcription(
                "",
                [PRIVATE_AUDIO_PATH],
            )

    assert transcripts == []
    assert "Here's what they said" not in enriched
    assert "speech-to-text did not produce usable transcript text" in enriched
    output = caplog.text
    assert "stt_result_class=success_empty" in output
    assert "transcript_non_empty=no" in output
    assert PRIVATE_AUDIO_PATH not in output


@pytest.mark.asyncio
async def test_stt_success_non_empty_logs_redacted_diagnostic(caplog):
    runner = _runner()

    with patch(
        "tools.transcription_tools.transcribe_audio",
        return_value={"success": True, "transcript": PRIVATE_TRANSCRIPT},
    ):
        with caplog.at_level(logging.INFO, logger="gateway.run"):
            enriched, transcripts = await runner._enrich_message_with_transcription(
                "",
                [PRIVATE_AUDIO_PATH],
            )

    assert transcripts == [PRIVATE_TRANSCRIPT]
    assert PRIVATE_TRANSCRIPT in enriched
    output = caplog.text
    assert "Telegram voice STT result diagnostic" in output
    assert "stt_result_class=success_non_empty" in output
    assert "transcript_non_empty=yes" in output
    assert "transcript_length_bucket=21-100" in output
    assert PRIVATE_AUDIO_PATH not in output
    # The transcript may appear in enriched text, but must never be logged.
    assert PRIVATE_TRANSCRIPT not in output


@pytest.mark.asyncio
async def test_stt_provider_unavailable_logs_redacted_class(caplog):
    runner = _runner()

    with patch(
        "tools.transcription_tools.transcribe_audio",
        return_value={"success": False, "error": "No STT provider configured"},
    ):
        with caplog.at_level(logging.INFO, logger="gateway.run"):
            enriched, transcripts = await runner._enrich_message_with_transcription(
                "",
                [PRIVATE_AUDIO_PATH],
            )

    assert transcripts == []
    assert "no STT provider is configured" in enriched
    output = caplog.text
    assert "Telegram voice STT result diagnostic" in output
    assert "stt_attempted=yes" in output
    assert "stt_provider_available=no" in output
    assert "stt_result_class=provider_unavailable" in output
    assert "transcript_non_empty=no" in output
    assert PRIVATE_AUDIO_PATH not in output
    assert "No STT provider configured" not in output
