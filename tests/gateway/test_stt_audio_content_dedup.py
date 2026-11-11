"""One clip that reaches a turn under two cache paths (Telegram: the current voice plus
``reply_to_message.voice``) is transcribed once; genuinely different clips stay distinct."""

from unittest.mock import call, patch

import pytest

from gateway.config import GatewayConfig


def _runner():
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(stt_enabled=True)
    runner._has_setup_skill = lambda: False
    return runner


@pytest.mark.asyncio
async def test_same_audio_under_two_paths_is_transcribed_once(tmp_path):
    current, replied = tmp_path / "audio_A.ogg", tmp_path / "audio_B.ogg"
    current.write_bytes(b"same telegram voice bytes")
    replied.write_bytes(b"same telegram voice bytes")

    with patch(
        "tools.transcription_tools.transcribe_audio",
        return_value={"success": True, "transcript": "one transcript", "provider": "local_command"},
    ) as transcribe:
        text, transcripts = await _runner()._enrich_message_with_transcription(
            "caption", [str(current), str(replied)])

    transcribe.assert_called_once_with(str(current), None, "gateway")
    assert text.count('"one transcript"') == 1 and text.endswith("caption")
    assert transcripts == ["one transcript"]


@pytest.mark.asyncio
async def test_distinct_and_unreadable_audio_keep_their_own_transcription(tmp_path):
    first, second = tmp_path / "first.ogg", tmp_path / "second.ogg"
    first.write_bytes(b"first telegram voice")  # same size as second, different bytes
    second.write_bytes(b"other telegram voice")
    paths = [str(first), str(second), "/missing/a.ogg", "/missing/b.ogg", str(first)]

    with patch(
        "tools.transcription_tools.transcribe_audio",
        side_effect=[{"success": True, "transcript": t, "provider": "local_command"}
                     for t in ("first", "second", "gone-a", "gone-b")],
    ) as transcribe:
        _text, transcripts = await _runner()._enrich_message_with_transcription("", paths)

    assert transcribe.call_args_list == [call(p, None, "gateway") for p in paths[:4]]
    assert transcripts == ["first", "second", "gone-a", "gone-b"]
