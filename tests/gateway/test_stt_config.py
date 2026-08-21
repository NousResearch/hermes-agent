"""Gateway STT config tests — honor stt.enabled: false from config.yaml."""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import yaml

from gateway.config import GatewayConfig, Platform, load_gateway_config
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource


def test_gateway_config_stt_disabled_from_dict_nested():
    config = GatewayConfig.from_dict({"stt": {"enabled": False}})
    assert config.stt_enabled is False


def test_load_gateway_config_bridges_stt_enabled_from_config_yaml(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        yaml.dump({"stt": {"enabled": False}}),
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    config = load_gateway_config()

    assert config.stt_enabled is False


@pytest.mark.asyncio
async def test_enrich_message_with_transcription_returns_tuple_for_empty_content_placeholder():
    """A successful transcription whose caption is the empty-content placeholder
    must still return the ``(text, transcripts)`` tuple.

    The Discord adapter delivers a captionless voice note as the literal
    ``"(The user sent a message with no text content)"`` placeholder. When STT
    succeeds we strip that redundant placeholder and return just the transcript
    prefix — but the method's contract (and every caller, which unpacks the
    result as ``text, transcripts = ...``) requires a 2-tuple. Returning a bare
    string here raised ``ValueError: too many values to unpack`` and dropped the
    whole voice message on the floor.
    """
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(stt_enabled=True)
    runner._has_setup_skill = lambda: False

    with patch(
        "tools.transcription_tools.transcribe_audio",
        return_value={
            "success": True,
            "transcript": "hello from a captionless voice note",
            "provider": "local_command",
        },
    ):
        result, transcripts = await runner._enrich_message_with_transcription(
            "(The user sent a message with no text content)",
            ["/tmp/voice.ogg"],
        )

    # The redundant placeholder is stripped, leaving only the transcript prefix.
    assert "hello from a captionless voice note" in result
    assert "(The user sent a message with no text content)" not in result
    # Crucially, the transcripts are still surfaced so callers can echo them.
    assert transcripts == ["hello from a captionless voice note"]


@pytest.mark.asyncio
async def test_enrich_message_with_transcription_guards_empty_transcript():
    """success=True with an empty/whitespace transcript must not emit empty
    quotes — it gets a sentinel note and is excluded from transcripts (#41603)."""
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(stt_enabled=True)
    runner._has_setup_skill = lambda: False

    with patch(
        "tools.transcription_tools.transcribe_audio",
        return_value={"success": True, "transcript": "   \n\t", "provider": "local_command"},
    ):
        result, transcripts = await runner._enrich_message_with_transcription(
            "caption",
            ["/tmp/voice.ogg"],
        )

    assert "empty or inaudible" in result
    assert '""' not in result
    assert transcripts == []


@pytest.mark.asyncio
async def test_successful_transcription_keeps_the_audio_path():
    """A SUCCESSFUL transcription must still record where the audio is.

    ``content`` is the only record of an attachment's path — the messages table
    has no media column, and consumers recover attachments by parsing these
    markers back out of the text. Every failure branch keeps the path; success
    used to drop it, so the better STT got, the more recordings became
    unreachable even though the bytes were still in the cache.
    """
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(stt_enabled=True)
    runner._has_setup_skill = lambda: False

    with patch(
        "tools.transcription_tools.transcribe_audio",
        return_value={
            "success": True,
            "transcript": "two cartons please",
            "provider": "local",
        },
    ):
        result, transcripts = await runner._enrich_message_with_transcription(
            "",
            ["/tmp/cache/audio/aud_abc123.ogg"],
        )

    # The transcript is still the message the model reads.
    assert '"two cartons please"' in result
    assert transcripts == ["two cartons please"]
    # ...and the recording is still locatable.
    assert "aud_abc123.ogg" in result


@pytest.mark.asyncio
async def test_successful_transcription_uses_the_media_placeholder_grammar():
    """The marker must be the one media consumers actually parse.

    ``_build_media_placeholder`` emits ``[User sent audio: <path>]`` and
    downstream consumers match on that. The stt-disabled branch emits prose
    ("[The user sent a voice message: ...]") which no media parser matches — so
    emitting the prose form here would put the path in front of the model while
    leaving the attachment unrecoverable, a fix that looks right in the diff and
    changes nothing for the consumer. This pins the grammar, not just the path.
    """
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(stt_enabled=True)
    runner._has_setup_skill = lambda: False

    with patch(
        "tools.transcription_tools.transcribe_audio",
        return_value={"success": True, "transcript": "hello", "provider": "local"},
    ):
        result, _ = await runner._enrich_message_with_transcription(
            "",
            ["/tmp/cache/audio/aud_abc123.ogg"],
        )

    assert "[User sent audio: /tmp/cache/audio/aud_abc123.ogg]" in result
    assert "The user sent a voice message" not in result


@pytest.mark.asyncio
async def test_transcript_precedes_the_media_marker():
    """Order matters: the transcript is the message, the marker is metadata.

    #41603 moved to a bare quoted transcript so the model replies to the words
    instead of narrating that a voice message arrived. Putting the marker first
    would reintroduce exactly that — the model leads with the attachment rather
    than the content — so the transcript stays on the first line.
    """
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(stt_enabled=True)
    runner._has_setup_skill = lambda: False

    with patch(
        "tools.transcription_tools.transcribe_audio",
        return_value={"success": True, "transcript": "send fifty", "provider": "local"},
    ):
        result, _ = await runner._enrich_message_with_transcription(
            "",
            ["/tmp/cache/audio/aud_abc123.ogg"],
        )

    assert result.index('"send fifty"') < result.index("[User sent audio:")


@pytest.mark.asyncio
async def test_empty_transcript_still_emits_no_media_marker():
    """The empty/inaudible sentinel (#41603) is unchanged by this fix.

    That branch deliberately does NOT surface a transcript, and it `continue`s
    before the append — so it must not start emitting a media marker either, or
    a silent clip would look to a consumer like a normal voice note.
    """
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(stt_enabled=True)
    runner._has_setup_skill = lambda: False

    with patch(
        "tools.transcription_tools.transcribe_audio",
        return_value={"success": True, "transcript": "   ", "provider": "local"},
    ):
        result, transcripts = await runner._enrich_message_with_transcription(
            "caption",
            ["/tmp/cache/audio/aud_abc123.ogg"],
        )

    assert "empty or inaudible" in result
    assert "[User sent audio:" not in result
    assert transcripts == []


@pytest.mark.asyncio
async def test_each_clip_in_a_batch_keeps_its_own_path():
    """Several voice notes in one turn must each stay individually locatable.

    A shared or last-wins marker would silently collapse a multi-clip turn to
    one recoverable attachment, which is the same data loss one layer in.
    """
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(stt_enabled=True)
    runner._has_setup_skill = lambda: False

    transcripts_by_path = {
        "/tmp/cache/audio/aud_one.ogg": "first clip",
        "/tmp/cache/audio/aud_two.ogg": "second clip",
    }

    def fake_transcribe(path, *args, **kwargs):
        return {
            "success": True,
            "transcript": transcripts_by_path[path],
            "provider": "local",
        }

    with patch("tools.transcription_tools.transcribe_audio", side_effect=fake_transcribe):
        result, transcripts = await runner._enrich_message_with_transcription(
            "",
            list(transcripts_by_path),
        )

    assert "[User sent audio: /tmp/cache/audio/aud_one.ogg]" in result
    assert "[User sent audio: /tmp/cache/audio/aud_two.ogg]" in result
    assert transcripts == ["first clip", "second clip"]
