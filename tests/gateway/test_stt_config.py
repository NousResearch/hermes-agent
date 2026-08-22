"""Gateway STT config tests — honor stt.enabled: false from config.yaml."""

from pathlib import Path
from unittest.mock import AsyncMock, call, patch

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
async def test_enrich_message_with_transcription_dedupes_identical_audio_content(tmp_path):
    """Distinct cache paths containing the same audio bytes get one STT call (#91513)."""
    from gateway.run import GatewayRunner

    current_audio = tmp_path / "current.ogg"
    replied_audio = tmp_path / "replied.ogg"
    current_audio.write_bytes(b"same telegram voice bytes")
    replied_audio.write_bytes(b"same telegram voice bytes")

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(stt_enabled=True)
    runner._has_setup_skill = lambda: False

    with patch(
        "tools.transcription_tools.transcribe_audio",
        return_value={"success": True, "transcript": "one transcript", "provider": "local_command"},
    ) as transcribe:
        result, transcripts = await runner._enrich_message_with_transcription(
            "caption", [str(current_audio), str(replied_audio)]
        )

    transcribe.assert_called_once_with(str(current_audio), None, "gateway")
    assert result.count('"one transcript"') == 1
    assert transcripts == ["one transcript"]


@pytest.mark.asyncio
async def test_enrich_message_with_transcription_offloads_content_deduplication(tmp_path):
    """Hashing cached audio must not block the gateway event loop (#91513)."""
    from gateway.run import GatewayRunner, _deduplicate_audio_paths

    audio = tmp_path / "voice.ogg"
    audio.write_bytes(b"telegram voice bytes")

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(stt_enabled=True)
    runner._has_setup_skill = lambda: False

    with (
        patch(
            "gateway.run.asyncio.to_thread",
            new_callable=AsyncMock,
            side_effect=[
                ([str(audio)], []),
                {"success": True, "transcript": "one", "provider": "local_command"},
            ],
        ) as to_thread,
        patch("tools.transcription_tools.transcribe_audio") as transcribe,
    ):
        _result, transcripts = await runner._enrich_message_with_transcription(
            "", [str(audio)]
        )

    assert to_thread.await_args_list[0] == call(_deduplicate_audio_paths, [str(audio)])
    assert to_thread.await_args_list[1] == call(transcribe, str(audio), None, "gateway")
    assert transcripts == ["one"]


@pytest.mark.asyncio
async def test_enrich_message_with_transcription_keeps_distinct_audio_content(tmp_path):
    """Content deduplication must not collapse two genuinely different clips."""
    from gateway.run import GatewayRunner

    first_audio = tmp_path / "first.ogg"
    second_audio = tmp_path / "second.ogg"
    first_audio.write_bytes(b"first telegram voice")
    second_audio.write_bytes(b"second telegram voice")

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(stt_enabled=True)
    runner._has_setup_skill = lambda: False

    with patch(
        "tools.transcription_tools.transcribe_audio",
        side_effect=[
            {"success": True, "transcript": "first", "provider": "local_command"},
            {"success": True, "transcript": "second", "provider": "local_command"},
        ],
    ) as transcribe:
        result, transcripts = await runner._enrich_message_with_transcription(
            "caption", [str(first_audio), str(second_audio)]
        )

    assert transcribe.call_args_list == [
        call(str(first_audio), None, "gateway"),
        call(str(second_audio), None, "gateway"),
    ]
    assert result.count('"first"') == 1
    assert result.count('"second"') == 1
    assert transcripts == ["first", "second"]


@pytest.mark.asyncio
async def test_enrich_message_with_transcription_keeps_distinct_unreadable_paths():
    """A failed identity probe falls back to path identity instead of dropping audio."""
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(stt_enabled=True)
    runner._has_setup_skill = lambda: False

    paths = ["/missing/first.ogg", "/missing/second.ogg"]
    with patch(
        "tools.transcription_tools.transcribe_audio",
        side_effect=[
            {"success": True, "transcript": "first", "provider": "local_command"},
            {"success": True, "transcript": "second", "provider": "local_command"},
        ],
    ) as transcribe:
        _result, transcripts = await runner._enrich_message_with_transcription("", paths)

    assert transcribe.call_args_list == [
        call(paths[0], None, "gateway"),
        call(paths[1], None, "gateway"),
    ]
    assert transcripts == ["first", "second"]


@pytest.mark.asyncio
async def test_enrich_message_with_transcription_dedupes_repeated_unreadable_path():
    """The existing exact-path guard still applies when content cannot be read."""
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(stt_enabled=True)
    runner._has_setup_skill = lambda: False

    path = "/missing/repeated.ogg"
    with patch(
        "tools.transcription_tools.transcribe_audio",
        return_value={"success": True, "transcript": "one", "provider": "local_command"},
    ) as transcribe:
        _result, transcripts = await runner._enrich_message_with_transcription("", [path, path])

    transcribe.assert_called_once_with(path, None, "gateway")
    assert transcripts == ["one"]


