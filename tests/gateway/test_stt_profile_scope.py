"""A served profile's ``stt.*`` switches govern its own voice notes on a multiplexed gateway.

``GatewayConfig`` is parsed once from the launch profile's ``config.yaml``, while inbound voice is
enriched inside the routed profile's runtime scope (the STT provider already resolves there). The
``stt.echo_transcripts`` / ``stt.enabled`` reads used to come from the launch config, so a served
profile that turned the 🎙️ transcript echo off still got it.
"""

import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from gateway.config import GatewayConfig
from gateway.platforms.helpers import spoken_mention_clips
from gateway.run import GatewayRunner, _profile_runtime_scope


@pytest.fixture
def runner_two_homes(tmp_path, monkeypatch):
    fake_home = tmp_path / "home"
    launch = fake_home / ".hermes"
    served = launch / "profiles" / "secondary"
    served.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
    monkeypatch.setenv("HERMES_HOME", str(launch))
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True, stt_enabled=True, stt_echo_transcripts=True)
    return runner, served


@pytest.mark.asyncio
async def test_served_profile_suppresses_echo_on_normal_and_pending_voice(runner_two_homes, tmp_path, monkeypatch):
    runner, served = runner_two_homes
    (served / "config.yaml").write_text("stt:\n  echo_transcripts: false\n", encoding="utf-8")
    transcript = "hello"
    monkeypatch.setattr("tools.transcription_tools.transcribe_audio", Mock(return_value={
        "success": True, "transcript": transcript, "provider": "test",
    }))
    adapter = object()
    monkeypatch.setattr(runner, "_delivery_adapter_for", lambda source: adapter)
    monkeypatch.setattr(runner, "_thread_metadata_for_source", lambda *args: {})
    monkeypatch.setattr(runner, "_reply_anchor_for_event", lambda event: None)
    echo = AsyncMock()
    monkeypatch.setattr(runner, "_echo_stt_transcripts", echo)
    clip = tmp_path / "voice.ogg"
    clip.write_bytes(b"OggS")
    source = SimpleNamespace(chat_id="test-chat")
    event = SimpleNamespace(source=source)
    with _profile_runtime_scope(served, hydrate_secrets=False):
        assert transcript in await runner._enrich_inbound_voice(event, source, "", [str(clip)])
        await runner._echo_pending_stt_transcripts_once(event, adapter, source, [transcript])
    echo.assert_not_awaited()

    await runner._enrich_inbound_voice(event, source, "", [str(clip)])
    echo.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("filename, content", [
    ("config.yaml", "stt:\n  enabled: false\n"),
    ("gateway.json", '{"stt_enabled": false}'),
])
async def test_served_profile_disables_inbound_and_gate_stt(runner_two_homes, tmp_path, monkeypatch, filename, content):
    runner, served = runner_two_homes
    (served / filename).write_text(content, encoding="utf-8")
    transcribe = Mock(return_value={"success": True, "transcript": "hello", "provider": "test"})
    monkeypatch.setattr("tools.transcription_tools.transcribe_audio", transcribe)
    # Isolate the gateway's gate from the transcription provider's own config check.
    monkeypatch.setattr("tools.transcription_tools.is_stt_enabled", lambda: True)
    monkeypatch.setattr("gateway.run._probe_audio_duration", AsyncMock(return_value=""))
    clip = tmp_path / "voice.ogg"
    clip.write_bytes(b"OggS")
    adapter = SimpleNamespace(gateway_runner=runner)
    with _profile_runtime_scope(served, hydrate_secrets=False):
        text, transcripts = await runner._enrich_message_with_transcription("", [str(clip)])
        assert "voice message" in text and transcripts == []
        assert await spoken_mention_clips([str(clip)], [re.compile("hello")], adapter=adapter) == []
    transcribe.assert_not_called()
