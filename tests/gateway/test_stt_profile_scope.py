"""A served profile's ``stt.*`` switches govern its own voice notes on a multiplexed gateway.

``GatewayConfig`` is parsed once from the launch profile's ``config.yaml``, while inbound voice is
enriched inside the routed profile's runtime scope (the STT provider already resolves there). The
``stt.echo_transcripts`` / ``stt.enabled`` reads used to come from the launch config, so a served
profile that turned the 🎙️ transcript echo off still got it.
"""

from pathlib import Path

import pytest

import gateway.run as gateway_run
from gateway.config import GatewayConfig


def _home(path: Path, stt: str) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.yaml").write_text(f"stt:\n{stt}", encoding="utf-8")
    return path


@pytest.fixture
def runner_two_homes(tmp_path, monkeypatch):
    fake_home = tmp_path / "home"
    launch = _home(fake_home / ".hermes", "  enabled: true\n  echo_transcripts: true\n")
    served = _home(launch / "profiles" / "dad", "  enabled: false\n  echo_transcripts: false\n")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
    monkeypatch.setenv("HERMES_HOME", str(launch))
    runner = gateway_run.GatewayRunner.__new__(gateway_run.GatewayRunner)
    runner.config = GatewayConfig.from_dict(gateway_run._load_gateway_config(launch / "config.yaml"))
    return runner, served


def test_served_profile_turns_off_its_own_transcript_echo(runner_two_homes):
    runner, served = runner_two_homes
    assert runner._should_echo_stt_transcripts() is True  # launch profile: echo on
    with gateway_run._profile_runtime_scope(served, hydrate_secrets=False):
        assert runner._should_echo_stt_transcripts() is False


@pytest.mark.asyncio
async def test_served_profile_stt_disabled_skips_transcription(runner_two_homes, tmp_path, monkeypatch):
    runner, served = runner_two_homes
    runner._has_setup_skill = lambda: False
    calls = []
    monkeypatch.setattr("tools.transcription_tools.transcribe_audio", lambda *a, **k: calls.append(a) or {
        "success": True, "transcript": "hi", "provider": "x"})
    clip = tmp_path / "v.ogg"
    clip.write_bytes(b"OggS")
    with gateway_run._profile_runtime_scope(served, hydrate_secrets=False):
        text, transcripts = await runner._enrich_message_with_transcription("", [str(clip)])
    assert transcripts == [] and not calls and "voice message" in text
