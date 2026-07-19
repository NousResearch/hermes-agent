"""Regression tests for Discord audio attachments that should be STT'd.

The iPhone/Shortcut ingress path can upload a voice note as a generic
``MessageType.AUDIO`` attachment instead of a native Discord voice message.  In
trusted capture channels, config must be bridged and the gateway must opt those
attachments into transcription so this path does not regress on update.
"""

from unittest.mock import MagicMock

from gateway.config import GatewayConfig, Platform, PlatformConfig, load_gateway_config
from gateway.platforms.base import SessionSource
from gateway.run import GatewayRunner


def _runner_with_discord_channels(channels):
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={
            Platform.DISCORD: PlatformConfig(
                extra={"transcribe_audio_attachment_channels": channels}
            )
        }
    )
    return runner


def _discord_source(chat_id="voice-channel", *, parent_chat_id=None, thread_id=None):
    return SessionSource(
        platform=Platform.DISCORD,
        chat_id=chat_id,
        user_id="user1",
        parent_chat_id=parent_chat_id,
        thread_id=thread_id,
    )


def test_config_bridges_discord_transcribe_audio_attachment_channels(monkeypatch, tmp_path):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "discord:\n"
        "  transcribe_audio_attachment_channels:\n"
        "    - \"1505333822944972842\"\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    config = load_gateway_config()

    assert config.platforms[Platform.DISCORD].extra[
        "transcribe_audio_attachment_channels"
    ] == ["1505333822944972842"]


def test_configured_discord_audio_attachment_channel_is_transcribed():
    runner = _runner_with_discord_channels(["voice-channel"])

    assert runner._should_transcribe_audio_attachment(
        _discord_source(chat_id="voice-channel")
    ) is True


def test_unconfigured_discord_audio_attachment_channel_remains_file_attachment():
    runner = _runner_with_discord_channels(["voice-channel"])

    assert runner._should_transcribe_audio_attachment(
        _discord_source(chat_id="general")
    ) is False


def test_discord_audio_attachment_thread_matches_parent_channel():
    runner = _runner_with_discord_channels(["voice-channel"])

    assert runner._should_transcribe_audio_attachment(
        _discord_source(
            chat_id="thread-id",
            parent_chat_id="voice-channel",
            thread_id="thread-id",
        )
    ) is True


def test_discord_audio_attachment_all_sentinel_transcribes_any_channel():
    runner = _runner_with_discord_channels(["all"])

    assert runner._should_transcribe_audio_attachment(
        _discord_source(chat_id="any-channel")
    ) is True


def test_missing_discord_audio_attachment_config_defaults_to_file_attachment():
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.DISCORD: PlatformConfig(extra={})}
    )

    assert runner._should_transcribe_audio_attachment(
        _discord_source(chat_id="voice-channel")
    ) is False


def test_non_discord_or_missing_platform_config_defaults_to_file_attachment():
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(platforms={})
    source = SessionSource(
        platform=MagicMock(),
        chat_id="voice-channel",
        user_id="user1",
    )

    assert runner._should_transcribe_audio_attachment(source) is False
