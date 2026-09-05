"""Regression tests for Discord audio attachments that should be STT'd.

The iPhone/Shortcut ingress path can upload a voice note as a generic
``MessageType.AUDIO`` attachment instead of a native Discord voice message.  In
trusted capture channels, config must be bridged and the gateway must opt those
attachments into transcription so this path does not regress on update.
"""

import logging
from unittest.mock import MagicMock

import pytest

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


@pytest.mark.parametrize(
    ("platform", "channels", "source_ids", "expected", "warning_type"),
    [
        (Platform.DISCORD, "ALL", {}, True, None),
        (Platform.TELEGRAM, [123], {"chat_id": "123"}, True, None),
        (Platform.DISCORD, ["parent"], {"parent_chat_id": "parent"}, True, None),
        (Platform.TELEGRAM, ["thread"], {"thread_id": "thread"}, True, None),
        (Platform.DISCORD, ["*"], {}, False, None),
        (Platform.DISCORD, {"channel": "voice-channel"}, {}, False, "dict"),
        (Platform.DISCORD, ["voice-channel", False], {}, False, "bool"),
    ],
    ids=["all", "integer", "parent", "thread", "asterisk", "root-type", "element-type"],
)
def test_audio_attachment_channel_policy_validates_and_matches(
    platform, channels, source_ids, expected, warning_type, caplog
):
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={
            platform: PlatformConfig(
                extra={"transcribe_audio_attachment_channels": channels}
            )
        }
    )
    source = SessionSource(
        platform=platform,
        chat_id=source_ids.get("chat_id", "voice-channel"),
        user_id="user1",
        parent_chat_id=source_ids.get("parent_chat_id"),
        thread_id=source_ids.get("thread_id"),
    )

    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        result = runner._should_transcribe_audio_attachment(source)

    assert result is expected
    if warning_type:
        assert any(
            "transcribe_audio_attachment_channels" in record.message
            and warning_type in record.message
            for record in caplog.records
        )
    else:
        assert not caplog.records
