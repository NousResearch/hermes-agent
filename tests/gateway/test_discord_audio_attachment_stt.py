"""Regression tests for Discord audio attachments that should be STT'd.

The iPhone/Shortcut ingress path can upload a voice note as a generic
``MessageType.AUDIO`` attachment instead of a native Discord voice message.  In
trusted capture channels, config must be bridged and the gateway must opt those
attachments into transcription so this path does not regress on update.
"""

import logging

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig, load_gateway_config
from gateway.platforms.base import SessionSource
from gateway.run import GatewayRunner


def _discord_source(
    chat_id="voice-channel",
    *,
    platform=Platform.DISCORD,
    parent_chat_id=None,
    thread_id=None,
):
    return SessionSource(
        platform=platform,
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


@pytest.mark.parametrize(
    (
        "source_platform",
        "configured_platform",
        "channels",
        "source_ids",
        "expected",
        "warning_type",
    ),
    [
        (
            Platform.DISCORD,
            Platform.DISCORD,
            ["voice-channel"],
            {"chat_id": "voice-channel"},
            True,
            None,
        ),
        (
            Platform.DISCORD,
            Platform.DISCORD,
            ["voice-channel"],
            {"chat_id": "general"},
            False,
            None,
        ),
        (
            Platform.DISCORD,
            Platform.DISCORD,
            ["parent"],
            {"chat_id": "thread", "parent_chat_id": "parent"},
            True,
            None,
        ),
        (
            Platform.TELEGRAM,
            Platform.TELEGRAM,
            ["thread"],
            {"thread_id": "thread"},
            True,
            None,
        ),
        (Platform.DISCORD, Platform.DISCORD, "ALL", {}, True, None),
        (Platform.TELEGRAM, Platform.TELEGRAM, [123], {"chat_id": "123"}, True, None),
        (Platform.DISCORD, Platform.DISCORD, ["*"], {}, False, None),
        (Platform.DISCORD, Platform.DISCORD, None, {}, False, None),
        (Platform.SLACK, None, None, {}, False, None),
        (
            Platform.DISCORD,
            Platform.DISCORD,
            {"channel": "voice-channel"},
            {},
            False,
            "dict",
        ),
        (
            Platform.DISCORD,
            Platform.DISCORD,
            ["voice-channel", False],
            {},
            False,
            "bool",
        ),
    ],
    ids=[
        "chat",
        "denied-chat",
        "parent",
        "thread",
        "all",
        "integer",
        "asterisk",
        "missing",
        "unsupported-platform",
        "root-type",
        "element-type",
    ],
)
def test_audio_attachment_channel_policy_validates_and_matches(
    source_platform,
    configured_platform,
    channels,
    source_ids,
    expected,
    warning_type,
    caplog,
):
    runner = object.__new__(GatewayRunner)
    platforms = {}
    if configured_platform is not None:
        extra = {}
        if channels is not None:
            extra["transcribe_audio_attachment_channels"] = channels
        platforms[configured_platform] = PlatformConfig(extra=extra)
    runner.config = GatewayConfig(platforms=platforms)
    source = _discord_source(
        platform=source_platform,
        chat_id=source_ids.get("chat_id", "voice-channel"),
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
