"""``discord.voice_silence_threshold_seconds``: silence that ends a spoken utterance in a voice channel.

It was hard-coded at 1.5 s, and that wait sits on the critical path of every voice reply (end of speech
-> transcription -> agent). It's now configurable (read on each join) and clamped to 0.3-5 s.
"""

from unittest.mock import MagicMock

import pytest


def _adapter():
    from plugins.platforms.discord.adapter import DiscordAdapter
    return object.__new__(DiscordAdapter)


def test_receiver_default_and_override():
    from plugins.platforms.discord.adapter import VoiceReceiver
    assert VoiceReceiver(MagicMock()).SILENCE_THRESHOLD == VoiceReceiver.SILENCE_THRESHOLD == 1.5
    assert VoiceReceiver(MagicMock(), silence_threshold=0.8).SILENCE_THRESHOLD == 0.8
    assert VoiceReceiver.SILENCE_THRESHOLD == 1.5  # the override is per instance


@pytest.mark.parametrize("raw,expected", [
    (None, 1.5), (1.0, 1.0), ("0.8", 0.8), (0.05, 0.3), (60, 5.0), ("junk", 1.5),
])
def test_config_value_is_clamped(monkeypatch, raw, expected):
    import hermes_cli.config as config
    discord_cfg = {} if raw is None else {"voice_silence_threshold_seconds": raw}
    monkeypatch.setattr(config, "read_raw_config", lambda: {"discord": discord_cfg})
    assert _adapter()._load_voice_silence_threshold() == expected


def test_missing_discord_section_uses_default(monkeypatch):
    import hermes_cli.config as config
    monkeypatch.setattr(config, "read_raw_config", lambda: {})
    assert _adapter()._load_voice_silence_threshold() == 1.5
