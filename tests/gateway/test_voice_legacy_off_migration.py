"""A `/voice off` saved before platform-prefixed voice keys (#12542) survives the upgrade.

The pre-#12542 file holds ``{"<chat_id>": "off"}``. Loading used to skip every unprefixed key,
so the chat lost its auto-TTS suppression and, with ``voice.auto_tts`` on, the next voice note
got a spoken reply the user had turned off.
"""

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gateway.config import Platform
from gateway.platforms.base import BasePlatformAdapter
from tests.gateway.test_voice_command import _make_event, _make_runner


def _restart(tmp_path, *, profile=None):
    runner = _make_runner(tmp_path)
    runner._voice_mode = runner._load_voice_modes()
    adapter = SimpleNamespace(platform=Platform.TELEGRAM, _auto_tts_default=False, _owner_profile=profile,
                              _auto_tts_enabled_chats=set(), _auto_tts_disabled_chats=set())
    with patch("hermes_cli.config.load_config", return_value={"voice": {"auto_tts": True}}):
        runner._sync_voice_mode_state_to_adapter(adapter)
    return runner, adapter


def _speaks(adapter, chat_id):
    return BasePlatformAdapter._should_auto_tts_for_chat(adapter, chat_id)


def test_a_legacy_voice_off_still_silences_auto_tts_after_restart(tmp_path):
    (tmp_path / "gateway_voice_mode.json").write_text(json.dumps({"123": "off", "456": "all"}))

    _, adapter = _restart(tmp_path)

    assert not _speaks(adapter, "123")  # the saved /voice off holds
    assert _speaks(adapter, "456")  # a legacy "all" is still dropped: voice.auto_tts decides
    assert _speaks(adapter, "789")  # untouched chats follow voice.auto_tts
    _, named = _restart(tmp_path, profile="work")  # prefixed keys never existed for named profiles
    assert _speaks(named, "123")


@pytest.mark.asyncio
async def test_an_explicit_voice_command_overrides_the_legacy_off(tmp_path):
    (tmp_path / "gateway_voice_mode.json").write_text(json.dumps({"123": "off"}))
    runner, _ = _restart(tmp_path)

    await runner._handle_voice_command(_make_event("/voice on", chat_id="123"))
    _, adapter = _restart(tmp_path)

    assert _speaks(adapter, "123")


@pytest.mark.parametrize(("platform", "chat_id"), [
    (Platform.MATRIX, "!room:example.org"),  # room ids carry the homeserver after a colon
    (Platform.SIGNAL, "group:aGVsbG8="),  # Signal group chats are "group:<id>"
    (Platform.SLACK, "C0123ABCD"),
    (Platform.TELEGRAM, "-1001234567890"),
])
def test_a_legacy_off_holds_for_every_chat_id_shape(tmp_path, platform, chat_id):
    (tmp_path / "gateway_voice_mode.json").write_text(json.dumps({chat_id: "off"}))
    runner = _make_runner(tmp_path)
    runner._voice_mode = runner._load_voice_modes()
    adapter = SimpleNamespace(platform=platform, _auto_tts_default=False, _owner_profile=None,
                              _auto_tts_enabled_chats=set(), _auto_tts_disabled_chats=set())
    with patch("hermes_cli.config.load_config", return_value={"voice": {"auto_tts": True}}):
        runner._sync_voice_mode_state_to_adapter(adapter)

    assert not _speaks(adapter, chat_id)


def test_a_prefixed_key_for_a_platform_not_registered_yet_survives_loading(tmp_path):
    """A named profile's plugin platform is discovered after the voice file loads: its explicit
    ``<profile>:<platform>:<chat>`` setting must not be mistaken for a legacy key and dropped."""
    (tmp_path / "gateway_voice_mode.json").write_text(json.dumps({"work:notyetloaded:42": "voice_only"}))

    assert _make_runner(tmp_path)._load_voice_modes() == {"work:notyetloaded:42": "voice_only"}
