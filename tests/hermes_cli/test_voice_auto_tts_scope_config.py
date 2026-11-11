"""Configuration contract for the gateway's global auto-TTS scope."""

import pytest

from hermes_cli.config import DEFAULT_CONFIG, validate_config_structure


@pytest.mark.parametrize("mode", ["all", "voice_only", " VOICE_ONLY "])
def test_supported_scopes_are_valid(mode):
    issues = validate_config_structure({"voice": {"auto_tts_mode": mode}})
    assert not any("voice.auto_tts_mode" in issue.message for issue in issues)


@pytest.mark.parametrize("mode", ["sometimes", "off", None, True, 1, [], {}])
def test_invalid_scope_is_reported_with_remedy(mode):
    issues = validate_config_structure({"voice": {"auto_tts_mode": mode}})
    assert any(
        issue.severity == "error" and "voice.auto_tts_mode" in issue.message
        and "all" in issue.hint and "voice_only" in issue.hint
        for issue in issues
    )


def test_registered_defaults_keep_auto_tts_off_and_legacy_scope():
    assert DEFAULT_CONFIG["voice"]["auto_tts"] is False
    assert DEFAULT_CONFIG["voice"]["auto_tts_mode"] == "all"
