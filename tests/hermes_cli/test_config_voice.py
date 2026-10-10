"""``voice.submit_mode`` structure check (hermes_cli/config_voice.py, reached via validate_config_structure)."""
from hermes_cli.config import validate_config_structure


def _voice_errors(config):
    return [i for i in validate_config_structure(config) if i.message.startswith("voice.")]


def test_an_unknown_submit_mode_is_an_error():
    assert [i.severity for i in _voice_errors({"voice": {"submit_mode": "shout"}})] == ["error"]


def test_valid_or_absent_submit_modes_pass():
    for config in ({"voice": {"submit_mode": " Draft "}}, {"voice": {"submit_mode": "direct"}}, {"voice": {}}, {}):
        assert _voice_errors(config) == []
