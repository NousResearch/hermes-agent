"""Malformed idle-channel rules are visible through the normal config check."""

from types import SimpleNamespace

import pytest

from hermes_cli import config as config_module
from hermes_cli.config import validate_config_structure


def test_config_check_reports_invalid_channel(monkeypatch, capsys):
    monkeypatch.setattr(config_module, "check_config_version", lambda **kwargs: (1, 1))
    monkeypatch.setattr(config_module, "load_config", lambda: {
        "compression": {"post_reply_idle": {"channels": [
            {"platform": "signal", "after_seconds": 300}
        ]}}
    })
    with pytest.raises(SystemExit) as exc:
        config_module._cmd_config_check(SimpleNamespace())
    assert exc.value.code == 1
    assert "post_reply_idle.channels[0].chat_id" in capsys.readouterr().out


def test_bad_post_reply_idle_rule_is_a_config_error():
    config = {"compression": {"post_reply_idle": {"channels": [
        {"platform": "signal", "after_seconds": 300}
    ]}}}
    issues = validate_config_structure(config)
    assert any(i.severity == "error" and "post_reply_idle.channels[0].chat_id" in i.message
               for i in issues)


def test_valid_post_reply_idle_rule_has_no_config_error():
    config = {"compression": {"post_reply_idle": {"channels": [
        {"platform": "signal", "chat_id": "group-id", "after_seconds": 300}
    ]}}}
    assert not [i for i in validate_config_structure(config) if i.severity == "error"]
