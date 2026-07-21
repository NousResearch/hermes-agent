"""Tests for CLI placeholder text in config/setup output."""

import os
from argparse import Namespace
from unittest.mock import patch

import pytest

from hermes_cli.config import config_command, show_config
from hermes_cli.setup import _print_setup_summary


def test_config_set_usage_marks_placeholders(capsys):
    args = Namespace(config_command="set", key=None, value=None)

    with pytest.raises(SystemExit) as exc:
        config_command(args)

    assert exc.value.code == 1
    out = capsys.readouterr().out
    # The usage line documents the optional --force flag added in #34067
    # (schema validation for unknown keys). Placeholder convention is preserved:
    # the literal ``<key>`` and ``<value>`` markers must still be present so
    # downstream tooling can detect placeholder syntax.
    assert "Usage: hermes config set" in out
    assert "<key>" in out
    assert "<value>" in out
    # --force escape hatch must be documented in the usage line.
    assert "--force" in out


def test_config_unknown_command_help_marks_placeholders(capsys):
    args = Namespace(config_command="wat")

    with pytest.raises(SystemExit) as exc:
        config_command(args)

    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "hermes config set <key> <value>   Set a config value" in out


def test_show_config_marks_placeholders(tmp_path, capsys):
    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
        show_config()

    out = capsys.readouterr().out
    assert "hermes config set <key> <value>" in out


def test_setup_summary_marks_placeholders(tmp_path, capsys):
    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
        _print_setup_summary({"tts": {"provider": "edge"}}, tmp_path)

    out = capsys.readouterr().out
    assert "hermes config set <key> <value>" in out

def test_show_config_displays_all_display_keys(tmp_path, capsys):
    """Verify show_config() renders tool_progress, skin, and show_cost."""
    import yaml
    from hermes_cli.config import _LOAD_CONFIG_CACHE

    config_dir = tmp_path / ".hermes"
    config_dir.mkdir()
    config_file = config_dir / "config.yaml"
    config_file.write_text(yaml.safe_dump({
        "display": {
            "tool_progress": "verbose",
            "skin": "cyberpunk",
            "show_cost": True,
            "show_reasoning": True,
        }
    }, sort_keys=False), encoding="utf-8")

    # Clear config cache to ensure fresh load
    _LOAD_CONFIG_CACHE.clear()

    with patch.dict(os.environ, {"HERMES_HOME": str(config_dir)}):
        show_config()

    out = capsys.readouterr().out
    assert "Tool progress:" in out
    assert "Skin:" in out
    assert "Cost:" in out
    # Use non-default values to verify config is actually read
    assert "verbose" in out
    assert "cyberpunk" in out
