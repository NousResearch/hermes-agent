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
    assert "Usage: hermes config set <key> <value>" in out


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


def test_show_config_handles_null_terminal_section(tmp_path, capsys):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("terminal: null\n", encoding="utf-8")

    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
        show_config()

    out = capsys.readouterr().out
    assert "◆ Terminal" in out
    assert "Backend:" in out
    assert "local" in out
    assert "Working dir:" in out


def test_show_config_handles_partial_terminal_section(tmp_path, capsys):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "terminal:\n  backend: local\n",
        encoding="utf-8",
    )

    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
        show_config()

    out = capsys.readouterr().out
    assert "◆ Terminal" in out
    assert "Backend:      local" in out
    assert "Timeout:      180s" in out


def test_setup_summary_marks_placeholders(tmp_path, capsys):
    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
        _print_setup_summary({"tts": {"provider": "edge"}}, tmp_path)

    out = capsys.readouterr().out
    assert "hermes config set <key> <value>" in out
