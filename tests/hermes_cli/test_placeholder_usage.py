"""Tests for CLI placeholder text in config/setup output."""

import os
from argparse import Namespace
from pathlib import Path
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


def test_show_config_surfaces_tool_governance_settings(tmp_path, capsys):
    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
        show_config()

    out = capsys.readouterr().out
    assert "Tool Governance" in out
    assert "security.tool_governance.skill_allowed_tools" in out
    assert "security.tool_governance.channel_tool_review" in out


def test_setup_summary_marks_placeholders(tmp_path, capsys):
    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
        _print_setup_summary({"tts": {"provider": "edge"}}, tmp_path)

    out = capsys.readouterr().out
    assert "hermes config set <key> <value>" in out


def test_setup_summary_mentions_tool_governance_examples(tmp_path, capsys):
    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
        _print_setup_summary({"tts": {"provider": "edge"}}, tmp_path)

    out = capsys.readouterr().out
    assert "security.tool_governance.skill_allowed_tools" in out
    assert "security.tool_governance.channel_tool_review" in out


def test_config_check_surfaces_tool_governance_settings(tmp_path, capsys):
    config_path = Path(tmp_path) / "config.yaml"
    config_path.write_text(
        """
security:
  tool_governance:
    skill_allowed_tools: true
    channel_tool_review: false
""".strip()
        + "\n",
        encoding="utf-8",
    )

    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
        config_command(Namespace(config_command="check"))

    out = capsys.readouterr().out
    assert "Tool Governance" in out
    assert "skill_allowed_tools: on" in out
    assert "channel_tool_review: off" in out


def test_config_governance_command_shows_focus_view(tmp_path, capsys):
    config_path = Path(tmp_path) / "config.yaml"
    config_path.write_text(
        """
security:
  tool_governance:
    skill_allowed_tools: false
    channel_tool_review: true
""".strip()
        + "\n",
        encoding="utf-8",
    )

    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
        config_command(Namespace(config_command="governance"))

    out = capsys.readouterr().out
    assert "Tool Governance" in out
    assert "skill_allowed_tools: off" in out
    assert "channel_tool_review: on" in out
    assert "hermes config set security.tool_governance.skill_allowed_tools true" in out
    assert "hermes config set security.tool_governance.channel_tool_review true" in out


def test_config_governance_enable_all_updates_config(tmp_path, capsys):
    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
        config_command(
            Namespace(
                config_command="governance",
                enable_all=True,
                disable_all=False,
                enable_skill_allowed_tools=False,
                disable_skill_allowed_tools=False,
                enable_channel_review=False,
                disable_channel_review=False,
            )
        )

    out = capsys.readouterr().out
    saved = (Path(tmp_path) / "config.yaml").read_text(encoding="utf-8")
    assert "skill_allowed_tools: true" in saved
    assert "channel_tool_review: true" in saved
    assert "skill_allowed_tools: on" in out
    assert "channel_tool_review: on" in out


def test_config_governance_can_toggle_individual_policy(tmp_path, capsys):
    config_path = Path(tmp_path) / "config.yaml"
    config_path.write_text(
        """
security:
  tool_governance:
    skill_allowed_tools: true
    channel_tool_review: true
""".strip()
        + "\n",
        encoding="utf-8",
    )

    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
        config_command(
            Namespace(
                config_command="governance",
                enable_all=False,
                disable_all=False,
                enable_skill_allowed_tools=False,
                disable_skill_allowed_tools=True,
                enable_channel_review=False,
                disable_channel_review=False,
                preset=None,
            )
        )

    out = capsys.readouterr().out
    saved = config_path.read_text(encoding="utf-8")
    assert "skill_allowed_tools: false" in saved
    assert "channel_tool_review: true" in saved
    assert "skill_allowed_tools: off" in out
    assert "channel_tool_review: on" in out


def test_config_governance_preset_messaging_safe_updates_config(tmp_path, capsys):
    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
        config_command(
            Namespace(
                config_command="governance",
                enable_all=False,
                disable_all=False,
                enable_skill_allowed_tools=False,
                disable_skill_allowed_tools=False,
                enable_channel_review=False,
                disable_channel_review=False,
                preset="messaging-safe",
            )
        )

    out = capsys.readouterr().out
    saved = (Path(tmp_path) / "config.yaml").read_text(encoding="utf-8")
    assert "skill_allowed_tools: false" in saved
    assert "channel_tool_review: true" in saved
    assert "skill_allowed_tools: off" in out
    assert "channel_tool_review: on" in out
