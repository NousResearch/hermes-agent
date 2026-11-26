"""False and 0 skill config values are shown, not treated as unset."""

import os
from pathlib import Path

import pytest

_SKILL = """---
name: deploy
description: Deploy helper.
metadata:
  hermes:
    config:
      - {key: deploy.dry_run, description: Dry run, default: true}
      - {key: deploy.retries, description: Retries, default: 3}
      - {key: deploy.note, description: Note}
      - {key: deploy.label, description: Label}
---
Deploy.
"""
_CONFIG = "skills:\n  config:\n    deploy:\n"
_CONFIG += "      dry_run: false\n      retries: 0\n"
_CONFIG += '      note: ""\n'


@pytest.fixture
def planted(_hermetic_environment):
    home = Path(os.environ["HERMES_HOME"])
    skill = home / "skills" / "deploy"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(_SKILL, encoding="utf-8")
    (home / "config.yaml").write_text(_CONFIG, encoding="utf-8")
    from agent.skill_utils import _raw_config_cache_clear
    from tools.skills_tool import clear_skills_cache

    _raw_config_cache_clear()
    clear_skills_cache()
    yield
    _raw_config_cache_clear()
    clear_skills_cache()


def test_invocation_shows_false_zero_and_blank(planted):
    from agent.skill_commands import build_skill_invocation_message

    message = build_skill_invocation_message("/deploy")
    assert "deploy.dry_run = False" in message
    assert "deploy.retries = 0" in message
    assert "deploy.note = (not set)" in message
    assert "deploy.label = (not set)" in message


def test_config_show_shows_false_zero_and_blank(planted, capsys):
    from hermes_cli.config_show import _show_skill_settings

    _show_skill_settings()
    out = capsys.readouterr().out
    assert "deploy.dry_run       False" in out
    assert "deploy.retries       0" in out
    assert "deploy.note          (not set)" in out
    assert "deploy.label         (not set)" in out
