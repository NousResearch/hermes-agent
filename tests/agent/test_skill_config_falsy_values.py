"""A skill config value of ``false`` or ``0`` is shown as itself, not "(not set)".

``resolve_skill_config_values`` returns the configured ``False`` / ``0``, but the
``[Skill config]`` block injected on ``/skill`` invocation and the "Skill Settings" section
of ``hermes config`` rendered every falsy value as "(not set)", so the agent read an explicit
``dry_run: false`` as unset and could fall back to the declared default.
"""

import pytest

SKILL_MD = """---
name: deploy
description: Deploys things
metadata:
  hermes:
    config:
      - key: deploy.dry_run
        description: Only print the plan
        default: true
      - key: deploy.retries
        description: Retry count
        default: 3
      - key: deploy.region
        description: Target region
---

Deploy the service.
"""


@pytest.fixture()
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    skill_dir = home / "skills" / "deploy"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(SKILL_MD, encoding="utf-8")
    (home / "config.yaml").write_text(
        "skills:\n  config:\n    deploy:\n      dry_run: false\n      retries: 0\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    from tools import skills_tool
    monkeypatch.setattr(skills_tool, "SKILLS_DIR", home / "skills")
    monkeypatch.setattr(skills_tool, "HERMES_HOME", home)
    return home


def test_skill_invocation_shows_false_and_zero(hermes_home):
    from agent.skill_commands import build_skill_invocation_message

    message = build_skill_invocation_message("/deploy", "ship it")

    assert message is not None
    assert "deploy.dry_run = False" in message
    assert "deploy.retries = 0" in message
    assert "deploy.region = (not set)" in message


def test_hermes_config_skill_settings_show_false_and_zero(hermes_home, capsys):
    from hermes_cli.config import _show_skill_settings

    _show_skill_settings()
    lines = {line.split()[0]: line for line in capsys.readouterr().out.splitlines() if line.strip()[:7] == "deploy."}

    assert "False" in lines["deploy.dry_run"] and "(not set)" not in lines["deploy.dry_run"]
    assert " 0 " in lines["deploy.retries"]
    assert "(not set)" in lines["deploy.region"]
