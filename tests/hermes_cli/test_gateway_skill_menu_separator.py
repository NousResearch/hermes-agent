"""Gateway slash menu must match skill paths with the platform separator.

``_collect_gateway_skill_entries`` admits a skill only when its
``skill_md_path`` starts with one of the allowed directory prefixes.  Those
prefixes were terminated with a hardcoded ``"/"`` (#18741), while
``skill_md_path`` is stored as a plain ``str(Path)``
(:func:`agent.skill_commands.get_skill_commands`), which is
backslash-separated on Windows.  Every comparison was therefore ``False``
there and the Telegram menu lost its entire skill tier.  ``/skill-name``
dispatch does not use this filter and kept working, so the loss was
silent.

The #8110 regression test does not cover this: it hand-builds
``skill_md_path`` with f-string ``/`` separators, so the first separator
after the skills dir is a forward slash on every platform.  The tests below
build the paths the way production does, with the platform separator.  On
POSIX both spellings are the same string, so these tests discriminate on
Windows only: there they fail against ``main`` and pass with the fix.
"""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

from hermes_cli.commands import telegram_menu_commands


def _skill_entry(root: Path, name: str) -> dict[str, str]:
    """Build a ``get_skill_commands`` entry with native separators."""
    skill_dir = os.path.join(str(root), name)
    return {
        "name": name,
        "description": f"The {name} skill",
        "skill_md_path": os.path.join(skill_dir, "SKILL.md"),
        "skill_dir": skill_dir,
    }


def test_native_separator_skill_paths_reach_the_menu(tmp_path, monkeypatch):
    """Local and external skills are admitted; a prefix sibling is not."""
    local_dir = tmp_path / "skills"
    local_dir.mkdir()
    external_dir = tmp_path / "my-skills"
    external_dir.mkdir()
    lookalike_dir = tmp_path / "my-skills-extra"
    lookalike_dir.mkdir()

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    fake_cmds = {
        "/local-one": _skill_entry(local_dir, "local-one"),
        "/morning-briefing": _skill_entry(external_dir, "morning-briefing"),
        "/lookalike-skill": _skill_entry(lookalike_dir, "lookalike-skill"),
    }

    with (
        patch("agent.skill_commands.get_skill_commands", return_value=fake_cmds),
        patch("tools.skills_tool.SKILLS_DIR", local_dir),
        patch(
            "agent.skill_utils.get_external_skills_dirs",
            return_value=[external_dir],
        ),
    ):
        menu, _ = telegram_menu_commands(max_commands=100)

    menu_names = {n for n, _ in menu}
    assert "local_one" in menu_names, (
        "a local skill path built with the platform separator must appear"
    )
    assert "morning_briefing" in menu_names, (
        "an external skill path built with the platform separator must appear"
    )
    assert "lookalike_skill" not in menu_names, (
        "prefix-match sibling directories must still be rejected"
    )


def test_hub_skills_stay_excluded_with_native_separators(tmp_path, monkeypatch):
    """The ``.hub`` exclusion survives the separator fix."""
    local_dir = tmp_path / "skills"
    hub_dir = local_dir / ".hub"
    hub_dir.mkdir(parents=True)

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    fake_cmds = {
        "/local-one": _skill_entry(local_dir, "local-one"),
        "/hub-installed": _skill_entry(hub_dir, "hub-installed"),
    }

    with (
        patch("agent.skill_commands.get_skill_commands", return_value=fake_cmds),
        patch("tools.skills_tool.SKILLS_DIR", local_dir),
        patch("agent.skill_utils.get_external_skills_dirs", return_value=[]),
    ):
        menu, _ = telegram_menu_commands(max_commands=100)

    menu_names = {n for n, _ in menu}
    assert "local_one" in menu_names, "the local skill must still appear"
    assert "hub_installed" not in menu_names, (
        "hub-installed skills must not reach the gateway menu"
    )
