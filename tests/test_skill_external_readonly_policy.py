"""Pinned external skills are read-only at the mutation boundary."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest


SKILL_MD = """---
name: atlas-load-bearing
description: Load-bearing Atlas skill fixture
---

# Atlas Load Bearing

Original guidance.
"""

EDITED_SKILL_MD = """---
name: atlas-load-bearing
description: Load-bearing Atlas skill fixture
---

# Atlas Load Bearing

Edited guidance.
"""


@pytest.fixture()
def external_skill_env(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes-home"
    local_skills = hermes_home / "skills"
    external_root = tmp_path / "atlas" / "shared" / "skills"
    skill_dir = external_root / "atlas-load-bearing"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(SKILL_MD, encoding="utf-8")
    (skill_dir / "references").mkdir()
    (skill_dir / "references" / "note.md").write_text("original note\n", encoding="utf-8")

    local_skills.mkdir(parents=True)
    (hermes_home / "config.yaml").write_text(
        f"skills:\n  external_dirs:\n    - {external_root}\n",
        encoding="utf-8",
    )
    (local_skills / ".usage.json").write_text(
        json.dumps({
            "atlas-load-bearing": {
                "created_by": "agent",
                "pinned": True,
                "state": "active",
            }
        }),
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    import agent.skill_utils as skill_utils
    skill_utils._external_dirs_cache_clear()

    import tools.skill_usage as skill_usage
    import tools.skill_manager_tool as skill_manager_tool
    skill_usage = importlib.reload(skill_usage)
    skill_manager_tool = importlib.reload(skill_manager_tool)
    monkeypatch.setattr(skill_manager_tool, "HERMES_HOME", hermes_home)
    monkeypatch.setattr(skill_manager_tool, "SKILLS_DIR", local_skills)

    return {
        "skill_manager_tool": skill_manager_tool,
        "skill_usage": skill_usage,
        "skill_dir": skill_dir,
    }


def _manage(tool, **kwargs):
    return json.loads(tool.skill_manage(**kwargs))


@pytest.mark.parametrize(
    "kwargs, expected_file, expected_content",
    [
        (
            {
                "action": "patch",
                "name": "atlas-load-bearing",
                "old_string": "Original guidance.",
                "new_string": "Patched guidance.",
            },
            "SKILL.md",
            SKILL_MD,
        ),
        (
            {
                "action": "edit",
                "name": "atlas-load-bearing",
                "content": EDITED_SKILL_MD,
            },
            "SKILL.md",
            SKILL_MD,
        ),
        (
            {
                "action": "write_file",
                "name": "atlas-load-bearing",
                "file_path": "references/new.md",
                "file_content": "new note\n",
            },
            "references/new.md",
            None,
        ),
        (
            {
                "action": "remove_file",
                "name": "atlas-load-bearing",
                "file_path": "references/note.md",
            },
            "references/note.md",
            "original note\n",
        ),
        (
            {
                "action": "delete",
                "name": "atlas-load-bearing",
                "absorbed_into": "",
            },
            "SKILL.md",
            SKILL_MD,
        ),
    ],
)
def test_pinned_external_skill_manage_mutations_fail_closed(
    external_skill_env, kwargs, expected_file, expected_content
):
    tool = external_skill_env["skill_manager_tool"]
    skill_dir = external_skill_env["skill_dir"]

    result = _manage(tool, **kwargs)

    assert result["success"] is False
    assert "pinned/read-only external skill" in result["error"]
    target = skill_dir / expected_file
    if expected_content is None:
        assert not target.exists()
    else:
        assert target.read_text(encoding="utf-8") == expected_content


def test_unpinned_external_skill_can_still_be_patched(external_skill_env):
    tool = external_skill_env["skill_manager_tool"]
    skill_usage = external_skill_env["skill_usage"]
    skill_dir = external_skill_env["skill_dir"]
    usage = skill_usage.load_usage()
    usage["atlas-load-bearing"]["pinned"] = False
    skill_usage.save_usage(usage)

    result = _manage(
        tool,
        action="patch",
        name="atlas-load-bearing",
        old_string="Original guidance.",
        new_string="Patched guidance.",
    )

    assert result["success"] is True
    assert "Patched guidance." in (skill_dir / "SKILL.md").read_text(encoding="utf-8")


def test_archive_skill_refuses_pinned_external_skill(external_skill_env):
    skill_usage = external_skill_env["skill_usage"]
    skill_dir = external_skill_env["skill_dir"]

    ok, message = skill_usage.archive_skill("atlas-load-bearing")

    assert ok is False
    assert "pinned/read-only external skill" in message
    assert skill_dir.exists()
    assert (skill_dir / "SKILL.md").read_text(encoding="utf-8") == SKILL_MD


def test_corrupt_usage_sidecar_fail_closes_external_skill_mutation(external_skill_env):
    tool = external_skill_env["skill_manager_tool"]
    skill_dir = external_skill_env["skill_dir"]
    usage_path = tool.SKILLS_DIR / ".usage.json"
    usage_path.write_text("{not json", encoding="utf-8")

    result = _manage(
        tool,
        action="patch",
        name="atlas-load-bearing",
        old_string="Original guidance.",
        new_string="Patched guidance.",
    )

    assert result["success"] is False
    assert "pin state cannot be verified safely" in result["error"]
    assert (skill_dir / "SKILL.md").read_text(encoding="utf-8") == SKILL_MD
