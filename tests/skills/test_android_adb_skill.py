"""tests/skills/test_android_adb_skill.py - Test suite for android-adb skill."""

from __future__ import annotations

import pathlib
import re


def test_android_adb_skill_file():
    skill_path = pathlib.Path("optional-skills/devops/android-adb/SKILL.md")
    assert skill_path.exists(), "android-adb SKILL.md must exist"

    content = skill_path.read_text(encoding="utf-8")

    # Verify description constraint (<= 60 chars)
    desc_match = re.search(r"^description:\s*(.*)$", content, re.MULTILINE)
    assert desc_match is not None, "description field missing from SKILL.md frontmatter"
    desc = desc_match.group(1).strip()
    assert len(desc) <= 60, f"description exceeds 60 chars limit: '{desc}' ({len(desc)} chars)"
    assert desc.endswith("."), "description must end with a period"

    # Verify key sections exist
    required_sections = [
        "# Android ADB Skill",
        "## When to Use",
        "## Prerequisites",
        "## How to Run",
        "## Quick Reference",
        "## Procedure",
        "## Pitfalls",
        "## Verification",
    ]
    for sec in required_sections:
        assert sec in content, f"Section '{sec}' missing from android-adb SKILL.md"
