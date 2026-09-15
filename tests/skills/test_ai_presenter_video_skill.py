"""Tests for the AI presenter video optional skill."""

import re
from pathlib import Path


SKILL_MD = (
    Path(__file__).resolve().parents[2]
    / "optional-skills/creative/ai-presenter-video/SKILL.md"
)


def test_modern_heading_sequence():
    headings = [
        line
        for line in SKILL_MD.read_text(encoding="utf-8").splitlines()
        if line.startswith("# ") or line.startswith("## ")
    ]
    assert headings == [
        "# AI Presenter Video Skill",
        "## When to Use",
        "## Prerequisites",
        "## How to Run",
        "## Quick Reference",
        "## Procedure",
        "## Pitfalls",
        "## Verification",
    ]


def test_native_tool_names_are_code_formatted():
    text = SKILL_MD.read_text(encoding="utf-8")
    for tool in ("image_generate", "text_to_speech", "vision_analyze"):
        assert not re.search(rf"(?<!`)\b{tool}\b(?!`)", text), tool
