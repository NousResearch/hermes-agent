from __future__ import annotations

import re
from pathlib import Path

SKILL_PATH = (
    Path(__file__).resolve().parents[2] / "skills" / "apple" / "apple-shortcuts" / "SKILL.md"
)

REQUIRED_SECTIONS = [
    "## When to Use",
    "## Prerequisites",
    "## How to Run",
    "## Quick Reference",
    "## Procedure",
    "## Pitfalls",
    "## Verification",
]


def _frontmatter() -> str:
    text = SKILL_PATH.read_text(encoding="utf-8")
    match = re.match(r"\A---\n(.*?)\n---\n", text, re.DOTALL)
    assert match is not None, "SKILL.md missing YAML frontmatter"
    return match.group(1)


def test_skill_file_exists() -> None:
    assert SKILL_PATH.is_file()


def test_description_is_hardline_compliant() -> None:
    match = re.search(r'^description: "?(.*?)"?$', _frontmatter(), re.MULTILINE)
    assert match is not None, "missing description"
    description = match.group(1)
    assert len(description) <= 60
    assert description.endswith(".")


def test_platform_gated_to_macos_only() -> None:
    match = re.search(r"^platforms: \[(.*?)\]$", _frontmatter(), re.MULTILINE)
    assert match is not None, "missing platforms gating"
    assert [p.strip() for p in match.group(1).split(",")] == ["macos"]


def test_platform_gate_matches_darwin_not_linux(monkeypatch) -> None:
    import agent.skill_utils as skill_utils

    monkeypatch.setattr(skill_utils.sys, "platform", "darwin")
    assert skill_utils.skill_matches_platform_list(["macos"])
    monkeypatch.setattr(skill_utils.sys, "platform", "linux")
    assert not skill_utils.skill_matches_platform_list(["macos"])


def test_prerequisites_declare_shortcuts_command() -> None:
    assert re.search(r"^\s+commands: \[shortcuts\]$", _frontmatter(), re.MULTILINE)


def test_body_sections_present_in_canonical_order() -> None:
    text = SKILL_PATH.read_text(encoding="utf-8")
    positions = [text.find(section) for section in REQUIRED_SECTIONS]
    assert -1 not in positions, (
        f"missing sections: {[s for s, p in zip(REQUIRED_SECTIONS, positions) if p == -1]}"
    )
    assert positions == sorted(positions), "sections out of canonical order"
