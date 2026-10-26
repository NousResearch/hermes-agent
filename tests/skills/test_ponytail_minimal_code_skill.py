"""Contract tests for the in-repository Ponytail minimal-code skill."""

from __future__ import annotations

import re
from pathlib import Path

SKILL_PATH = (
    Path(__file__).resolve().parents[2]
    / "skills"
    / "software-development"
    / "ponytail-minimal-code"
    / "SKILL.md"
)
REQUIRED_SECTIONS = [
    "When to Use",
    "Prerequisites",
    "How to Run",
    "Quick Reference",
    "Procedure",
    "Pitfalls",
    "Verification",
]


def _read_skill() -> str:
    return SKILL_PATH.read_text(encoding="utf-8")


def _frontmatter(content: str) -> dict[str, object]:
    match = re.match(r"\A---\n(.*?)\n---\n", content, re.DOTALL)
    assert match, "SKILL.md must begin with YAML frontmatter"
    fields: dict[str, object] = {}
    for line in match.group(1).splitlines():
        if not line or line.startswith(" ") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        value = value.strip()
        if value.startswith('"') and value.endswith('"'):
            value = value[1:-1]
        elif value.startswith("["):
            value = [
                item.strip().strip("'\"")
                for item in value[1:-1].split(",")
                if item.strip()
            ]
        fields[key] = value
    return fields


def test_skill_path_exists() -> None:
    assert SKILL_PATH.is_file()


def test_frontmatter_contract() -> None:
    frontmatter = _frontmatter(_read_skill())

    assert frontmatter["name"] == "ponytail-minimal-code"
    description = frontmatter["description"]
    assert isinstance(description, str)
    assert description.startswith("Use when ")
    assert len(description) <= 60
    assert re.fullmatch(r"[^.!?]+\.", description)
    assert frontmatter["author"] == "Het / @hetdev, Hermes Agent"
    assert frontmatter["version"] == "1.0.0"
    assert frontmatter["license"] == "MIT"
    assert frontmatter["platforms"] == ["linux", "macos", "windows"]


def test_exact_title_and_section_order() -> None:
    content = _read_skill()
    body = content.split("---\n", 2)[-1]
    assert body.startswith("\n# Ponytail Minimal Code Skill\n")
    headings = re.findall(r"^## (.+)$", content, re.MULTILINE)
    assert headings == REQUIRED_SECTIONS


def test_decision_ladder_and_attribution_are_preserved() -> None:
    content = _read_skill()
    for marker in (
        "Does this need to exist?",
        "Does the standard library do it?",
        "Does the native platform feature do it?",
        "Does an installed dependency do it?",
        "Is it a one-liner?",
        "Is new code still required?",
    ):
        assert marker in content
    assert "**Credits:**" in content
    assert (
        "[Ponytail project](https://github.com/DietrichGebert/ponytail)" in content
    )
    assert "[Dietrich Gebert](https://github.com/DietrichGebert)" in content
    assert "MIT-licensed" in content


def test_skill_has_no_external_runtime_dependency() -> None:
    content = _read_skill().lower()
    assert "no setup or external runtime dependency is required" in content
    for marker in ("pip install", "npm install", "brew install", "third-party runtime"):
        assert marker not in content
