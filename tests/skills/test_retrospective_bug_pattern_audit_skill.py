"""Tests for optional-skills/software-development/retrospective-bug-pattern-audit/SKILL.md."""

import re
import sys
from pathlib import Path

import pytest
import hermes_yaml as yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_DIR = REPO_ROOT / "optional-skills" / "software-development" / "retrospective-bug-pattern-audit"
SKILL_FILE = SKILL_DIR / "SKILL.md"

# Patterns from test_authoring_standards.py
MARKETING = re.compile(
    r"\b(powerful|comprehensive|seamless|revolutionary|cutting-edge|state-of-the-art)\b", re.I
)
MACHINE_LOCAL = re.compile(r"/home/(?!runner\b)[a-z0-9_-]+/|[A-Z]:\\+Users\\+(?!<)")


def _all_skill_names() -> set[str]:
    names = set()
    for p in REPO_ROOT.glob("skills/**/SKILL.md"):
        names.add(p.parent.name)
    for p in REPO_ROOT.glob("optional-skills/**/SKILL.md"):
        names.add(p.parent.name)
    return names


ALL_SKILL_NAMES = _all_skill_names()


def _load_frontmatter():
    content = SKILL_FILE.read_text(encoding="utf-8")
    assert content.startswith("---"), "SKILL.md must start with ---"
    m = re.search(r"\n---\s*\n", content[3:])
    assert m, "unclosed frontmatter"
    fm = yaml.safe_load(content[3 : m.start() + 3])
    assert isinstance(fm, dict), "frontmatter must be a YAML mapping"
    return fm, content


def test_skill_file_exists():
    assert SKILL_FILE.exists(), f"Skill file not found at {SKILL_FILE}"


def test_required_frontmatter_fields():
    fm, _ = _load_frontmatter()
    missing = [
        f
        for f in ("name", "description", "version", "author", "license", "platforms")
        if f not in fm
    ]
    assert not missing, f"missing frontmatter fields: {missing}"


def test_name_matches_directory():
    fm, _ = _load_frontmatter()
    assert fm.get("name") == SKILL_DIR.name, (
        f"frontmatter name {fm.get('name')!r} != dir {SKILL_DIR.name!r}"
    )


def test_description_hardline():
    fm, _ = _load_frontmatter()
    desc = str(fm.get("description") or "")
    assert len(desc) <= 60, f"description {len(desc)} chars (hardline 60)"
    assert desc.rstrip().endswith("."), "description must end with a period"
    m = MARKETING.search(desc)
    assert not m, f"marketing word in description: {m.group(0)!r}"


def test_author_credits_human_first():
    fm, _ = _load_frontmatter()
    author = str(fm.get("author") or "")
    assert "Hermes Agent" in author, "author must credit Hermes Agent"
    # Human name should appear before "Hermes Agent"
    human_part = author.split("Hermes Agent")[0].strip()
    assert human_part, "author must credit a human before 'Hermes Agent'"
    assert "," in human_part or human_part.endswith(")"), "author format should be 'Name (handle), Hermes Agent'"


def test_platforms_audited():
    fm, _ = _load_frontmatter()
    platforms = fm.get("platforms")
    assert isinstance(platforms, list) and platforms, "platforms must be a non-empty list"
    # This skill uses git + terminal only — cross-platform
    assert set(platforms) == {"linux", "macos", "windows"}, (
        f"expected [linux, macos, windows] for cross-platform git/terminal skill, got {platforms}"
    )


def test_tags_present():
    fm, _ = _load_frontmatter()
    hermes = (fm.get("metadata") or {}).get("hermes") or {}
    tags = hermes.get("tags") or fm.get("tags")
    assert tags, "no tags (metadata.hermes.tags or top-level tags)"
    assert isinstance(tags, list) and len(tags) > 0


def test_related_skills_resolve():
    fm, _ = _load_frontmatter()
    hermes = (fm.get("metadata") or {}).get("hermes") or {}
    dangling = [
        rs for rs in (hermes.get("related_skills") or []) if rs not in ALL_SKILL_NAMES
    ]
    assert not dangling, f"dangling related_skills: {dangling}"


def test_no_machine_local_paths():
    _, content = _load_frontmatter()
    m = MACHINE_LOCAL.search(content)
    assert not m, f"machine-local path {m.group(0)!r}"


def test_size_limit():
    _, content = _load_frontmatter()
    assert len(content) <= 100_000, f"{len(content)} chars > 100k"


def test_when_to_use_section_present():
    _, content = _load_frontmatter()
    assert "## When to Use" in content, "missing '## When to Use' section"


def test_prerequisites_section_present():
    _, content = _load_frontmatter()
    assert "## Prerequisites" in content, "missing '## Prerequisites' section"


def test_how_to_run_section_present():
    _, content = _load_frontmatter()
    assert "## How to Run" in content, "missing '## How to Run' section"


def test_pitfalls_section_present():
    _, content = _load_frontmatter()
    assert "## Pitfalls" in content, "missing '## Pitfalls' section"


def test_verification_section_present():
    _, content = _load_frontmatter()
    assert "## Verification" in content, "missing '## Verification' section"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))