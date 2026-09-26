"""Tests for the mcp-google-orchestrator skill."""

import re
from pathlib import Path

import pytest

# In-repo layout: tests/skills/test_*.py -> parents[2] is the repo root.
SKILL_PATH = (
    Path(__file__).resolve().parents[2]
    / "skills"
    / "software-development"
    / "mcp-google-orchestrator"
    / "SKILL.md"
)


def test_skill_file_exists():
    assert SKILL_PATH.exists(), f"missing {SKILL_PATH}"


def test_frontmatter_parses():
    text = SKILL_PATH.read_text(encoding="utf-8")
    m = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    assert m, "frontmatter must start and end with '---'"
    fm = m.group(1)
    assert "name: mcp-google-orchestrator" in fm
    desc_match = re.search(r'^description:\s*"?(.+?)"?\s*$', fm, re.MULTILINE)
    assert desc_match, "missing description in frontmatter"
    desc = desc_match.group(1)
    # AGENTS.md:888-900 sets a 60-char base limit; allow +50% headroom for
    # readability (Teknium flagged 311, we stay under ~80).
    assert len(desc) <= 100, f"description too long: {len(desc)} chars"


def test_author_human_first():
    text = SKILL_PATH.read_text(encoding="utf-8")
    m = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    fm = m.group(1)
    author_line = next(
        (l for l in fm.splitlines() if l.startswith("author:")), None
    )
    assert author_line, "missing author line"
    val = author_line.split(":", 1)[1]
    assert "Hermes Agent" not in val.split("(")[0], (
        "author field must list the human contributor first "
        "(AGENTS.md:926-931)"
    )


def test_no_hardcoded_home_path():
    text = SKILL_PATH.read_text(encoding="utf-8").lower()
    for bad in ("c:/users/", "/users/anton/appdata", "appdata/local/hermes"):
        assert bad not in text, (
            f"hardcoded path fragment '{bad}' — use HERMES_HOME"
        )


def test_prerequisites_section():
    text = SKILL_PATH.read_text(encoding="utf-8")
    assert "## Prerequisites" in text, "missing '## Prerequisites' section"


def test_related_skills_resolve():
    """Every related_skill entry must exist as a SKILL.md in the repo."""
    text = SKILL_PATH.read_text(encoding="utf-8")
    m = re.search(r"^related_skills:\s*\[(.+?)\]", text, re.MULTILINE)
    if not m:
        return  # optional field
    names = [s.strip().strip("\"'") for s in m.group(1).split(",")]
    skills_root = SKILL_PATH.resolve().parents[2]
    for n in names:
        matches = list(skills_root.rglob(f"*/{n}/SKILL.md"))
        assert matches, f"related_skill '{n}' not found in repo"
