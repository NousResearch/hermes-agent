"""Metadata sanity tests for the 2chat-whatsapp optional skill.

Uses only the standard library + pytest. No live network calls.
"""
import re
from pathlib import Path

SKILL = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "communication"
    / "2chat-whatsapp"
    / "SKILL.md"
)


def _frontmatter() -> str:
    text = SKILL.read_text(encoding="utf-8")
    assert text.startswith("---\n"), "SKILL.md must start with YAML frontmatter"
    return text.split("---\n", 2)[1]


def test_skill_file_exists():
    assert SKILL.is_file(), f"missing skill file: {SKILL}"


def test_required_frontmatter_fields():
    fm = _frontmatter()
    for field in ("name:", "description:", "version:", "author:", "license:"):
        assert field in fm, f"frontmatter missing {field!r}"


def test_description_is_short_single_sentence():
    fm = _frontmatter()
    m = re.search(r"^description:\s*(.+)$", fm, re.MULTILINE)
    assert m, "no description field found"
    desc = m.group(1).strip()
    assert len(desc) <= 60, f"description must be <= 60 chars, got {len(desc)}"
    assert desc.endswith("."), "description must end with a period"


def test_references_present():
    ref_dir = SKILL.parent / "references"
    assert (ref_dir / "mcp-server.yaml").is_file()
    assert (ref_dir / "tools.md").is_file()
