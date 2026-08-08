"""Contract tests for the bundled contact-routing skill."""

from pathlib import Path


SKILL_PATH = (
    Path(__file__).resolve().parents[2]
    / "skills"
    / "productivity"
    / "contact-routing"
    / "SKILL.md"
)


def _frontmatter_value(text: str, key: str) -> str:
    prefix = f"{key}:"
    for line in text.splitlines():
        if line.startswith(prefix):
            return line.removeprefix(prefix).strip().strip('"')
    raise AssertionError(f"missing {key!r} frontmatter in {SKILL_PATH}")


def test_contact_routing_skill_meets_metadata_contract():
    text = SKILL_PATH.read_text(encoding="utf-8")
    description = _frontmatter_value(text, "description")
    author = _frontmatter_value(text, "author")

    assert len(description) <= 60
    assert description.endswith(".")
    assert author.startswith("Julian Albou (@julianships)")


def test_contact_routing_skill_uses_modern_section_order():
    text = SKILL_PATH.read_text(encoding="utf-8")
    headings = [
        "# Contact Routing Skill",
        "## When to Use",
        "## Prerequisites",
        "## How to Run",
        "## Quick Reference",
        "## Procedure",
        "## Pitfalls",
        "## Verification",
    ]

    positions = [text.index(heading) for heading in headings]
    assert positions == sorted(positions)