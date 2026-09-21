"""Tests for the ip-as-logo optional skill."""

import re
from pathlib import Path

import yaml

SKILL_PATH = (
    Path(__file__).resolve().parents[2] / "optional-skills" / "creative" / "ip-as-logo" / "SKILL.md"
)


def _frontmatter_and_body():
    content = SKILL_PATH.read_text(encoding="utf-8")
    assert content.startswith("---")
    m = re.search(r"\n---\s*\n", content[3:])
    assert m, "frontmatter must close with ---"
    return yaml.safe_load(content[3 : m.start() + 3]), content[m.end() + 3 :]


def test_frontmatter_fields_and_section_order():
    fm, body = _frontmatter_and_body()
    for field in ("name", "description", "version", "author", "license", "platforms"):
        assert field in fm, f"missing frontmatter field: {field}"
    assert fm["name"] == "ip-as-logo"
    assert fm["metadata"]["hermes"]["category"] == "creative"
    headings = re.findall(r"^## (.+)$", body, re.MULTILINE)
    expected = ["When to Use", "Prerequisites", "Procedure"]
    assert [h for h in headings if h in expected] == expected, headings
    assert headings[-1] == "Verification"


def test_generation_routes_through_native_tool():
    """The upstream skill targeted another agent's image pipeline; the port must
    route through Hermes' `image_generate` and carry no residue of that harness."""
    _, body = _frontmatter_and_body()
    assert "`image_generate`" in body
    assert 'aspect_ratio="square"' in body
    assert not re.search(r"\bImageGen\b|\bCodex\b|\bClaude\b", body)
