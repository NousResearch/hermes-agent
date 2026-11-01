"""Tests for the skill-certification bundled skill."""
import re
from pathlib import Path

import yaml

SKILL_PATH = (
    Path(__file__).resolve().parents[2]
    / "skills"
    / "software-development"
    / "skill-certification"
    / "SKILL.md"
)


def _frontmatter_and_body():
    content = SKILL_PATH.read_text(encoding="utf-8")
    assert content.startswith("---")
    m = re.search(r"\n---\s*\n", content[3:])
    assert m, "frontmatter must close with ---"
    fm = yaml.safe_load(content[3 : m.start() + 3])
    body = content[m.end() + 3 :]
    return fm, body


def test_skill_file_exists():
    assert SKILL_PATH.is_file()


def test_frontmatter_required_fields():
    fm, _ = _frontmatter_and_body()
    for field in ("name", "description", "version", "author", "license", "platforms"):
        assert field in fm, f"missing frontmatter field: {field}"
    assert fm["name"] == "skill-certification"
    hermes = fm["metadata"]["hermes"]
    assert hermes["tags"]
    assert "requires_toolsets" in hermes


def test_description_hardline():
    fm, _ = _frontmatter_and_body()
    desc = fm["description"]
    assert len(desc) <= 60, f"description is {len(desc)} chars; hardline is 60"
    assert desc.endswith(".")


def test_body_has_expected_sections():
    _, body = _frontmatter_and_body()
    for section in ("When to Use", "Prerequisites", "How to Run", "Quick Reference", "Pitfalls", "Verification"):
        assert section in body, f"missing section: {section}"


def test_api_endpoints_documented():
    content = SKILL_PATH.read_text(encoding="utf-8")
    for endpoint in ("/api/verify", "/submit", "/api/report", "/health"):
        assert endpoint in content, f"missing endpoint: {endpoint}"


def test_no_marketing_words():
    _, body = _frontmatter_and_body()
    for word in ("powerful", "comprehensive", "seamless", "revolutionary"):
        assert word.lower() not in body.lower(), f"marketing word: {word}"
