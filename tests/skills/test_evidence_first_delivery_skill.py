"""Behavioral contract tests for the evidence-first delivery skill."""
import re
from pathlib import Path

import yaml

SKILL_PATH = (
    Path(__file__).resolve().parents[2]
    / "skills"
    / "productivity"
    / "evidence-first-delivery"
    / "SKILL.md"
)


def _frontmatter_and_body():
    content = SKILL_PATH.read_text(encoding="utf-8")
    assert content.startswith("---")
    match = re.search(r"\n---\s*\n", content[3:])
    assert match, "frontmatter must close with ---"
    frontmatter = yaml.safe_load(content[3 : match.start() + 3])
    body = content[match.end() + 3 :]
    return frontmatter, body


def test_skill_file_exists():
    assert SKILL_PATH.is_file()


def test_frontmatter_has_required_metadata():
    frontmatter, _ = _frontmatter_and_body()
    for field in ("name", "description", "version", "author", "license", "platforms"):
        assert field in frontmatter, f"missing frontmatter field: {field}"
    assert frontmatter["name"] == "evidence-first-delivery"
    assert frontmatter["metadata"]["hermes"]["tags"]
    assert frontmatter["metadata"]["hermes"]["category"] == "productivity"


def test_description_and_attribution_follow_authoring_policy():
    frontmatter, _ = _frontmatter_and_body()
    assert len(frontmatter["description"]) <= 60
    assert frontmatter["description"].endswith(".")
    assert frontmatter["author"].startswith("Ryan (ryankr)")


def test_skill_keeps_execution_and_proof_separate():
    _, body = _frontmatter_and_body()
    for heading in ("## When to Use", "## Procedure", "## Pitfalls", "## Verification"):
        assert heading in body
    for state in ("Planned", "Executed, unverified", "Verified", "Blocked or partial"):
        assert state in body
    assert "read the exact target back" in body
    assert "subagent report as unverified" in body


def test_procedure_has_completion_criteria_and_no_local_paths():
    _, body = _frontmatter_and_body()
    steps = re.findall(r"^### \d+\..*?(?=^### \d+\.|^## )", body, re.MULTILINE | re.DOTALL)
    assert len(steps) == 5
    assert all("Done when" in step for step in steps)
    content = SKILL_PATH.read_text(encoding="utf-8")
    assert "/Users/" not in content
    assert "/home/" not in content
