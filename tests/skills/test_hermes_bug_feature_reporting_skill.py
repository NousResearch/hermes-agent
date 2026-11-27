"""Test for the hermes-bug-feature-reporting skill metadata."""

import yaml
from pathlib import Path


def test_hermes_bug_feature_reporting_skill_metadata():
    skill_path = Path(__file__).parents[2] / "optional-skills/devops/hermes-bug-feature-reporting/SKILL.md"
    content = skill_path.read_text(encoding="utf-8")
    # Extract frontmatter
    if not content.startswith("---\"):
        raise AssertionError("SKILL.md must start with frontmatter delimiter")
    _, frontmatter, _ = content.split("---\", 2)
    data = yaml.safe_load(frontmatter)
    assert data["name"] == "hermes-bug-feature-reporting"
    # Description must be <= 60 characters
    desc = data["description"]
    assert isinstance(desc, str), "description must be a string"
    assert len(desc) <= 60, f"description too long: {len(desc)} chars: {desc}"
    # related_skills should include obsidian
    related = data.get("metadata", {}).get("hermes", {}).get("related_skills", [])
    assert "obsidian" in related, f"related_skills must contain obsidian, got {related}"
