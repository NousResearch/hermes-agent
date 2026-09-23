"""Verify the codebase-semantic-graph SKILL.md matches current Hermes format."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

SKILL_MD = (
    Path(__file__).resolve().parents[2]
    / "skills"
    / "software-development"
    / "codebase-semantic-graph"
    / "SKILL.md"
)


class TestCodebaseSemanticGraphSkill(unittest.TestCase):
    def test_skill_md_exists_with_yaml_frontmatter(self) -> None:
        self.assertTrue(SKILL_MD.exists(), f"Expected {SKILL_MD} to exist")
        content = SKILL_MD.read_text(encoding="utf-8")
        self.assertTrue(content.startswith("---"))
        self.assertIn("\n---\n", content)
        _, _, body = content.split("---", 2)
        self.assertTrue(body.strip(), "SKILL.md must have content after frontmatter")

    def test_skill_md_has_required_frontmatter_fields(self) -> None:
        content = SKILL_MD.read_text(encoding="utf-8")
        _, frontmatter, _ = content.split("---", 2)
        self.assertIn("name: codebase-semantic-graph", frontmatter)
        self.assertIn("description:", frontmatter)
        self.assertIn("version: 1.0.0", frontmatter)
        self.assertIn("author:", frontmatter)
        self.assertIn("license: MIT", frontmatter)
        self.assertIn("metadata:", frontmatter)

    def test_skill_md_includes_documentation_sections(self) -> None:
        body = SKILL_MD.read_text(encoding="utf-8").split("---", 2)[-1]
        required = ["## When to Use", "## Quick Reference", "## Procedure", "## Pitfalls", "## Verification"]
        for heading in required:
            self.assertIn(heading, body)

    def test_skill_md_description_is_concise(self) -> None:
        content = SKILL_MD.read_text(encoding="utf-8")
        match = re.search(r"^description:\s*(.+)$", content, re.MULTILINE)
        self.assertIsNotNone(match)
        self.assertLessEqual(len(match.group(1)), 70)


if __name__ == "__main__":
    unittest.main()
