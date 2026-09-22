#!/usr/bin/env python3
"""Dependency-free publication smoke test for the optional skill."""
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "SKILL.md"


class SkillPublicationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = SKILL.read_text(encoding="utf-8")

    def test_has_required_frontmatter(self):
        self.assertTrue(self.text.startswith("---\n"))
        frontmatter = self.text.split("---\n", 2)[1]
        for key in (
            "name:",
            "description:",
            "version:",
            "author:",
            "license:",
            "platforms:",
            "metadata:",
            "tags:",
        ):
            self.assertIn(key, frontmatter)
        self.assertIn("name: citation-integrity", frontmatter)

    def test_public_skill_frontmatter_contract(self):
        expected = {
            "citation-integrity": "Use when verifying citation ledgers, quotes, and evidence.",
            "evidence-reports": "Use when writing research reports with checked claims.",
            "merge-reconciler": "Use when reconciling conflicting Git branches safely.",
        }
        for name, description in expected.items():
            skill = ROOT.parent / name / "SKILL.md"
            self.assertTrue(skill.is_file(), skill)
            text = skill.read_text(encoding="utf-8")
            frontmatter = text.split("---\n", 2)[1]
            self.assertIn(f"name: {name}", frontmatter)
            self.assertIn(f'description: "{description}"', frontmatter)
            self.assertLessEqual(len(description), 60)
            self.assertTrue(description.endswith("."))

    def test_covers_required_topics(self):
        for phrase in (
            "duplicate",
            "exact substring",
            "evidence identity",
            "Strict verification",
            "content digest",
        ):
            self.assertRegex(self.text.lower(), re.escape(phrase.lower()))

    def test_has_no_private_or_incident_specific_references(self):
        forbidden = (
            ".hermes/profiles/",
            "cache/scratch",
            "incident history",
            "internal path",
            "private tool",
            "execute_",
            "incident-specific",
        )
        lowered = self.text.lower()
        for term in forbidden:
            self.assertNotIn(term.lower(), lowered, term)

    def test_is_self_contained(self):
        self.assertIn("tool-agnostic", self.text)
        self.assertNotIn("grounded-citations", self.text)
        self.assertNotIn("sources.py", self.text)


if __name__ == "__main__":
    unittest.main()
