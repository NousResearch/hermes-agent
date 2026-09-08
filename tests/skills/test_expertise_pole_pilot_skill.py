"""Regression contract for the experimental expertise-pole pilot skill.

The checks validate structural relationships in the reference pack; they do not
claim SEO outcomes, remote-source retrieval, or a completed independent review.
"""
from __future__ import annotations

import importlib.util
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SKILL_DIR = ROOT / "optional-skills" / "autonomous-ai-agents" / "expertise-pole-pilot"
SKILL_MD = SKILL_DIR / "SKILL.md"
PACK_ROOT = SKILL_DIR / "references" / "pilot-v0.1"


class ExpertisePolePilotSkillTests(unittest.TestCase):
    def skill_text(self) -> str:
        return SKILL_MD.read_text(encoding="utf-8")

    def test_skill_frontmatter_and_modern_sections(self) -> None:
        text = self.skill_text()
        frontmatter = re.match(r"^---\n(.*?)\n---\n", text, flags=re.S)
        self.assertIsNotNone(frontmatter, "SKILL.md must start with YAML frontmatter")
        header = frontmatter.group(1) if frontmatter else ""
        description = re.search(r"^description: (.+)$", header, flags=re.M)
        self.assertIn("name: expertise-pole-pilot", header)
        self.assertIsNotNone(description)
        assert description is not None
        self.assertLessEqual(len(description.group(1).strip('"')), 60)
        self.assertTrue(description.group(1).strip('"').endswith("."))
        self.assertIn("author: Vincent HERON", header)
        self.assertIn("platforms: [linux, macos, windows]", header)
        for heading in (
            "## When to Use",
            "## Prerequisites",
            "## How to Run",
            "## Quick Reference",
            "## Procedure",
            "## Pitfalls",
            "## Verification",
        ):
            self.assertIn(heading, text)

    def test_reference_pack_has_canonical_support_and_seo_contracts(self) -> None:
        support = {
            "README.md", "AGENTS.md", "GOVERNANCE.md", "DOCUMENT-TAXONOMY.md",
            "NAMING-AND-STRUCTURE.md", "LIFECYCLE.md", "QUALITY-GATES.md",
            "ORCHESTRATION.md", "RESEARCH-PROTOCOL.md", "SECURITY-AND-PERMISSIONS.md",
            "EVALUATION-FRAMEWORK.md", "CHANGELOG.md",
        }
        seo = {
            "README.md", "CAPABILITIES.md", "EXPERTS.md", "CONTRACTS.md",
            "EVALUATION.md", "SOURCES.md", "SHOULD.md", "AGENTS.md", "WORKFLOW.md",
        }
        self.assertTrue(all((PACK_ROOT / "support-pole" / name).is_file() for name in support))
        self.assertTrue(all((PACK_ROOT / "domains" / "seo" / name).is_file() for name in seo))
        self.assertGreaterEqual(len(list((PACK_ROOT / "domains" / "seo" / "agents").glob("*.md"))), 6)
        self.assertGreaterEqual(
            len(list((PACK_ROOT / "domains" / "seo" / "skills").glob("*/SKILL-CONTRACT.md"))),
            3,
        )
        self.assertEqual([], list(PACK_ROOT.rglob("SKILL.md")))
        packaging_note = (PACK_ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("SKILL-CONTRACT.md", packaging_note)
        self.assertIn("not loadable", packaging_note)
        self.assertTrue((PACK_ROOT / "domains" / "seo" / "workflows" / "evidence-to-action.md").is_file())

    def test_reference_pack_validator_passes_and_discloses_blocked_gates(self) -> None:
        audit_path = PACK_ROOT / "evaluations" / "audit_pack.py"
        spec = importlib.util.spec_from_file_location("expertise_pole_audit", audit_path)
        self.assertIsNotNone(spec)
        assert spec is not None
        module = importlib.util.module_from_spec(spec)
        self.assertIsNotNone(spec.loader)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        report = module.audit(PACK_ROOT)
        for key in (
            "missing_required", "missing_controls", "duplicate_responsibilities",
            "workflow_contract_gaps", "skill_contract_gaps", "capability_gaps",
            "evaluation_gaps", "safety_claim_gaps", "profile_contract_gaps",
            "case_execution_gaps",
        ):
            self.assertEqual([], report[key], report)
        self.assertEqual("blocked-no-spend", report["research_status"])
        self.assertEqual("blocked-no-spend", report["independent_review_status"])


if __name__ == "__main__":
    unittest.main()
