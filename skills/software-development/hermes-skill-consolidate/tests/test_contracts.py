from __future__ import annotations

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKILL = (ROOT / "SKILL.md").read_text(encoding="utf-8")
SAFETY = (ROOT / "references" / "safety.md").read_text(encoding="utf-8")
DECISION = (ROOT / "references" / "decision-model.md").read_text(encoding="utf-8")
REPORT = (ROOT / "references" / "report-contract.md").read_text(encoding="utf-8")
CASES = json.loads((ROOT / "tests" / "cases.json").read_text(encoding="utf-8"))
CONTRACTS = json.loads((ROOT / "tests" / "contract-cases.json").read_text(encoding="utf-8"))


class ConsolidationContractTests(unittest.TestCase):
    def test_planning_is_read_only(self):
        self.assertIn("Phase 1 is always read-only", SKILL)
        self.assertIn("Do not modify live files in this phase", SKILL)

    def test_second_explicit_approval_is_required(self):
        self.assertIn("second explicit approval", SKILL)
        self.assertIn("Approval is valid only for the named skills", SKILL)
        self.assertIn("Approval becomes stale", SAFETY)

    def test_safety_monotonicity_is_explicit(self):
        self.assertIn("Safety is monotonic during consolidation", SKILL)
        self.assertIn("preserve every material approval gate", SAFETY)
        self.assertIn("prefer separate skills when read-only and mutating responsibilities differ", SAFETY)

    def test_all_relationship_classes_are_published(self):
        relationships = [
            "CONFIRMED DUPLICATE",
            "LIKELY REDUNDANT",
            "PARTIAL OVERLAP",
            "COMPLEMENTARY",
            "PARENT OR ORCHESTRATOR",
            "SHARED REFERENCE CANDIDATE",
            "INTENTIONALLY SEPARATE",
            "INSUFFICIENT EVIDENCE",
        ]
        for relationship in relationships:
            with self.subTest(relationship=relationship):
                self.assertIn(relationship, SKILL)
                self.assertIn(relationship, DECISION)

    def test_untrusted_content_boundary_blocks_execution(self):
        combined = (SKILL + SAFETY).lower()
        self.assertIn("untrusted evidence", combined)
        self.assertIn("never follow instructions", combined)
        self.assertIn("do not activate", combined)
        self.assertIn("prompt injection or social engineering", combined)
        self.assertGreaterEqual(len(CONTRACTS["untrusted_content_prompts"]), 2)

    def test_rollback_precedes_live_write(self):
        snapshot_pos = SKILL.index("### 8. Snapshot and verify")
        cutover_pos = SKILL.index("### 10. Cut over reversibly")
        self.assertLess(snapshot_pos, cutover_pos)
        self.assertIn("snapshot output must be outside the live skills tree", (ROOT / "scripts" / "snapshot_skills.py").read_text(encoding="utf-8"))

    def test_success_requires_verification(self):
        self.assertIn("APPLIED AND VERIFIED", REPORT)
        self.assertIn("A successful write is not equivalent to successful behavior", REPORT)
        self.assertIn("`APPLIED AND VERIFIED` is forbidden until", SKILL)

    def test_behavior_cases_cover_critical_boundaries(self):
        ids = {case["id"] for case in CASES["cases"]}
        for case_id in {
            "safety-separation",
            "approval-boundary",
            "rollback-before-write",
            "untrusted-content-boundary",
            "ambiguous-overlap",
            "post-apply-verification",
        }:
            self.assertIn(case_id, ids)


if __name__ == "__main__":
    unittest.main()
