"""Behavioural regression tests for the support-pole + SEO pilot pack.

The checks intentionally validate document contracts rather than prose snapshots.
They require no network, credentials, LLM, or external side effect.
"""
from __future__ import annotations

import pathlib
import unittest

from evaluations.audit_pack import audit

ROOT = pathlib.Path(__file__).resolve().parents[1]


class AuditPackTests(unittest.TestCase):
    def test_pack_has_all_required_documents_and_controls(self) -> None:
        report = audit(ROOT)
        self.assertEqual([], report["missing_required"], report)
        self.assertEqual([], report["missing_controls"], report)
        self.assertEqual([], report["duplicate_responsibilities"], report)

    def test_workflow_and_skills_have_executable_contract_markers(self) -> None:
        report = audit(ROOT)
        self.assertEqual([], report["workflow_contract_gaps"], report)
        self.assertEqual([], report["skill_contract_gaps"], report)

    def test_capability_traceability_and_evaluation_coverage_are_complete(self) -> None:
        report = audit(ROOT)
        self.assertEqual([], report["capability_gaps"], report)
        self.assertEqual([], report["evaluation_gaps"], report)

    def test_no_external_or_billable_execution_claim_is_made(self) -> None:
        report = audit(ROOT)
        self.assertEqual([], report["safety_claim_gaps"], report)

    def test_every_required_profile_and_fixture_run_has_a_contract(self) -> None:
        report = audit(ROOT)
        self.assertEqual([], report["profile_contract_gaps"], report)
        self.assertEqual([], report["case_execution_gaps"], report)

    def test_blocked_research_and_independence_are_disclosed_not_faked(self) -> None:
        report = audit(ROOT)
        self.assertEqual("blocked-no-spend", report["research_status"], report)
        self.assertEqual("blocked-no-spend", report["independent_review_status"], report)


if __name__ == "__main__":
    unittest.main()
