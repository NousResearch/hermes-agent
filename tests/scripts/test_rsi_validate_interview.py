#!/usr/bin/env python3
"""Regression tests for deterministic RSI interview reconciliation."""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "rsi-validate-interview.py"
BUILDER = Path(__file__).resolve().parents[2] / "scripts" / "rsi-build-interview.py"
# The full required installed roster: default + every installed profile dir.
FULL_ROSTER = [
    "default", "buggy", "coder", "jade", "jade-ops", "product", "qa",
    "research", "reviewer", "rsi", "x", "yuki", "yuki-ops",
]
PRODUCT_IDS = [
    "65333767b7964fc3986351e2a0ba2c02",
    "d8f9a32809414530803297153c6ee3e4",
    "50ca24c42feb4cfdb6a3deeabfc22a47",
    "5359e5e602ba48cfb3b6604816122246",
    "e157b0fb99a1440eb279be22f7ab5cac",
    "0485af1762214200bf69c69bd1afcbab",
]
CODER_IDS = [
    "20260902_180438_d8c6cd",
    "20260902_175149_c1fce9",
    "20260902_174221_00e278",
    "20260902_173236_2f74c7",
    "20260902_171518_7dc0f9",
    "20260902_162325_d3131c",
    "20260902_155533_7e6f23",
    "20260902_151527_ccdc42",
]


def _audit() -> dict:
    return {
        "profiles": {
            "product": {
                "session_failures": [],
                "cron_failures": [
                    {
                        "execution_id": execution_id,
                        "name": "eng-completion",
                        "status": "failed",
                    }
                    for execution_id in PRODUCT_IDS
                ],
                "kanban_failures": [],
            },
            "coder": {
                "session_failures": [
                    {
                        "id": session_id,
                        "failed": True,
                        "claimed_ok": True,
                        "fail_hits": ["lifecycle:needs_input"],
                    }
                    for session_id in CODER_IDS
                ],
                "cron_failures": [],
                "kanban_failures": [],
            },
        }
    }


def _report(profile: str, ids: list[str], detail_field: str) -> dict:
    report = {
        "profile": profile,
        "autonomous_failures": [],
        "incomplete_tasks": [],
        "incidents": [],
        "correction_feedback": [],
        "accounted_session_ids": list(ids),
    }
    report[detail_field] = [
        ({"id": item, "summary": f"failed execution {item}", "evidence": item, "suggested_fix": "retry"}
         if detail_field == "autonomous_failures"
         else {"id": item, "title": "audited task", "summary": item, "why_incomplete": "needs_input"})
        for item in ids
    ]
    return report


class InterviewValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("rsi_validate_interview", SCRIPT)
        assert spec is not None and spec.loader is not None
        cls.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.mod)

    def test_product_grouped_label_does_not_replace_six_execution_ids(self):
        report = _report("product", [], "autonomous_failures")
        report["autonomous_failures"] = [{
            "summary": "eng-completion failed six times",
            "evidence": "grouped under eng-completion",
            "suggested_fix": "retry",
        }]
        report["accounted_session_ids"] = ["eng-completion"]

        result = self.mod.validate_interview("product", report, _audit())

        self.assertFalse(result["valid"])
        self.assertEqual(result["missing_accounted_ids"], PRODUCT_IDS)
        self.assertEqual(result["missing_detail_ids"]["autonomous_failures"], PRODUCT_IDS)

    def test_coder_claimed_ok_does_not_exempt_eight_needs_input_ids(self):
        report = _report("coder", [], "incomplete_tasks")

        result = self.mod.validate_interview("coder", report, _audit())

        self.assertFalse(result["valid"])
        self.assertEqual(result["missing_accounted_ids"], CODER_IDS)
        self.assertEqual(result["missing_detail_ids"]["incomplete_tasks"], CODER_IDS)

    def test_incomplete_task_requires_exact_id_field(self):
        required_id = CODER_IDS[0]
        audit = _audit()
        audit["profiles"]["coder"]["session_failures"] = [
            audit["profiles"]["coder"]["session_failures"][0]
        ]
        report = _report("coder", [required_id], "incomplete_tasks")
        report["incomplete_tasks"] = [{
            "id": "wrong-id",
            "title": "unrelated task",
            "summary": required_id,
            "why_incomplete": f"see {required_id}-extra",
        }]

        result = self.mod.validate_interview("coder", report, audit)

        self.assertFalse(result["valid"])
        self.assertEqual(result["missing_detail_ids"]["incomplete_tasks"], [required_id])

    def test_autonomous_failure_rejects_prefixed_and_suffixed_ids(self):
        """A near-match in every documented field still counts as missing."""
        required_id = PRODUCT_IDS[0]
        audit = _audit()
        audit["profiles"]["product"]["cron_failures"] = [
            audit["profiles"]["product"]["cron_failures"][0]
        ]
        for near_miss in (f"prefix-{required_id}", f"{required_id}-extra"):
            with self.subTest(near_miss=near_miss):
                report = _report("product", [required_id], "autonomous_failures")
                report["autonomous_failures"] = [{
                    "summary": f"eng-completion execution {near_miss} stopped",
                    "evidence": near_miss,
                    "suggested_fix": "retry",
                }]

                result = self.mod.validate_interview("product", report, audit)

                self.assertFalse(result["valid"])
                self.assertEqual(
                    result["missing_detail_ids"]["autonomous_failures"],
                    [required_id],
                )

    def test_exact_id_field_wins_even_when_evidence_has_a_near_match(self):
        required_id = PRODUCT_IDS[0]
        audit = _audit()
        audit["profiles"]["product"]["cron_failures"] = [
            audit["profiles"]["product"]["cron_failures"][0]
        ]
        report = _report("product", [required_id], "autonomous_failures")
        report["autonomous_failures"] = [{
            "id": required_id,
            "summary": f"eng-completion execution {required_id} stopped",
            "evidence": f"{required_id}-extra context",
            "suggested_fix": "retry",
        }]

        result = self.mod.validate_interview("product", report, audit)

        self.assertTrue(result["valid"], result["errors"])

    def test_summary_id_does_not_substitute_for_documented_exact_id_field(self):
        required_id = PRODUCT_IDS[0]
        audit = _audit()
        audit["profiles"]["product"]["cron_failures"] = [
            audit["profiles"]["product"]["cron_failures"][0]
        ]
        report = _report("product", [required_id], "autonomous_failures")
        report["autonomous_failures"] = [{
            "summary": f"eng-completion execution {required_id} stopped autonomously",
            "evidence": "cron runner reported failure",
            "suggested_fix": "retry",
        }]

        result = self.mod.validate_interview("product", report, audit)

        self.assertFalse(result["valid"])
        self.assertEqual(result["missing_detail_ids"]["autonomous_failures"], [required_id])

    def test_autonomous_failure_rejects_near_match_in_summary(self):
        required_id = PRODUCT_IDS[0]
        audit = _audit()
        audit["profiles"]["product"]["cron_failures"] = [
            audit["profiles"]["product"]["cron_failures"][0]
        ]
        report = _report("product", [required_id], "autonomous_failures")
        report["autonomous_failures"] = [{
            "summary": f"eng-completion execution {required_id}-extra stopped",
            "evidence": "cron runner reported failure",
            "suggested_fix": "retry",
        }]

        result = self.mod.validate_interview("product", report, audit)

        self.assertFalse(result["valid"])
        self.assertEqual(result["missing_detail_ids"]["autonomous_failures"], [required_id])

    def test_wrong_id_in_record_never_satisfies_required_id(self):
        required_id = PRODUCT_IDS[0]
        audit = _audit()
        audit["profiles"]["product"]["cron_failures"] = [
            audit["profiles"]["product"]["cron_failures"][0]
        ]
        report = _report("product", [required_id], "autonomous_failures")
        report["autonomous_failures"] = [{
            "summary": "some other execution entirely",
            "evidence": "execution_id=different_run_001",
            "suggested_fix": "retry",
        }]

        result = self.mod.validate_interview("product", report, audit)

        self.assertFalse(result["valid"])
        self.assertEqual(result["missing_detail_ids"]["autonomous_failures"], [required_id])

    def test_structured_lifecycle_marker_not_lexical_words_selects_incomplete_field(self):
        session_id = "20260902_190251_66cec0"
        audit = {
            "profiles": {
                "qa": {
                    "session_failures": [{
                        "id": session_id,
                        "title": "user wrote failed and needs_input",
                        "fail_hits": ["tool:terminal:exit_code=1"],
                    }],
                    "cron_failures": [],
                    "kanban_failures": [],
                }
            }
        }
        report = _report("qa", [session_id], "autonomous_failures")

        result = self.mod.validate_interview("qa", report, audit)

        self.assertTrue(result["valid"])
        self.assertEqual(result["missing_detail_ids"]["incomplete_tasks"], [])

    def test_valid_full_product_reconciliation(self):
        result = self.mod.validate_interview(
            "product",
            _report("product", PRODUCT_IDS, "autonomous_failures"),
            _audit(),
        )
        self.assertTrue(result["valid"])
        self.assertEqual(result["required_ids"], PRODUCT_IDS)
        self.assertEqual(result["errors"], [])

    def test_valid_full_coder_reconciliation(self):
        report = _report("coder", CODER_IDS, "incomplete_tasks")
        report["autonomous_failures"] = _report(
            "coder", CODER_IDS, "autonomous_failures"
        )["autonomous_failures"]

        result = self.mod.validate_interview("coder", report, _audit())

        self.assertTrue(result["valid"], result["errors"])

    def test_missing_profile_audit_slice_cannot_be_accepted_clean(self):
        result = self.mod.validate_interview(
            "product",
            _report("product", [], "autonomous_failures"),
            {"profiles": {}},
        )
        self.assertFalse(result["valid"])
        self.assertIn("audit has no structured profile slice", result["errors"])

    def test_every_roster_profile_with_empty_slice_validates_clean_report(self):
        """All-installed-profile contract: an empty slice must validate a
        clean report for every one of the 13 installed profiles."""
        audit = {
            "profiles": {
                name: {"sessions": [], "session_failures": [], "cron_failures": [], "kanban_failures": []}
                for name in FULL_ROSTER
            }
        }
        for name in FULL_ROSTER:
            with self.subTest(profile=name):
                result = self.mod.validate_interview(
                    name, _report(name, [], "autonomous_failures"), audit
                )
                self.assertTrue(result["valid"], result["errors"])

    def test_malformed_json_is_rejected_and_builds_grill_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = root / "audit.json"
            report = root / "report.json"
            grill_base = root / "grill.txt"
            grill_output = root / "generated-grill.txt"
            audit.write_text(json.dumps(_audit()), encoding="utf-8")
            report.write_text('{"profile":"product"', encoding="utf-8")
            grill_base.write_text("GRILL BASE\n", encoding="utf-8")

            run = subprocess.run(
                [
                    "python3", str(SCRIPT), "product", str(report),
                    "--audit", str(audit),
                    "--grill-prompt", str(grill_base),
                    "--grill-output", str(grill_output),
                ],
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(run.returncode, 1)
            payload = json.loads(run.stdout)
            self.assertFalse(payload["valid"])
            self.assertIn("malformed interview JSON", payload["errors"][0])
            self.assertIn("GRILL BASE", grill_output.read_text(encoding="utf-8"))
            self.assertIn("malformed interview JSON", grill_output.read_text(encoding="utf-8"))

    def test_build_interview_includes_exact_structured_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            store = home / ".hermes" / "rsi"
            (store / "audit").mkdir(parents=True)
            (store / "interview-prompt.txt").write_text("BASE\n", encoding="utf-8")
            (store / "audit" / "latest.json").write_text(
                json.dumps(_audit()), encoding="utf-8"
            )
            env = dict(os.environ, HOME=str(home))

            run = subprocess.run(
                ["python3", str(BUILDER), "product"],
                text=True,
                capture_output=True,
                check=False,
                env=env,
            )

            self.assertEqual(run.returncode, 0, run.stderr)
            for execution_id in PRODUCT_IDS:
                self.assertIn(execution_id, run.stdout)

    def test_validation_is_read_only_without_grill_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = root / "audit.json"
            report = root / "report.json"
            audit.write_text(json.dumps(_audit(), sort_keys=True), encoding="utf-8")
            report.write_text(
                json.dumps(_report("product", PRODUCT_IDS, "autonomous_failures"), sort_keys=True),
                encoding="utf-8",
            )
            before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in (audit, report)}

            run = subprocess.run(
                ["python3", str(SCRIPT), "product", str(report), "--audit", str(audit)],
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertTrue(json.loads(run.stdout)["valid"])
            after = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in (audit, report)}
            self.assertEqual(before, after)
            self.assertEqual(sorted(root.iterdir()), [audit, report])


if __name__ == "__main__":
    unittest.main()
