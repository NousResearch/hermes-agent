#!/usr/bin/env python3
"""End-to-end privacy flow: brief → receipt → envelope → learning ledger (stdlib)."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
PY = sys.executable

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import privacy_runtime  # noqa: E402


def run_cli(args: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [PY, str(SCRIPT_DIR / "neon_genie.py"), *args],
        cwd=cwd or ROOT,
        capture_output=True,
        text=True,
    )


class PrivacyIntegrationTests(unittest.TestCase):
    def test_brief_receipt_envelope_ledger_local_only_recipe(self):
        tmp = Path(tempfile.mkdtemp(prefix="ng-priv-int-"))
        out = tmp / "product-audit"
        try:
            r = run_cli(["do", "recipe", "--name", "product-audit", "--out", str(out)])
            self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
            receipt = json.loads((out / "run-receipt.json").read_text(encoding="utf-8"))
            envelope = json.loads((out / "run-envelope.json").read_text(encoding="utf-8"))

            self.assertEqual(receipt["privacy_mode"], "local_only")
            self.assertEqual(receipt["telemetry_status"], "disabled")
            self.assertIn("privacy", receipt["profiles_loaded"])
            self.assertEqual(receipt["privacy"]["contract_version"], "1.0.0")
            self.assertEqual(
                receipt["privacy_unknowns"]["provider_retention"], "NOT_COMPUTABLE"
            )
            self.assertEqual(receipt["external_actions"], [])
            self.assertTrue(receipt.get("learning_ledger_disclosure"))

            self.assertEqual(envelope["privacy_mode"], "local_only")
            self.assertEqual(envelope["telemetry_status"], "disabled")
            self.assertEqual(envelope["authority"], "advisory_only")
            self.assertFalse(envelope["grants_execution"])
            privacy_runtime.assert_envelope_privacy(envelope["privacy"])

            ledger = tmp / "learning-ledger.jsonl"
            lr = run_cli(
                [
                    "do",
                    "learn",
                    "--class",
                    "other",
                    "--summary",
                    "integration privacy disclosure check",
                    "--envelope",
                    str(out / "run-envelope.json"),
                    "--ledger",
                    str(ledger),
                ]
            )
            self.assertEqual(lr.returncode, 0, lr.stderr + lr.stdout)
            entry = json.loads(ledger.read_text(encoding="utf-8").strip().splitlines()[-1])
            self.assertEqual(entry["privacy_mode"], "local_only")
            self.assertTrue(entry["learning_local_only"])
            self.assertTrue(entry["learning_auto_apply_forbidden"])
            self.assertEqual(entry["host_provider_retention"], "NOT_COMPUTABLE")
            self.assertEqual(entry["privacy_contract_version"], "1.0.0")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_external_research_brief_flows_consents(self):
        tmp = Path(tempfile.mkdtemp(prefix="ng-priv-ext-"))
        try:
            receipt_path = tmp / "run-receipt.json"
            brief = ROOT / "examples" / "privacy-external-research.brief.yaml"
            r = run_cli(
                [
                    "do",
                    "receipt",
                    "--profiles",
                    "core,privacy,commercial",
                    "--brief",
                    str(brief),
                    "--out",
                    str(receipt_path),
                ]
            )
            self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            self.assertEqual(receipt["privacy_mode"], "external_research_allowed")
            self.assertTrue(receipt["privacy"]["egress"]["allowed"])
            self.assertIn("example.com", receipt["privacy"]["egress"]["approved_domains"])
            self.assertEqual(len(receipt["privacy"]["consents"]), 1)
            self.assertEqual(receipt["privacy"]["consents"][0]["scope"], "purpose_bound")

            # Build a minimal run dir + envelope
            out = tmp / "run"
            out.mkdir()
            shutil.copy2(receipt_path, out / "run-receipt.json")
            (out / "recipe-summary.json").write_text(
                json.dumps(
                    {
                        "recipe": "privacy-smoke",
                        "brief": "examples/privacy-external-research.brief.yaml",
                        "status": "PASS",
                        "authority": "advisory_only",
                        "grants_execution": False,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            er = run_cli(["do", "envelope", "--out-dir", str(out), "--write"])
            self.assertEqual(er.returncode, 0, er.stderr + er.stdout)
            env = json.loads((out / "run-envelope.json").read_text(encoding="utf-8"))
            self.assertEqual(env["privacy_mode"], "external_research_allowed")
            self.assertEqual(len(env["privacy"]["consents"]), 1)
            privacy_runtime.assert_envelope_privacy(env["privacy"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_external_action_enumeration_on_receipt(self):
        tmp = Path(tempfile.mkdtemp(prefix="ng-priv-ea-"))
        try:
            action = privacy_runtime.build_external_action(
                provider="host",
                tool_class="web_search",
                destination="example.com",
                purpose="public market research for pricing band",
                source_class="public_research",
                classification="public",
                decision="ALLOW",
                redaction_status="not_required",
                recorded_at="2026-07-31T12:00:00Z",
                safe_query="developer tools SaaS public pricing examples 2026",
            )
            actions_path = tmp / "actions.json"
            actions_path.write_text(json.dumps([action], indent=2) + "\n", encoding="utf-8")
            receipt_path = tmp / "run-receipt.json"
            r = run_cli(
                [
                    "do",
                    "receipt",
                    "--profiles",
                    "core,privacy",
                    "--privacy-mode",
                    "external_research_allowed",
                    "--external-actions",
                    str(actions_path),
                    "--out",
                    str(receipt_path),
                ]
            )
            self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            self.assertEqual(len(receipt["external_actions"]), 1)
            ea = receipt["external_actions"][0]
            for field in privacy_runtime.EXTERNAL_ACTION_REQUIRED:
                self.assertIn(field, ea)
                self.assertTrue(ea[field])
            self.assertFalse(ea.get("private_source_persisted", False))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_local_only_rejects_external_actions_on_receipt(self):
        tmp = Path(tempfile.mkdtemp(prefix="ng-priv-block-"))
        try:
            action = privacy_runtime.build_external_action(
                provider="host",
                tool_class="web_search",
                destination="example.com",
                purpose="research",
                source_class="public_research",
                classification="public",
                decision="ALLOW",
                redaction_status="not_required",
                recorded_at="2026-07-31T12:00:00Z",
            )
            actions_path = tmp / "actions.json"
            actions_path.write_text(json.dumps([action]) + "\n", encoding="utf-8")
            r = run_cli(
                [
                    "do",
                    "receipt",
                    "--profiles",
                    "core,privacy",
                    "--privacy-mode",
                    "local_only",
                    "--external-actions",
                    str(actions_path),
                    "--out",
                    str(tmp / "run-receipt.json"),
                ]
            )
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("NG-PRIVACY-001", r.stderr + r.stdout)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_sample_privacy_context_validates(self):
        r = run_cli(
            [
                "do",
                "validate",
                "--packet",
                str(ROOT / "examples" / "packets" / "sample-privacy-context.json"),
                "--type",
                "privacy",
            ]
        )
        self.assertEqual(r.returncode, 0, r.stderr + r.stdout)

    def test_redact_then_allow_safe_query_not_persisted_in_action(self):
        secret = "sk_abcdefghijklmnopqrstuvwxyz123456"
        payload = f"public market size developer tools contact notes {secret}"
        ctx = privacy_runtime.default_privacy_context(mode="external_research_allowed")
        # secret forces BLOCK, not REDACT
        self.assertEqual(
            privacy_runtime.privacy_egress_check(payload, "example.com", "research", ctx),
            "BLOCK",
        )
        mixed = "public market size developer tools alice@example.com"
        prep = privacy_runtime.prepare_egress(
            mixed,
            "example.com",
            "public market research",
            ctx,
            recorded_at="2026-07-31T12:00:00Z",
        )
        self.assertEqual(prep["decision"], "REDACT_THEN_ALLOW")
        self.assertNotIn("alice@example.com", prep["safe_query"] or "")
        self.assertIn("public market size", prep["safe_query"] or "")
        blob = json.dumps(prep["external_action"])
        self.assertNotIn("alice@example.com", blob)
        self.assertFalse(prep["external_action"]["private_source_persisted"])


if __name__ == "__main__":
    unittest.main()
