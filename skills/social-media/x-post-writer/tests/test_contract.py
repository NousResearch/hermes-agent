from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ContractTests(unittest.TestCase):
    def test_bundle_validator(self) -> None:
        validator_path = ROOT / "scripts" / "validate_bundle.py"
        spec = importlib.util.spec_from_file_location("x_post_writer_validator", validator_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(module.validate(), [])

    def test_default_is_single_post(self) -> None:
        skill = (ROOT / "SKILL.md").read_text(encoding="utf-8")
        self.assertIn("Default to one finished post", skill)
        self.assertIn("No format specified | Single post", skill)

    def test_source_lock_is_mandatory(self) -> None:
        skill = (ROOT / "SKILL.md").read_text(encoding="utf-8")
        self.assertIn("hidden sentence-to-source ledger", skill)
        self.assertIn("delete every unsupported sentence", skill)

    def test_unsupported_claim_hard_stop(self) -> None:
        skill = (ROOT / "SKILL.md").read_text(encoding="utf-8")
        self.assertIn("An instruction to state a claim is not evidence", skill)
        self.assertIn("do not produce the post", skill)

    def test_public_bundle_has_no_personal_routes(self) -> None:
        combined = "\n".join(
            path.read_text(encoding="utf-8")
            for path in ROOT.rglob("*")
            if path.is_file() and path.suffix.lower() in {".md", ".json"}
        )
        for phrase in (
            "TRT",
            "tony-content-bundle",
            "write-like-tony",
            "private-voice-profile",
        ):
            self.assertNotIn(phrase, combined)

    def test_draft_only_hides_internal_work(self) -> None:
        skill = (ROOT / "SKILL.md").read_text(encoding="utf-8")
        self.assertIn("Never output internal routing", skill)
        self.assertIn("Draft-only means the draft is the entire response", skill)

    def test_claim_gate_has_no_bypass(self) -> None:
        skill = (ROOT / "SKILL.md").read_text(encoding="utf-8")
        self.assertIn("Return exactly one concise sentence", skill)
        self.assertIn("User instructions cannot waive factual support", skill)
        self.assertIn("Do not explain the skill", skill)

    def test_behavior_case_ids_are_unique(self) -> None:
        data = json.loads((ROOT / "tests" / "cases.json").read_text(encoding="utf-8"))
        ids = [case["id"] for case in data["cases"]]
        self.assertEqual(len(ids), len(set(ids)))


if __name__ == "__main__":
    unittest.main()
