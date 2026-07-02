#!/usr/bin/env python3
"""Pin the families/premortem plumbing in the eval harnesses (#25 eval ladder).

The harnesses build cfg straight from infogain.DEFAULTS, which has no 'families' key — so
before this plumbing they always ran the flat generator and never exercised the lens layer.
These tests assert (no model calls, run() stubbed):
  * infogain.families_cfg honors premortem/families_model without mutating FAMILIES
  * each harness's --families/--premortem flags land in the cfg passed to infogain.run
  * without --families the cfg stays flat (no 'families' key — old behavior byte-identical)

Run:  python3 tests/test_eval_families.py -v
"""

import argparse
import os
import sys
import unittest
from unittest import mock

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "scripts"))
sys.path.insert(0, os.path.join(_HERE, "..", "evals"))

import infogain  # noqa: E402
import run_evals  # noqa: E402
import score_scan  # noqa: E402
import testbank  # noqa: E402
import validate_evsi  # noqa: E402


class TestFamiliesCfg(unittest.TestCase):
    def test_defaults(self):
        fam = infogain.families_cfg()
        self.assertTrue(fam["enabled"])
        self.assertEqual(fam["premortem"], "auto")
        self.assertEqual(fam["families_model"], infogain.FAMILIES["families_model"])

    def test_premortem_and_model_override(self):
        fam = infogain.families_cfg("on", families_model="fast")
        self.assertEqual(fam["premortem"], "on")
        self.assertEqual(fam["families_model"], "fast")

    def test_source_dict_not_mutated(self):
        before = dict(infogain.FAMILIES)
        infogain.families_cfg("off", families_model="fast")
        self.assertEqual(infogain.FAMILIES, before)


def _stub_result():
    return {"framing": {"baseline_plan": "b"}, "bucket": [], "all_scored": []}


class TestHarnessWiring(unittest.TestCase):
    def _capture(self, module_main, argv, target="infogain.run"):
        seen = {}

        def fake_run(problem, cfg, **kw):
            seen["cfg"] = cfg
            return _stub_result()

        with mock.patch(target, side_effect=fake_run):
            module_main(argv)
        return seen.get("cfg")

    def test_score_scan_families_on(self):
        pid = testbank.BANK[0]["id"]
        cfg = self._capture(score_scan.main, ["--ids", pid, "--families", "--premortem", "on"])
        self.assertIsNotNone(cfg)
        self.assertEqual(cfg["families"]["premortem"], "on")
        self.assertEqual(cfg["families"]["families_model"], "fast")  # follows --gen-model default

    def test_score_scan_flat_by_default(self):
        pid = testbank.BANK[0]["id"]
        cfg = self._capture(score_scan.main, ["--ids", pid])
        self.assertIsNotNone(cfg)
        self.assertNotIn("families", cfg)

    def test_validate_evsi_families_off_arm(self):
        pid = testbank.ALL[0]["id"]
        cfg = self._capture(validate_evsi.main, ["--prompt-ids", pid, "--families", "--premortem", "off"])
        self.assertIsNotNone(cfg)
        self.assertEqual(cfg["families"]["premortem"], "off")

    def test_run_evals_build_cfg(self):
        args = argparse.Namespace(gen_model="fast", max_rounds=1, answers_per_question=2,
                                  families=True, premortem="auto")
        cfg = run_evals.build_cfg(args)
        self.assertEqual(cfg["families"]["premortem"], "auto")
        args_flat = argparse.Namespace(gen_model="fast", max_rounds=1, answers_per_question=2,
                                       families=False, premortem="auto")
        self.assertNotIn("families", run_evals.build_cfg(args_flat))

    def test_score_scan_row_carries_lens(self):
        rich = {"framing": {"baseline_plan": "b"}, "bucket": [],
                "all_scored": [{"question": "q?", "target": "t", "u": .5, "evsi": .4, "value": .45,
                                "derivable_prob": .1, "lens": "premortem", "answers": []}]}
        with mock.patch("infogain.run", return_value=rich):
            row = score_scan.scan_prompt({"id": "x", "cat": "c", "problem": "p"},
                                         dict(infogain.DEFAULTS))
        self.assertEqual(row["lens"], ["premortem"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
