#!/usr/bin/env python3
"""Tests for the adjudicator eval harness.

  * structural_checks / adjudicate / evaluate_case logic — judge mocked, no network.
  * Live (-gated)  — run one underspecified case end-to-end and adjudicate it.

Run:  python3 tests/test_evals.py -v
      uv run --with pytest python3 -m pytest tests/test_evals.py -v -k "not live"
"""

import os
import sys
import unittest
from unittest import mock

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "scripts"))
sys.path.insert(0, os.path.join(_HERE, "..", "evals"))

try:
    import pipeline  # noqa: E402
    import infogain  # noqa: E402
    import adjudicator  # noqa: E402
    _OK = True
except SystemExit:
    _OK = False


def _good_result():
    return {
        "framing": {"goal": "g", "decision": "d", "baseline_plan": "do X then Y"},
        "config": {"hard_cap": 7, "pre_answer_threshold": 0.60, "discard_threshold": 0.40},
        "discarded_count": 1,
        "bucket": [
            {"question": "Q1", "target": "scope", "value": 0.72, "recommendation": "PRE_ANSWER",
             "modal_answer": {"answer": "a1"}},
            {"question": "Q2", "target": "data", "value": 0.50, "recommendation": "ASSUME_DEFAULT",
             "modal_answer": {"answer": "a2"}},
        ],
    }


@unittest.skipUnless(_OK, "skill scripts not importable")
class TestStructuralChecks(unittest.TestCase):
    def setUp(self):
        self.case = {"id": "c", "expectation": "underspecified",
                     "expect_min_bucket": 2, "expect_max_bucket": 7}

    def test_good_passes(self):
        out = adjudicator.structural_checks(_good_result(), self.case)
        self.assertTrue(out["passed"], out["failures"])

    def test_empty_baseline_fails(self):
        r = _good_result(); r["framing"]["baseline_plan"] = ""
        self.assertFalse(adjudicator.structural_checks(r, self.case)["passed"])

    def test_value_out_of_range_and_unsorted(self):
        r = _good_result()
        r["bucket"][0]["value"] = 1.5
        r["bucket"][1]["value"] = 0.9  # now higher than the (clamped-context) first
        self.assertFalse(adjudicator.structural_checks(r, self.case)["passed"])

    def test_pre_answer_below_threshold_fails(self):
        r = _good_result(); r["bucket"][0]["value"] = 0.45  # PRE_ANSWER but < 0.60
        fails = adjudicator.structural_checks(r, self.case)["failures"]
        self.assertTrue(any("PRE_ANSWER" in f for f in fails))

    def test_duplicate_target_fails(self):
        r = _good_result(); r["bucket"][1]["target"] = "scope"  # dup of q0
        fails = adjudicator.structural_checks(r, self.case)["failures"]
        self.assertTrue(any("duplicate target" in f for f in fails))

    def test_calibration_band(self):
        # well-specified case expecting ≤1, but bucket has 2 → fail
        case = {"id": "w", "expectation": "well-specified",
                "expect_min_bucket": 0, "expect_max_bucket": 1}
        fails = adjudicator.structural_checks(_good_result(), case)["failures"]
        self.assertTrue(any("calibration" in f for f in fails))


@unittest.skipUnless(_OK, "skill scripts not importable")
class TestAdjudicatorMocked(unittest.TestCase):
    def _judge(self, scores):
        crit = {k: {"score": v, "reason": "r"} for k, v in scores.items()}
        return {"criteria": crit, "summary": "ok"}

    def test_accept_when_required_clear_floor(self):
        good = self._judge({"framing_accuracy": 0.8, "question_relevance": 0.7,
                            "value_justified": 0.4, "diversity": 0.9, "calibration": 0.7})
        with mock.patch.object(pipeline, "_call_json", return_value=(good, None)):
            v = adjudicator.adjudicate({"problem": "p", "expectation": "underspecified"},
                                       _good_result(), "deepseek")
        self.assertTrue(v["acceptable"])  # advisory value_justified=0.4 doesn't block

    def test_reject_when_required_below_floor(self):
        bad = self._judge({"framing_accuracy": 0.3, "question_relevance": 0.7,
                           "value_justified": 0.9, "diversity": 0.9, "calibration": 0.7})
        with mock.patch.object(pipeline, "_call_json", return_value=(bad, None)):
            v = adjudicator.adjudicate({"problem": "p", "expectation": "underspecified"},
                                       _good_result(), "deepseek")
        self.assertFalse(v["acceptable"])

    def test_judge_error_not_acceptable(self):
        with mock.patch.object(pipeline, "_call_json", return_value=(None, "boom")):
            v = adjudicator.adjudicate({"problem": "p", "expectation": "underspecified"},
                                       _good_result(), "deepseek")
        self.assertFalse(v["acceptable"])
        self.assertTrue(v["error"])

    def test_evaluate_case_combines(self):
        good = self._judge({"framing_accuracy": 0.8, "question_relevance": 0.8,
                            "value_justified": 0.8, "diversity": 0.8, "calibration": 0.8})
        case = {"id": "c", "expectation": "underspecified",
                "expect_min_bucket": 2, "expect_max_bucket": 7}
        with mock.patch.object(pipeline, "_call_json", return_value=(good, None)):
            v = adjudicator.evaluate_case(case, _good_result(), "deepseek")
        self.assertTrue(v["acceptable"])
        # structural failure should veto even a happy judge
        bad_struct = _good_result(); bad_struct["framing"]["goal"] = ""
        with mock.patch.object(pipeline, "_call_json", return_value=(good, None)):
            v2 = adjudicator.evaluate_case(case, bad_struct, "deepseek")
        self.assertFalse(v2["acceptable"])


@unittest.skipUnless(_OK, "skill scripts not importable")
class TestEvalLive(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reachable = pipeline.ollama_reachable(timeout=5)

    def setUp(self):
        if not self.reachable:
            self.skipTest("Ollama not reachable")

    def test_underspecified_case_end_to_end(self):
        cfg = dict(infogain.DEFAULTS)
        for k in ("plan_model", "question_gen_model", "answer_model", "value_judge_model"):
            cfg[k] = "fast"
        cfg.update(max_rounds=1, questions_per_round=4, answers_per_question=3, min_bucket_size=1)
        case = {"id": "live", "expectation": "underspecified",
                "expect_min_bucket": 1, "expect_max_bucket": 7,
                "problem": "Set up monitoring and alerting for our microservices."}
        result = infogain.run(case["problem"], cfg)
        verdict = adjudicator.evaluate_case(case, result, judge_model="fast", timeout=120)
        # deterministic structural checks must pass; judge must return scored criteria
        self.assertTrue(verdict["structural"]["passed"], verdict["structural"]["failures"])
        self.assertIsNone(verdict["judged"].get("error"))
        self.assertEqual(set(verdict["judged"]["criteria"]), set(adjudicator.ALL_CRITERIA))
        self.assertIsInstance(verdict["acceptable"], bool)


if __name__ == "__main__":
    unittest.main(verbosity=2)
