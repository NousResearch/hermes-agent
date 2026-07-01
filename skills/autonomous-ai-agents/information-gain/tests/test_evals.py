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
    import analyze_evsi  # noqa: E402
    import validate_evsi  # noqa: E402
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
class TestAnalyzeEvsi(unittest.TestCase):
    """The #24 gate metric — ranks WITHIN-TASK on realized_stakes (change is within-task-dead)."""

    def _row(self, prompt, q, method, prob, pdelta, stakes, qv, rc, rstk):
        return {"prompt": prompt, "question": q, "method": method, "prob": prob,
                "projected_delta": pdelta, "stakes": stakes, "q_u": 0.8, "q_evsi": qv, "q_value": qv,
                "realized_change": rc, "realized_stakes": rstk,
                "realized_regret": round(rc * rstk, 3)}

    def test_by_question_aggregates_realized_and_projected_stakes(self):
        rows = [self._row("P", "q", "absolute", 0.75, 0.8, 0.6, 0.5, 1.0, 0.8),
                self._row("P", "q", "absolute", 0.25, 0.4, 0.2, 0.5, 0.0, 0.4)]
        qs = analyze_evsi.by_question(rows)
        self.assertEqual(len(qs), 1)
        # P'-weighted: 0.75*0.8 + 0.25*0.4 = 0.7 (realized_stakes); 0.75*0.6+0.25*0.2 = 0.5 (mean_stakes)
        self.assertAlmostEqual(qs[0]["realized_stakes"], 0.7, places=6)
        self.assertAlmostEqual(qs[0]["mean_stakes"], 0.5, places=6)

    def test_stakes_only_formula_reads_mean_stakes(self):
        self.assertIn("stakes-only", analyze_evsi.FORMULAS)
        self.assertEqual(analyze_evsi.FORMULAS["stakes-only"]({"mean_stakes": 0.42}), 0.42)

    def test_within_task_rhos_are_per_prompt(self):
        # two prompts, each 2 questions; q_value perfectly orders realized_stakes -> ρ=+1 each
        rows = []
        for pr in ("P1", "P2"):
            rows += [self._row(pr, "qa", "absolute", 1.0, 0.5, 0.9, 0.9, 0.5, 0.9),
                     self._row(pr, "qb", "absolute", 1.0, 0.5, 0.2, 0.2, 0.5, 0.2)]
        rhos = analyze_evsi.within_task_rhos(analyze_evsi.by_question(rows), "realized_stakes")
        self.assertEqual(set(rhos), {"P1", "P2"})
        for v in rhos.values():
            self.assertAlmostEqual(v, 1.0, places=6)

    def test_paired_guard_rejects_narrow_win(self):
        # a mean lifted by one outlier, with losses -> not "broad", not beyond ~1 SE
        st = analyze_evsi._paired([+0.9, -0.1, -0.1, -0.1])
        self.assertEqual((st["wins"], st["losses"]), (1, 3))
        self.assertLess(st["mean"], st["se"])  # beyond_noise is False -> gate keeps absolute

    def test_gate_runs_and_keys_on_regret(self):
        # smoke: two methods present -> ab_within_task executes without error
        rows = []
        for pr in ("P1", "P2", "P3"):
            for m in ("absolute", "pairwise"):
                rows += [self._row(pr, "qa", m, 1.0, 0.5, 0.9, 0.9, 0.5, 0.9),
                         self._row(pr, "qb", m, 1.0, 0.5, 0.2, 0.2, 0.5, 0.2)]
        analyze_evsi.ab_within_task(rows)  # must not raise
        # primary target = realized_regret (realized EVSI); change/stakes reported alongside
        self.assertEqual(analyze_evsi._GATE_TARGETS[0][0], "realized_regret")
        self.assertEqual({t[0] for t in analyze_evsi._GATE_TARGETS},
                         {"realized_regret", "realized_stakes", "realized_change"})


@unittest.skipUnless(_OK, "skill scripts not importable")
class TestValidateEvsiRows(unittest.TestCase):
    """The realized rows must carry lens/family so analyze_evsi can attribute value per lens."""

    def test_rows_carry_lens_and_family(self):
        record = {"question": "What breaks on token theft?", "target": "token_theft",
                  "lens": "premortem", "family": "Failure Hazards",
                  "answers": [{"answer": "rotate keys", "prob": 0.6, "delta_plan": 0.8, "stakes": 0.9}]}
        fake_result = {"framing": {"baseline_plan": "B"}, "all_scored": [record], "bucket": [record]}
        with mock.patch.object(validate_evsi.infogain, "run", return_value=fake_result), \
             mock.patch.object(validate_evsi.pipeline, "resolve_alias", side_effect=lambda m: m), \
             mock.patch.object(validate_evsi.pipeline, "frame_and_plan",
                               return_value=({"baseline_plan": "B'"}, None)), \
             mock.patch.object(validate_evsi, "change_judge", return_value=0.7), \
             mock.patch.object(validate_evsi, "stakes_judge", return_value=0.5):
            cfg = {"plan_model": "fast", "value_judge_model": "fast", "judge_timeout": 10}
            rows, _ = validate_evsi.run_prompt({"id": "add-auth", "problem": "p"}, cfg,
                                               judge_model="fast", max_answers=3, timeout=10,
                                               source="all_scored")
        self.assertTrue(rows)
        self.assertEqual(rows[0]["lens"], "premortem")
        self.assertEqual(rows[0]["family"], "Failure Hazards")
        self.assertEqual(rows[0]["realized_regret"], round(0.7 * 0.5, 3))

    def test_flat_generator_rows_have_empty_lens(self):
        record = {"question": "Q", "target": "t",  # no lens/family (flat generator)
                  "answers": [{"answer": "a", "prob": 0.5, "delta_plan": 0.4, "stakes": 0.3}]}
        fake_result = {"framing": {"baseline_plan": "B"}, "all_scored": [record]}
        with mock.patch.object(validate_evsi.infogain, "run", return_value=fake_result), \
             mock.patch.object(validate_evsi.pipeline, "resolve_alias", side_effect=lambda m: m), \
             mock.patch.object(validate_evsi.pipeline, "frame_and_plan",
                               return_value=({"baseline_plan": "B2"}, None)), \
             mock.patch.object(validate_evsi, "change_judge", return_value=0.2), \
             mock.patch.object(validate_evsi, "stakes_judge", return_value=0.2):
            cfg = {"plan_model": "fast", "value_judge_model": "fast", "judge_timeout": 10}
            rows, _ = validate_evsi.run_prompt({"id": "gmail-triage", "problem": "p"}, cfg,
                                               judge_model="fast", max_answers=3, timeout=10,
                                               source="all_scored")
        self.assertEqual(rows[0]["lens"], "")
        self.assertEqual(rows[0]["family"], "")


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
