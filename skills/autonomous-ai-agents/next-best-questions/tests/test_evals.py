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
    import rejudge  # noqa: E402
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

    def test_value_out_of_range_fails(self):
        r = _good_result()
        r["bucket"][0]["value"] = 1.5
        fails = adjudicator.structural_checks(r, self.case)["failures"]
        self.assertTrue(any("out of [0,1]" in f for f in fails))

    def test_unsorted_bucket_fails(self):
        # The old combined test never exercised this branch: an out-of-range q0 takes the
        # leading `if`, so `last` stays 2.0 and q1 can never exceed it. In-range values that
        # genuinely invert the order are required to reach `elif v > last`.
        r = _good_result()
        r["bucket"][0]["value"] = 0.45
        r["bucket"][0]["recommendation"] = "ASSUME_DEFAULT"   # keep other checks quiet
        r["bucket"][1]["value"] = 0.62
        r["bucket"][1]["recommendation"] = "PRE_ANSWER"
        fails = adjudicator.structural_checks(r, self.case)["failures"]
        self.assertTrue(any("not sorted" in f for f in fails), fails)

    def test_hard_cap_bad_rec_and_below_discard_fail(self):
        r = _good_result()
        r["result_config"] = None
        cfg = r.setdefault("config", {})
        cfg["hard_cap"] = 1                                    # bucket of 2 exceeds it
        r["bucket"][1]["recommendation"] = "MAYBE"             # invalid tag
        r["bucket"][1]["value"] = 0.1                          # kept but below discard 0.30
        fails = adjudicator.structural_checks(r, self.case)["failures"]
        self.assertTrue(any("exceeds hard_cap" in f for f in fails), fails)
        self.assertTrue(any("bad recommendation" in f for f in fails), fails)
        self.assertTrue(any("below discard" in f or "< discard" in f or "discard" in f
                            for f in fails), fails)

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

    def test_empty_bucket_relevance_is_vacuous(self):
        # bucket=0 on a well-specified control: question_relevance is N/A and how a judge
        # encodes that is model luck (fast scored it 0.0 while its reason said the behavior
        # was correct — the reverse-string CI flake). It must not decide acceptance.
        na_low = self._judge({"framing_accuracy": 0.9, "question_relevance": 0.0,
                              "value_justified": 1.0, "diversity": 1.0, "calibration": 1.0})
        empty = _good_result(); empty["bucket"] = []
        with mock.patch.object(pipeline, "_call_json", return_value=(na_low, None)):
            v = adjudicator.adjudicate({"problem": "p", "expectation": "well-specified"},
                                       empty, "deepseek")
        self.assertTrue(v["acceptable"])
        # ...but with kept questions the same scores still reject (relevance is real again)
        with mock.patch.object(pipeline, "_call_json", return_value=(na_low, None)):
            v2 = adjudicator.adjudicate({"problem": "p", "expectation": "underspecified"},
                                        _good_result(), "deepseek")
        self.assertFalse(v2["acceptable"])

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

    def _gate_rows(self, control_tracks_realized):
        """4 prompts × 3 questions × 2 methods; realized_regret 0.1/0.5/0.9 shared. One method's
        q_value tracks realized perfectly, the other inverts — a maximal, unambiguous gate case."""
        rows = []
        for pr in ("P1", "P2", "P3", "P4"):
            for qi, regret in enumerate((0.1, 0.5, 0.9)):
                tracking, inverted = regret, 1.0 - regret
                for method in ("absolute", "pairwise"):
                    is_control = method == "absolute"
                    qv = (tracking if (is_control == control_tracks_realized) else inverted)
                    rows.append(self._row(pr, f"q{qi}", method, 1.0, 0.5, regret, qv,
                                          regret, 1.0))
        return rows

    def _capture_gate(self, rows):
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            analyze_evsi.ab_within_task(rows)
        return buf.getvalue()

    def test_gate_verdict_adopts_a_clear_broad_winner(self):
        out = self._capture_gate(self._gate_rows(control_tracks_realized=False))
        self.assertIn("ADOPT pairwise", out)         # challenger wins 4/4, Δρ +2.0

    def test_gate_verdict_keeps_control_when_challenger_loses(self):
        out = self._capture_gate(self._gate_rows(control_tracks_realized=True))
        self.assertIn("keep absolute", out)
        self.assertNotIn("ADOPT", out)

    def test_split_methods_control_detection(self):
        # the shipped default present in the run is the control; everything else challenges it
        self.assertEqual(analyze_evsi.split_methods({"absolute", "pairwise"}),
                         ("absolute", ["pairwise"]))
        self.assertEqual(analyze_evsi.split_methods({"stated", "sampled"}),
                         ("stated", ["sampled"]))
        self.assertEqual(analyze_evsi.split_methods({"absolute", "solution"}),
                         ("absolute", ["solution"]))

    def test_gate_generalizes_to_probs_methods(self):
        # #26 shape: methods stated/sampled — the verdict block must run with control=stated
        rows = []
        for pr in ("P1", "P2", "P3"):
            for m in ("stated", "sampled"):
                rows += [self._row(pr, "qa", m, 1.0, 0.5, 0.9, 0.9, 0.5, 0.9),
                         self._row(pr, "qb", m, 1.0, 0.5, 0.2, 0.2, 0.5, 0.2)]
        analyze_evsi.ab_within_task(rows)  # must not raise


@unittest.skipUnless(_OK, "skill scripts not importable")
class TestSelectionPolicies(unittest.TestCase):
    """#23 evidence machinery: policy pickers + realized_regret capture accounting."""

    def _q(self, prompt, qv, regret):
        return {"prompt": prompt, "question": f"q{qv}", "q_value": qv, "realized_regret": regret}

    def test_pickers(self):
        qs = [self._q("P", v, v) for v in (0.9, 0.6, 0.5, 0.2, 0.1)]
        self.assertEqual({q["q_value"] for q in analyze_evsi._pick_abs(qs, 0.30)}, {0.9, 0.6, 0.5})
        self.assertEqual(len(analyze_evsi._pick_topk(qs, 3)), 3)
        # rel 0.6*top = 0.54 -> keeps 0.9, 0.6; hybrid floor cannot go below the abs floor
        self.assertEqual({q["q_value"] for q in analyze_evsi._pick_rel(qs, 0.6)}, {0.9, 0.6})
        self.assertEqual({q["q_value"] for q in analyze_evsi._pick_rel(qs, 0.1, abs_floor=0.30)},
                         {0.9, 0.6, 0.5})

    def test_capture_accounting(self):
        # one prompt, 4 questions; all positive regret lives in the top two by q_value
        qs = [self._q("P", 0.9, 0.8), self._q("P", 0.7, 0.4),
              self._q("P", 0.2, 0.0), self._q("P", 0.1, 0.0)]
        out = analyze_evsi.selection_policies(qs, min_q=4)
        self.assertIsNotNone(out)
        self.assertAlmostEqual(out["top3-rank"]["capture"], 1.0, places=6)
        self.assertAlmostEqual(out["rel>=0.6*top"]["capture"], 1.0, places=6)
        self.assertEqual(out["top3-rank"]["mean_kept"], 3.0)
        self.assertEqual(out["rel>=0.6*top"]["mean_kept"], 2.0)  # tighter set, same capture

    def test_returns_none_when_too_few_questions(self):
        qs = [self._q("P", 0.9, 0.8)]
        self.assertIsNone(analyze_evsi.selection_policies(qs, min_q=4))


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
class TestAnalysisDrivers(unittest.TestCase):
    """p1a / p1c / per_lens are the printing drivers whose numbers decide shipping verdicts —
    previously only their pure helpers were unit-tested."""

    @staticmethod
    def _capture(fn, *a, **kw):
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            fn(*a, **kw)
        return buf.getvalue()

    def _q(self, prompt, qv, evsi, u, target, lens=""):
        return {"prompt": prompt, "question": f"q{qv}", "n_ans": 1, "lens": lens, "family": "",
                "q_u": u, "q_evsi": evsi, "q_value": qv, "max_delta": 0.5, "mean_delta": 0.5,
                "mean_stakes": 0.5, "realized_change": target, "realized_evsi": target,
                "realized_regret": target, "realized_stakes": target}

    def test_p1a_perfect_calibration_reports_unity(self):
        rows = [{"prompt": "P", "question": f"q{i}", "prob": 1.0,
                 "projected_delta": v, "realized_change": v, "stakes": 0.5,
                 "realized_stakes": v, "realized_regret": v, "q_u": 0.5, "q_evsi": v,
                 "q_value": v} for i, v in enumerate((0.1, 0.3, 0.5, 0.7, 0.9))]
        out = self._capture(analyze_evsi.p1a, rows, analyze_evsi.by_question(rows))
        self.assertIn("Pearson  r = +1.000", out)
        self.assertIn("Spearman ρ = +1.000", out)

    def test_p1c_ranks_the_formula_that_tracks_realized(self):
        qs = []
        for pr in ("P1", "P2"):
            for v in (0.2, 0.5, 0.8):
                # q_value tracks the target perfectly; EVSI inverts; U constant
                qs.append(self._q(pr, qv=v, evsi=1.0 - v, u=0.5, target=v))
        out = self._capture(analyze_evsi.p1c, qs, target_key="realized_regret")
        value_line = [ln for ln in out.splitlines() if "value √(U·EVSI)" in ln][0]
        self.assertIn("+1.000", value_line)
        self.assertIn("<- best", value_line)
        evsi_line = [ln for ln in out.splitlines() if "EVSI-only" in ln][0]
        self.assertIn("-1.000", evsi_line)

    def test_per_lens_attribution_means(self):
        qs = ([self._q("P", 0.5, 0.5, 0.5, target=0.8, lens="reach")] * 2
              + [self._q("P", 0.5, 0.5, 0.5, target=0.2, lens="scoped")] * 2)
        out = self._capture(analyze_evsi.per_lens, qs)
        reach_line = [ln for ln in out.splitlines() if "reach" in ln][0]
        scoped_line = [ln for ln in out.splitlines() if "scoped" in ln][0]
        self.assertIn("0.800", reach_line)
        self.assertIn("0.200", scoped_line)


@unittest.skipUnless(_OK, "skill scripts not importable")
class TestRealizedInstruments(unittest.TestCase):
    """change_judge + stakes_judge are the SHIPPED realized-measurement instruments every P1
    calibration and A/B gate rests on — previously only the graded variant was asserted."""

    def test_change_judge_prompt_parse_and_clamp(self):
        seen = {}

        def fake(model, p, timeout, num_predict=0, **kw):
            seen["prompt"] = p
            return {"change": 1.7}, None
        with mock.patch.object(validate_evsi.pipeline, "_call_json", side_effect=fake):
            out = validate_evsi.change_judge("P", "base", "new", "m", 10)
        self.assertEqual(out, 1.0)                          # clamped
        self.assertIn("RESPONSE A (baseline)", seen["prompt"])
        self.assertIn('{"change": 0.0}', seen["prompt"])    # the JSON contract
        with mock.patch.object(validate_evsi.pipeline, "_call_json", return_value=(None, "err")):
            self.assertIsNone(validate_evsi.change_judge("P", "b", "n", "m", 10))

    def test_stakes_judge_prompt_parse_and_clamp(self):
        seen = {}

        def fake(model, p, timeout, num_predict=0, **kw):
            seen["prompt"] = p
            return {"stakes": -0.3, "reason": "r"}, None
        with mock.patch.object(validate_evsi.pipeline, "_call_json", side_effect=fake):
            out = validate_evsi.stakes_judge("P", "base", "new", "m", 10)
        self.assertEqual(out, 0.0)                          # clamped
        self.assertIn("IGNORING how large the", seen["prompt"])  # size-independence anchor
        self.assertIn("0.6 = clearly wants the better one", seen["prompt"])
        with mock.patch.object(validate_evsi.pipeline, "_call_json", return_value=(None, "err")):
            self.assertIsNone(validate_evsi.stakes_judge("P", "b", "n", "m", 10))


@unittest.skipUnless(_OK, "skill scripts not importable")
class TestPreflight(unittest.TestCase):
    """Judge preflight: reasoning-channel models return empty content via raw_chat and would
    silently null every judged value (the #26 gpt-oss:20b burned smoke) — abort before any rows."""

    def test_empty_content_exits_2_naming_the_model(self):
        with mock.patch.object(validate_evsi.pipeline, "raw_chat",
                               return_value={"content": "", "elapsed": 0.1, "error": None}):
            with self.assertRaises(SystemExit) as ctx:
                validate_evsi.preflight_model("gpt-oss:20b", "judge")
        self.assertEqual(ctx.exception.code, 2)

    def test_normal_content_proceeds(self):
        with mock.patch.object(validate_evsi.pipeline, "raw_chat",
                               return_value={"content": "OK", "elapsed": 0.1, "error": None}):
            validate_evsi.preflight_model("fast", "judge")  # must not raise


@unittest.skipUnless(_OK, "skill scripts not importable")
class TestAbProbs(unittest.TestCase):
    """#26 probs A/B machinery: per-method prob rows, the swap lever, union answer selection."""

    def _record(self):
        return {"question": "Which DB?", "target": "db",
                "derivable_prob": 0.2, "prob_mode_used": "sampled",
                "answers": [{"answer": "pg", "prob": 0.9, "stated_prob": 0.5,
                             "delta_plan": 0.8, "stakes": 0.7},
                            {"answer": "mongo", "prob": 0.1, "stated_prob": 0.5,
                             "delta_plan": 0.3, "stakes": 0.2}]}

    def test_swap_probs_roundtrip_and_stated_mode_noop(self):
        rec = self._record()
        validate_evsi._swap_probs([rec])
        self.assertEqual((rec["answers"][0]["prob"], rec["answers"][0]["stated_prob"]), (0.5, 0.9))
        validate_evsi._swap_probs([rec])
        self.assertEqual((rec["answers"][0]["prob"], rec["answers"][0]["stated_prob"]), (0.9, 0.5))
        plain = {"answers": [{"answer": "x", "prob": 0.7}]}  # stated-mode record: no stated_prob
        validate_evsi._swap_probs([plain])
        self.assertEqual(plain["answers"][0]["prob"], 0.7)

    def test_tested_answers_union_of_top_n(self):
        a1, a2 = {"answer": "a1"}, {"answer": "a2"}
        q = {"answers": [a1, a2]}
        methods = {"sampled": {"a": {id(q): {id(a1): (0, 0, 0.9), id(a2): (0, 0, 0.1)}}},
                   "stated": {"a": {id(q): {id(a1): (0, 0, 0.2), id(a2): (0, 0, 0.8)}}}}
        got = validate_evsi._tested_answers(q, methods, max_answers=1)
        self.assertEqual({a["answer"] for a in got}, {"a1", "a2"})  # union of the two top-1s
        got = validate_evsi._tested_answers(q, {"stated": methods["stated"]}, max_answers=1)
        self.assertEqual([a["answer"] for a in got], ["a2"])  # single method = plain top-N

    def test_run_prompt_ab_probs_emits_per_method_prob(self):
        record = self._record()
        validate_evsi.voi.score_record(record)  # as the sampled run would have scored it
        fake_result = {"framing": {"baseline_plan": "B"}, "all_scored": [record]}
        with mock.patch.object(validate_evsi.infogain, "run", return_value=fake_result), \
             mock.patch.object(validate_evsi.pipeline, "resolve_alias", side_effect=lambda m: m), \
             mock.patch.object(validate_evsi.pipeline, "frame_and_plan",
                               return_value=({"baseline_plan": "B2"}, None)), \
             mock.patch.object(validate_evsi, "change_judge", return_value=0.6), \
             mock.patch.object(validate_evsi, "stakes_judge", return_value=0.5):
            cfg = {"plan_model": "fast", "value_judge_model": "fast", "judge_timeout": 10,
                   "answer_prob_mode": "sampled"}
            rows, _ = validate_evsi.run_prompt({"id": "add-auth", "problem": "p"}, cfg,
                                               judge_model="fast", max_answers=3, timeout=10,
                                               source="all_scored", ab_probs=True)
        by_m = {}
        for r in rows:
            by_m.setdefault(r["method"], {})[r["answer"]] = r
        self.assertEqual(set(by_m), {"sampled", "stated"})
        # prob is per-method: the same answer carries its arm's P(a)
        self.assertEqual(by_m["sampled"]["pg"]["prob"], 0.9)
        self.assertEqual(by_m["stated"]["pg"]["prob"], 0.5)
        # q-level scores move with P (entropy + EVSI weighting differ between arms)
        self.assertNotEqual(by_m["sampled"]["pg"]["q_value"], by_m["stated"]["pg"]["q_value"])
        # realized fields are shared (method-independent measurement)
        self.assertEqual(by_m["sampled"]["pg"]["realized_regret"],
                         by_m["stated"]["pg"]["realized_regret"])
        # and the record is restored to the sampled state after the stated re-score
        self.assertEqual(record["answers"][0]["prob"], 0.9)

    def test_run_prompt_ab_solution_emits_both_methods(self):
        # #27 shape: absolute snapshot first, then the solution re-judge mutates delta_plan and
        # the "solution" snapshot picks it up — two rows per pair with per-method projected_delta.
        record = {"question": "Q", "target": "t", "derivable_prob": 0.2,
                  "answers": [{"answer": "a", "prob": 1.0, "delta_plan": 0.8, "stakes": 0.7}]}
        validate_evsi.voi.score_record(record)
        fake_result = {"framing": {"baseline_plan": "B"}, "all_scored": [record]}

        def fake_solution_judge(p, f, b, recs, *a, **k):
            self.assertEqual(k.get("solutions"), ["B", "alt"])
            for q in recs:
                for ans in q["answers"]:
                    ans["delta_plan"] = 0.5  # 1 of 2 solutions invalidated
            return recs

        with mock.patch.object(validate_evsi.infogain, "run", return_value=fake_result), \
             mock.patch.object(validate_evsi.pipeline, "resolve_alias", side_effect=lambda m: m), \
             mock.patch.object(validate_evsi.pipeline, "sample_solutions",
                               return_value=["B", "alt"]), \
             mock.patch.object(validate_evsi.pipeline, "judge_plan_change_solution_batch",
                               side_effect=fake_solution_judge), \
             mock.patch.object(validate_evsi.pipeline, "frame_and_plan",
                               return_value=({"baseline_plan": "B2"}, None)), \
             mock.patch.object(validate_evsi, "change_judge", return_value=0.6), \
             mock.patch.object(validate_evsi, "stakes_judge", return_value=0.5):
            cfg = {"plan_model": "fast", "value_judge_model": "fast", "judge_timeout": 10,
                   "solution_samples": 2, "solution_temperature": 0.8, "plan_timeout": 10}
            rows, _ = validate_evsi.run_prompt({"id": "add-auth", "problem": "p"}, cfg,
                                               judge_model="fast", max_answers=3, timeout=10,
                                               source="all_scored", ab_solution=True)
        by_m = {r["method"]: r for r in rows}
        self.assertEqual(set(by_m), {"absolute", "solution"})
        self.assertEqual(by_m["absolute"]["projected_delta"], 0.8)
        self.assertEqual(by_m["solution"]["projected_delta"], 0.5)
        self.assertEqual(by_m["absolute"]["realized_regret"], by_m["solution"]["realized_regret"])


@unittest.skipUnless(_OK, "skill scripts not importable")
class TestGradedJudgeAndRejudge(unittest.TestCase):
    """De-saturated realized-change instrument (opt-in) + the offline instrument A/B."""

    def test_graded_judge_prompt_has_midscale_anchors(self):
        sent = {}

        def fake_call(model, prompt, timeout, **kw):
            sent["prompt"] = prompt
            return {"reason": "r", "change": 1.7}, None  # out-of-range -> must clamp

        with mock.patch.object(validate_evsi.pipeline, "_call_json", side_effect=fake_call):
            v = validate_evsi.change_judge_graded("p", "A", "B", "fast", 10)
        self.assertEqual(v, 1.0)  # clamped
        for anchor in ("0.2", "0.4", "0.7", "FULL scale"):
            self.assertIn(anchor, sent["prompt"])

    def test_run_prompt_uses_change_fn_and_keeps_responses(self):
        record = {"question": "Q", "target": "t",
                  "answers": [{"answer": "a", "prob": 0.5, "delta_plan": 0.4, "stakes": 0.3}]}
        fake_result = {"framing": {"baseline_plan": "B"}, "all_scored": [record]}
        with mock.patch.object(validate_evsi.infogain, "run", return_value=fake_result), \
             mock.patch.object(validate_evsi.pipeline, "resolve_alias", side_effect=lambda m: m), \
             mock.patch.object(validate_evsi.pipeline, "frame_and_plan",
                               return_value=({"baseline_plan": "B2"}, None)), \
             mock.patch.object(validate_evsi, "change_judge", return_value=0.2) as orig, \
             mock.patch.object(validate_evsi, "change_judge_graded", return_value=0.4) as graded, \
             mock.patch.object(validate_evsi, "stakes_judge", return_value=0.5):
            cfg = {"plan_model": "fast", "value_judge_model": "fast", "judge_timeout": 10}
            rows, _ = validate_evsi.run_prompt(
                {"id": "x", "problem": "p"}, cfg, judge_model="fast", max_answers=3, timeout=10,
                source="all_scored", keep_responses=True,
                change_fn=validate_evsi.change_judge_graded)
        graded.assert_called_once()
        orig.assert_not_called()
        self.assertEqual(rows[0]["realized_change"], 0.4)
        self.assertEqual(rows[0]["baseline_resp"], "B")
        self.assertEqual(rows[0]["new_resp"], "B2")

    def test_rejudge_compare_stats(self):
        pairs = [{"prompt": "P", "question": "q1", "q_value": 0.9, "orig": 1.0, "graded": 0.7},
                 {"prompt": "P", "question": "q2", "q_value": 0.5, "orig": 1.0, "graded": 0.4},
                 {"prompt": "P", "question": "q3", "q_value": 0.1, "orig": 0.0, "graded": 0.2}]
        stats = rejudge.compare(pairs)
        self.assertEqual(stats["n"], 3)
        # original: all three at the endpoints; graded: none
        self.assertEqual(stats["orig"]["frac_endpoints"], 1.0)
        self.assertEqual(stats["graded"]["frac_endpoints"], 0.0)
        self.assertIsNotNone(stats["agreement_rho"])
        self.assertIsNotNone(stats["qvalue_vs_graded_rho"])

    def test_rejudge_rows_calls_graded_judge_on_stored_texts(self):
        rows = [{"prompt": "P", "question": "q", "answer": "a", "q_value": 0.5,
                 "realized_change": 1.0, "baseline_resp": "base", "new_resp": "new"}]
        with mock.patch.object(rejudge.validate_evsi, "change_judge_graded",
                               return_value=0.6) as gj:
            pairs = rejudge.rejudge_rows(rows, "fast", 10, progress=False)
        gj.assert_called_once_with("P", "base", "new", "fast", 10)
        self.assertEqual(pairs[0]["orig"], 1.0)
        self.assertEqual(pairs[0]["graded"], 0.6)


@unittest.skipUnless(_OK, "skill scripts not importable")
class TestArchivalHelpers(unittest.TestCase):
    """The pure math inside the archival analysis scripts (saturation_scan, compare_domains,
    analyze_validity) produced SHIPPED evidence — the knee that justified 'modest breadth', the
    reordering count behind the U-inertness verdicts. Their drivers stay archival (see
    evals/README §Coverage); the math gets pinned here."""

    def test_saturation_knees(self):
        import saturation_scan
        steps = [{"samples": 1, "distinct_targets": 4, "max_value": 0.50},
                 {"samples": 2, "distinct_targets": 7, "max_value": 0.62},
                 {"samples": 3, "distinct_targets": 7, "max_value": 0.63},
                 {"samples": 5, "distinct_targets": 8, "max_value": 0.63}]
        self.assertEqual(saturation_scan._knee(steps), 2)        # coverage stalls at s=3
        self.assertEqual(saturation_scan._value_knee(steps), 2)  # value plateaus (+0.01 < eps)
        self.assertEqual(saturation_scan._knee([]), 0)
        self.assertEqual(saturation_scan._value_knee([]), 0)

    def test_compare_domains_reorderings_and_stats(self):
        import compare_domains
        # U constant -> value and EVSI order identically -> 0 flips
        inert = [{"prompt": "P", "q_value": v, "q_evsi": v} for v in (0.2, 0.5, 0.8)]
        flips, total = compare_domains.reorderings(inert)
        self.assertEqual((flips, total), (0, 3))
        # U flips one pair's order -> exactly that pair counts
        active = [{"prompt": "P", "q_value": 0.9, "q_evsi": 0.1},
                  {"prompt": "P", "q_value": 0.1, "q_evsi": 0.9}]
        self.assertEqual(compare_domains.reorderings(active)[0], 1)
        self.assertAlmostEqual(compare_domains._mean([1, 2, 3]), 2.0)
        self.assertAlmostEqual(compare_domains._frac([1, 2, 3, 4], lambda x: x > 2), 0.5)

    def test_analyze_validity_by_question_weighting(self):
        import analyze_validity
        rows = [{"prompt": "P", "question": "q", "cat": "devops", "prob": 0.75,
                 "realized_change": 1.0, "realized_regret": 0.8, "realized_stakes": 0.8,
                 "q_value": 0.5, "q_evsi": 0.5, "q_u": 0.5, "projected_delta": 0.5,
                 "stakes": 0.5},
                {"prompt": "P", "question": "q", "cat": "devops", "prob": 0.25,
                 "realized_change": 0.0, "realized_regret": 0.4, "realized_stakes": 0.4,
                 "q_value": 0.5, "q_evsi": 0.5, "q_u": 0.5, "projected_delta": 0.5,
                 "stakes": 0.5}]
        qs = analyze_validity.by_question(rows)
        self.assertEqual(len(qs), 1)
        self.assertAlmostEqual(qs[0]["rc"], 0.75, places=6)      # P'-weighted change
        self.assertAlmostEqual(qs[0]["regret"], 0.7, places=6)   # 0.75*0.8 + 0.25*0.4


@unittest.skipUnless(os.environ.get("INFOGAIN_TEST_LIVE"),
                     "live suite: set INFOGAIN_TEST_LIVE=1 or run tests/run.py live")
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
