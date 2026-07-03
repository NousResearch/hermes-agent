#!/usr/bin/env python3
"""Tests for the information-gain skill.

Layers:
  * voi.*            — pure math, no imports beyond voi (run anywhere).
  * pipeline.*       — model-calling stages with raw_chat mocked (needs the `ask`
                       skill's model_utils importable).
  * infogain.run     — the bucket-fill loop with the pipeline stages mocked.
  * Live (-gated)    — real Ollama calls, skipped unless the daemon is reachable.

Run:  python3 tests/test_infogain.py -v
      uv run --with pytest python3 -m pytest tests/ -v -k "not live"
"""

import json
import math
import os
import re
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))

import voi  # noqa: E402
import pairwise  # noqa: E402


def _answer(prob, dp, st):
    return {"answer": "a", "prob": prob, "delta_plan": dp, "stakes": st}


# ── pure math ────────────────────────────────────────────────────────────────


class TestVoiMath(unittest.TestCase):
    def test_clamp01(self):
        self.assertEqual(voi.clamp01(-1), 0.0)
        self.assertEqual(voi.clamp01(2), 1.0)
        self.assertEqual(voi.clamp01("x"), 0.0)
        self.assertEqual(voi.clamp01(0.5), 0.5)

    def test_normalize_probs(self):
        self.assertEqual(voi.normalize_probs([]), [])
        self.assertEqual(voi.normalize_probs([0, 0]), [0.5, 0.5])
        self.assertEqual(voi.normalize_probs([1, 3]), [0.25, 0.75])
        self.assertEqual(voi.normalize_probs([-5, 5]), [0.0, 1.0])

    def test_normalized_entropy(self):
        self.assertEqual(voi.normalized_entropy([1, 0]), 0.0)
        self.assertEqual(voi.normalized_entropy([0.5, 0.5]), 1.0)
        self.assertAlmostEqual(voi.normalized_entropy([1]), 0.0)
        self.assertTrue(0.9 < voi.normalized_entropy([0.4, 0.35, 0.25]) <= 1.0)

    def test_uncertainty_derivable_discount(self):
        ans = [_answer(0.5, 0, 0), _answer(0.5, 0, 0)]
        self.assertAlmostEqual(voi.uncertainty(ans, 0.0), 1.0)
        self.assertAlmostEqual(voi.uncertainty(ans, 1.0), 0.0)
        self.assertAlmostEqual(voi.uncertainty(ans, 0.5), 0.5)

    def test_evsi_probability_weighting(self):
        # a big plan change under a 10% answer is worth less than a moderate one at 90%
        low = voi.evsi([_answer(0.1, 1.0, 1.0), _answer(0.9, 0.0, 0.0)])
        high = voi.evsi([_answer(0.9, 0.5, 0.5), _answer(0.1, 0.0, 0.0)])
        self.assertAlmostEqual(low, 0.1)
        self.assertAlmostEqual(high, 0.225)
        self.assertGreater(high, low)

    def test_value_is_geometric_mean(self):
        self.assertAlmostEqual(voi.question_value(0.81, 0.49), math.sqrt(0.81 * 0.49))
        self.assertEqual(voi.question_value(0.0, 0.9), 0.0)
        self.assertEqual(voi.question_value(0.9, 0.0), 0.0)

    def test_gate(self):
        self.assertTrue(voi.is_gated_out(0.0, 0.5))
        self.assertTrue(voi.is_gated_out(0.5, 0.0))
        self.assertFalse(voi.is_gated_out(0.5, 0.5))

    def test_classify(self):
        self.assertEqual(voi.classify(0.7, 0.6, 0.4), "PRE_ANSWER")
        self.assertEqual(voi.classify(0.5, 0.6, 0.4), "ASSUME_DEFAULT")
        self.assertEqual(voi.classify(0.3, 0.6, 0.4), "SKIP")

    def test_modal_answer(self):
        self.assertIsNone(voi.modal_answer([]))
        m = voi.modal_answer([{"answer": "x", "prob": 0.2}, {"answer": "y", "prob": 0.8}])
        self.assertEqual(m["answer"], "y")

    def test_score_record_no_sensitivity_gated(self):
        rec = {"answers": [_answer(0.5, 0, 0.9), _answer(0.5, 0, 0.9)], "derivable_prob": 0.1}
        voi.score_record(rec)
        self.assertEqual(rec["evsi"], 0.0)
        self.assertTrue(rec["gated_out"])
        self.assertEqual(rec["value"], 0.0)

    def test_score_breakdown_matches_and_explains(self):
        rec = {"answers": [_answer(0.6, 0.8, 0.7), _answer(0.4, 0.2, 0.3)],
               "derivable_prob": 0.1}
        voi.score_record(rec)
        b = voi.score_breakdown(rec)
        # breakdown reproduces the canonical score, never drifts
        self.assertAlmostEqual(b["u"], rec["u"], places=4)
        self.assertAlmostEqual(b["value"], rec["value"], places=3)
        # per-answer EVSI terms sum to EVSI
        self.assertAlmostEqual(sum(t["term"] for t in b["evsi_terms"]), b["evsi"], places=3)
        self.assertEqual(len(b["evsi_terms"]), 2)


class TestPairwiseAggregator(unittest.TestCase):
    """pairwise.py — pure Bradley-Terry / win-count aggregation for comparative elicitation (#24)."""

    # items: 0=FLOOR, 1=weak answer, 2=strong answer, 3=CEILING ; transitive chain
    _COMPS = [(3, 2, 1.0), (3, 1, 1.0), (3, 0, 1.0),
              (2, 1, 1.0), (2, 0, 1.0), (1, 0, 1.0)]

    def test_bradley_terry_monotone_transitive(self):
        s = pairwise.bradley_terry(4, self._COMPS)
        self.assertTrue(s[3] > s[2] > s[1] > s[0])  # more wins -> higher strength

    def test_anchored_scores_pin_anchors_and_order_reals(self):
        sc = pairwise.anchored_scores(pairwise.bradley_terry(4, self._COMPS), 0, 3)
        self.assertAlmostEqual(sc[0], 0.0, places=6)   # FLOOR -> 0
        self.assertAlmostEqual(sc[3], 1.0, places=6)   # CEILING -> 1
        self.assertTrue(0.0 < sc[1] < sc[2] < 1.0)     # reals ordered strictly between

    def test_scale_preserved_across_questions(self):
        # the between-task safeguard: a question whose reals merely TIE the floor must land LOWER than
        # a question whose reals strongly beat it — even though both normalize against the same anchors.
        low = [(3, 2, 1.0), (3, 1, 1.0), (3, 0, 1.0), (2, 1, 0.5), (2, 0, 0.5), (1, 0, 0.5)]
        high = [(3, 2, 1.0), (3, 1, 1.0), (3, 0, 1.0), (2, 1, 0.5), (2, 0, 1.0), (1, 0, 1.0)]
        sl = pairwise.anchored_scores(pairwise.bradley_terry(4, low), 0, 3)
        sh = pairwise.anchored_scores(pairwise.bradley_terry(4, high), 0, 3)
        self.assertLess(max(sl[1:3]), max(sh[1:3]))

    def test_win_fractions_monotone_and_off_rails(self):
        wf = pairwise.win_fractions(4, self._COMPS)
        self.assertTrue(wf[3] > wf[2] > wf[1] > wf[0])
        self.assertTrue(0.0 < wf[0] and wf[3] < 1.0)  # Laplace prior keeps extremes off 0/1

    def test_degenerate_and_robustness(self):
        self.assertEqual(pairwise.bradley_terry(0, []), [])
        self.assertEqual(pairwise.bradley_terry(1, []), [1.0])
        self.assertEqual(pairwise.anchored_scores([1.0, 1.0], 0, 1), [0.0, 0.0])  # no spread
        self.assertEqual(pairwise.anchored_scores([], 0, 1), [])
        self.assertEqual(pairwise.all_pairs(3), [(0, 1), (0, 2), (1, 2)])
        # malformed comparisons are skipped, not fatal
        s = pairwise.bradley_terry(2, [("x", 1, 1.0), (0, 1, 1.0), (5, 9, 1.0)])
        self.assertTrue(s[0] > s[1])


class TestSimilarityAndSelection(unittest.TestCase):
    def test_similarity_same_target(self):
        a = {"question": "Which datastore?", "target": "datastore"}
        b = {"question": "Which DB engine?", "target": "datastore"}
        c = {"question": "Who are the users?", "target": "audience"}
        self.assertEqual(voi.question_similarity(a, b), 1.0)
        self.assertLess(voi.question_similarity(a, c), 0.5)

    def test_dedupe_keeps_first(self):
        recs = [{"question": "q1", "target": "t", "value": 0.6},
                {"question": "q2", "target": "t", "value": 0.5}]
        kept = voi.dedupe(recs)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["question"], "q1")

    def _scored(self, q, t, val):
        return {"question": q, "target": t, "value": val, "u": 0.8, "evsi": val,
                "gated_out": False, "answers": [], "modal_answer": None}

    def test_rank_collapses_redundant_and_classifies(self):
        recs = [self._scored("Which datastore?", "datastore", 0.7),
                self._scored("Which DB engine?", "datastore", 0.65),  # redundant
                self._scored("Who are the users?", "audience", 0.5),
                self._scored("trivial", "noise", 0.1)]  # below discard
        bucket, discarded = voi.rank_and_select(
            recs, discard_threshold=0.4, pre_answer_threshold=0.6, hard_cap=7)
        targets = [r["target"] for r in bucket]
        self.assertEqual(targets, ["datastore", "audience"])
        self.assertEqual(bucket[0]["recommendation"], "PRE_ANSWER")
        self.assertEqual(bucket[1]["recommendation"], "ASSUME_DEFAULT")
        recs_by = {r["recommendation"] for r in discarded}
        self.assertIn("REDUNDANT", recs_by)
        self.assertIn("SKIP", recs_by)

    def test_hard_cap_overflow(self):
        recs = [self._scored(f"q{i}", f"t{i}", 0.9 - i * 0.05) for i in range(5)]
        bucket, discarded = voi.rank_and_select(
            recs, discard_threshold=0.4, pre_answer_threshold=0.6, hard_cap=3)
        self.assertEqual(len(bucket), 3)
        self.assertTrue(any(r["recommendation"] == "OVERFLOW" for r in discarded))

    def test_best_value(self):
        self.assertEqual(voi.best_value([]), 0.0)
        self.assertEqual(voi.best_value([{"value": 0.2}, {"value": 0.7}]), 0.7)

    def test_hierarchical_similarity_tiers(self):
        # same target -> 1.0 (regardless of family)
        self.assertEqual(voi.hierarchical_similarity(
            {"target": "x", "family": "A"}, {"target": "x", "family": "B"}), 1.0)
        # same family, different target -> family_sim
        self.assertEqual(voi.hierarchical_similarity(
            {"target": "x", "family": "A", "question": "aa"},
            {"target": "y", "family": "A", "question": "bb"}, family_sim=0.5), 0.5)
        # different family + target + text -> token jaccard (0 here)
        self.assertEqual(voi.hierarchical_similarity(
            {"target": "x", "family": "A", "question": "aa"},
            {"target": "y", "family": "B", "question": "bb"}), 0.0)
        # no family -> reduces to question_similarity (same target still 1.0)
        self.assertEqual(voi.hierarchical_similarity({"target": "x"}, {"target": "x"}), 1.0)

    def test_rank_spreads_across_families_with_sim_fn(self):
        # family A has the two highest-value reps; B has lower. hard_cap=2.
        recs = [self._fam("a1", "t1", "A", 0.9), self._fam("a2", "t2", "A", 0.85),
                self._fam("b1", "t3", "B", 0.8), self._fam("b2", "t4", "B", 0.75)]
        simf = lambda a, b: voi.hierarchical_similarity(a, b, 0.5)
        buck, _ = voi.rank_and_select([dict(r) for r in recs], discard_threshold=0.4,
                                      pre_answer_threshold=0.6, hard_cap=2, sim_fn=simf)
        self.assertEqual(sorted(r["family"] for r in buck), ["A", "B"])  # spread, not both A
        # default sim_fn (no family tier) picks the two highest-value -> both A
        buck0, _ = voi.rank_and_select([dict(r) for r in recs], discard_threshold=0.4,
                                       pre_answer_threshold=0.6, hard_cap=2)
        self.assertEqual([r["family"] for r in buck0], ["A", "A"])

    def test_selection_floor_modes(self):
        recs = [self._scored("a", "t", 0.6), self._scored("b", "t2", 0.3)]
        self.assertEqual(voi.selection_floor(recs, 0.40), 0.40)                       # rel_frac=0 -> absolute
        self.assertAlmostEqual(voi.selection_floor(recs, 0.40, rel_frac=0.5), 0.30)   # max(0.10, 0.5*0.6)
        self.assertAlmostEqual(voi.selection_floor(recs, 0.40, rel_frac=0.1, abs_floor=0.15), 0.15)  # backstop

    def test_relative_knee_keeps_more_in_low_value_domain(self):
        # values all run low; absolute 0.40 wipes most, the relative knee scales to the top value
        recs = [self._scored("q1", "t1", 0.50), self._scored("q2", "t2", 0.35),
                self._scored("q3", "t3", 0.28), self._scored("q4", "t4", 0.12)]
        abs_bucket, _ = voi.rank_and_select([dict(r) for r in recs], discard_threshold=0.40,
                                            pre_answer_threshold=0.60, hard_cap=7)            # rel_frac=0
        rel_bucket, _ = voi.rank_and_select([dict(r) for r in recs], discard_threshold=0.40,
                                            pre_answer_threshold=0.60, hard_cap=7, rel_frac=0.5)
        self.assertEqual(len(abs_bucket), 1)   # only q1 (>=0.40)
        self.assertEqual(len(rel_bucket), 3)   # floor=max(0.10, 0.25)=0.25 -> q1,q2,q3

    def _fam(self, q, t, fam, val):
        return {"question": q, "target": t, "family": fam, "value": val, "u": 0.8,
                "evsi": val, "gated_out": False, "answers": [], "modal_answer": None}


# ── pipeline (mocked Ollama) — requires model_utils (ask skill) importable ────

try:
    import pipeline  # noqa: E402
    import infogain  # noqa: E402
    _PIPELINE_OK = True
except SystemExit:
    _PIPELINE_OK = False


@unittest.skipUnless(_PIPELINE_OK, "ask skill / model_utils not importable")
class TestPipelineMocked(unittest.TestCase):
    def test_extract_json_fenced_and_prose(self):
        self.assertEqual(pipeline.extract_json('```json\n{"a":1}\n```'), {"a": 1})
        self.assertEqual(pipeline.extract_json('blah {"a": 2} trailing'), {"a": 2})
        self.assertEqual(pipeline.extract_json('[1,2,3]'), [1, 2, 3])
        with self.assertRaises(ValueError):
            pipeline.extract_json("no json here")

    def _mock_raw(self, content):
        return mock.patch.object(pipeline, "raw_chat",
                                 return_value={"content": content, "error": None, "elapsed": 0.0})

    def test_frame_and_plan(self):
        payload = '{"goal":"g","decision":"d","success_criteria":["s"],"baseline_plan":"p"}'
        with self._mock_raw(payload):
            fr, err = pipeline.frame_and_plan("problem", "fast")
        self.assertIsNone(err)
        self.assertEqual(fr["baseline_plan"], "p")

    def test_generate_questions_filters_empty(self):
        payload = ('{"questions":[{"question":"Q1","type":"scope","why":"w","target":"t1"},'
                   '{"question":"","type":"x"}]}')
        with self._mock_raw(payload):
            qs, err = pipeline.generate_questions("p", {"goal": "g"}, "fast", 6)
        self.assertEqual(len(qs), 1)
        self.assertEqual(qs[0]["target"], "t1")

    # ── families layer ──
    def test_generate_families_parses_and_defaults_lens(self):
        obj = {"families": [{"name": "Scope", "scope": "s", "lens": "scoped"},
                            {"name": "Approach", "scope": "s"},   # missing lens -> default scoped
                            {"name": "", "scope": "x"}]}           # empty name dropped
        with mock.patch.object(pipeline, "_call_json", return_value=(obj, None)):
            fams, err = pipeline.generate_families("p", {"goal": "g", "decision": "d"}, "fast")
        self.assertEqual([f["name"] for f in fams], ["Scope", "Approach"])
        self.assertEqual(fams[1]["lens"], "scoped")
        self.assertIsNone(err)

    def test_generate_families_empty_returns_error(self):
        with mock.patch.object(pipeline, "_call_json", return_value=(None, "bad json")):
            fams, err = pipeline.generate_families("p", {"goal": "g"}, "fast")
        self.assertEqual(fams, [])
        self.assertIsNotNone(err)  # so run() can fall back

    def test_generate_family_questions_tags_and_early_return(self):
        obj = {"questions": [{"question": "Q1", "type": "scope", "why": "w", "target": "t1"}]}
        families = [{"name": "Approach", "scope": "s", "lens": "contrarian"}]
        with mock.patch.object(pipeline, "_call_json", return_value=(obj, None)):
            out = pipeline.generate_family_questions("p", {"goal": "g"}, families, "fast", n_per=2)
        self.assertEqual(out[0]["questions"][0]["family"], "Approach")
        self.assertEqual(out[0]["questions"][0]["lens"], "contrarian")
        self.assertEqual(pipeline.generate_family_questions("p", {}, [], "fast"), [])  # empty -> []

    def test_vantage_auto_gate(self):
        self.assertTrue(pipeline._vantage_relevant({"goal": "deploy a service to a server", "decision": ""}))
        self.assertFalse(pipeline._vantage_relevant({"goal": "should I buy or rent a home", "decision": "advice"}))

    def test_hint_matching_is_word_boundary(self):
        # prefix-at-token-start keeps inflections ("deployment") but mid-word substrings no longer
        # trip the gates ("report"⊅"repo", "rapid"⊅"api", "product"⊅"prod", "dropdown"⊅"drop")
        self.assertTrue(pipeline._vantage_relevant({"goal": "the deployment environment", "decision": ""}))
        self.assertTrue(pipeline._vantage_relevant({"goal": "clone the repo", "decision": ""}))
        self.assertTrue(pipeline._vantage_relevant({"goal": "connect to the db", "decision": ""}))  # old "db " missed this
        self.assertFalse(pipeline._vantage_relevant({"goal": "summarize the quarterly report", "decision": "rapid draft"}))
        self.assertTrue(pipeline._premortem_relevant({"goal": "push to prod", "decision": ""}))
        self.assertTrue(pipeline._premortem_relevant({"goal": "drop the orders table", "decision": ""}))
        self.assertFalse(pipeline._premortem_relevant({"goal": "improve the product page", "decision": ""}))
        self.assertFalse(pipeline._premortem_relevant({"goal": "add a dropdown to the form", "decision": ""}))

    def test_questions_prompt_family_lens_directive(self):
        contra = pipeline.questions_prompt("p", {"goal": "g", "decision": "d"}, 3,
                                           family={"name": "F", "scope": "s", "lens": "contrarian"})
        self.assertIn("CHALLENGE", contra)
        vant = pipeline.questions_prompt("p", {"goal": "g", "decision": "d"}, 3,
                                         family={"name": "F", "scope": "s", "lens": "vantage"})
        self.assertIn("vantage", vant.lower())
        self.assertNotIn("FAMILY", pipeline.questions_prompt("p", {"goal": "g", "decision": "d"}, 3))

    def test_premortem_lens_directive_and_gate(self):
        # directive registered
        self.assertIn("premortem", pipeline._LENS_DIRECTIVE)
        pm = pipeline.questions_prompt("p", {"goal": "g", "decision": "d"}, 3,
                                       family={"name": "F", "scope": "s", "lens": "premortem"})
        self.assertIn("FAILED", pm)  # the failure-mode directive is injected
        # conservative auto-gate: fires on a failure surface, quiet on read-only
        self.assertTrue(pipeline._premortem_relevant({"goal": "deploy to production", "decision": "ship"}))
        self.assertTrue(pipeline._premortem_relevant({"goal": "send the payment", "decision": "charge card"}))
        self.assertFalse(pipeline._premortem_relevant({"goal": "summarize this research paper",
                                                       "decision": "explain the findings"}))
        # retrieval tasks over high-stakes ARTIFACTS must not trip the gate (#25 tier-1 eval:
        # gmail-triage fired on the bare noun "email"; acting tasks all carry a verb hint)
        self.assertFalse(pipeline._premortem_relevant(
            {"goal": "summarize important unread email from this week", "decision": "which to flag"}))
        self.assertFalse(pipeline._premortem_relevant(
            {"goal": "get the top customers from the database", "decision": "which query to run"}))
        self.assertTrue(pipeline._premortem_relevant(
            {"goal": "send an email to the whole customer list", "decision": "what to say"}))
        self.assertTrue(pipeline._premortem_relevant(
            {"goal": "migrate the orders table", "decision": "how to run the migration"}))

    def test_gates_match_raw_problem_text(self):
        # framing is model-paraphrased, so a hint verb in the PROMPT can vanish before the gate
        # sees it (post-rename smoke: "migrate prod DB" framed by fast -> no premortem family).
        # The gates now match the raw problem text too.
        paraphrased = {"goal": "move customer data to the new schema", "decision": "a plan"}
        self.assertFalse(pipeline._premortem_relevant(paraphrased))  # framing alone: no hint
        self.assertTrue(pipeline._premortem_relevant(paraphrased, "migrate the prod DB and deploy"))
        self.assertTrue(pipeline._vantage_relevant(
            {"goal": "improve the workflow", "decision": ""}, "fix the login on the staging server"))
        # read-only prompts stay quiet even with the problem text in the blob
        self.assertFalse(pipeline._premortem_relevant(
            {"goal": "explain the findings", "decision": ""}, "summarize this research paper"))
        # word boundaries hold on the problem text too
        self.assertFalse(pipeline._premortem_relevant(
            {"goal": "", "decision": ""}, "improve the product page dropdown"))

    def test_generate_families_gates_on_problem(self):
        # call-site threading: a hint verb only in the raw problem must reach families_prompt
        captured = {}

        def fake_call_json(model, prompt, timeout, num_predict=600, sink=None):
            captured["prompt"] = prompt
            return {"families": [{"name": "F", "scope": "s", "lens": "scoped"}]}, None

        framing = {"goal": "move customer data to the new schema", "decision": "a plan"}
        with mock.patch.object(pipeline, "_call_json", side_effect=fake_call_json):
            pipeline.generate_families("migrate the prod DB", framing, "fast",
                                       vantage="auto", premortem="auto")
        self.assertIn("PRE-MORTEM", captured["prompt"])

    def test_families_prompt_premortem_block_is_gated(self):
        on = pipeline.families_prompt("p", {"goal": "g", "decision": "d"}, 3, True, True, premortem=True)
        off = pipeline.families_prompt("p", {"goal": "g", "decision": "d"}, 3, True, True, premortem=False)
        self.assertIn("PRE-MORTEM", on)
        self.assertIn('"premortem"', on)           # schema enum advertises it
        self.assertNotIn("PRE-MORTEM", off)         # absent when off
        self.assertNotIn('"premortem"', off)        # enum stays the original 3 lenses

    def test_reach_lens_directive_gate_and_prompt_block(self):
        # #29: directive registered + injected; families_prompt block + enum gated; the auto
        # gate is the vantage gate (shared systems/access surface, raw problem text included).
        self.assertIn("reach", pipeline._LENS_DIRECTIVE)
        qp = pipeline.questions_prompt("p", {"goal": "g", "decision": "d"}, 3,
                                       family={"name": "F", "scope": "s", "lens": "reach"})
        self.assertIn("CHAINED hops", qp)
        on = pipeline.families_prompt("p", {"goal": "g"}, 3, True, True, reach=True)
        off = pipeline.families_prompt("p", {"goal": "g"}, 3, True, True, reach=False)
        self.assertIn("REACH", on)
        self.assertIn('"reach"', on)
        self.assertNotIn('"reach"', off)
        self.assertIs(pipeline._reach_relevant, pipeline._vantage_relevant)
        self.assertTrue(pipeline._reach_relevant(
            {"goal": "improve the workflow", "decision": ""}, "debug the service in the k8s cluster"))
        self.assertFalse(pipeline._reach_relevant(
            {"goal": "explain the findings", "decision": ""}, "summarize a paper"))

    def test_generate_families_reach_knob(self):
        captured = {}

        def fake_call_json(model, prompt, timeout, num_predict=600, sink=None):
            captured["prompt"] = prompt
            return {"families": [{"name": "F", "scope": "s", "lens": "reach"}]}, None

        framing = {"goal": "fix the login flow", "decision": "a plan"}
        with mock.patch.object(pipeline, "_call_json", side_effect=fake_call_json):
            pipeline.generate_families("fix the login on the staging server", framing, "fast",
                                       reach="auto")
        self.assertIn("REACH", captured["prompt"])          # auto fires on systems surface
        with mock.patch.object(pipeline, "_call_json", side_effect=fake_call_json):
            pipeline.generate_families("fix the login on the staging server", framing, "fast",
                                       reach="off")
        self.assertNotIn("REACH", captured["prompt"])       # explicit off wins

    def test_reach_cli_env_resolution(self):
        cfg = infogain.resolve_config(infogain.build_parser().parse_args(["--reach", "on", "p"]))
        self.assertEqual(cfg["families"]["reach"], "on")
        self.assertEqual(infogain.resolve_config(
            infogain.build_parser().parse_args(["p"]))["families"]["reach"], "auto")
        old = os.environ.get("INFOGAIN_REACH")
        try:
            os.environ["INFOGAIN_REACH"] = "off"
            self.assertEqual(infogain.resolve_config(
                infogain.build_parser().parse_args(["p"]))["families"]["reach"], "off")
        finally:
            os.environ.pop("INFOGAIN_REACH", None) if old is None else os.environ.update(
                {"INFOGAIN_REACH": old})
        self.assertEqual(infogain.families_cfg(reach="on")["reach"], "on")
        self.assertEqual(infogain.families_cfg()["reach"], "auto")

    def test_generate_families_premortem_auto_gate(self):
        obj = {"families": [{"name": "Hazards", "scope": "s", "lens": "premortem"}]}
        with mock.patch.object(pipeline, "_call_json", return_value=(obj, None)) as m:
            pipeline.generate_families("p", {"goal": "deploy to production", "decision": "ship"},
                                       "fast", premortem="auto")
            self.assertIn("PRE-MORTEM", m.call_args[0][1])   # gate fired -> block present
            m.reset_mock()
            pipeline.generate_families("p", {"goal": "summarize a paper", "decision": "explain"},
                                       "fast", premortem="auto")
            self.assertNotIn("PRE-MORTEM", m.call_args[0][1])  # gate quiet -> block absent
            m.reset_mock()
            pipeline.generate_families("p", {"goal": "summarize a paper", "decision": "explain"},
                                       "fast", premortem="on")
            self.assertIn("PRE-MORTEM", m.call_args[0][1])   # forced on overrides the gate

    def test_generate_questions_multisample_union_dedup(self):
        # two independent samples; t2 appears in both -> deduped; union covers t1,t2,t3
        obj1 = {"questions": [{"question": "Q-A", "type": "scope", "why": "w", "target": "t1"},
                              {"question": "Q-B", "type": "data", "why": "w", "target": "t2"}]}
        obj2 = {"questions": [{"question": "Q-B variant", "type": "data", "why": "w", "target": "t2"},
                              {"question": "Q-C", "type": "risk", "why": "w", "target": "t3"}]}
        with mock.patch.object(pipeline, "_call_json", side_effect=[(obj1, None), (obj2, None)]):
            qs, err = pipeline.generate_questions("p", {"goal": "g"}, "fast", 4,
                                                  samples=2, temperature=0.9)
        self.assertIsNone(err)
        self.assertEqual(sorted(q["target"] for q in qs), ["t1", "t2", "t3"])

    def test_consolidate_questions_merges_synonyms(self):
        cands = [
            {"question": "How fresh must data be?", "type": "constraint", "why": "w", "target": "freshness"},
            {"question": "What update latency is acceptable?", "type": "constraint", "why": "w", "target": "latency"},
            {"question": "Who are the users?", "type": "audience", "why": "w", "target": "users"},
        ]
        merged = {"questions": [
            {"question": "What data freshness / update latency is required?",
             "type": "constraint", "why": "w", "target": "freshness", "merged_count": 2},
            {"question": "Who are the users?", "type": "audience", "why": "w",
             "target": "users", "merged_count": 1}]}
        with mock.patch.object(pipeline, "_call_json", return_value=(merged, None)):
            out = pipeline.consolidate_questions("p", cands, "fast")
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0].get("merged_count"), 2)

    def test_consolidate_questions_never_drops_on_failure(self):
        cands = [{"question": "q1", "target": "t1"}, {"question": "q2", "target": "t2"}]
        with mock.patch.object(pipeline, "_call_json", return_value=({"questions": []}, "boom")):
            out = pipeline.consolidate_questions("p", cands, "fast")
        self.assertEqual(out, cands)  # fall back to input — never lose questions

    def test_project_answers_parses_derivable(self):
        rec = {"question": "Q"}
        with self._mock_raw('{"derivable_prob":0.2,"answers":[{"answer":"a","prob":1.0}]}'):
            pipeline.project_answers("p", {"goal": "g"}, rec, "fast", 5)
        self.assertEqual(rec["derivable_prob"], 0.2)
        self.assertEqual(len(rec["answers"]), 1)

    def test_usage_counters_reset(self):
        pipeline.reset_usage()
        u = pipeline.get_usage()
        self.assertEqual(u["calls"], 0)
        self.assertEqual(u["input_tokens"], 0)
        self.assertIn("output_tokens", u)
        self.assertIn("model_seconds", u)

    def test_evidence_woven_into_prompts(self):
        ev = ["budget is $0", "users are coaches"]
        fp = pipeline.frame_prompt("problem", ev)
        qp = pipeline.questions_prompt("problem", {"goal": "g"}, 6, evidence=ev)
        ap = pipeline.answers_prompt("problem", {"goal": "g"}, "Q", 5, ev)
        for prompt in (fp, qp, ap):
            self.assertIn("budget is $0", prompt)
            self.assertIn("ALREADY ESTABLISHED", prompt)
        # no evidence → no block
        self.assertNotIn("ALREADY ESTABLISHED", pipeline.frame_prompt("problem"))

    def test_judge_misaligned_short_list_defaults_zero(self):
        # judge per-answer alignment is POSITIONAL (judged[i] -> answers[i]); a short/misaligned judged
        # list must default trailing answers to 0.0 rather than crash or mis-score silently.
        rec = {"question": "Q", "answers": [{"answer": "a", "prob": 0.4},
                                            {"answer": "b", "prob": 0.4},
                                            {"answer": "c", "prob": 0.2}]}
        with self._mock_raw('{"answers":[{"delta_plan":0.9,"stakes":0.8},{"delta_plan":0.3,"stakes":0.2}]}'):
            pipeline.judge_plan_change("p", {"goal": "g"}, "baseline", rec, "fast")
        a = rec["answers"]
        self.assertEqual((a[0]["delta_plan"], a[0]["stakes"]), (0.9, 0.8))   # aligned
        self.assertEqual((a[1]["delta_plan"], a[1]["stakes"]), (0.3, 0.2))   # aligned
        self.assertEqual((a[2]["delta_plan"], a[2]["stakes"]), (0.0, 0.0))   # short list -> safe default
        empty = {"question": "Q", "answers": []}                            # no answers -> rec unchanged
        self.assertIs(pipeline.judge_plan_change("p", {"goal": "g"}, "b", empty, "fast"), empty)

    def test_project_then_judge_roundtrip(self):
        rec = {"question": "Which DB?"}
        with self._mock_raw('{"derivable_prob":0.2,"answers":[{"answer":"pg","prob":0.6},'
                            '{"answer":"mongo","prob":0.4}]}'):
            pipeline.project_answers("p", {"goal": "g"}, rec, "fast", 5)
        self.assertEqual(len(rec["answers"]), 2)
        self.assertEqual(rec["derivable_prob"], 0.2)
        with self._mock_raw('{"answers":[{"delta_plan":0.8,"stakes":0.7},'
                            '{"delta_plan":0.2,"stakes":0.3}]}'):
            pipeline.judge_plan_change("p", {"goal": "g"}, "baseline", rec, "fast")
        self.assertEqual(rec["answers"][0]["delta_plan"], 0.8)
        voi.score_record(rec)
        self.assertGreater(rec["value"], 0.0)
        self.assertFalse(rec["gated_out"])

    # ── pairwise (comparative) judge, #24 ──
    def test_pairwise_judge_writes_ordered_delta_plan(self):
        # items 1=FLOOR, 2=pg, 3=mongo, 4=CEILING. pg beats floor+mongo, mongo only ties floor →
        # pg.delta_plan > mongo.delta_plan, on the SAME fields the absolute judge writes.
        rec = {"question": "Which DB?", "answers": [{"answer": "pg", "prob": 0.6},
                                                    {"answer": "mongo", "prob": 0.4}]}
        comp = ('{"comparisons":[{"a":1,"b":2,"winner":2},{"a":1,"b":3,"winner":"tie"},'
                '{"a":1,"b":4,"winner":4},{"a":2,"b":3,"winner":2},{"a":2,"b":4,"winner":4},'
                '{"a":3,"b":4,"winner":4}]}')
        with mock.patch.object(pipeline, "_call_json", return_value=(json.loads(comp), None)):
            pipeline.judge_plan_change_pairwise("p", {"goal": "g"}, "baseline", rec, "fast")
        a = rec["answers"]
        self.assertGreater(a[0]["delta_plan"], a[1]["delta_plan"])
        for x in a:  # in-range and present on both dimensions
            self.assertTrue(0.0 <= x["delta_plan"] <= 1.0 and 0.0 <= x["stakes"] <= 1.0)
        voi.score_record(rec)
        self.assertGreater(rec["value"], 0.0)

    def test_pairwise_judge_safe_zero_on_failure(self):
        rec = {"question": "Q", "answers": [{"answer": "x", "prob": 1.0}]}
        with mock.patch.object(pipeline, "_call_json", return_value=(None, "bad json")):
            pipeline.judge_plan_change_pairwise("p", {"goal": "g"}, "b", rec, "fast")
        self.assertEqual(rec["answers"][0]["delta_plan"], 0.0)
        self.assertEqual(rec["answers"][0]["stakes"], 0.0)
        empty = {"question": "Q", "answers": []}  # no answers -> rec returned unchanged
        self.assertIs(pipeline.judge_plan_change_pairwise("p", {"goal": "g"}, "b", empty, "fast"), empty)

    # ── sampled P(a), #26 ──
    def _sample_rec(self):
        return {"question": "Which DB?", "answers": [{"answer": "pg", "prob": 0.7},
                                                     {"answer": "mongo", "prob": 0.3}],
                "derivable_prob": 0.2}

    def test_sampled_probs_frequencies_and_smoothing(self):
        # 6 parseable draws → Laplace-smoothed frequencies (α=0.5, m=2): prob = (c+0.5)/7,
        # counts sum to n, probs sum to 1, and the stated probs survive as stated_prob.
        rec = self._sample_rec()
        with self._mock_raw("1"):
            pipeline.sample_answer_distribution("p", {"goal": "g"}, rec, "fast", n_samples=6)
        self.assertEqual(rec["prob_mode_used"], "sampled")
        self.assertEqual(sum(rec["sample_counts"]), 6)
        self.assertEqual([a["stated_prob"] for a in rec["answers"]], [0.7, 0.3])
        self.assertAlmostEqual(sum(a["prob"] for a in rec["answers"]), 1.0)
        for c, a in zip(rec["sample_counts"], rec["answers"]):
            self.assertAlmostEqual(a["prob"], (c + 0.5) / 7.0)

    def test_sampled_probs_shuffle_maps_back(self):
        # The model always votes for the option labelled "pg", wherever the per-sample shuffle
        # put it — every count must land on answer 0 (mapping shuffled position → canonical).
        rec = self._sample_rec()

        def vote_pg(model, prompt, **k):
            n = re.search(r"(\d+)\. pg\b", prompt).group(1)
            return {"content": n, "error": None, "elapsed": 0.0}

        with mock.patch.object(pipeline, "raw_chat", side_effect=vote_pg):
            pipeline.sample_answer_distribution("p", {"goal": "g"}, rec, "fast", n_samples=6)
        self.assertEqual(rec["sample_counts"], [6, 0])
        self.assertGreater(rec["answers"][0]["prob"], rec["answers"][1]["prob"])

    def test_sampled_probs_fallback_keeps_stated(self):
        # unparseable ("no idea") and out-of-range ("9") replies don't count; below ⌈N/2⌉ valid
        # the stated distribution must survive untouched, tagged stated-fallback.
        for bad in ("no idea", "9"):
            rec = self._sample_rec()
            with self._mock_raw(bad):
                pipeline.sample_answer_distribution("p", {"goal": "g"}, rec, "fast", n_samples=6)
            self.assertEqual(rec["prob_mode_used"], "stated-fallback")
            self.assertEqual([a["prob"] for a in rec["answers"]], [0.7, 0.3])
            self.assertEqual([a["stated_prob"] for a in rec["answers"]], [0.7, 0.3])

    def test_sampled_probs_single_answer_short_circuits(self):
        # a 0/1-option support IS its distribution — no sampling calls, tagged stated.
        one = {"question": "Q", "answers": [{"answer": "x", "prob": 1.0}]}
        with mock.patch.object(pipeline, "raw_chat",
                               side_effect=AssertionError("must not call the model")):
            pipeline.sample_answer_distribution("p", {"goal": "g"}, one, "fast", n_samples=6)
        self.assertEqual(one["prob_mode_used"], "stated")
        self.assertEqual(one["answers"][0]["stated_prob"], 1.0)

    # ── solution-space Δplan, #27 ──
    def test_sample_solutions_reuses_baseline_first(self):
        framing = {"goal": "g", "decision": "d", "baseline_plan": "the baseline"}
        with mock.patch.object(pipeline, "_call_json",
                               return_value=({"solution": "an alternative"}, None)) as m:
            sols = pipeline.sample_solutions("p", framing, "fast", k=3)
        self.assertEqual(sols[0], "the baseline")   # solution 1 is free
        self.assertEqual(len(sols), 3)
        self.assertEqual(m.call_count, 2)           # only k-1 sampled
        # failures shrink the set rather than padding it with empties
        with mock.patch.object(pipeline, "_call_json", return_value=(None, "boom")):
            sols = pipeline.sample_solutions("p", framing, "fast", k=3)
        self.assertEqual(sols, ["the baseline"])

    def test_solution_judge_delta_is_invalidated_fraction(self):
        sols = ["s1", "s2", "s3", "s4"]
        rec = {"question": "Q", "answers": [{"answer": "a", "prob": 0.6},
                                            {"answer": "b", "prob": 0.4}]}
        judged = {"answers": [{"viable": [1, 3], "stakes": 0.7},          # 2 of 4 invalidated
                              {"viable": [1, 2, 3, 4], "stakes": 0.1}]}   # nothing invalidated
        with mock.patch.object(pipeline, "_call_json", return_value=(judged, None)):
            pipeline.judge_plan_change_solution("p", {"goal": "g"}, sols, rec, "fast")
        a = rec["answers"]
        self.assertEqual(a[0]["delta_plan"], 0.5)
        self.assertEqual(a[0]["viable_solutions"], [1, 3])
        self.assertEqual(a[0]["stakes"], 0.7)                             # stakes passthrough
        self.assertEqual(a[1]["delta_plan"], 0.0)
        voi.score_record(rec)                                             # drop-in for the formula
        self.assertGreaterEqual(rec["value"], 0.0)

    def test_solution_judge_safe_zero_and_bounds(self):
        sols = ["s1", "s2"]
        rec = {"question": "Q", "answers": [{"answer": "a", "prob": 1.0}]}
        with mock.patch.object(pipeline, "_call_json", return_value=(None, "bad json")):
            pipeline.judge_plan_change_solution("p", {"goal": "g"}, sols, rec, "fast")
        self.assertEqual(rec["answers"][0]["delta_plan"], 0.0)            # safe-zero
        self.assertEqual(rec["answers"][0]["stakes"], 0.0)
        # out-of-range / junk viable entries are ignored, not counted as invalidations of nothing
        rec2 = {"question": "Q", "answers": [{"answer": "a", "prob": 1.0}]}
        judged = {"answers": [{"viable": [0, 1, 9, "x"], "stakes": 0.5}]}  # only 1 is in-range
        with mock.patch.object(pipeline, "_call_json", return_value=(judged, None)):
            pipeline.judge_plan_change_solution("p", {"goal": "g"}, sols, rec2, "fast")
        self.assertEqual(rec2["answers"][0]["delta_plan"], 0.5)           # (2-1)/2
        self.assertEqual(rec2["answers"][0]["viable_solutions"], [1])
        empty = {"question": "Q", "answers": []}                          # no answers -> unchanged
        self.assertIs(pipeline.judge_plan_change_solution("p", {"goal": "g"}, sols, empty, "fast"),
                      empty)

    def test_project_answers_sampled_roundtrip(self):
        # composition: projection call first (answers + derivable), then N forced-choice draws.
        rec = {"question": "Which DB?"}
        replies = iter(['{"derivable_prob":0.2,"answers":[{"answer":"pg","prob":0.6},'
                        '{"answer":"mongo","prob":0.4}]}'] + ["1"] * 4)
        with mock.patch.object(pipeline, "raw_chat",
                               side_effect=lambda *a, **k: {"content": next(replies),
                                                            "error": None, "elapsed": 0.0}):
            pipeline.project_answers_sampled("p", {"goal": "g"}, rec, "fast", 5, n_samples=4)
        self.assertEqual(rec["prob_mode_used"], "sampled")
        self.assertEqual(len(rec["answers"]), 2)
        self.assertEqual([a["stated_prob"] for a in rec["answers"]], [0.6, 0.4])
        voi.score_record(rec)   # the swapped probs feed the frozen formula unchanged
        self.assertGreaterEqual(rec["value"], 0.0)


@unittest.skipUnless(_PIPELINE_OK, "ask skill / model_utils not importable")
class TestOrchestrationMocked(unittest.TestCase):
    def _cfg(self, **over):
        cfg = {k: v for k, v in infogain.DEFAULTS.items()}
        cfg.update(over)
        return cfg

    def _fake_round(self, n, base_target="t"):
        # n distinct high-value questions
        out = []
        for i in range(n):
            out.append({
                "question": f"Q{i}", "type": "scope", "why": "w", "target": f"{base_target}{i}",
                "answers": [{"answer": "A", "prob": 0.5, "delta_plan": 0.8, "stakes": 0.8},
                            {"answer": "B", "prob": 0.5, "delta_plan": 0.2, "stakes": 0.3}],
                "derivable_prob": 0.1,
            })
        return out

    _FAM_CFG = {"enabled": True, "n_scoped": 2, "contrarian": True, "vantage": "off",
                "questions_per_family": 2, "family_sim": 0.5, "families_model": "fast"}

    def _fake_families(self):
        return [{"name": "Scope", "scope": "s", "lens": "scoped"},
                {"name": "Approach", "scope": "s", "lens": "contrarian"}]

    def _fake_family_questions(self, *a, **k):
        fams = a[2] if len(a) > 2 else k["families"]
        out = []
        for fi, fam in enumerate(fams):
            qs = self._fake_round(2, base_target=f"f{fi}")
            for q in qs:
                q["family"], q["lens"] = fam["name"], fam.get("lens", "scoped")
            out.append({**fam, "questions": qs})
        return out

    def test_families_branch_runs_and_tags(self):
        cfg = self._cfg(max_rounds=2, target_bucket_size=3, min_bucket_size=1, families=self._FAM_CFG)
        with mock.patch.object(pipeline, "frame_and_plan",
                               return_value=({"goal": "g", "decision": "d", "success_criteria": [],
                                              "baseline_plan": "p"}, None)), \
             mock.patch.object(pipeline, "generate_families",
                               side_effect=lambda *a, **k: (self._fake_families(), None)), \
             mock.patch.object(pipeline, "generate_family_questions", side_effect=self._fake_family_questions), \
             mock.patch.object(pipeline, "project_answers_batch", side_effect=lambda p, f, recs, *a, **k: recs), \
             mock.patch.object(pipeline, "judge_plan_change_batch", side_effect=lambda p, f, b, recs, *a, **k: recs):
            result = infogain.run("vague systems task", cfg)
        self.assertEqual({f["lens"] for f in result["families"]}, {"scoped", "contrarian"})
        self.assertTrue(result["bucket"])
        self.assertTrue(all(r.get("family") for r in result["bucket"]))  # every kept record is tagged

    def test_families_empty_falls_back_to_flat(self):
        # the regression guard: a stage-1a failure must degrade to flat, not produce an empty run
        cfg = self._cfg(max_rounds=2, questions_per_round=4, target_bucket_size=2, min_bucket_size=1,
                        families=self._FAM_CFG)
        flat = mock.MagicMock(side_effect=lambda *a, **k: (self._fake_round(4), None))
        with mock.patch.object(pipeline, "frame_and_plan",
                               return_value=({"goal": "g", "decision": "d", "success_criteria": [],
                                              "baseline_plan": "p"}, None)), \
             mock.patch.object(pipeline, "generate_families", side_effect=lambda *a, **k: ([], "boom")), \
             mock.patch.object(pipeline, "generate_family_questions", side_effect=lambda *a, **k: []), \
             mock.patch.object(pipeline, "generate_questions", flat), \
             mock.patch.object(pipeline, "project_answers_batch", side_effect=lambda p, f, recs, *a, **k: recs), \
             mock.patch.object(pipeline, "judge_plan_change_batch", side_effect=lambda p, f, b, recs, *a, **k: recs):
            result = infogain.run("task", cfg)
        self.assertTrue(flat.called)        # flat fallback fired
        self.assertTrue(result["bucket"])   # NOT empty (the bug would give an empty bucket)

    def test_loop_stops_at_target_in_one_round(self):
        cfg = self._cfg(question_gen_model="fast", answer_model="fast",
                        value_judge_model="fast", max_rounds=3, questions_per_round=6,
                        target_bucket_size=5, min_bucket_size=3)
        with mock.patch.object(pipeline, "frame_and_plan",
                               return_value=({"goal": "g", "decision": "d",
                                              "success_criteria": [], "baseline_plan": "p"}, None)), \
             mock.patch.object(pipeline, "generate_questions",
                               side_effect=lambda *a, **k: (self._fake_round(6), None)), \
             mock.patch.object(pipeline, "project_answers_batch", side_effect=lambda p, f, recs, *a, **k: recs), \
             mock.patch.object(pipeline, "judge_plan_change_batch", side_effect=lambda p, f, b, recs, *a, **k: recs):
            result = infogain.run("vague problem", cfg)
        self.assertEqual(result["rounds_used"], 1)
        self.assertGreaterEqual(len(result["bucket"]), cfg["target_bucket_size"])
        self.assertTrue(result["min_met"])

    def test_loop_reports_underfilled_bucket(self):
        # only 1 distinct valuable question per round, same target every round -> never fills
        cfg = self._cfg(max_rounds=2, questions_per_round=3, min_bucket_size=3,
                        target_bucket_size=5)
        with mock.patch.object(pipeline, "frame_and_plan",
                               return_value=({"goal": "g", "decision": "", "success_criteria": [],
                                              "baseline_plan": "p"}, None)), \
             mock.patch.object(pipeline, "generate_questions",
                               side_effect=lambda *a, **k: (self._fake_round(1, base_target="solo"), None)), \
             mock.patch.object(pipeline, "project_answers_batch", side_effect=lambda p, f, recs, *a, **k: recs), \
             mock.patch.object(pipeline, "judge_plan_change_batch", side_effect=lambda p, f, b, recs, *a, **k: recs):
            result = infogain.run("nearly specified", cfg)
        self.assertFalse(result["min_met"])
        md = infogain.render_markdown(result)
        self.assertIn("below the minimum", md)

    def test_render_has_sections(self):
        cfg = self._cfg(max_rounds=1, questions_per_round=4, target_bucket_size=2,
                        min_bucket_size=1)
        with mock.patch.object(pipeline, "frame_and_plan",
                               return_value=({"goal": "g", "decision": "d", "success_criteria": ["s"],
                                              "baseline_plan": "p"}, None)), \
             mock.patch.object(pipeline, "generate_questions",
                               side_effect=lambda *a, **k: (self._fake_round(4), None)), \
             mock.patch.object(pipeline, "project_answers_batch", side_effect=lambda p, f, recs, *a, **k: recs), \
             mock.patch.object(pipeline, "judge_plan_change_batch", side_effect=lambda p, f, b, recs, *a, **k: recs):
            result = infogain.run("p", cfg)
        md = infogain.render_markdown(result)
        self.assertIn("Key Questions to Improve the Response", md)
        self.assertIn("ranked by weight", md)
        self.assertIn("what the weight means", md)

    def test_ranked_list_shows_weight_and_clarification(self):
        rec = voi.score_record({
            "question": "Which datastore?", "target": "datastore", "recommendation": "PRE_ANSWER",
            "answers": [{"answer": "Postgres", "prob": 0.6, "delta_plan": 0.8, "stakes": 0.7},
                        {"answer": "Mongo", "prob": 0.4, "delta_plan": 0.2, "stakes": 0.3}],
            "derivable_prob": 0.1})
        rec["recommendation"] = "PRE_ANSWER"
        md = infogain._ranked_list([rec])
        self.assertIn("weight", md)
        self.assertIn("what the weight means", md)
        self.assertIn("Postgres", md)  # modal answer named in the clarification
        self.assertIn("already specified", infogain._ranked_list([]))  # empty bucket

    def test_evidence_in_result_and_render(self):
        cfg = self._cfg(max_rounds=1, questions_per_round=4, target_bucket_size=2,
                        min_bucket_size=1)
        with mock.patch.object(pipeline, "frame_and_plan",
                               return_value=({"goal": "g", "decision": "d", "success_criteria": [],
                                              "baseline_plan": "p"}, None)), \
             mock.patch.object(pipeline, "generate_questions",
                               side_effect=lambda *a, **k: (self._fake_round(4), None)), \
             mock.patch.object(pipeline, "project_answers_batch", side_effect=lambda p, f, recs, *a, **k: recs), \
             mock.patch.object(pipeline, "judge_plan_change_batch", side_effect=lambda p, f, b, recs, *a, **k: recs):
            result = infogain.run("p", cfg, evidence=["budget is $0"])
        self.assertEqual(result["evidence"], ["budget is $0"])
        self.assertIn("usage", result)
        self.assertIn("wall_seconds", result["usage"])
        md = infogain.render_markdown(result)
        self.assertIn("budget is $0", md)
        self.assertIn("answer-value", md)  # EVSI column header

    def test_trace_captures_show_your_work(self):
        cfg = self._cfg(max_rounds=1, questions_per_round=4, target_bucket_size=2,
                        min_bucket_size=1)
        with mock.patch.object(pipeline, "frame_and_plan",
                               return_value=({"goal": "g", "decision": "d",
                                              "success_criteria": [], "baseline_plan": "p"}, None)), \
             mock.patch.object(pipeline, "generate_questions",
                               side_effect=lambda *a, **k: (self._fake_round(4), None)), \
             mock.patch.object(pipeline, "project_answers_batch", side_effect=lambda p, f, recs, *a, **k: recs), \
             mock.patch.object(pipeline, "judge_plan_change_batch", side_effect=lambda p, f, b, recs, *a, **k: recs):
            result = infogain.run("p", cfg, trace=True)
        self.assertIn("trace", result)
        tr = result["trace"]
        self.assertIn("models", tr)
        self.assertTrue(tr["rounds"])
        q0 = tr["rounds"][0]["questions"][0]
        self.assertIn("breakdown", q0)
        self.assertIn("evsi_terms", q0["breakdown"])
        md = infogain.render_trace(result)
        self.assertIn("show your work", md)
        self.assertIn("value = √(U", md)
        self.assertIn("value-of-answering", md)


# ── live (real Ollama) ────────────────────────────────────────────────────────


@unittest.skipUnless(_PIPELINE_OK, "ask skill / model_utils not importable")
class TestModeConfig(unittest.TestCase):
    def _cfg(self, argv):
        return infogain.resolve_config(infogain.build_parser().parse_args(argv))

    def test_focus_default_matches_defaults(self):
        cfg = self._cfg(["a problem"])
        self.assertEqual(cfg["mode"], "focus")
        self.assertEqual(cfg["questions_per_round"], infogain.DEFAULTS["questions_per_round"])
        self.assertEqual(cfg["target_bucket_size"], infogain.DEFAULTS["target_bucket_size"])

    def test_breadth_preset_applied(self):
        cfg = self._cfg(["--mode", "breadth", "a problem"])
        self.assertEqual(cfg["mode"], "breadth")
        self.assertEqual(cfg["questions_per_round"],
                         infogain.MODES["breadth"]["questions_per_round"])
        self.assertEqual(cfg["discard_threshold"], 0.30)
        self.assertEqual(cfg["hard_cap"], 18)
        # breadth gets its coverage from sampling the model's distribution
        self.assertEqual(cfg["gen_samples"], 3)
        self.assertGreater(cfg["gen_temperature"], 0.0)

    def test_cli_flag_overrides_mode(self):
        cfg = self._cfg(["--mode", "breadth", "--questions-per-round", "3", "a problem"])
        self.assertEqual(cfg["questions_per_round"], 3)

    def test_families_default_on(self):
        self.assertTrue(self._cfg(["a problem"])["families"]["enabled"])

    def test_no_families_flag(self):
        self.assertFalse(self._cfg(["--no-families", "a problem"])["families"]["enabled"])

    def test_families_env_toggle(self):
        old = os.environ.get("INFOGAIN_FAMILIES")
        try:
            os.environ["INFOGAIN_FAMILIES"] = "off"
            self.assertFalse(self._cfg(["a problem"])["families"]["enabled"])
            os.environ["INFOGAIN_FAMILIES"] = "on"
            self.assertTrue(self._cfg(["a problem"])["families"]["enabled"])
            # explicit CLI beats env
            self.assertFalse(self._cfg(["--no-families", "a problem"])["families"]["enabled"])
        finally:
            os.environ.pop("INFOGAIN_FAMILIES", None) if old is None else os.environ.update(
                {"INFOGAIN_FAMILIES": old})

    # ── premortem lens (auto-on by design) ──
    def test_premortem_default_auto(self):
        self.assertEqual(self._cfg(["a problem"])["families"]["premortem"], "auto")

    def test_premortem_cli_override(self):
        self.assertEqual(self._cfg(["--premortem", "off", "p"])["families"]["premortem"], "off")
        self.assertEqual(self._cfg(["--premortem", "on", "p"])["families"]["premortem"], "on")

    def test_premortem_env_and_cli_precedence(self):
        old = os.environ.get("INFOGAIN_PREMORTEM")
        try:
            os.environ["INFOGAIN_PREMORTEM"] = "on"
            self.assertEqual(self._cfg(["p"])["families"]["premortem"], "on")
            self.assertEqual(self._cfg(["--premortem", "off", "p"])["families"]["premortem"], "off")
        finally:
            os.environ.pop("INFOGAIN_PREMORTEM", None) if old is None else os.environ.update(
                {"INFOGAIN_PREMORTEM": old})

    def test_premortem_key_stays_inside_families_dict(self):
        # the new lens knob must live in the families dict, NOT leak into the scalar cfg (the DEFAULTS
        # auto-loop / byte-identical-cfg safety invariant that keeps the live default composition safe)
        cfg = self._cfg(["a problem"])
        self.assertNotIn("premortem", {k: v for k, v in cfg.items() if k != "families"})
        self.assertIn("premortem", cfg["families"])
        self.assertNotIn("premortem", infogain.DEFAULTS)

    def test_families_model_default_and_cli_override(self):
        # default: the FAMILIES constant; --question-gen-model must NOT leak into it
        cfg = self._cfg(["--question-gen-model", "fast", "a problem"])
        self.assertEqual(cfg["families"]["families_model"], infogain.FAMILIES["families_model"])
        cfg = self._cfg(["--families-model", "fast", "a problem"])
        self.assertEqual(cfg["families"]["families_model"], "fast")

    def test_families_model_env_and_cli_precedence(self):
        old = os.environ.get("INFOGAIN_FAMILIES_MODEL")
        try:
            os.environ["INFOGAIN_FAMILIES_MODEL"] = "qwen"
            self.assertEqual(self._cfg(["p"])["families"]["families_model"], "qwen")
            self.assertEqual(self._cfg(["--families-model", "fast", "p"])["families"]["families_model"],
                             "fast")
        finally:
            os.environ.pop("INFOGAIN_FAMILIES_MODEL", None) if old is None else os.environ.update(
                {"INFOGAIN_FAMILIES_MODEL": old})

    # ── value_judge_mode selector (#24, off by default) ──
    def test_value_judge_mode_defaults_absolute(self):
        self.assertEqual(self._cfg(["a problem"])["value_judge_mode"], "absolute")

    def test_value_judge_mode_cli(self):
        self.assertEqual(self._cfg(["--value-judge-mode", "pairwise", "p"])["value_judge_mode"],
                         "pairwise")

    def test_value_judge_mode_env(self):
        old = os.environ.get("INFOGAIN_VALUE_JUDGE_MODE")
        try:
            os.environ["INFOGAIN_VALUE_JUDGE_MODE"] = "pairwise"
            self.assertEqual(self._cfg(["p"])["value_judge_mode"], "pairwise")
            # explicit CLI beats env
            self.assertEqual(self._cfg(["--value-judge-mode", "absolute", "p"])["value_judge_mode"],
                             "absolute")
        finally:
            os.environ.pop("INFOGAIN_VALUE_JUDGE_MODE", None) if old is None else os.environ.update(
                {"INFOGAIN_VALUE_JUDGE_MODE": old})

    # ── solution judge mode (#27, off by default) ──
    def test_value_judge_mode_solution_cli_and_knobs(self):
        cfg = self._cfg(["--value-judge-mode", "solution", "p"])
        self.assertEqual(cfg["value_judge_mode"], "solution")
        self.assertEqual(cfg["solution_samples"], 4)
        self.assertEqual(cfg["solution_temperature"], 0.8)
        self.assertEqual(self._cfg(["--solution-samples", "6", "p"])["solution_samples"], 6)

    def test_run_solution_mode_samples_once_and_binds_judge(self):
        cfg = {k: v for k, v in infogain.DEFAULTS.items()}
        cfg["families"] = {"enabled": False}
        cfg["max_rounds"] = 2          # 2 rounds must NOT re-sample the solution set
        cfg["target_bucket_size"] = 99  # force both rounds to run
        cfg["min_bucket_size"] = 99
        cfg["value_judge_mode"] = "solution"
        calls = {"sample": 0, "judge": 0, "sols": None}

        def fake_sample(*a, **k):
            calls["sample"] += 1
            return ["baseline", "alt1", "alt2", "alt3"]

        def fake_solution_batch(p, f, b, recs, *a, **kw):
            calls["judge"] += 1
            calls["sols"] = kw.get("solutions")
            return recs

        gen_n = {"i": 0}

        def fake_gen(*a, **k):
            gen_n["i"] += 1
            return ([{"question": f"Q{gen_n['i']}", "target": f"t{gen_n['i']}",
                      "answers": [], "derivable_prob": 0.1}], None)

        with mock.patch.object(pipeline, "frame_and_plan",
                               return_value=({"goal": "g", "decision": "d", "success_criteria": [],
                                              "baseline_plan": "baseline"}, None)), \
             mock.patch.object(pipeline, "sample_solutions", side_effect=fake_sample), \
             mock.patch.object(pipeline, "generate_questions", side_effect=fake_gen), \
             mock.patch.object(pipeline, "project_answers_batch",
                               side_effect=lambda p, f, recs, *a, **k: recs), \
             mock.patch.object(pipeline, "judge_plan_change_solution_batch",
                               side_effect=fake_solution_batch):
            infogain.run("vague task", cfg)
        self.assertEqual(calls["sample"], 1)                 # sampled ONCE per run
        self.assertEqual(calls["judge"], 2)                  # bound judge used every round
        self.assertEqual(calls["sols"], ["baseline", "alt1", "alt2", "alt3"])

    # ── answer_prob_mode selector (#26, off by default) ──
    def test_answer_prob_mode_defaults_stated(self):
        cfg = self._cfg(["a problem"])
        self.assertEqual(cfg["answer_prob_mode"], "stated")
        self.assertEqual(cfg["answer_samples"], 6)
        self.assertEqual(cfg["answer_sample_temperature"], 1.0)

    def test_answer_prob_mode_cli(self):
        self.assertEqual(self._cfg(["--answer-prob-mode", "sampled", "p"])["answer_prob_mode"],
                         "sampled")
        self.assertEqual(self._cfg(["--answer-samples", "4", "p"])["answer_samples"], 4)

    def test_answer_prob_mode_env(self):
        old = os.environ.get("INFOGAIN_ANSWER_PROB_MODE")
        try:
            os.environ["INFOGAIN_ANSWER_PROB_MODE"] = "sampled"
            self.assertEqual(self._cfg(["p"])["answer_prob_mode"], "sampled")
            # CLI beats env
            self.assertEqual(self._cfg(["--answer-prob-mode", "stated", "p"])["answer_prob_mode"],
                             "stated")
        finally:
            os.environ.pop("INFOGAIN_ANSWER_PROB_MODE", None) if old is None else os.environ.update(
                {"INFOGAIN_ANSWER_PROB_MODE": old})

    def test_run_routes_projection_by_answer_prob_mode(self):
        # the SAFETY INVARIANT, asserted: a cfg straight from DEFAULTS (no answer_prob_mode key)
        # must route to the stated projection batch; "sampled" must route to the sampled one.
        def _go(cfg):
            seen = {"fn": None}
            with mock.patch.object(pipeline, "frame_and_plan",
                                   return_value=({"goal": "g", "decision": "d",
                                                  "success_criteria": [], "baseline_plan": "p"},
                                                 None)), \
                 mock.patch.object(pipeline, "generate_questions",
                                   side_effect=lambda *a, **k: ([{"question": "Q", "target": "t",
                                                                 "answers": [],
                                                                 "derivable_prob": 0.1}], None)), \
                 mock.patch.object(pipeline, "project_answers_batch",
                                   side_effect=lambda p, f, recs, *a, **k: (
                                       seen.__setitem__("fn", "stated"), recs)[1]), \
                 mock.patch.object(pipeline, "project_answers_sampled_batch",
                                   side_effect=lambda p, f, recs, *a, **k: (
                                       seen.__setitem__("fn", "sampled"), recs)[1]), \
                 mock.patch.object(pipeline, "judge_plan_change_batch",
                                   side_effect=lambda p, f, b, recs, *a, **k: recs):
                infogain.run("vague task", cfg)
            return seen["fn"]

        cfg = {k: v for k, v in infogain.DEFAULTS.items()}
        cfg["families"] = {"enabled": False}
        cfg["max_rounds"] = 1
        self.assertEqual(_go(dict(cfg)), "stated")
        self.assertEqual(_go(dict(cfg, answer_prob_mode="sampled")), "sampled")

    def test_default_run_uses_absolute_judge_batch(self):
        # the SAFETY INVARIANT, asserted: a cfg straight from DEFAULTS (no value_judge_mode key) must
        # route to the absolute batch — the experiment is inert unless explicitly switched on.
        cfg = {k: v for k, v in infogain.DEFAULTS.items()}
        cfg["families"] = {"enabled": False}
        cfg["max_rounds"] = 1
        seen = {"fn": None}
        with mock.patch.object(pipeline, "frame_and_plan",
                               return_value=({"goal": "g", "decision": "d", "success_criteria": [],
                                              "baseline_plan": "p"}, None)), \
             mock.patch.object(pipeline, "generate_questions",
                               side_effect=lambda *a, **k: ([{"question": "Q", "target": "t",
                                                             "answers": [], "derivable_prob": 0.1}], None)), \
             mock.patch.object(pipeline, "project_answers_batch",
                               side_effect=lambda p, f, recs, *a, **k: recs), \
             mock.patch.object(pipeline, "judge_plan_change_batch",
                               side_effect=lambda *a, **k: (seen.__setitem__("fn", "absolute"), a[3])[1]), \
             mock.patch.object(pipeline, "judge_plan_change_pairwise_batch",
                               side_effect=lambda *a, **k: (seen.__setitem__("fn", "pairwise"), a[3])[1]):
            infogain.run("vague task", cfg)
        self.assertEqual(seen["fn"], "absolute")

    def test_ranked_list_groups_by_family(self):
        bucket = [{"question": "q-contra", "value": 0.6, "family": "Approach", "lens": "contrarian",
                   "recommendation": "PRE_ANSWER", "target": "t1", "evsi": 0.5, "modal_answer": None},
                  {"question": "q-scoped", "value": 0.5, "family": "Data", "lens": "scoped",
                   "recommendation": "ASSUME_DEFAULT", "target": "t2", "evsi": 0.4, "modal_answer": None}]
        md = infogain._ranked_list(bucket)
        self.assertIn("### Approach", md)
        self.assertIn("contrarian lens", md)
        self.assertIn("### Data", md)
        self.assertLess(md.index("### Approach"), md.index("### Data"))  # higher-value family first

    def test_ranked_list_flat_when_no_family(self):
        bucket = [{"question": "q1", "value": 0.6, "recommendation": "PRE_ANSWER", "target": "t",
                   "evsi": 0.5, "modal_answer": None}]
        md = infogain._ranked_list(bucket)
        self.assertNotIn("###", md)
        self.assertIn("[weight 0.60]", md)


@unittest.skipUnless(_PIPELINE_OK, "ask skill / model_utils not importable")
class TestPlumbing(unittest.TestCase):
    """The real transport layer everything else mocks away: raw_chat's payload/accounting,
    _call_json's retry+trace, _parallel's order preservation. A silent defect here corrupts
    every run, which is exactly why these get direct tests despite being 'infrastructure'."""

    def _fake_urlopen(self, captured, response_obj):
        import io, contextlib

        @contextlib.contextmanager
        def fake(req, timeout=None):
            captured["url"] = req.full_url
            captured["timeout"] = timeout
            captured["payload"] = json.loads(req.data.decode("utf-8"))
            yield io.BytesIO(json.dumps(response_obj).encode("utf-8"))
        return fake

    def test_raw_chat_payload_accounting_and_content(self):
        captured = {}
        resp = {"message": {"content": "  hello  "}, "prompt_eval_count": 11, "eval_count": 7}
        pipeline.reset_usage()
        with mock.patch.object(pipeline.urllib.request, "urlopen",
                               self._fake_urlopen(captured, resp)):
            out = pipeline.raw_chat("some-model", "hi", timeout=42, temperature=0.7,
                                    num_predict=123)
        p = captured["payload"]
        self.assertEqual(p["model"], "some-model")
        self.assertFalse(p["stream"])
        self.assertFalse(p["think"])                       # reasoning-channel suppression
        self.assertEqual(p["options"], {"temperature": 0.7, "num_predict": 123})
        self.assertEqual(captured["timeout"], 42)
        self.assertEqual(out["content"], "hello")          # stripped
        self.assertIsNone(out["error"])
        self.assertEqual((out["input_tokens"], out["output_tokens"]), (11, 7))
        u = pipeline.get_usage()
        self.assertEqual((u["calls"], u["input_tokens"], u["output_tokens"]), (1, 11, 7))

    def test_raw_chat_error_as_data_never_raises(self):
        def boom(req, timeout=None):
            raise OSError("connection refused")
        with mock.patch.object(pipeline.urllib.request, "urlopen", boom):
            out = pipeline.raw_chat("m", "hi")
        self.assertEqual(out["content"], "")
        self.assertIn("connection refused", out["error"])

    def test_call_json_retry_nudge_and_sink(self):
        calls = []

        def fake_raw(model, content, timeout=0, num_predict=0, temperature=0.0):
            calls.append(content)
            if len(calls) == 1:
                return {"content": "sorry, no json here", "elapsed": 0.1, "error": None}
            return {"content": '{"ok": 1}', "elapsed": 0.1, "error": None}

        sink = []
        with mock.patch.object(pipeline, "raw_chat", side_effect=fake_raw):
            parsed, err = pipeline._call_json("m", "PROMPT", 10, 100, sink=sink)
        self.assertEqual(parsed, {"ok": 1})
        self.assertIsNone(err)
        self.assertEqual(len(calls), 2)
        self.assertIn("Return ONLY valid JSON", calls[1])   # the retry nudge
        self.assertEqual(sink[0]["attempts"], 2)

    def test_call_json_exhausted_sinks_the_error(self):
        with mock.patch.object(pipeline, "raw_chat",
                               return_value={"content": "still prose", "elapsed": 0.1,
                                             "error": None}):
            sink = []
            parsed, err = pipeline._call_json("m", "PROMPT", 10, 100, sink=sink)
        self.assertIsNone(parsed)
        self.assertIn("no parseable JSON", err)
        self.assertEqual(sink[0]["attempts"], 2)
        self.assertIsNotNone(sink[0]["error"])

    def test_parallel_preserves_input_order_under_staggered_completion(self):
        import time as _t
        items = [{"i": 0, "sleep": 0.05}, {"i": 1, "sleep": 0.0}, {"i": 2, "sleep": 0.02}]

        def work(it):
            _t.sleep(it["sleep"])              # completion order 1,2,0 — result order must be 0,1,2
            return dict(it, done=True)

        out = pipeline._parallel(work, items)
        self.assertEqual([r["i"] for r in out], [0, 1, 2])
        self.assertTrue(all(r["done"] for r in out))

    def test_parallel_exception_marks_item_without_killing_batch(self):
        items = [{"i": 0}, {"i": 1}]

        def work(it):
            if it["i"] == 1:
                raise RuntimeError("boom")
            return dict(it, done=True)

        out = pipeline._parallel(work, items)
        self.assertTrue(out[0]["done"])
        self.assertEqual(out[1]["error"], "boom")           # item survives, tagged
        self.assertEqual(out[1]["i"], 1)


@unittest.skipUnless(_PIPELINE_OK, "ask skill / model_utils not importable")
class TestBehaviorJudge(unittest.TestCase):
    """#28: Δplan elicited as BEHAVIOR change of the result (consequence, not code size)."""

    def test_prompt_pins_consequence_not_code_size(self):
        p = pipeline.judge_behavior_prompt("prob", {"goal": "g"}, "base", "Q?",
                                           [{"answer": "A"}, {"answer": "B"}])
        self.assertIn("consequence, not code size", p)
        self.assertIn("0.2 or less", p)          # the boilerplate anchor
        self.assertIn("one-token change", p)     # the flip anchor
        self.assertIn('{"answers": [{"delta_plan": float, "stakes": float}', p)  # same contract

    def test_behavior_judge_same_contract_as_absolute(self):
        rec = {"question": "Q?", "answers": [{"answer": "A"}, {"answer": "B"}]}
        judged = {"answers": [{"delta_plan": 0.9, "stakes": 0.8},
                              {"delta_plan": 0.1, "stakes": 0.2}]}
        with mock.patch.object(pipeline, "_call_json", return_value=(judged, None)):
            out = pipeline.judge_plan_change_behavior("p", {"goal": "g"}, "base", rec, "m")
        self.assertEqual(out["answers"][0]["delta_plan"], 0.9)
        self.assertEqual(out["answers"][1]["stakes"], 0.2)
        # malformed reply -> safe zeros, same as absolute
        with mock.patch.object(pipeline, "_call_json", return_value=(None, "boom")):
            out = pipeline.judge_plan_change_behavior("p", {"goal": "g"}, "base",
                                                      {"question": "Q?",
                                                       "answers": [{"answer": "A"}]}, "m")
        self.assertEqual(out["answers"][0]["delta_plan"], 0.0)

    def test_run_routes_behavior_mode_and_default_stays_absolute(self):
        def _go(cfg):
            seen = {"fn": None}
            with mock.patch.object(pipeline, "frame_and_plan",
                                   return_value=({"goal": "g", "decision": "d",
                                                  "success_criteria": [], "baseline_plan": "p"},
                                                 None)), \
                 mock.patch.object(pipeline, "generate_questions",
                                   side_effect=lambda *a, **k: ([{"question": "Q", "target": "t",
                                                                 "answers": [],
                                                                 "derivable_prob": 0.1}], None)), \
                 mock.patch.object(pipeline, "project_answers_batch",
                                   side_effect=lambda p, f, recs, *a, **k: recs), \
                 mock.patch.object(pipeline, "judge_plan_change_batch",
                                   side_effect=lambda *a, **k: (seen.__setitem__("fn", "absolute"),
                                                                a[3])[1]), \
                 mock.patch.object(pipeline, "judge_plan_change_behavior_batch",
                                   side_effect=lambda *a, **k: (seen.__setitem__("fn", "behavior"),
                                                                a[3])[1]):
                infogain.run("vague task", cfg)
            return seen["fn"]

        cfg = {k: v for k, v in infogain.DEFAULTS.items()}
        cfg["families"] = {"enabled": False}
        cfg["max_rounds"] = 1
        self.assertEqual(_go(dict(cfg)), "absolute")                       # byte-identical pin
        self.assertEqual(_go(dict(cfg, value_judge_mode="behavior")), "behavior")

    def test_cli_accepts_behavior(self):
        cfg = infogain.resolve_config(infogain.build_parser().parse_args(
            ["--value-judge-mode", "behavior", "p"]))
        self.assertEqual(cfg["value_judge_mode"], "behavior")


@unittest.skipUnless(_PIPELINE_OK, "ask skill / model_utils not importable")
class TestDeriveOrAsk(unittest.TestCase):
    """Derive-or-ask: derivability claims are tested, not trusted. Derived answers become
    tombstones in the evidence context; failed claims restore the question's uncertainty."""

    def _cfg(self, argv):
        return infogain.resolve_config(infogain.build_parser().parse_args(argv))

    @staticmethod
    def _q(name, derivable, target=None):
        return {"question": name, "target": target or name,
                "derivable_prob": derivable,
                "answers": [{"answer": "A1", "prob": 0.6, "delta_plan": 0.5, "stakes": 0.5},
                            {"answer": "A2", "prob": 0.4, "delta_plan": 0.3, "stakes": 0.4}]}

    def _run(self, cfg, rounds_qs, derive_fn):
        """Drive run() with scripted generation rounds and a scripted derivation oracle.
        rounds_qs: list of question-lists, one per generation call (then [] forever).
        Returns (result, gen_evidence_snapshots, derive_calls)."""
        gen_i = {"i": 0}
        gen_ev = []
        derive_calls = []

        def fake_gen(problem, framing, model, n, avoid, timeout, sink=None, samples=1,
                     temperature=0.0, evidence=None):
            gen_ev.append(list(evidence or []))
            qs = rounds_qs[gen_i["i"]] if gen_i["i"] < len(rounds_qs) else []
            gen_i["i"] += 1
            return list(qs), None

        def fake_derive(problem, framing, rec, model, evidence=None, timeout=60, sink=None):
            derive_calls.append(rec["question"])
            return derive_fn(rec)

        with mock.patch.object(pipeline, "frame_and_plan",
                               return_value=({"goal": "g", "decision": "d", "success_criteria": [],
                                              "baseline_plan": "base"}, None)), \
             mock.patch.object(pipeline, "generate_questions", side_effect=fake_gen), \
             mock.patch.object(pipeline, "project_answers_batch",
                               side_effect=lambda p, f, recs, *a, **k: recs), \
             mock.patch.object(pipeline, "judge_plan_change_batch",
                               side_effect=lambda p, f, b, recs, *a, **k: recs), \
             mock.patch.object(pipeline, "attempt_derivation", side_effect=fake_derive):
            result = infogain.run("vague task", cfg)
        return result, gen_ev, derive_calls

    def _base_cfg(self, **kw):
        cfg = {k: v for k, v in infogain.DEFAULTS.items()}
        cfg["families"] = {"enabled": False}
        cfg["max_rounds"] = 1
        cfg.update(kw)
        return cfg

    def test_derived_becomes_tombstone_and_leaves_bucket(self):
        cfg = self._base_cfg(auto_derive="on")
        result, gen_ev, calls = self._run(
            cfg, [[self._q("Which gateway?", 0.9), self._q("What limits?", 0.1, "limits")]],
            lambda rec: {"answer": "Kong", "derived": True})
        self.assertEqual(calls, ["Which gateway?"])          # only the high claim tested
        self.assertEqual(len(result["derived"]), 1)
        self.assertEqual(result["derived"][0]["answer"], "Kong")
        self.assertNotIn("Which gateway?", [r["question"] for r in result["bucket"]])
        self.assertIn("What limits?", [r["question"] for r in result["bucket"]])
        self.assertEqual(result["evidence"], [])             # caller-provided facts only
        rec = [r for r in result["all_scored"] if r["question"] == "Which gateway?"][0]
        self.assertEqual(rec["recommendation"], "DERIVED")
        self.assertEqual(rec["derivability_tested"], "derived")
        # final-round derivation grants ONE extra refill round, which re-plans against the
        # tombstone (the round-2 generation call must see it as evidence)
        self.assertEqual(len(gen_ev), 2)
        self.assertTrue(any("Kong" in e for e in gen_ev[1]))

    def test_cannot_derive_restores_uncertainty(self):
        cfg = self._base_cfg(auto_derive="on")
        result, gen_ev, calls = self._run(
            cfg, [[self._q("Which gateway?", 0.9)]],
            lambda rec: {"answer": "", "derived": False})
        rec = [r for r in result["all_scored"] if r["question"] == "Which gateway?"][0]
        self.assertEqual(rec["derivability_tested"], "failed")
        self.assertEqual(rec["derivable_prob"], infogain.DEFAULTS["cannot_derive_cap"])
        self.assertNotEqual(rec.get("recommendation"), "DERIVED")
        # re-scored with honest uncertainty -> clears the floor and ranks
        self.assertIn("Which gateway?", [r["question"] for r in result["bucket"]])
        self.assertEqual(len(gen_ev), 1)                     # no extra round without a derivation
        self.assertEqual(result["derived"], [])

    def test_absent_key_is_off_and_flag_off_is_inert(self):
        # the SAFETY INVARIANT (families/#24/#26 precedent): a cfg straight from DEFAULTS has
        # no auto_derive key -> the pass must not run; explicit "off" likewise.
        for cfg in (self._base_cfg(), self._base_cfg(auto_derive="off")):
            result, _, calls = self._run(cfg, [[self._q("Which gateway?", 0.9)]],
                                         lambda rec: {"answer": "X", "derived": True})
            self.assertEqual(calls, [])
            self.assertEqual(result["derived"], [])
            self.assertNotIn("DERIVED",
                             [r.get("recommendation") for r in result["all_scored"]])

    def test_threshold_and_per_round_cap(self):
        cfg = self._base_cfg(auto_derive="on", derive_max_per_round=2)
        qs = [self._q("q95", 0.95), self._q("q90", 0.90), self._q("q85", 0.85),
              self._q("q50", 0.50)]
        result, _, calls = self._run(cfg, [qs], lambda rec: {"answer": "", "derived": False})
        self.assertEqual(calls, ["q95", "q90"])              # top claims first, cap respected
        rec85 = [r for r in result["all_scored"] if r["question"] == "q85"][0]
        self.assertIsNone(rec85.get("derivability_tested"))  # past cap: today's behavior
        rec50 = [r for r in result["all_scored"] if r["question"] == "q50"][0]
        self.assertIsNone(rec50.get("derivability_tested"))  # below threshold: untouched

    def test_extra_round_granted_at_most_once(self):
        cfg = self._base_cfg(auto_derive="on")
        result, gen_ev, calls = self._run(
            cfg, [[self._q("d1", 0.9)], [self._q("d2", 0.9)]],
            lambda rec: {"answer": "ans-" + rec["question"], "derived": True})
        # round 1 derives -> extra round; round 2 derives too but NO second extension
        self.assertEqual(len(gen_ev), 2)
        self.assertEqual(len(result["derived"]), 2)
        self.assertEqual(result["rounds_used"], 2)

    def test_attempt_derivation_escape_and_malformed(self):
        # hedges are non-answers wearing an answer's clothes — the do-no-harm check caught
        # fast tombstoning "The prompt does not specify whether ..." as a derivation
        rec = {"question": "Q?"}
        for content, expect in [("", False), ("CANNOT_DERIVE", False),
                                ("cannot_derive — no info", False),
                                ("The prompt does not specify whether tone matters.", False),
                                ("Not specified in the context.", False),
                                ("There is insufficient information to say.", False),
                                ("Kong", True),
                                ("Use HTTP-only cookies for browser SPAs.", True)]:
            with mock.patch.object(pipeline, "raw_chat",
                                   return_value={"content": content, "elapsed": 0.1,
                                                 "error": None}):
                got = pipeline.attempt_derivation("p", {"goal": "g"}, rec, "m")
            self.assertEqual(got["derived"], expect, content)

    def test_derive_prompt_is_knowledge_inclusive(self):
        # pinned: a strict "from the prompt alone" wording re-creates the bucket-flooding
        # failure mode (knowledge-derivable claims failing derivation) — see design-decisions.
        p = pipeline.derive_answer_prompt("prob", {"goal": "g"}, "Q?", ["f1"])
        self.assertIn("your own general knowledge", p)
        self.assertIn("CANNOT_DERIVE", p)
        self.assertIn("f1", p)

    def test_report_renders_derived_section(self):
        result = {"problem": "p", "evidence": [], "framing": {"goal": "g", "decision": "d",
                  "success_criteria": [], "baseline_plan": "b"},
                  "derived": [{"question": "Which gateway?", "answer": "Kong",
                               "derivable_prob": 0.9, "round": 1}],
                  "config": dict(infogain.DEFAULTS, mode="focus"), "rounds_used": 1,
                  "candidates_considered": 2, "all_scored": [], "bucket": [],
                  "discarded_count": 0, "min_met": False, "pre_answer": [], "usage": {}}
        md = infogain.render_markdown(result)
        self.assertIn("Resolved during analysis", md)
        self.assertIn("Kong", md)
        self.assertIn("tombstoned", md)

    def test_config_resolution_auto_derive(self):
        self.assertEqual(self._cfg(["p"])["auto_derive"], "on")          # CLI default ON
        self.assertEqual(self._cfg(["--auto-derive", "off", "p"])["auto_derive"], "off")
        old = os.environ.get("INFOGAIN_AUTO_DERIVE")
        try:
            os.environ["INFOGAIN_AUTO_DERIVE"] = "off"
            self.assertEqual(self._cfg(["p"])["auto_derive"], "off")
            self.assertEqual(self._cfg(["--auto-derive", "on", "p"])["auto_derive"], "on")
        finally:
            os.environ.pop("INFOGAIN_AUTO_DERIVE", None) if old is None else os.environ.update(
                {"INFOGAIN_AUTO_DERIVE": old})
        cfg = self._cfg(["--derive-model", "glm", "--derive-threshold", "0.7", "p"])
        self.assertEqual(cfg["derive_model"], "glm")
        self.assertEqual(cfg["derive_threshold"], 0.7)
        self.assertEqual(self._cfg(["p"])["derive_model"], "")           # "" -> judge model


@unittest.skipUnless(_PIPELINE_OK, "ask skill / model_utils not importable")
class TestCliAndDryRun(unittest.TestCase):
    """infogain.main — the live entry point: exit codes, evidence-file ingestion, output
    dispatch — plus the _dry_run prompt assembly and the breadth-consolidation branch."""

    def _fake_result(self):
        return {"problem": "p", "evidence": [], "derived": [], "usage": {},
                "framing": {"goal": "g", "decision": "d", "success_criteria": [],
                            "baseline_plan": "b"},
                "framing_error": None, "config": dict(infogain.DEFAULTS, mode="focus"),
                "rounds_used": 1, "families": [], "candidates_considered": 0,
                "all_scored": [], "bucket": [], "discarded_count": 0, "min_met": False,
                "pre_answer": []}

    def test_main_no_problem_exits_3(self):
        self.assertEqual(infogain.main([]), 3)

    def test_main_ollama_unreachable_exits_2(self):
        with mock.patch.object(pipeline, "ollama_reachable", return_value=False):
            self.assertEqual(infogain.main(["some problem"]), 2)

    def test_main_evidence_file_json_and_output_write(self):
        import tempfile
        seen = {}

        def fake_run(problem, cfg, progress=None, trace=False, evidence=None):
            seen["evidence"] = evidence
            return self._fake_result()

        with tempfile.TemporaryDirectory() as td:
            ev = os.path.join(td, "facts.txt")
            with open(ev, "w") as f:
                f.write("# a comment\nbudget: free tier\n\nusers: coaches\n")
            out_path = os.path.join(td, "report.md")
            with mock.patch.object(pipeline, "ollama_reachable", return_value=True), \
                 mock.patch.object(infogain, "run", side_effect=fake_run):
                rc = infogain.main(["problem text", "--evidence-file", ev,
                                    "-o", out_path, "--quiet"])
            self.assertEqual(rc, 0)
            self.assertEqual(seen["evidence"], ["budget: free tier", "users: coaches"])
            with open(out_path) as f:
                self.assertIn("Key Questions", f.read())     # markdown written to -o

    def test_main_json_dispatch(self):
        import io
        from contextlib import redirect_stdout
        with mock.patch.object(pipeline, "ollama_reachable", return_value=True), \
             mock.patch.object(infogain, "run", return_value=self._fake_result()):
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = infogain.main(["problem", "--json", "--quiet"])
        self.assertEqual(rc, 0)
        parsed = json.loads(buf.getvalue())
        self.assertEqual(parsed["problem"], "p")

    def test_dry_run_assembles_all_stage_prompts(self):
        import io
        from contextlib import redirect_stdout

        def go(argv):
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = infogain.main(argv)
            self.assertEqual(rc, 0)
            return buf.getvalue()

        out = go(["a problem", "--dry-run"])
        for marker in ("STAGE 0", "STAGE 1a", "STAGE 1b", "STAGE 2", "STAGE 3",
                       "STAGE 3b"):                            # families default-on, derive on
            self.assertIn(marker, out)
        out_off = go(["a problem", "--dry-run", "--auto-derive", "off", "--no-families"])
        self.assertNotIn("STAGE 3b", out_off)
        self.assertNotIn("STAGE 1a", out_off)
        self.assertIn("STAGE 1 —", out_off)                    # flat generator prompt instead

    def test_breadth_consolidation_branch_runs(self):
        # the only run() branch previously uncovered: gen_samples>1 routes candidates
        # through consolidate_questions
        cfg = {k: v for k, v in infogain.DEFAULTS.items()}
        cfg["families"] = {"enabled": False}
        cfg["max_rounds"] = 1
        cfg["gen_samples"] = 2
        called = {"n": 0}

        def fake_consolidate(problem, qs, model, timeout, sink=None):
            called["n"] += 1
            return qs

        with mock.patch.object(pipeline, "frame_and_plan",
                               return_value=({"goal": "g", "decision": "d",
                                              "success_criteria": [], "baseline_plan": "b"},
                                             None)), \
             mock.patch.object(pipeline, "generate_questions",
                               side_effect=lambda *a, **k: ([{"question": "Q1", "target": "t1",
                                                             "answers": [], "derivable_prob": 0.1},
                                                            {"question": "Q2", "target": "t2",
                                                             "answers": [], "derivable_prob": 0.1}],
                                                            None)), \
             mock.patch.object(pipeline, "consolidate_questions",
                               side_effect=fake_consolidate), \
             mock.patch.object(pipeline, "project_answers_batch",
                               side_effect=lambda p, f, recs, *a, **k: recs), \
             mock.patch.object(pipeline, "judge_plan_change_batch",
                               side_effect=lambda p, f, b, recs, *a, **k: recs):
            infogain.run("vague task", cfg)
        self.assertEqual(called["n"], 1)


@unittest.skipUnless(os.environ.get("INFOGAIN_TEST_LIVE"),
                     "live suite: set INFOGAIN_TEST_LIVE=1 or run tests/run.py live")
@unittest.skipUnless(_PIPELINE_OK, "ask skill / model_utils not importable")
class TestLive(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reachable = pipeline.ollama_reachable(timeout=5)

    def setUp(self):
        if not self.reachable:
            self.skipTest("Ollama not reachable at " + pipeline.OLLAMA_URL)

    def test_live_small_run(self):
        cfg = {k: v for k, v in infogain.DEFAULTS.items()}
        cfg.update(question_gen_model="fast", answer_model="fast", value_judge_model="fast",
                   max_rounds=1, questions_per_round=3, answers_per_question=3,
                   min_bucket_size=1)
        result = infogain.run("Build a tool to summarize our team's documents.", cfg)
        self.assertIn("baseline_plan", result["framing"])
        self.assertIsInstance(result["bucket"], list)
        for r in result["bucket"]:
            self.assertIn("value", r)
            self.assertGreaterEqual(r["value"], 0.0)
            self.assertLessEqual(r["value"], 1.0)
        # report renders without error
        md = infogain.render_markdown(result)
        self.assertIn("Key Questions", md)
        self.assertIn("Prompt:", md)


if __name__ == "__main__":
    unittest.main(verbosity=2)
