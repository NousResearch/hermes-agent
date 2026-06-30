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

import math
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))

import voi  # noqa: E402


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

    def test_exploration_value_answerability_discount(self):
        base = math.sqrt(0.81 * 0.49)
        self.assertAlmostEqual(voi.exploration_value(0.81, 0.49, 1.0), base)  # default unchanged
        self.assertAlmostEqual(voi.exploration_value(0.81, 0.49, 0.5), 0.5 * base)
        self.assertEqual(voi.exploration_value(0.9, 0.9, 0.0), 0.0)  # unanswerable → 0

    def test_score_record_answerability_scales_value(self):
        ans = [_answer(0.6, 0.8, 0.7), _answer(0.4, 0.2, 0.3)]
        r1 = {"answers": [dict(a) for a in ans], "derivable_prob": 0.1, "answerability": 1.0}
        r2 = {"answers": [dict(a) for a in ans], "derivable_prob": 0.1, "answerability": 0.5}
        voi.score_record(r1)
        voi.score_record(r2)
        self.assertAlmostEqual(r2["value"], 0.5 * r1["value"], places=4)
        self.assertEqual(r1["answerability"], 1.0)
        # default (missing answerability) behaves as 1.0
        r3 = {"answers": [dict(a) for a in ans], "derivable_prob": 0.1}
        voi.score_record(r3)
        self.assertAlmostEqual(r3["value"], r1["value"], places=4)

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

    def test_project_answers_parses_answerability(self):
        rec = {"question": "Q"}
        with self._mock_raw('{"derivable_prob":0.2,"answerability":0.4,'
                            '"answers":[{"answer":"a","prob":1.0}]}'):
            pipeline.project_answers("p", {"goal": "g"}, rec, "fast", 5)
        self.assertEqual(rec["answerability"], 0.4)
        self.assertEqual(rec["derivable_prob"], 0.2)

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
        self.assertIn("Information-Gain Analysis", md)
        self.assertIn("Pre-answer these", md)
        self.assertIn("exploration value", md)

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
        md = infogain.render_markdown(result)
        self.assertIn("budget is $0", md)
        self.assertIn("answerable", md)  # new column header

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
        self.assertIn("exploration value =", md)
        self.assertIn("answerability", md)


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
        self.assertIn("Information-Gain Analysis", infogain.render_markdown(result))


if __name__ == "__main__":
    unittest.main(verbosity=2)
