#!/usr/bin/env python3
"""Unit tests for the iterate-context loop logic — deterministic, no network.

Monkeypatches iterate.rank so the loop's stop/cap/tombstone/eligibility logic is tested in
isolation (no Ollama, no hermes). Run:
    python3 -m pytest tests/test_iterate.py -v        (or)   python3 tests/test_iterate.py
"""

import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "scripts"))

import iterate  # noqa: E402


def q(text, value):
    return {"question": text, "value": value, "target": text}


def found_answerer(qq, problem, evidence, cfg):
    return True, f"answer:{qq['question']}"


def notfound_answerer(qq, problem, evidence, cfg):
    return False, "not discoverable"


def mock_responder(problem, evidence, cfg):
    return f"resp/{len(evidence)}"


class LoopLogic(unittest.TestCase):
    def setUp(self):
        self._orig = iterate.rank
        self.calls = []

    def tearDown(self):
        iterate.rank = self._orig

    def _patch(self, sequence):
        """sequence: list of question-lists, one per rank() call (last repeats)."""
        seq = list(sequence)

        def fake(problem, evidence, rank_cfg):
            self.calls.append(list(evidence))
            return seq[min(len(self.calls) - 1, len(seq) - 1)]
        iterate.rank = fake

    def test_converged_when_all_below_floor(self):
        self._patch([[q("a", 0.05), q("b", 0.01)]])
        out = iterate.iterate("p", {"k": 2, "max_rounds": 3, "floor": 0.12},
                              answerer=found_answerer, responder=mock_responder)
        self.assertEqual(out["rounds"], 1)
        self.assertIn("converged", out["stop_reason"])
        self.assertEqual(out["n_answered"], 0)
        self.assertFalse(out["artificial_cap_bound"])

    def test_k_caps_research_per_round(self):
        # 5 above floor, K=2 -> only 2 researched in the (single, then converged) round
        self._patch([[q("a", .9), q("b", .8), q("c", .7), q("d", .6), q("e", .5)]])
        # after answering a,b the same list returns; c,d,e still above floor -> more rounds
        out = iterate.iterate("p", {"k": 2, "max_rounds": 1, "floor": 0.12},
                              answerer=found_answerer, responder=mock_responder)
        self.assertEqual(out["n_answered"], 2)          # only K researched
        self.assertTrue(out["k_capped"])                # 5 > 2
        self.assertEqual(out["stop_reason"], "max_rounds reached")
        self.assertTrue(out["artificial_cap_bound"])

    def test_answered_filter_drives_convergence(self):
        # same 3 questions every round; once all answered -> converged (no new above floor)
        same = [q("a", .9), q("b", .8), q("c", .7)]
        self._patch([same])
        out = iterate.iterate("p", {"k": 2, "max_rounds": 5, "floor": 0.12},
                              answerer=found_answerer, responder=mock_responder)
        self.assertEqual(out["n_answered"], 3)          # a,b then c, then converge
        self.assertIn("converged", out["stop_reason"])
        self.assertEqual(out["rounds"], 2 + 1)          # r1: a,b · r2: c · r3: converged

    def test_max_rounds_with_fresh_questions(self):
        # fresh distinct questions each round -> never converges -> hits max_rounds
        self._patch([[q("r1a", .9), q("r1b", .8)],
                     [q("r2a", .9), q("r2b", .8)],
                     [q("r3a", .9), q("r3b", .8)],
                     [q("r4a", .9)]])
        out = iterate.iterate("p", {"k": 2, "max_rounds": 3, "floor": 0.12},
                              answerer=found_answerer, responder=mock_responder)
        self.assertEqual(out["rounds"], 3)
        self.assertEqual(out["n_answered"], 6)
        self.assertEqual(out["stop_reason"], "max_rounds reached")

    def test_notfound_tombstones(self):
        self._patch([[q("a", .9)], [q("b", .8)], [q("c", .7)]])
        out = iterate.iterate("p", {"k": 1, "max_rounds": 3, "floor": 0.12},
                              answerer=notfound_answerer, responder=mock_responder)
        self.assertEqual(out["n_gaps"], 3)
        self.assertEqual(out["n_answered"], 0)
        self.assertTrue(all(t["status"] == "NOT_FOUND" for t in out["tombstones"]))
        self.assertIn("known gap", out["tombstones"][0]["evidence"])

    def test_context_grows_monotonically(self):
        # each rank() call should see one more fact than the last (append-only context)
        self._patch([[q("a", .9), q("b", .8)], [q("c", .7), q("d", .6)], [q("e", .5)]])
        iterate.iterate("p", {"k": 2, "max_rounds": 3, "floor": 0.12},
                        answerer=found_answerer, responder=mock_responder)
        sizes = [len(ev) for ev in self.calls]
        self.assertEqual(sizes, sorted(sizes))          # non-decreasing
        self.assertEqual(sizes, [0, 2, 4])              # facts accrue across rounds

    def test_extract_recovers_long_false_positive(self):
        # is_api_error mislabels a long real answer (about errors) -> recover from `error`
        text, err = iterate._extract({"content": "", "error": "API error: " + ("x. " * 100)})
        self.assertIsNone(err)
        self.assertGreater(len(text), 200)

    def test_extract_keeps_short_real_error(self):
        text, err = iterate._extract({"content": "", "error": "API error: rate limit exceeded"})
        self.assertEqual(text, "")
        self.assertIn("rate limit", err)

    def test_extract_strips_suggestion_block(self):
        raw = "Here is the answer.\n\nSUGGESTION:{\"options\": [{\"label\": \"x\"}]}"
        text, err = iterate._extract({"content": raw})
        self.assertEqual(text, "Here is the answer.")
        self.assertNotIn("SUGGESTION", text)

    def test_validate_selection_picks_ends(self):
        ranked = [q("top1", .9), q("top2", .8), q("mid", .5), q("bot2", .2), q("bot1", .1)]
        self._patch([ranked])
        top = iterate.validate_selection("p", "top", 2, answerer=found_answerer, responder=mock_responder)
        bot = iterate.validate_selection("p", "bottom", 2, answerer=found_answerer, responder=mock_responder)
        self.assertEqual(top["selected"], ["top1", "top2"])
        self.assertEqual(bot["selected"], ["bot1", "bot2"])  # reversed: worst-first


if __name__ == "__main__":
    unittest.main(verbosity=2)
