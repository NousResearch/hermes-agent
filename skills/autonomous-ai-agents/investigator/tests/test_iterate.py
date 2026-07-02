#!/usr/bin/env python3
"""Unit tests for the Investigator loop — deterministic, no network.

Monkeypatches iterate.rank so the loop's stop/cap/tombstone/eligibility logic is tested in
isolation (no Ollama, no hermes). Resolves the sibling information-gain ranker via
INFOGAIN_SCRIPTS_DIR. Run:
    python3 tests/test_iterate.py
"""

import os
import sys
import unittest
from unittest import mock

_HERE = os.path.dirname(os.path.abspath(__file__))
# Point the investigator at the sibling information-gain ranker (source tree) + make iterate importable.
os.environ.setdefault("INFOGAIN_SCRIPTS_DIR",
                      os.path.abspath(os.path.join(_HERE, "..", "..", "information-gain", "scripts")))
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
        self._patch([[q("a", .9), q("b", .8), q("c", .7), q("d", .6), q("e", .5)]])
        out = iterate.iterate("p", {"k": 2, "max_rounds": 1, "floor": 0.12},
                              answerer=found_answerer, responder=mock_responder)
        self.assertEqual(out["n_answered"], 2)          # only K researched
        self.assertTrue(out["k_capped"])                # 5 > 2
        self.assertEqual(out["stop_reason"], "max_rounds reached")
        self.assertTrue(out["artificial_cap_bound"])

    def test_answered_filter_drives_convergence(self):
        same = [q("a", .9), q("b", .8), q("c", .7)]
        self._patch([same])
        out = iterate.iterate("p", {"k": 2, "max_rounds": 5, "floor": 0.12},
                              answerer=found_answerer, responder=mock_responder)
        self.assertEqual(out["n_answered"], 3)          # a,b then c, then converge
        self.assertIn("converged", out["stop_reason"])
        self.assertEqual(out["rounds"], 2 + 1)          # r1: a,b · r2: c · r3: converged

    def test_max_rounds_with_fresh_questions(self):
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
        self._patch([[q("a", .9), q("b", .8)], [q("c", .7), q("d", .6)], [q("e", .5)]])
        iterate.iterate("p", {"k": 2, "max_rounds": 3, "floor": 0.12},
                        answerer=found_answerer, responder=mock_responder)
        sizes = [len(ev) for ev in self.calls]
        self.assertEqual(sizes, sorted(sizes))          # non-decreasing
        self.assertEqual(sizes, [0, 2, 4])              # facts accrue across rounds

    def test_extract_recovers_long_false_positive(self):
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

    # ── seed evidence (relentless-solve prerequisite) ──
    def test_seed_evidence_reaches_rank_and_responder(self):
        self._patch([[q("a", .9)], []])
        captured = {}

        def resp(problem, evidence, cfg):
            captured["ev"] = list(evidence)
            return "r"
        out = iterate.iterate("p", {"k": 1, "max_rounds": 2, "floor": 0.12},
                              answerer=found_answerer, responder=resp,
                              seed_evidence=["Tried alfa: failed — 503"])
        self.assertEqual(self.calls[0], ["Tried alfa: failed — 503"])   # rank round 1 sees seeds
        self.assertEqual(captured["ev"][0], "Tried alfa: failed — 503")  # responder sees seeds first
        self.assertEqual(len(out["tombstones"]), 1)                      # seeds are NOT tombstones

    def test_seed_evidence_blank_lines_dropped(self):
        self._patch([[]])
        iterate.iterate("p", {"k": 1, "max_rounds": 1, "floor": 0.12},
                        answerer=found_answerer, responder=mock_responder,
                        seed_evidence=["  ", "", "fact one"])
        self.assertEqual(self.calls[0], ["fact one"])

    def test_no_seeds_is_backward_compatible(self):
        self._patch([[q("a", .9)]])
        iterate.iterate("p", {"k": 1, "max_rounds": 1, "floor": 0.12},
                        answerer=found_answerer, responder=mock_responder)
        self.assertEqual(self.calls[0], [])

    def test_evidence_file_flag_seeds_main(self):
        import contextlib
        import io
        import tempfile
        self._patch([[]])
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
            fh.write("# a comment\nTried alfa: failed — 503\n\n")
            path = fh.name
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                rc = iterate.main(["--problem", "p", "--dry-run", "--json",
                                   "--evidence-file", path])
            self.assertEqual(rc, 0)
            self.assertEqual(self.calls[0], ["Tried alfa: failed — 503"])  # comment + blank dropped
        finally:
            os.unlink(path)

    # ── capability ladder ──
    def test_capability_act_is_full_default(self):
        cfg = iterate.apply_capability({}, "act")
        self.assertIn("terminal", cfg["answer_toolsets"])
        self.assertEqual(cfg["answer_directive"], "")

    def test_capability_read_downscopes(self):
        cfg = iterate.apply_capability({}, "read")
        self.assertNotIn("terminal", cfg["answer_toolsets"])
        self.assertIn("READ-ONLY", cfg["answer_directive"])

    def test_capability_experiment_reversible_directive(self):
        cfg = iterate.apply_capability({}, "experiment")
        self.assertIn("terminal", cfg["answer_toolsets"])
        self.assertIn("REVERSIBLE", cfg["answer_directive"])


@unittest.skipUnless(getattr(iterate, "_HAVE_ASK", False), "model_utils (ask skill) not importable")
class GroundedAnswer(unittest.TestCase):
    """grounded_answer with a mocked dispatch_single — prompt assembly, directive, NOT_FOUND parse."""

    def _cfg(self, **over):
        cfg = dict(iterate.DEFAULTS)
        cfg.update(over)
        return cfg

    def test_directive_prepended_and_normal_answer(self):
        cfg = iterate.apply_capability(self._cfg(), "read")  # read -> directive + file,web toolsets
        ds = mock.MagicMock(return_value={"content": "The stack is FastAPI + Postgres.", "error": None})
        with mock.patch.object(iterate, "dispatch_single", ds), \
             mock.patch.object(iterate, "resolve_alias", lambda m: m):
            found, text = iterate.grounded_answer("What's the stack?", "Add auth", [], cfg)
        self.assertTrue(found)
        self.assertIn("FastAPI", text)
        # dispatch_single(model, PROMPT, "", toolsets, ...): directive prepended, toolsets downscoped
        self.assertIn("READ-ONLY", ds.call_args[0][1])
        self.assertEqual(ds.call_args[0][3], cfg["answer_toolsets"])
        self.assertNotIn("terminal", ds.call_args[0][3])

    def test_not_found_parsed_and_act_has_no_directive(self):
        cfg = self._cfg()  # act default -> empty directive, full toolsets
        ds = mock.MagicMock(return_value={"content": "NOT_FOUND: no credentials available", "error": None})
        with mock.patch.object(iterate, "dispatch_single", ds), \
             mock.patch.object(iterate, "resolve_alias", lambda m: m):
            found, text = iterate.grounded_answer("Do you have creds?", "task", [], cfg)
        self.assertFalse(found)
        self.assertIn("no credentials", text)
        self.assertNotIn("READ-ONLY", ds.call_args[0][1])     # act default -> no directive prepended
        self.assertEqual(ds.call_args[0][3], "file,web,terminal")

    def test_research_error_returns_not_found(self):
        cfg = self._cfg()
        ds = mock.MagicMock(return_value={"content": "", "error": "boom"})
        with mock.patch.object(iterate, "dispatch_single", ds), \
             mock.patch.object(iterate, "resolve_alias", lambda m: m):
            found, text = iterate.grounded_answer("q", "task", [], cfg)
        self.assertFalse(found)
        self.assertIn("research error", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
