#!/usr/bin/env python3
"""Unit tests for the Investigator loop — deterministic, no network.

Monkeypatches iterate.rank so the loop's stop/cap/tombstone/eligibility logic is tested in
isolation (no Ollama, no hermes). Resolves the sibling next-best-questions ranker via
INFOGAIN_SCRIPTS_DIR. Run:
    python3 tests/test_iterate.py
"""

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

_HERE = os.path.dirname(os.path.abspath(__file__))
# Point the investigator at the sibling next-best-questions ranker (source tree) + make iterate importable.
os.environ.setdefault("INFOGAIN_SCRIPTS_DIR",
                      os.path.abspath(os.path.join(_HERE, "..", "..", "next-best-questions", "scripts")))
sys.path.insert(0, os.path.join(_HERE, "..", "scripts"))

import answerer  # noqa: E402
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

    def test_tombstones_carry_ranked_value_stakes_and_recommendation(self):
        ranked = {**q("deployment target", .73), "recommendation": "ASK",
                  "answers": [{"stakes": .25}, {"stakes": None}, {"stakes": .81}, "invalid"]}
        answered = iterate._tombstone(ranked, True, "production")
        gap = iterate._tombstone(ranked, False, "no access")
        for tomb in (answered, gap):
            self.assertEqual(tomb["value"], .73)
            self.assertEqual(tomb["stakes"], .81)
            self.assertEqual(tomb["recommendation"], "ASK")
        self.assertIsNone(iterate._tombstone({**q("minimal", .1), "answers": {}},
                                             False, "unknown")["stakes"])

    def test_unresolved_key_questions_threshold_and_non_top_exclusion(self):
        self._patch([[q("highest", .8), q("highest", .7), q("at threshold", .4),
                      q("low", .2)]])
        out = iterate.iterate(
            "p", {"k": 4, "max_rounds": 1, "floor": .1, "key_gap_threshold": .4},
            answerer=notfound_answerer, responder=mock_responder)
        self.assertEqual(out["unresolved_key_questions"], [
            {"question": "highest", "value": .8, "stakes": None,
             "gap_reason": "not discoverable"},
            {"question": "at threshold", "value": .4, "stakes": None,
             "gap_reason": "not discoverable"},
        ])

    def test_unresolved_key_questions_always_includes_highest_below_threshold(self):
        self._patch([[q("highest", .3), q("lower", .2)]])
        out = iterate.iterate(
            "p", {"k": 2, "max_rounds": 1, "floor": .1, "key_gap_threshold": .4},
            answerer=notfound_answerer, responder=mock_responder)
        self.assertEqual([gap["question"] for gap in out["unresolved_key_questions"]],
                         ["highest"])

    def test_no_notfound_tombstones_has_no_unresolved_key_questions(self):
        self._patch([[q("answered", .9)]])
        out = iterate.iterate("p", {"k": 1, "max_rounds": 1, "floor": .1},
                              answerer=found_answerer, responder=mock_responder)
        self.assertEqual(out["unresolved_key_questions"], [])

    def test_stakes_aware_responder_receives_tombstones_and_key_gaps(self):
        self._patch([[q("answered", .9), q("key gap", .8)]])
        captured = {}

        def mixed_answerer(qq, problem, evidence, cfg):
            if qq["question"] == "answered":
                return True, "known"
            return False, "not discoverable"

        def responder(problem, evidence, cfg):
            captured.update(cfg)
            return "response"

        out = iterate.iterate(
            "p", {"k": 2, "max_rounds": 1, "floor": .1,
                  "key_gap_threshold": .4, "stakes_aware_respond": True},
            answerer=mixed_answerer, responder=responder)
        self.assertEqual(captured["tombstones"], out["tombstones"])
        self.assertEqual(captured["unresolved_key_questions"], [
            {"question": "key gap", "value": .8, "stakes": None,
             "gap_reason": "not discoverable"},
        ])

    def test_default_responder_cfg_does_not_gain_bucketing_keys(self):
        self._patch([[q("gap", .8)]])
        captured = {}

        def responder(problem, evidence, cfg):
            captured.update(cfg)
            return "response"

        iterate.iterate(
            "p", {"k": 1, "max_rounds": 1, "floor": .1, "key_gap_threshold": .4},
            answerer=notfound_answerer, responder=responder)
        self.assertNotIn("tombstones", captured)
        self.assertNotIn("unresolved_key_questions", captured)

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


class Durability(unittest.TestCase):
    """The tombstone journal: resume, stale-problem guard, tolerant parse, fp dedup."""

    def setUp(self):
        self._orig = iterate.rank
        self.calls = []
        self.tmp = tempfile.mkdtemp(prefix="inv-journal-")

    def tearDown(self):
        iterate.rank = self._orig
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _patch(self, sequence):
        seq = list(sequence)

        def fake(problem, evidence, rank_cfg):
            self.calls.append(list(evidence))
            return seq[min(len(self.calls) - 1, len(seq) - 1)]
        iterate.rank = fake

    def _journal_lines(self):
        with open(os.path.join(self.tmp, iterate.JOURNAL), encoding="utf-8") as fh:
            return [ln for ln in fh.read().splitlines() if ln.strip()]

    def test_resume_skips_answered_and_converges(self):
        self._patch([[q("a", .9), q("b", .8)]])
        out1 = iterate.iterate("p", {"k": 2, "max_rounds": 1, "floor": 0.12, "run_dir": self.tmp},
                               answerer=found_answerer, responder=mock_responder)
        self.assertEqual((out1["n_answered"], out1["n_resumed"]), (2, 0))
        self.assertEqual(len(self._journal_lines()), 3)  # header + 2 tombstones

        out2 = iterate.iterate("p", {"k": 2, "max_rounds": 3, "floor": 0.12, "run_dir": self.tmp},
                               answerer=found_answerer, responder=mock_responder)
        self.assertEqual(out2["n_resumed"], 2)
        self.assertEqual(out2["n_answered"], 2)          # resumed tombstones count
        self.assertEqual(out2["rounds"], 1)              # both already answered → converge round 1
        self.assertIn("converged", out2["stop_reason"])
        self.assertEqual(len(self._journal_lines()), 3)  # no duplicate journal lines
        # resumed evidence reached rank() on the very first call of run 2
        self.assertEqual(len(self.calls[-1]), 2)

    def test_stale_problem_rotates_and_clears_artifacts(self):
        self._patch([[q("a", .9)]])
        iterate.iterate("OLD problem", {"k": 1, "max_rounds": 1, "floor": 0.12, "run_dir": self.tmp},
                        answerer=found_answerer, responder=mock_responder)
        leftover = os.path.join(self.tmp, "answer-deadbeef00000000.json")
        with open(leftover, "w", encoding="utf-8") as fh:
            fh.write('{"answer": "stale"}')
        out = iterate.iterate("NEW problem", {"k": 1, "max_rounds": 1, "floor": 0.12,
                                              "run_dir": self.tmp},
                              answerer=found_answerer, responder=mock_responder)
        self.assertEqual(out["n_resumed"], 0)
        self.assertTrue(os.path.exists(os.path.join(self.tmp, iterate.JOURNAL + ".stale")))
        self.assertFalse(os.path.exists(leftover))
        header = json.loads(self._journal_lines()[0])
        self.assertEqual(header["problem_fp"], answerer.fp("NEW problem"))

    def test_tolerant_parse_skips_torn_tail(self):
        self._patch([[q("a", .9)]])
        iterate.iterate("p", {"k": 1, "max_rounds": 1, "floor": 0.12, "run_dir": self.tmp},
                        answerer=found_answerer, responder=mock_responder)
        with open(os.path.join(self.tmp, iterate.JOURNAL), "a", encoding="utf-8") as fh:
            fh.write('{"question": "torn line, no clos')  # crash mid-append
        out = iterate.iterate("p", {"k": 1, "max_rounds": 1, "floor": 0.12, "run_dir": self.tmp},
                              answerer=found_answerer, responder=mock_responder)
        self.assertEqual(out["n_resumed"], 1)  # the good tombstone survived; torn line skipped

    def test_resume_old_tombstone_without_rank_metadata(self):
        old_gap = {
            "question": "legacy gap", "status": "NOT_FOUND", "fact": "not recorded",
            "evidence": "legacy gap -> (known gap: not recorded)", "via": "research",
        }
        with open(os.path.join(self.tmp, iterate.JOURNAL), "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"kind": "header", "problem_fp": answerer.fp("p")}) + "\n")
            fh.write(json.dumps(old_gap) + "\n")
        self._patch([[]])
        out = iterate.iterate(
            "p", {"max_rounds": 1, "run_dir": self.tmp, "key_gap_threshold": .4},
            answerer=found_answerer, responder=mock_responder)
        self.assertEqual(out["n_resumed"], 1)
        self.assertEqual(out["unresolved_key_questions"], [{
            "question": "legacy gap", "value": None, "stakes": None,
            "gap_reason": "not recorded",
        }])

    def test_fp_dedup_catches_reworded_question(self):
        self._patch([[q("What's the Stack?", .9)], [q("what's   the STACK!!", .9)]])
        out = iterate.iterate("p", {"k": 1, "max_rounds": 2, "floor": 0.12, "run_dir": self.tmp},
                              answerer=found_answerer, responder=mock_responder)
        self.assertEqual(out["n_answered"], 1)  # punct/case variant is the same question
        self.assertIn("converged", out["stop_reason"])

    def test_no_run_dir_is_in_memory_and_unjournaled(self):
        self._patch([[q("a", .9)]])
        out = iterate.iterate("p", {"k": 1, "max_rounds": 1, "floor": 0.12},
                              answerer=found_answerer, responder=mock_responder)
        self.assertEqual((out["n_resumed"], out["run_dir"]), (0, None))
        self.assertFalse(os.path.exists(os.path.join(self.tmp, iterate.JOURNAL)))


class DerivedConsumption(unittest.TestCase):
    def setUp(self):
        self._orig = iterate.rank
        self.calls = []

    def tearDown(self):
        iterate.rank = self._orig

    def _patch(self, sequence):
        seq = list(sequence)

        def fake(problem, evidence, rank_cfg):
            self.calls.append({"evidence": list(evidence), "rank_cfg": dict(rank_cfg)})
            return seq[min(len(self.calls) - 1, len(seq) - 1)]
        iterate.rank = fake

    @staticmethod
    def _derived(text, value, answer):
        return {**q(text, value), "recommendation": "DERIVED", "derived_answer": answer}

    def test_derived_fact_is_consumed_once_without_research(self):
        derived = self._derived("What port?", .9, "5432")
        self._patch([[derived, q("What host?", .8)], [derived]])
        researched = []

        def answer(qq, problem, evidence, cfg):
            researched.append(qq["question"])
            self.assertIn("What port? -> 5432 (derived during analysis)", evidence)
            return True, "db.internal"

        out = iterate.iterate("p", {"triage": True, "k": 1, "max_rounds": 2, "floor": .12},
                              answerer=answer, responder=mock_responder)
        derived_tombs = [t for t in out["tombstones"] if t["via"] == "derived"]
        self.assertEqual(len(derived_tombs), 1)
        self.assertEqual(derived_tombs[0], {
            "question": "What port?", "status": "ANSWERED", "fact": "5432",
            "evidence": "What port? -> 5432 (derived during analysis)", "via": "derived",
            "value": .9, "stakes": None, "recommendation": "DERIVED",
        })
        self.assertEqual(researched, ["What host?"])
        self.assertEqual(out["n_derived"], 1)
        self.assertIn("converged", out["stop_reason"])

    def test_derived_journal_round_trip_preserves_via(self):
        tmp = tempfile.mkdtemp(prefix="inv-derived-")
        try:
            derived = self._derived("What port?", .9, "5432")
            self._patch([[derived]])
            iterate.iterate("p", {"triage": True, "max_rounds": 1, "run_dir": tmp},
                            answerer=found_answerer, responder=mock_responder)
            out = iterate.iterate("p", {"triage": True, "max_rounds": 1, "run_dir": tmp},
                                  answerer=found_answerer, responder=mock_responder)
            self.assertEqual(out["n_resumed"], 1)
            self.assertEqual(out["n_derived"], 1)
            self.assertEqual(out["tombstones"][0]["via"], "derived")
            self.assertNotIn("rationale", out["tombstones"][0])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_triage_controls_auto_derive_rank_flag(self):
        self._patch([[]])
        iterate.iterate("p", {"max_rounds": 1},
                        answerer=found_answerer, responder=mock_responder)
        self.assertNotIn("auto_derive", self.calls[-1]["rank_cfg"])
        iterate.iterate("p", {"triage": True, "max_rounds": 1},
                        answerer=found_answerer, responder=mock_responder)
        self.assertEqual(self.calls[-1]["rank_cfg"]["auto_derive"], "on")


class TriageRouting(unittest.TestCase):
    def setUp(self):
        self._orig = iterate.rank
        self.calls = []

    def tearDown(self):
        iterate.rank = self._orig

    def _patch(self, sequence):
        seq = list(sequence)

        def fake(problem, evidence, rank_cfg):
            self.calls.append(list(evidence))
            return seq[min(len(self.calls) - 1, len(seq) - 1)]
        iterate.rank = fake

    def test_judgment_routes_to_judge_not_answerer(self):
        self._patch([[q("Choose a color?", .9)]])
        researched, judged = [], []

        def research(qq, problem, evidence, cfg):
            researched.append(qq["question"])
            return True, "researched"

        def judge(question, problem, evidence, cfg):
            judged.append(question)
            return True, "blue", "standard default"

        out = iterate.iterate(
            "p", {"triage": True, "k": 1, "max_rounds": 1},
            answerer=research, responder=mock_responder,
            triager=lambda *args: {answerer.fp("Choose a color?"): "JUDGMENT"},
            judge=judge)
        self.assertEqual(judged, ["Choose a color?"])
        self.assertEqual(researched, [])
        self.assertEqual(out["n_assumed"], 1)
        self.assertEqual(out["tombstones"][0]["evidence"],
                         "Choose a color? -> blue (assumed: standard default)")

    def test_findable_routes_to_answerer_not_judge(self):
        self._patch([[q("What port?", .9)]])
        researched, judged = [], []

        def research(qq, problem, evidence, cfg):
            researched.append(qq["question"])
            return True, "5432"

        def judge(question, problem, evidence, cfg):
            judged.append(question)
            return True, "unused", "unused"

        iterate.iterate(
            "p", {"triage": True, "k": 1, "max_rounds": 1},
            answerer=research, responder=mock_responder,
            triager=lambda *args: {answerer.fp("What port?"): "FINDABLE"},
            judge=judge)
        self.assertEqual(researched, ["What port?"])
        self.assertEqual(judged, [])

    def test_triage_absent_never_calls_triager(self):
        self._patch([[q("a", .9), q("b", .8)]])
        researched = []

        def forbidden(*args):
            raise AssertionError("triager called while triage is disabled")

        def research(qq, problem, evidence, cfg):
            researched.append(qq["question"])
            return True, f"answer:{qq['question']}"

        out = iterate.iterate(
            "p", {"k": 2, "max_rounds": 1},
            answerer=research, responder=mock_responder,
            triager=forbidden, judge=forbidden)
        self.assertEqual(researched, ["a", "b"])
        self.assertEqual(out["n_answered"], 2)
        self.assertEqual(out["n_assumed"], 0)

    def test_empty_routes_fail_open_to_research(self):
        self._patch([[q("a", .9), q("b", .8)]])
        researched, judged = [], []

        def research(qq, problem, evidence, cfg):
            researched.append(qq["question"])
            return True, "found"

        def judge(question, problem, evidence, cfg):
            judged.append(question)
            return True, "unused", "unused"

        iterate.iterate(
            "p", {"triage": True, "k": 2, "max_rounds": 1},
            answerer=research, responder=mock_responder,
            triager=lambda *args: {}, judge=judge)
        self.assertEqual(researched, ["a", "b"])
        self.assertEqual(judged, [])

    def test_max_assumes_sends_overflow_to_research(self):
        questions = [q("choice 1", .9), q("choice 2", .8), q("choice 3", .7)]
        self._patch([questions])
        researched, judged, triaged = [], [], []

        def triage(problem, batch, evidence, cfg):
            triaged.append([qq["question"] for qq in batch])
            return {answerer.fp(qq["question"]): "JUDGMENT" for qq in batch}

        def research(qq, problem, evidence, cfg):
            researched.append(qq["question"])
            return True, "researched"

        def judge(question, problem, evidence, cfg):
            judged.append(question)
            return True, "default", "conservative"

        out = iterate.iterate(
            "p", {"triage": True, "k": 3, "max_rounds": 1, "max_assumes": 2},
            answerer=research, responder=mock_responder, triager=triage, judge=judge)
        self.assertEqual(triaged, [["choice 1", "choice 2", "choice 3"]])
        self.assertEqual(judged, ["choice 1", "choice 2"])
        self.assertEqual(researched, ["choice 3"])
        self.assertEqual(out["n_assumed"], 2)

    def test_judge_failure_falls_back_without_assumed_tombstone(self):
        self._patch([[q("choice", .9)]])
        researched = []

        def research(qq, problem, evidence, cfg):
            researched.append(qq["question"])
            return True, "researched"

        out = iterate.iterate(
            "p", {"triage": True, "k": 1, "max_rounds": 1},
            answerer=research, responder=mock_responder,
            triager=lambda *args: {answerer.fp("choice"): "JUDGMENT"},
            judge=lambda *args: (False, "", "reason"))
        self.assertEqual(researched, ["choice"])
        self.assertEqual(out["tombstones"][0].get("via", "research"), "research")
        self.assertEqual(out["n_assumed"], 0)

    def test_resumed_assumption_counts_toward_cap(self):
        tmp = tempfile.mkdtemp(prefix="inv-assumed-")
        try:
            iterate._append_journal(
                tmp, {"schema": 1, "kind": "header", "problem_fp": answerer.fp("p")})
            iterate._append_journal(tmp, {
                "question": "prior choice", "status": "ANSWERED", "fact": "default",
                "evidence": "prior choice -> default (assumed: conservative)",
                "via": "assumed", "rationale": "conservative",
            })
            fresh = [q("fresh choice 1", .9), q("fresh choice 2", .8)]
            self._patch([fresh])
            researched, judged = [], []

            def triage(problem, batch, evidence, cfg):
                return {answerer.fp(qq["question"]): "JUDGMENT" for qq in batch}

            def research(qq, problem, evidence, cfg):
                researched.append(qq["question"])
                return True, "researched"

            def judge(question, problem, evidence, cfg):
                judged.append(question)
                return True, "new default", "conservative"

            out = iterate.iterate(
                "p", {"triage": True, "k": 2, "max_rounds": 1,
                      "max_assumes": 1, "run_dir": tmp},
                answerer=research, responder=mock_responder, triager=triage, judge=judge)
            self.assertEqual(judged, [])
            self.assertEqual(researched, ["fresh choice 1", "fresh choice 2"])
            self.assertEqual(out["n_resumed"], 1)
            self.assertEqual(out["n_assumed"], 1)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class AssumptionLedger(unittest.TestCase):
    def setUp(self):
        self._orig = iterate.rank
        self.calls = []

    def tearDown(self):
        iterate.rank = self._orig

    def _patch(self, sequence):
        seq = list(sequence)

        def fake(problem, evidence, rank_cfg):
            self.calls.append(list(evidence))
            return seq[min(len(self.calls) - 1, len(seq) - 1)]
        iterate.rank = fake

    def test_return_includes_assumed_decision_and_rationale(self):
        self._patch([[q("Choose a database?", .9)]])
        out = iterate.iterate(
            "p", {"triage": True, "k": 1, "max_rounds": 1},
            answerer=found_answerer, responder=mock_responder,
            triager=lambda *args: {answerer.fp("Choose a database?"): "JUDGMENT"},
            judge=lambda *args: (True, "use SQLite", "simplest reversible default"))
        self.assertEqual(out["n_assumed"], 1)
        self.assertEqual(out["assumptions"], [{
            "question": "Choose a database?",
            "decision": "use SQLite",
            "rationale": "simplest reversible default",
        }])

    def test_resumed_assumption_is_in_returned_ledger(self):
        tmp = tempfile.mkdtemp(prefix="inv-assumption-ledger-")
        try:
            iterate._append_journal(
                tmp, {"schema": 1, "kind": "header", "problem_fp": answerer.fp("p")})
            iterate._append_journal(tmp, {
                "question": "prior choice", "status": "ANSWERED", "fact": "default",
                "evidence": "prior choice -> default (assumed: conservative)",
                "via": "assumed", "rationale": "conservative",
            })
            self._patch([[]])
            out = iterate.iterate(
                "p", {"triage": True, "max_rounds": 1, "run_dir": tmp},
                answerer=found_answerer, responder=mock_responder)
            self.assertEqual(out["n_resumed"], 1)
            self.assertEqual(out["n_assumed"], 1)
            self.assertEqual(out["assumptions"], [{
                "question": "prior choice", "decision": "default",
                "rationale": "conservative",
            }])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    @unittest.skipUnless(getattr(answerer, "_HAVE_ASK", False),
                         "model_utils (ask skill) not importable")
    def test_respond_requests_ledger_for_assumed_and_derived_markers(self):
        cfg = dict(iterate.DEFAULTS)
        for evidence in (["choice -> blue (assumed: standard default)"],
                         ["port -> 5432 (derived during analysis)"]):
            with self.subTest(evidence=evidence):
                ds = mock.MagicMock(return_value={"content": "response", "error": None})
                with mock.patch.object(answerer, "dispatch_single", ds), \
                     mock.patch.object(answerer, "resolve_alias", lambda m: m):
                    answerer.respond("task", evidence, cfg)
                prompt = ds.call_args[0][1]
                self.assertIn("Assumptions", prompt)
                self.assertIn("Known gaps", prompt)

    @unittest.skipUnless(getattr(answerer, "_HAVE_ASK", False),
                         "model_utils (ask skill) not importable")
    def test_respond_without_markers_preserves_prompt_bytes(self):
        cfg = dict(iterate.DEFAULTS)
        problem = "Build the thing"
        evidence = ["stack -> Python", "deployment -> (known gap: unspecified)"]
        facts = "\n".join(f"- {e}" for e in evidence) or "(none)"
        expected = (f"TASK: {problem}\n\nEstablished facts and known gaps:\n{facts}\n\n"
                    f"Produce the best possible response to the task using what's established. "
                    f"State any assumptions you make for unresolved gaps. Be direct and useful.")
        ds = mock.MagicMock(return_value={"content": "response", "error": None})
        with mock.patch.object(answerer, "dispatch_single", ds), \
             mock.patch.object(answerer, "resolve_alias", lambda m: m):
            answerer.respond(problem, evidence, cfg)
        self.assertEqual(ds.call_args[0][1], expected)


class StakesAwareRespond(unittest.TestCase):
    def _respond_prompt(self, problem, evidence, cfg):
        ds = mock.MagicMock(return_value={"content": "response", "error": None})
        with mock.patch.object(answerer, "dispatch_single", ds), \
             mock.patch.object(answerer, "resolve_alias", lambda m: m):
            answerer.respond(problem, evidence, cfg)
        ds.assert_called_once()
        return ds.call_args[0][1]

    def test_explicit_off_preserves_prompt_bytes(self):
        cfg = {**iterate.DEFAULTS, "stakes_aware_respond": False}
        problem = "Build the thing"
        evidence = ["stack -> Python", "deployment -> (known gap: unspecified)"]
        facts = "\n".join(f"- {e}" for e in evidence) or "(none)"
        expected = (f"TASK: {problem}\n\nEstablished facts and known gaps:\n{facts}\n\n"
                    f"Produce the best possible response to the task using what's established. "
                    f"State any assumptions you make for unresolved gaps. Be direct and useful.")
        self.assertEqual(self._respond_prompt(problem, evidence, cfg), expected)

    def test_on_buckets_established_minor_and_key_gaps(self):
        answered = iterate._tombstone(q("Which stack?", .9), True, "Python")
        minor = iterate._tombstone(q("Which color?", .2), False, "not discoverable")
        key = iterate._tombstone(q("Which deployment target?", .8), False, "not discoverable")
        cfg = {
            **iterate.DEFAULTS,
            "stakes_aware_respond": True,
            "tombstones": [answered, minor, key],
            "unresolved_key_questions": [{
                "question": "Which deployment target?", "value": .8,
                "stakes": None, "gap_reason": "not discoverable",
            }],
        }
        prompt = self._respond_prompt(
            "Build the thing", ["Caller seed fact", answered["evidence"],
                                minor["evidence"], key["evidence"]], cfg)
        self.assertIn("Established facts", prompt)
        self.assertIn("Minor open gaps", prompt)
        self.assertIn("⚠️ Unresolved key questions", prompt)
        self.assertIn("Which deployment target?", prompt)
        self.assertIn("Material risks — assumptions to confirm", prompt)
        self.assertIn("Caller seed fact", prompt)

    def test_on_without_key_gaps_omits_risk_framing(self):
        answered = iterate._tombstone(q("Which stack?", .9), True, "Python")
        minor = iterate._tombstone(q("Which color?", .2), False, "not discoverable")
        cfg = {
            **iterate.DEFAULTS,
            "stakes_aware_respond": True,
            "tombstones": [answered, minor],
            "unresolved_key_questions": [],
        }
        prompt = self._respond_prompt(
            "Build the thing", [answered["evidence"], minor["evidence"]], cfg)
        self.assertIn("Established facts", prompt)
        self.assertIn("Minor open gaps", prompt)
        self.assertNotIn("⚠️ Unresolved key questions", prompt)
        self.assertNotIn("Material risks — assumptions to confirm", prompt)


class RefinedPrompt(unittest.TestCase):
    def setUp(self):
        self._orig = iterate.rank
        self.calls = []

    def tearDown(self):
        iterate.rank = self._orig

    def _patch(self, sequence):
        seq = list(sequence)

        def fake(problem, evidence, rank_cfg):
            self.calls.append(list(evidence))
            return seq[min(len(self.calls) - 1, len(seq) - 1)]
        iterate.rank = fake

    def test_prompt_mode_calls_only_refiner(self):
        self._patch([[]])
        refined, responded = [], []

        def refiner(problem, evidence, cfg):
            refined.append(problem)
            return f"REFINED: {problem}"

        def responder(problem, evidence, cfg):
            responded.append(problem)
            return "response"

        out = iterate.iterate("original", {"output": "prompt", "max_rounds": 1},
                              answerer=found_answerer, responder=responder, refiner=refiner)
        self.assertEqual(refined, ["original"])
        self.assertEqual(responded, [])
        self.assertEqual(out["refined_prompt"], "REFINED: original")
        self.assertIsNone(out["final"])

    def test_both_mode_responds_from_refined_prompt(self):
        self._patch([[]])
        order, responder_problems = [], []

        def refiner(problem, evidence, cfg):
            order.append("refiner")
            return "REFINED TASK"

        def responder(problem, evidence, cfg):
            order.append("responder")
            responder_problems.append(problem)
            return "final response"

        out = iterate.iterate("original", {"output": "both", "max_rounds": 1},
                              answerer=found_answerer, responder=responder, refiner=refiner)
        self.assertEqual(order, ["refiner", "responder"])
        self.assertEqual(responder_problems, ["REFINED TASK"])
        self.assertEqual(out["final"], "final response")

    def test_absent_output_preserves_response_mode(self):
        self._patch([[]])
        refined, responder_problems = [], []

        def refiner(problem, evidence, cfg):
            refined.append(problem)
            return "unexpected"

        def responder(problem, evidence, cfg):
            responder_problems.append(problem)
            return "response"

        out = iterate.iterate("ORIGINAL", {"max_rounds": 1}, answerer=found_answerer,
                              responder=responder, refiner=refiner)
        self.assertEqual(responder_problems, ["ORIGINAL"])
        self.assertEqual(refined, [])
        self.assertIsNone(out["refined_prompt"])

    def test_refined_prompt_is_written_to_run_dir(self):
        tmp = tempfile.mkdtemp(prefix="inv-refined-")
        try:
            self._patch([[]])
            out = iterate.iterate(
                "p", {"output": "prompt", "max_rounds": 1, "run_dir": tmp},
                answerer=found_answerer, responder=mock_responder,
                refiner=lambda problem, evidence, cfg: "REFINED CONTENT")
            path = os.path.join(tmp, iterate.REFINED_PROMPT_FILE)
            self.assertEqual(out["refined_prompt"], "REFINED CONTENT")
            self.assertTrue(os.path.exists(path))
            with open(path, encoding="utf-8") as fh:
                self.assertEqual(fh.read(), "REFINED CONTENT")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_stale_problem_rotation_clears_refined_and_answer_artifacts(self):
        tmp = tempfile.mkdtemp(prefix="inv-refined-stale-")
        try:
            self._patch([[q("a", .9)]])
            iterate.iterate(
                "OLD problem", {"output": "prompt", "k": 1, "max_rounds": 1,
                                "floor": .12, "run_dir": tmp},
                answerer=found_answerer, responder=mock_responder,
                refiner=lambda *args: "OLD REFINED")
            refined_path = os.path.join(tmp, iterate.REFINED_PROMPT_FILE)
            answer_path = os.path.join(tmp, "answer-deadbeef00000000.json")
            with open(answer_path, "w", encoding="utf-8") as fh:
                fh.write('{"answer": "stale"}')
            self.assertTrue(os.path.exists(refined_path))
            iterate.iterate(
                "NEW problem", {"output": "response", "k": 1, "max_rounds": 1,
                                "floor": .12, "run_dir": tmp},
                answerer=found_answerer, responder=mock_responder)
            self.assertFalse(os.path.exists(refined_path))
            self.assertFalse(os.path.exists(answer_path))
            self.assertTrue(os.path.exists(os.path.join(tmp, iterate.JOURNAL + ".stale")))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class EnvConfig(unittest.TestCase):
    def _captured_cfg(self, argv):
        with mock.patch("iterate.iterate", return_value={"tombstones": []}) as run, \
             contextlib.redirect_stdout(io.StringIO()):
            iterate.main(["--problem", "p", "--json", *argv])
        return run.call_args.args[1]

    def test_max_rounds_from_env(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("INVESTIGATOR_MAX_ROUNDS", None)
            os.environ.pop("INVESTIGATOR_MAX_ASSUMES", None)
            os.environ["INVESTIGATOR_MAX_ROUNDS"] = "5"
            cfg = self._captured_cfg([])
        self.assertEqual(cfg["max_rounds"], 5)

    def test_max_rounds_cli_wins_over_env(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("INVESTIGATOR_MAX_ROUNDS", None)
            os.environ.pop("INVESTIGATOR_MAX_ASSUMES", None)
            os.environ["INVESTIGATOR_MAX_ROUNDS"] = "2"
            cfg = self._captured_cfg(["--max-rounds", "8"])
        self.assertEqual(cfg["max_rounds"], 8)

    def test_max_rounds_defaults_when_env_unset(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("INVESTIGATOR_MAX_ROUNDS", None)
            os.environ.pop("INVESTIGATOR_MAX_ASSUMES", None)
            cfg = self._captured_cfg([])
        self.assertEqual(cfg["max_rounds"], 3)

    def test_invalid_max_rounds_env_exits_2_and_names_var(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("INVESTIGATOR_MAX_ROUNDS", None)
            os.environ.pop("INVESTIGATOR_MAX_ASSUMES", None)
            os.environ["INVESTIGATOR_MAX_ROUNDS"] = "abc"
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
                iterate.main(["--problem", "p"])
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("INVESTIGATOR_MAX_ROUNDS", stderr.getvalue())

    def test_max_assumes_from_env(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("INVESTIGATOR_MAX_ROUNDS", None)
            os.environ.pop("INVESTIGATOR_MAX_ASSUMES", None)
            os.environ["INVESTIGATOR_MAX_ASSUMES"] = "9"
            cfg = self._captured_cfg([])
        self.assertEqual(cfg["max_assumes"], 9)

    def test_max_assumes_cli_wins_over_env(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("INVESTIGATOR_MAX_ROUNDS", None)
            os.environ.pop("INVESTIGATOR_MAX_ASSUMES", None)
            os.environ["INVESTIGATOR_MAX_ASSUMES"] = "2"
            cfg = self._captured_cfg(["--max-assumes", "8"])
        self.assertEqual(cfg["max_assumes"], 8)

    def test_max_assumes_defaults_when_env_unset(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("INVESTIGATOR_MAX_ROUNDS", None)
            os.environ.pop("INVESTIGATOR_MAX_ASSUMES", None)
            cfg = self._captured_cfg([])
        self.assertEqual(cfg["max_assumes"], 6)

    def test_invalid_max_assumes_env_exits_2_and_names_var(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("INVESTIGATOR_MAX_ROUNDS", None)
            os.environ.pop("INVESTIGATOR_MAX_ASSUMES", None)
            os.environ["INVESTIGATOR_MAX_ASSUMES"] = "abc"
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
                iterate.main(["--problem", "p"])
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("INVESTIGATOR_MAX_ASSUMES", stderr.getvalue())

    def test_key_gap_threshold_from_env(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("INVESTIGATOR_KEY_GAP_THRESHOLD", None)
            os.environ["INVESTIGATOR_KEY_GAP_THRESHOLD"] = "0.65"
            cfg = self._captured_cfg([])
        self.assertEqual(cfg["key_gap_threshold"], .65)

    def test_key_gap_threshold_cli_wins_over_env(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ["INVESTIGATOR_KEY_GAP_THRESHOLD"] = "0.65"
            cfg = self._captured_cfg(["--key-gap-threshold", "0.55"])
        self.assertEqual(cfg["key_gap_threshold"], .55)

    def test_key_gap_threshold_defaults_when_env_unset(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("INVESTIGATOR_KEY_GAP_THRESHOLD", None)
            cfg = self._captured_cfg([])
        self.assertEqual(cfg["key_gap_threshold"], .40)

    def test_invalid_key_gap_threshold_env_exits_2_and_names_var(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ["INVESTIGATOR_KEY_GAP_THRESHOLD"] = "abc"
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
                iterate.main(["--problem", "p"])
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("INVESTIGATOR_KEY_GAP_THRESHOLD", stderr.getvalue())

    def test_stakes_aware_respond_defaults_off(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("INVESTIGATOR_STAKES_AWARE_RESPOND", None)
            cfg = self._captured_cfg([])
        self.assertIs(cfg["stakes_aware_respond"], False)

    def test_stakes_aware_respond_from_env(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ["INVESTIGATOR_STAKES_AWARE_RESPOND"] = "on"
            cfg = self._captured_cfg([])
        self.assertIs(cfg["stakes_aware_respond"], True)

    def test_stakes_aware_respond_cli_wins_over_env(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ["INVESTIGATOR_STAKES_AWARE_RESPOND"] = "on"
            cfg = self._captured_cfg(["--stakes-aware-respond", "off"])
        self.assertIs(cfg["stakes_aware_respond"], False)


@unittest.skipUnless(getattr(answerer, "_HAVE_ASK", False), "model_utils (ask skill) not importable")
class JudgmentCall(unittest.TestCase):
    def _cfg(self, **over):
        cfg = dict(iterate.DEFAULTS)
        cfg.update(over)
        return cfg

    def _call(self, ds, question="choice", problem="task"):
        with mock.patch.object(answerer, "dispatch_single", ds), \
             mock.patch.object(answerer, "resolve_alias", lambda m: m):
            return answerer.judgment_call(question, problem, [], self._cfg())

    def test_valid_json_returns_decision_and_rationale(self):
        ds = mock.MagicMock(return_value={
            "content": '{"decision": "use SQLite", "rationale": "simplest default"}',
            "error": None,
        })
        ok, decision, rationale = self._call(ds)
        self.assertTrue(ok)
        self.assertEqual(decision, "use SQLite")
        self.assertEqual(rationale, "simplest default")

    def test_cannot_decide_returns_false(self):
        ds = mock.MagicMock(return_value={
            "content": "CANNOT_DECIDE: the options are equivalent", "error": None})
        ok, decision, rationale = self._call(ds)
        self.assertFalse(ok)
        self.assertEqual(decision, "")
        self.assertIn("equivalent", rationale)

    def test_hedged_decision_returns_false(self):
        ds = mock.MagicMock(return_value={
            "content": ('{"decision": "does not specify a preference", '
                        '"rationale": "no requirement chooses one"}'),
            "error": None,
        })
        ok, decision, rationale = self._call(ds)
        self.assertFalse(ok)
        self.assertEqual(decision, "")
        self.assertEqual(rationale, "no requirement chooses one")

    def test_malformed_json_returns_false(self):
        ds = mock.MagicMock(return_value={"content": "{not valid json", "error": None})
        ok, decision, rationale = self._call(ds)
        self.assertFalse(ok)
        self.assertEqual(decision, "")
        self.assertTrue(rationale)


@unittest.skipUnless(getattr(answerer, "_HAVE_ASK", False), "model_utils (ask skill) not importable")
class GroundedAnswer(unittest.TestCase):
    """grounded_answer with a mocked dispatch_single — prompt assembly, directive, NOT_FOUND
    parse, and the answer-artifact capture path (mocks live in the answerer module now)."""

    def _cfg(self, **over):
        cfg = dict(iterate.DEFAULTS)
        cfg.update(over)
        return cfg

    def _call(self, ds, question, cfg, problem="task"):
        with mock.patch.object(answerer, "dispatch_single", ds), \
             mock.patch.object(answerer, "resolve_alias", lambda m: m):
            return answerer.grounded_answer(question, problem, [], cfg)

    def test_directive_prepended_and_normal_answer(self):
        cfg = iterate.apply_capability(self._cfg(), "read")  # read -> directive + file,web toolsets
        ds = mock.MagicMock(return_value={"content": "The stack is FastAPI + Postgres.", "error": None})
        found, text = self._call(ds, "What's the stack?", cfg, problem="Add auth")
        self.assertTrue(found)
        self.assertIn("FastAPI", text)
        # dispatch_single(model, PROMPT, "", toolsets, ...): directive prepended, toolsets downscoped
        self.assertIn("READ-ONLY", ds.call_args[0][1])
        self.assertEqual(ds.call_args[0][3], cfg["answer_toolsets"])
        self.assertNotIn("terminal", ds.call_args[0][3])

    def test_not_found_parsed_and_act_has_no_directive(self):
        cfg = self._cfg()  # act default -> empty directive, full toolsets
        ds = mock.MagicMock(return_value={"content": "NOT_FOUND: no credentials available", "error": None})
        found, text = self._call(ds, "Do you have creds?", cfg)
        self.assertFalse(found)
        self.assertIn("no credentials", text)
        self.assertNotIn("READ-ONLY", ds.call_args[0][1])     # act default -> no directive prepended
        self.assertEqual(ds.call_args[0][3], "file,web,terminal")

    def test_research_error_returns_not_found(self):
        ds = mock.MagicMock(return_value={"content": "", "error": "boom"})
        found, text = self._call(ds, "q", self._cfg())
        self.assertFalse(found)
        self.assertIn("research error", text)

    def test_dict_question_interpolates_text_not_repr(self):
        ds = mock.MagicMock(return_value={"content": "ok.", "error": None})
        self._call(ds, {"question": "What's the stack?", "value": 0.9}, self._cfg())
        prompt = ds.call_args[0][1]
        self.assertIn("What's the stack?", prompt)
        self.assertNotIn("{'question'", prompt)


@unittest.skipUnless(getattr(answerer, "_HAVE_ASK", False), "model_utils (ask skill) not importable")
class AnswerArtifacts(unittest.TestCase):
    """Artifact-beats-stdout: instruction gating by capability + read precedence."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="inv-artifact-")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _cfg(self, capability="act", **over):
        cfg = iterate.apply_capability(dict(iterate.DEFAULTS), capability)
        cfg["run_dir"] = self.tmp
        cfg.update(over)
        return cfg

    def _call(self, ds, cfg, question="q1"):
        with mock.patch.object(answerer, "dispatch_single", ds), \
             mock.patch.object(answerer, "resolve_alias", lambda m: m):
            return answerer.grounded_answer(question, "task", [], cfg)

    def test_artifact_beats_misclassified_stdout(self):
        apath = answerer.artifact_path(self.tmp, "q1")

        def ds(*a, **kw):  # agent wrote the artifact; stdout came back as a bogus error
            with open(apath, "w", encoding="utf-8") as fh:
                json.dump({"answer": "The port is 5432."}, fh)
            return {"content": "", "error": "API error: rate limit exceeded"}
        found, text = self._call(ds, self._cfg())
        self.assertTrue(found)
        self.assertEqual(text, "The port is 5432.")

    def test_artifact_not_found_still_judged_in_code(self):
        apath = answerer.artifact_path(self.tmp, "q1")

        def ds(*a, **kw):
            with open(apath, "w", encoding="utf-8") as fh:
                json.dump({"answer": "NOT_FOUND: no access to prod"}, fh)
            return {"content": "irrelevant", "error": None}
        found, text = self._call(ds, self._cfg())
        self.assertFalse(found)
        self.assertIn("no access", text)

    def test_missing_artifact_falls_back_to_stdout(self):
        ds = mock.MagicMock(return_value={"content": "From stdout.", "error": None})
        found, text = self._call(ds, self._cfg())
        self.assertTrue(found)
        self.assertEqual(text, "From stdout.")

    def test_act_prompt_carries_instruction_with_absolute_path(self):
        ds = mock.MagicMock(return_value={"content": "ok", "error": None})
        self._call(ds, self._cfg("act"))
        prompt = ds.call_args[0][1]
        self.assertIn(answerer.artifact_path(self.tmp, "q1"), prompt)
        self.assertIn('"answer"', prompt)

    def test_read_capability_omits_instruction_and_ignores_artifact(self):
        apath = answerer.artifact_path(self.tmp, "q1")
        with open(apath, "w", encoding="utf-8") as fh:
            json.dump({"answer": "should be ignored"}, fh)
        ds = mock.MagicMock(return_value={"content": "From stdout.", "error": None})
        found, text = self._call(ds, self._cfg("read"))
        self.assertNotIn("answer-", ds.call_args[0][1])  # no artifact instruction in prompt
        self.assertEqual(text, "From stdout.")           # artifact not read

    def test_malformed_artifact_falls_back(self):
        apath = answerer.artifact_path(self.tmp, "q1")
        with open(apath, "w", encoding="utf-8") as fh:
            fh.write("{not json")
        ds = mock.MagicMock(return_value={"content": "Salvaged.", "error": None})
        found, text = self._call(ds, self._cfg())
        self.assertTrue(found)
        self.assertEqual(text, "Salvaged.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
