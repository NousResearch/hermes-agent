#!/usr/bin/env python3
"""Unit tests for harvest.py — pure dict-driven folds, no container, no network. Run:
    python3 tests/test_harvest.py
"""

import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "scripts"))

import harvest  # noqa: E402


def tk(tid, method, crit="observably done"):
    return {"id": tid, "method": method, "description": f"do {method}",
            "success_criterion": crit, "depends_on": [], "status": "pending"}


def pl(*tasks):
    return {"schema": 1, "slug": "s", "cycle": 0, "disposition": "tasks",
            "rationale": "r", "question": None, "tasks": list(tasks)}


def res(tid, method, verdict, evidence="e", learnings=None):
    r = {"id": tid, "method": method, "verdict": verdict, "evidence": evidence}
    if learnings is not None:
        r["learnings"] = learnings
    return r


class HarvestTasks(unittest.TestCase):
    def test_needs_split_becomes_a_split_hint_fact(self):
        plan = pl(tk("t1", "alfa"))
        r = {**res("t1", "alfa", "needs_split", "two systems in one task"),
             "split": ["db half", "api half"]}
        out = harvest.harvest_tasks(plan, [r], cycle=1)
        self.assertEqual(len(out), 1)
        rec = out[0]
        self.assertEqual(rec["kind"], "fact")  # the method is NOT dead
        self.assertEqual(rec["fp"], harvest.fp("split alfa"))
        self.assertIn("SPLIT HINT", rec["text"])
        self.assertIn("db half; api half", rec["text"])

    def test_needs_split_without_split_falls_back_to_evidence(self):
        out = harvest.harvest_tasks(pl(tk("t1", "alfa")),
                                    [res("t1", "alfa", "needs_split", "too coarse")],
                                    cycle=0)
        self.assertIn("too coarse", out[0]["text"])

    def test_failed_becomes_dead_end(self):
        plan = pl(tk("t1", "source A (authoritative)", crit="rows match"))
        out = harvest.harvest_tasks(plan, [res("t1", "source A (authoritative)",
                                               "failed", "HTTP 503")], cycle=2)
        self.assertEqual(len(out), 1)
        rec = out[0]
        self.assertEqual(rec["kind"], "dead-end")
        self.assertEqual(rec["text"], "Tried source A (authoritative): failed — HTTP 503")
        self.assertEqual(rec["cycle"], 2)
        self.assertEqual(rec["source"], "harvest")
        self.assertEqual(rec["fp"], harvest.fp("source A (authoritative)"))
        self.assertEqual(rec["meta"], {"task": "t1", "criterion": "rows match"})

    def test_worked_becomes_fact(self):
        plan = pl(tk("t1", "alfa"))
        out = harvest.harvest_tasks(plan, [res("t1", "alfa", "worked", "saw it")], 0)
        self.assertEqual(out[0]["kind"], "fact")
        self.assertEqual(out[0]["text"], "Done alfa: saw it")
        self.assertEqual(out[0]["fp"], harvest.fp("ok alfa"))

    def test_skipped_emits_nothing(self):
        plan = pl(tk("t1", "alfa"), tk("t2", "beta"))
        out = harvest.harvest_tasks(plan, [res("t1", "alfa", "failed"),
                                           res("t2", "beta", "skipped",
                                               "dependency failed")], 0)
        self.assertEqual([r["kind"] for r in out], ["dead-end"])

    def test_fail_then_work_records_both_transitions(self):
        # distinct fp namespaces: a method that failed in c0 and worked in c2 keeps BOTH
        plan = pl(tk("t1", "alfa"))
        dead = harvest.harvest_tasks(plan, [res("t1", "alfa", "failed")], 0)[0]
        done = harvest.harvest_tasks(plan, [res("t1", "alfa", "worked")], 2)[0]
        self.assertNotEqual(dead["fp"], done["fp"])

    def test_fp_keys_on_method_not_reason(self):
        self.assertEqual(harvest.fp("Source A (authoritative)"),
                         harvest.fp("source a — authoritative!"))
        self.assertNotEqual(harvest.fp("source a"), harvest.fp("source b"))
        plan = pl(tk("t1", "source a"))
        out1 = harvest.harvest_tasks(plan, [res("t1", "source a", "failed",
                                                "HTTP 503; first try")], 0)
        out2 = harvest.harvest_tasks(plan, [res("t1", "source a", "failed",
                                                "HTTP 502; second attempt")], 1)
        self.assertEqual(out1[0]["fp"], out2[0]["fp"])  # reworded reason → same fp

    def test_criterion_defaults_empty_for_unknown_task(self):
        # a dependency-skip synthesis may reference a task id; unknown ids stay safe
        out = harvest.harvest_tasks(pl(), [res("tx", "ghost", "failed")], 0)
        self.assertEqual(out[0]["meta"]["criterion"], "")

    def test_learnings_fold_as_separate_facts_alongside_the_verdict_record(self):
        plan = pl(tk("t1", "alfa"))
        learnings = ["billing API is async via webhook, not synchronous",
                    "target has a spare read replica useful for verify"]
        out = harvest.harvest_tasks(plan, [res("t1", "alfa", "worked", "saw it",
                                               learnings=learnings)], 0)
        self.assertEqual(len(out), 3)  # 1 "Done" fact + 2 learning facts
        self.assertEqual(out[0]["kind"], "fact")
        self.assertEqual(out[0]["text"], "Done alfa: saw it")
        for rec, text in zip(out[1:], learnings):
            self.assertEqual(rec["kind"], "fact")
            self.assertEqual(rec["text"], text)
            self.assertEqual(rec["fp"], harvest.fp(text))
            self.assertEqual(rec["meta"]["learning_from"], "t1")

    def test_learnings_fold_even_on_a_failed_task(self):
        plan = pl(tk("t1", "alfa"))
        out = harvest.harvest_tasks(
            plan, [res("t1", "alfa", "failed", "HTTP 503",
                      learnings=["source system requires an API key rotated hourly"])],
            0)
        kinds = [r["kind"] for r in out]
        self.assertEqual(kinds, ["dead-end", "fact"])  # failure AND its learning, both
        self.assertIn("API key rotated hourly", out[1]["text"])

    def test_no_learnings_key_is_a_no_op(self):
        plan = pl(tk("t1", "alfa"))
        out = harvest.harvest_tasks(plan, [res("t1", "alfa", "worked")], 0)  # no learnings=
        self.assertEqual(len(out), 1)

    def test_duplicate_learning_text_across_cycles_dedups_by_fp(self):
        plan = pl(tk("t1", "alfa"))
        same_learning = "the target enforces OAuth2, not basic auth"
        out1 = harvest.harvest_tasks(
            plan, [res("t1", "alfa", "worked", learnings=[same_learning])], 0)
        out2 = harvest.harvest_tasks(
            plan, [res("t1", "alfa", "worked", learnings=[same_learning])], 1)
        self.assertEqual(out1[1]["fp"], out2[1]["fp"])  # same text -> same fp -> dedupes
                                                        # once folded through fold_records


if __name__ == "__main__":
    unittest.main(verbosity=2)
