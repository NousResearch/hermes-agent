#!/usr/bin/env python3
"""Unit tests for harvest.py — fixture-driven, no container, no network.

Fixtures are copies of real resilient-planner artifacts (demo-exhaustion, test-guard-halt,
demo-backtrack); the guard-halt journal is split across journal.tick0.jsonl + journal.jsonl
to pin drive.py's archive-merge order. Run:
    python3 tests/test_harvest.py
"""

import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "scripts"))

import harvest  # noqa: E402

FIX = os.path.join(_HERE, "fixtures")


def read(name):
    with open(os.path.join(FIX, name), encoding="utf-8") as fh:
        return fh.read()


class PlanTreeParsing(unittest.TestCase):
    def test_state_detection(self):
        self.assertEqual(harvest.parse_plan_tree(read("exhaustion/plan-tree.md"))["state"],
                         "EXHAUSTION-STOP")
        self.assertEqual(harvest.parse_plan_tree(read("guard-halt/plan-tree.md"))["state"],
                         "GUARD-HALT")
        self.assertEqual(harvest.parse_plan_tree(read("success/plan-tree.md"))["state"],
                         "SUCCESS")
        self.assertEqual(harvest.parse_state("# Plan-Tree: x   STATE: active"), "active")
        self.assertIsNone(harvest.parse_state("no header here"))

    def test_dead_nodes_carry_method_without_id(self):
        p = harvest.parse_plan_tree(read("guard-halt/plan-tree.md"))
        self.assertEqual(len(p["dead"]), 3)
        self.assertEqual(p["dead"][0]["id"], "S1")
        self.assertEqual(p["dead"][0]["method"], "source A (authoritative, freshest)")
        self.assertIn("HTTP 503", p["dead"][0]["reason"])
        self.assertNotIn("S1", p["dead"][0]["method"])

    def test_done_nodes_parsed(self):
        p = harvest.parse_plan_tree(read("success/plan-tree.md"))
        self.assertEqual([d["id"] for d in p["dead"]], ["S1", "S1b"])
        self.assertEqual([d["id"] for d in p["done"]], ["S2", "S3"])
        self.assertIn("verified valid JSON", p["done"][1]["receipt"])

    def test_frontier_list_and_empty(self):
        gh = harvest.parse_plan_tree(read("guard-halt/plan-tree.md"))
        self.assertEqual(gh["frontier"], ["S1d", "S1e", "S1f", "S1g", "S1h"])
        ex = harvest.parse_plan_tree(read("exhaustion/plan-tree.md"))
        self.assertEqual(ex["frontier"], [])

    def test_guard_text_extracted_only_for_guard_halt(self):
        gh = harvest.parse_plan_tree(read("guard-halt/plan-tree.md"))
        self.assertIn("budget", gh["guard_text"])
        self.assertIn("Smallest budget bump", gh["guard_text"])
        self.assertNotIn("INTENT", gh["guard_text"])
        self.assertIsNone(harvest.parse_plan_tree(read("success/plan-tree.md"))["guard_text"])


class ForkExtraction(unittest.TestCase):
    def test_fork_only_for_guard_halt(self):
        self.assertIsNone(harvest.extract_fork(read("exhaustion/plan-tree.md")))
        self.assertIsNone(harvest.extract_fork(read("success/plan-tree.md")))
        fork = harvest.extract_fork(read("guard-halt/plan-tree.md"))
        self.assertIn("S1d", fork)
        self.assertIn("budget", fork)
        self.assertTrue(fork.rstrip().endswith("?"))


class JournalLoading(unittest.TestCase):
    def test_tolerant_parse_skips_garbage(self):
        recs = harvest.parse_journal(['{"node":"S1","verdict":"fail"}', "not json {",
                                      "", '["a","list"]'])
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["node"], "S1")

    def test_load_run_merges_ticks_before_live_journal(self):
        tree, recs = harvest.load_run(FIX, "guard-halt")
        self.assertIn("GUARD-HALT", tree)
        self.assertEqual([r["node"] for r in recs], ["S1", "S1b", "S1c"])  # tick0 then live

    def test_load_run_missing_dir(self):
        tree, recs = harvest.load_run(FIX, "no-such-slug")
        self.assertIsNone(tree)
        self.assertEqual(recs, [])


class Harvesting(unittest.TestCase):
    def test_dead_ends_with_journal_evidence_merged(self):
        tree, recs = harvest.load_run(FIX, "guard-halt")
        out = harvest.harvest(tree, recs, cycle=0)
        dead = [r for r in out if r["kind"] == "dead-end"]
        self.assertEqual(len(dead), 3)
        self.assertTrue(dead[0]["text"].startswith("Tried source A"))
        self.assertIn("failed —", dead[0]["text"])
        self.assertEqual(dead[0]["cycle"], 0)
        self.assertEqual(dead[0]["source"], "harvest")
        self.assertIn("connection timeout", dead[1]["meta"]["journal_evidence"][0])

    def test_guard_halt_adds_one_fact(self):
        tree, recs = harvest.load_run(FIX, "guard-halt")
        facts = [r for r in harvest.harvest(tree, recs, 0) if r["kind"] == "fact"]
        self.assertEqual(len(facts), 1)
        self.assertIn("budget guard", facts[0]["text"])
        self.assertIn("S1d", facts[0]["text"])

    def test_exhaustion_adds_one_fact(self):
        tree, recs = harvest.load_run(FIX, "exhaustion")
        out = harvest.harvest(tree, recs, 1)
        facts = [r for r in out if r["kind"] == "fact"]
        self.assertEqual(len(facts), 1)
        self.assertIn("exhausted every method", facts[0]["text"])
        self.assertEqual(len([r for r in out if r["kind"] == "dead-end"]), 1)

    def test_success_yields_dead_ends_only(self):
        tree, recs = harvest.load_run(FIX, "success")
        out = harvest.harvest(tree, recs, 0)
        self.assertEqual([r["kind"] for r in out], ["dead-end", "dead-end"])

    def test_fp_keys_on_method_not_reason(self):
        self.assertEqual(harvest.fp("Source A (authoritative)"),
                         harvest.fp("source a — authoritative!"))
        self.assertNotEqual(harvest.fp("source a"), harvest.fp("source b"))
        tree, recs = harvest.load_run(FIX, "guard-halt")
        out1 = harvest.harvest(tree, recs, 0)
        retried = tree.replace("HTTP 503; scenario r-source-a", "HTTP 502; second attempt")
        out2 = harvest.harvest(retried, recs, 1)
        self.assertEqual(out1[0]["fp"], out2[0]["fp"])  # same method, reworded reason → same fp


if __name__ == "__main__":
    unittest.main(verbosity=2)
