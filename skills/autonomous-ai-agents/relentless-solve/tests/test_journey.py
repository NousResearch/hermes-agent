#!/usr/bin/env python3
"""Unit tests for journey.py — the consolidated decision record. Pure module: no ctx,
no engine, no LLM, no file IO. Run: python3 tests/test_journey.py

The fixture is a canned failure→retry→delegate-declined→replan→success run built from
hand-written trace events + ledger records, exercising every unification the design
locked: a failed branch IS dead-end evidence (with a `from` pointer), a superseded
replan tail IS a not_taken option, retries/delegations are nodes, evidence lives ONLY
as per-node deltas (positional horizon), and the flat ledger round-trips by
concatenation."""

import copy
import os
import re
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "scripts"))

import journey  # noqa: E402


def rec(cycle, kind, text, fpv=None, source="harvest", task=None):
    meta = {"task": task} if task else {}
    return {"cycle": cycle, "source": source, "kind": kind, "text": text,
            "fp": fpv or journey.fp(text), "meta": meta}


def tk(tid, method):
    return {"id": tid, "method": method, "description": f"do {method}",
            "success_criterion": "observably done", "depends_on": [],
            "status": "pending"}


def fixture():
    """ledger indices: 0 clarify fact · 1 dead-end(t1/env-override) · 2 learning(t1)
    · 3+4 Done facts (t2/t3). Events: plan(knew 0..0) → retry(knew 0..0) →
    replan(knew 0..2, old tail superseded) → terminal(3..4)."""
    ledger = [
        rec(0, "fact", "target db is postgres 14", source="clarify"),
        rec(0, "dead-end", "Tried env-override: failed — perms denied",
            fpv=journey.fp("env-override"), task="t1"),
        {"cycle": 0, "source": "harvest", "kind": "fact",
         "text": "loader reads /etc/app.yaml at boot",
         "fp": journey.fp("loader reads /etc/app.yaml at boot"),
         "meta": {"task": "t1", "learning_from": "t1"}},
        rec(0, "fact", "Done patch-loader: config patched",
            fpv=journey.fp("ok patch-loader"), task="t2"),
        rec(0, "fact", "Done verify: criterion holds",
            fpv=journey.fp("ok verify"), task="t3"),
    ]
    plan0 = {"disposition": "tasks", "rationale": "try the env override first",
             "tasks": [tk("t1", "env-override"), tk("t9", "restart-daemon")],
             "alternatives": [{"method": "patch-loader",
                               "why_not_now": "riskier, edits a config file"}]}
    replan0 = {"disposition": "tasks", "rationale": "patch the loader config instead",
               "tasks": [tk("t2", "patch-loader"), tk("t3", "verify")]}
    trace = [
        journey.plan_event("c0/plan", "plan", 0, plan0,
                           {"elapsed": 0, "budget_remaining": 1800, "share": 900,
                            "capability": "act"}, 1),
        journey.retry_event("c0/t/t1/retry1", 0, tk("t1", "env-override"), 1,
                            {"verdict": "failed", "evidence": "still denied"}, 1),
        journey.plan_event("c0/replan/after-t1", "replan", 0, replan0,
                           {"after": "t1", "done": ["t1"], "failed": ["t1"]}, 3,
                           superseded=[tk("t9", "restart-daemon")],
                           stale_reason="dead-method-reuse"),
    ]
    receipts = {"route": "full", "cycles": 1, "stop_reason": "success"}
    return journey.fold_journey("probe", "success", "all 3 plan tasks verified worked",
                                receipts, trace, ledger), ledger


class Fold(unittest.TestCase):
    def setUp(self):
        self.j, self.ledger = fixture()

    def test_chain_shape_and_keys(self):
        self.assertEqual([n["key"] for n in self.j["nodes"]], ["S0", "S1", "S2", "S3"])
        self.assertEqual([n["kind"] for n in self.j["nodes"]],
                         ["plan", "retry", "replan", "terminal"])

    def test_evidence_lives_only_as_deltas_and_horizon_is_positional(self):
        deltas = [[e["fp"] for e in n["evidence"]] for n in self.j["nodes"]]
        self.assertEqual(deltas, [[self.ledger[0]["fp"]], [],
                                  [self.ledger[1]["fp"], self.ledger[2]["fp"]],
                                  [self.ledger[3]["fp"], self.ledger[4]["fp"]]])
        # "known at S_k" == ledger[:watermark] — by concatenating deltas up to k
        known_at_s2 = [e["fp"] for n in self.j["nodes"][:3] for e in n["evidence"]]
        self.assertEqual(known_at_s2, [r["fp"] for r in self.ledger[:3]])

    def test_derive_ledger_round_trips(self):
        flat = journey.derive_ledger(self.j)
        self.assertEqual([(e["kind"], e["text"], e["fp"]) for e in flat],
                         [(r["kind"], r["text"], r["fp"]) for r in self.ledger])

    def test_failed_branch_is_evidence_with_a_from_pointer(self):
        # `from` resolves to the DECISION COORDINATE (node:option) that planned the
        # task; the bare task id survives as from_task — tree edges by reference.
        dead = [e for e in journey.derive_ledger(self.j) if e["kind"] == "dead-end"]
        self.assertEqual(len(dead), 1)
        self.assertEqual(dead[0]["from"], "S0:env-override")
        self.assertEqual(dead[0]["from_task"], "t1")

    def test_learning_evidence_is_tagged_via_learning(self):
        learn = [e for e in journey.derive_ledger(self.j) if e.get("via") == "learning"]
        self.assertEqual(len(learn), 1)
        self.assertIn("loader reads", learn[0]["text"])
        self.assertEqual(learn[0]["from_task"], "t1")

    def test_success_path_is_the_worked_tasks_in_order(self):
        self.assertEqual(self.j["success_path"],
                         [{"node": "S2", "task": "t2", "method": "patch-loader"},
                          {"node": "S2", "task": "t3", "method": "verify"}])

    def test_superseded_tail_is_a_not_taken_option_with_the_stale_reason(self):
        opts = self.j["nodes"][2]["options"]
        old = [o for o in opts if o["method"].startswith("continue-old-tail")]
        self.assertEqual(len(old), 1)
        self.assertFalse(old[0]["taken"])
        self.assertEqual(old[0]["why_not"], "dead-method-reuse")
        self.assertIn("restart-daemon", old[0]["method"])

    def test_alternatives_are_not_taken_options_with_contemporaneous_why(self):
        alts = [o for o in self.j["nodes"][0]["options"]
                if o["method"] == "patch-loader"]
        self.assertEqual(len(alts), 1)
        self.assertFalse(alts[0]["taken"])
        self.assertIn("riskier", alts[0]["why_not"])

    def test_malformed_alternatives_drop_silently(self):
        plan = {"disposition": "tasks", "rationale": "r", "tasks": [tk("t1", "m")],
                "alternatives": ["just a string", {"why_not_now": "no method"},
                                 {"method": "ok-one"}, {"method": "x"},
                                 {"method": "y"}, {"method": "z"}]}
        ev = journey.plan_event("c0/plan", "plan", 0, plan, {}, 0)
        methods = [o["method"] for o in ev["options"] if not o["taken"]]
        self.assertNotIn("just a string", methods)
        self.assertIn("ok-one", methods)  # survivors capped at MAX_ALTERNATIVES
        self.assertLessEqual(len(methods), journey.MAX_ALTERNATIVES)

    def test_task_outcomes_annotated_by_fingerprint(self):
        s0 = {t["method"]: t["outcome"]
              for o in self.j["nodes"][0]["options"] if o.get("taken")
              for t in o["tasks"]}
        self.assertEqual(s0, {"env-override": "failed", "restart-daemon": "not-run"})
        s2 = {t["method"]: t["outcome"]
              for o in self.j["nodes"][2]["options"] if o.get("taken")
              for t in o.get("tasks") or []}
        self.assertEqual(s2, {"patch-loader": "worked", "verify": "worked"})

    def test_path_is_the_chain_of_takens_ending_at_terminal(self):
        self.assertEqual(self.j["path"][0].split(":", 1)[0], "S0")
        self.assertEqual(self.j["path"][-1], "S3")
        self.assertIn("S1:retry: env-override", self.j["path"])

    def test_exploration_counts_are_neutral(self):
        x = self.j["exploration"]
        self.assertEqual(x["options_taken"], 3)   # plan, retry, replan
        self.assertEqual(x["dead_ends"], 1)
        self.assertGreater(x["options_recorded"], x["options_taken"])

    def test_mid_cycle_fork_keeps_the_original_tail_as_taken(self):
        ev = journey.plan_event(
            "c0/replan/after-t1", "replan", 0,
            {"disposition": "needs_decision", "question": "which way?"},
            {}, 0, superseded=[tk("t9", "restart-daemon")], stale_reason="forced")
        by_taken = {o["taken"]: o for o in ev["options"]}
        self.assertTrue(by_taken[True]["method"].startswith("continue-old-tail"))
        self.assertEqual(ev["chose"]["method"], by_taken[True]["method"])

    def test_failure_run_keeps_the_same_skeleton(self):
        trace = [journey.plan_event("c0/plan", "plan", 0,
                                    {"disposition": "exhausted", "rationale": "dead"},
                                    {}, 0)]
        j = journey.fold_journey("probe", "information-dry", "zero fresh facts",
                                 {"route": "run", "cycles": 1,
                                  "stop_reason": "information-dry"}, trace, [])
        self.assertEqual(j["verdict"], "information-dry")
        self.assertEqual([n["kind"] for n in j["nodes"]], ["plan", "terminal"])
        self.assertEqual(j["success_path"], [])
        text = journey.render_journey(j, level="FULL")
        self.assertIn("## Where it stopped", text)
        self.assertNotIn("## The path that worked", text)

    def test_degenerate_route_is_one_decision_plus_terminal(self):
        j = journey.degenerate("probe", "answered", "answered",
                               {"route": "trivial", "cycles": 0,
                                "stop_reason": "answered"},
                               "direct-answer", "trivial gate verdict", "42.")
        self.assertEqual([n["kind"] for n in j["nodes"]], ["plan", "terminal"])
        self.assertEqual(j["path"], ["S0:direct-answer", "S1"])
        self.assertEqual(j["nodes"][0]["evidence"][0]["text"], "42.")
        self.assertEqual(j["success_path"],
                         [{"node": "S0", "task": "t0", "method": "direct-answer"}])

    def test_evidence_coordinates_are_cycle_aware(self):
        ledger = [
            rec(0, "dead-end", "method-a failed", fpv=journey.fp("method-a"), task="t1"),
            rec(0, "fact", "Done method-a", fpv=journey.fp("ok method-a"), task="t1"),
            rec(1, "dead-end", "method-b failed", fpv=journey.fp("method-b"), task="t1"),
            rec(1, "fact", "Done method-b", fpv=journey.fp("ok method-b"), task="t1"),
        ]
        trace = [
            journey.plan_event("c0/plan", "plan", 0,
                               {"disposition": "tasks", "rationale": "a",
                                "tasks": [tk("t1", "method-a")]}, {}, 2),
            journey.plan_event("c1/plan", "plan", 1,
                               {"disposition": "tasks", "rationale": "b",
                                "tasks": [tk("t1", "method-b")]}, {}, 4),
        ]
        j = journey.fold_journey("cycles", "success", "done", {"cycles": 2},
                                 trace, ledger)
        evidence = [e for n in j["nodes"] for e in n["evidence"]]
        self.assertEqual([e["from"] for e in evidence],
                         ["S0:method-a", "S0:method-a",
                          "S1:method-b", "S1:method-b"])
        self.assertEqual([e["from_task"] for e in evidence], ["t1"] * 4)
        self.assertEqual(j["success_path"],
                         [{"node": "S0", "task": "t1", "method": "method-a"},
                          {"node": "S1", "task": "t1", "method": "method-b"}])

    def test_within_cycle_replan_splice_provenance_points_to_splice(self):
        ledger = [
            rec(0, "fact", "Done method-a", fpv=journey.fp("ok method-a"), task="t1"),
            rec(0, "fact", "Done spliced-t2: complete",
                fpv=journey.fp("ok spliced-t2"), task="t2"),
        ]
        trace = [
            journey.plan_event(
                "c0/plan", "plan", 0,
                {"disposition": "tasks", "rationale": "initial plan",
                 "tasks": [tk("t1", "method-a"), tk("t2", "orig-t2")]}, {}, 0),
            journey.plan_event(
                "c0/replan/after-t1", "replan", 0,
                {"disposition": "tasks", "rationale": "splice replacement tail",
                 "tasks": [tk("t2", "spliced-t2")]}, {}, 1,
                superseded=[tk("t2", "orig-t2")], stale_reason="stale-tail"),
        ]
        j = journey.fold_journey("splice", "success", "done", {"cycles": 1},
                                 trace, ledger)
        evidence = next(e for n in j["nodes"] for e in n["evidence"]
                        if e["text"].startswith("Done spliced-t2"))
        self.assertEqual(evidence["from"], "S1:spliced-t2")
        self.assertEqual(evidence["from_task"], "t2")

    def test_task_outcomes_are_cycle_aware_when_ids_repeat(self):
        ledger = [
            rec(0, "dead-end", "method-a failed", fpv=journey.fp("method-a"), task="t1"),
            rec(1, "dead-end", "method-b failed", fpv=journey.fp("method-b"), task="t1"),
            rec(2, "fact", "Done method-c", fpv=journey.fp("ok method-c"), task="t1"),
        ]
        trace = [
            journey.plan_event(f"c{i}/plan", "plan", i,
                               {"disposition": "tasks", "rationale": method,
                                "tasks": [tk("t1", method)]}, {}, i + 1)
            for i, method in enumerate(("method-a", "method-b", "method-c"))
        ]
        j = journey.fold_journey("cycles", "success", "done", {"cycles": 3},
                                 trace, ledger)
        outcomes = [n["options"][0]["tasks"][0]["outcome"] for n in j["nodes"][:3]]
        self.assertEqual(outcomes, ["failed", "failed", "worked"])
        self.assertEqual(j["success_path"],
                         [{"node": "S2", "task": "t1", "method": "method-c"}])

    def test_outcomes_are_scoped_to_task_identity(self):
        method = "shared-method"
        trace = [journey.plan_event(
            "c0/plan", "plan", 0,
            {"disposition": "tasks", "rationale": "r",
             "tasks": [tk("t1", method), tk("t2", method)]}, {}, 1)]
        ledger = [rec(0, "fact", "Done shared-method",
                      fpv=journey.fp("ok " + method), task="t1")]
        j = journey.fold_journey("identity", "success", "done", {"cycles": 1},
                                 trace, ledger)
        tasks = j["nodes"][0]["options"][0]["tasks"]
        self.assertEqual([(t["id"], t["outcome"]) for t in tasks],
                         [("t1", "worked"), ("t2", "not-run")])
        self.assertEqual(j["success_path"],
                         [{"node": "S0", "task": "t1", "method": method}])

    def test_unicode_only_methods_do_not_alias_empty_fingerprint(self):
        first, second = "修復🚀", "診断🔥"
        self.assertEqual(journey.fp(first), journey.fp(""))
        self.assertEqual(journey.fp(second), journey.fp(""))
        trace = [journey.plan_event(
            "c0/plan", "plan", 0,
            {"disposition": "tasks", "rationale": "r",
             "tasks": [tk("t1", first), tk("t2", second)]}, {}, 1)]
        ledger = [rec(0, "fact", "Done first", fpv=journey.fp("ok " + first),
                      task="t1")]
        j = journey.fold_journey("unicode", "success", "done", {"cycles": 1},
                                 trace, ledger)
        tasks = j["nodes"][0]["options"][0]["tasks"]
        self.assertEqual([(t["id"], t["outcome"]) for t in tasks],
                         [("t1", "worked"), ("t2", "not-run")])

    def test_fold_does_not_mutate_trace(self):
        trace = [journey.plan_event(
            "c0/plan", "plan", 0,
            {"disposition": "tasks", "rationale": "r",
             "tasks": [tk("t1", "method")]}, {}, 1)]
        before = copy.deepcopy(trace)
        journey.fold_journey(
            "pure", "success", "done", {"cycles": 1}, trace,
            [rec(0, "fact", "Done method", fpv=journey.fp("ok method"), task="t1")])
        self.assertEqual(trace, before)

    def test_decreasing_and_oversized_watermarks_are_clamped(self):
        ledger = [rec(0, "fact", f"fact {i}") for i in range(3)]
        trace = [
            journey.plan_event("c0/plan", "plan", 0,
                               {"disposition": "exhausted", "rationale": "first"},
                               {}, 5),
            journey.plan_event("c0/replan", "replan", 0,
                               {"disposition": "exhausted", "rationale": "second"},
                               {}, 2),
            journey.plan_event("c0/replan-corrupt", "replan", 0,
                               {"disposition": "exhausted", "rationale": "third"},
                               {}, "corrupt"),
        ]
        j = journey.fold_journey("clamp", "failed", "done", {"cycles": 1},
                                 trace, ledger)
        self.assertEqual([len(n["evidence"]) for n in j["nodes"]], [3, 0, 0, 0])
        cumulative = []
        total = 0
        for node in j["nodes"]:
            total += len(node["evidence"])
            cumulative.append(total)
        self.assertEqual(cumulative, [3, 3, 3, 3])
        self.assertLessEqual(max(cumulative), len(ledger))

    def test_empty_trace_folds_to_terminal_only_journey(self):
        ledger = [rec(0, "fact", "orphan fact"), rec(0, "gap", "orphan gap")]
        j = journey.fold_journey("empty", "failed", "no decisions", {"cycles": 0},
                                 [], ledger)
        self.assertEqual(len(j["nodes"]), 1)
        self.assertEqual(j["nodes"][0]["kind"], "terminal")
        self.assertEqual([e["fp"] for e in j["nodes"][0]["evidence"]],
                         [r["fp"] for r in ledger])
        self.assertEqual(j["path"], ["S0"])
        self.assertEqual(j["success_path"], [])
        for level in journey.LEVELS:
            self.assertIsInstance(journey.render_journey(j, level), str)

    def test_two_halt_events_preserve_taken_semantics(self):
        trace = [
            journey.plan_event("c0/plan", "plan", 0,
                               {"disposition": "needs_decision", "question": "ask"},
                               {}, 0),
            journey.plan_event("c0/replan", "replan", 0,
                               {"disposition": "exhausted", "rationale": "done"},
                               {}, 0),
        ]
        j = journey.fold_journey("halts", "failed", "stopped", {"cycles": 1},
                                 trace, [])
        self.assertTrue(j["nodes"][0]["options"][0]["taken"])
        self.assertEqual(j["nodes"][0]["options"][0]["method"], "halt: ask a human")
        self.assertTrue(j["nodes"][1]["options"][0]["taken"])
        self.assertEqual(j["nodes"][1]["options"][0]["method"],
                         "halt: declare exhaustion")
        self.assertEqual(j["path"],
                         ["S0:halt: ask a human", "S1:halt: declare exhaustion", "S2"])
        for level in journey.LEVELS:
            self.assertIsInstance(journey.render_journey(j, level), str)


class HindsightSupport(unittest.TestCase):
    def setUp(self):
        self.j, self.ledger = fixture()

    def _claim(self, node="S0", option="patch-loader", fpv=None):
        return {"node": node, "option": option,
                "enabling_evidence_fp": fpv or self.ledger[0]["fp"], "why": "w"}

    def _hs(self, avoidable=None, unavoidable=None):
        return {"schema": 1, "optimality": "acceptable",
                "hindsight_path": [], "avoidable_branches": avoidable or [],
                "unavoidable_branches": unavoidable or [], "promoted_learnings": []}

    def test_valid_hindsight_passes(self):
        hs = self._hs([self._claim()])
        hs["hindsight_path"] = [{"method": "patch-loader"}]
        self.assertEqual(journey.validate_hindsight(hs, self.j), [])

    def test_hindsight_path_requires_list_of_non_empty_methods(self):
        for malformed in ("bad", ["bad"], [{}], [{"method": ""}]):
            with self.subTest(hindsight_path=malformed):
                hs = self._hs()
                hs["hindsight_path"] = malformed
                self.assertTrue(journey.validate_hindsight(hs, self.j))

    def test_branch_claim_option_must_be_string(self):
        self.assertTrue(journey.validate_hindsight(
            self._hs([self._claim(option=123)]), self.j))

    def test_branch_claim_node_must_be_string(self):
        self.assertTrue(journey.validate_hindsight(
            self._hs([self._claim(node=[])]), self.j))

    def test_branch_claim_lookup_fp_must_be_string(self):
        claim = self._claim()
        claim["enabling_evidence_fp"] = []
        self.assertTrue(journey.validate_hindsight(self._hs([claim]), self.j))

    def test_citation_contract_rejects_dangling_node_and_fp(self):
        v = journey.validate_hindsight(self._hs([self._claim(node="S99")]), self.j)
        self.assertTrue(any("names no node" in x for x in v))
        v = journey.validate_hindsight(self._hs([self._claim(fpv="feedfeedfeedfeed")]),
                                       self.j)
        self.assertTrue(any("resolves to no evidence fp" in x for x in v))

    def test_optimality_enum_enforced(self):
        hs = self._hs()
        hs["optimality"] = "great"
        self.assertTrue(journey.validate_hindsight(hs, self.j))

    def test_tier_genuinely_avoidable(self):
        # patch-loader was RECORDED as a not_taken option at S0, and the enabling
        # evidence (the clarify fact) was already in S0's delta → tier (a).
        hs = journey.stamp_tiers(self.j, self._hs([self._claim()]))
        c = hs["avoidable_branches"][0]
        self.assertEqual(c["tier"], "genuinely-avoidable")
        self.assertTrue(c["seen_at_the_time"])

    def test_tier_blind_spot(self):
        hs = journey.stamp_tiers(
            self.j, self._hs([self._claim(option="use-config-mgmt-tool")]))
        c = hs["avoidable_branches"][0]
        self.assertEqual(c["tier"], "blind-spot")
        self.assertFalse(c["seen_at_the_time"])

    def test_tier_conservatively_rejects_judge_paraphrase(self):
        # Exact-fp identity is the safer authoritative rule: a paraphrase that does
        # not normalize equally is conservatively classified as a blind spot.
        hs = journey.stamp_tiers(
            self.j, self._hs([self._claim(option="patch the loader")]))
        c = hs["avoidable_branches"][0]
        self.assertEqual(c["tier"], "blind-spot")
        self.assertFalse(c["seen_at_the_time"])

    def test_tier_rejects_subset_label(self):
        j = copy.deepcopy(self.j)
        alternative = next(o for o in j["nodes"][0]["options"] if not o["taken"])
        alternative["method"] = "verify config loader"
        hs = journey.stamp_tiers(
            j, self._hs([self._claim(option="verify config")]))
        c = hs["avoidable_branches"][0]
        self.assertEqual(c["tier"], "blind-spot")
        self.assertFalse(c["seen_at_the_time"])

    def test_tier_distinct_unicode_only_labels_do_not_match(self):
        j = copy.deepcopy(self.j)
        alternative = next(o for o in j["nodes"][0]["options"] if not o["taken"])
        alternative["method"] = "修復🚀"
        hs = journey.stamp_tiers(
            j, self._hs([self._claim(option="診断🔥")]))
        c = hs["avoidable_branches"][0]
        self.assertEqual(c["tier"], "blind-spot")
        self.assertFalse(c["seen_at_the_time"])

    def test_tier_option_horizon_is_cumulative(self):
        hs = journey.stamp_tiers(
            self.j, self._hs([self._claim(node="S2", option="patch-loader")]))
        c = hs["avoidable_branches"][0]
        self.assertEqual(c["tier"], "genuinely-avoidable")
        self.assertTrue(c["seen_at_the_time"])

    def test_tier_honest_exploration_when_evidence_was_born_later(self):
        # the "Done patch-loader" fact is born at the TERMINAL node (S3) — claiming it
        # justified a different choice back at S0 is hindsight over-reach → tier (c).
        hs = journey.stamp_tiers(
            self.j, self._hs([self._claim(fpv=journey.fp("ok patch-loader"))]))
        self.assertEqual(hs["avoidable_branches"][0]["tier"], "honest-exploration")


class Render(unittest.TestCase):
    def setUp(self):
        self.j, self.ledger = fixture()

    def test_pure_and_idempotent(self):
        for level in journey.LEVELS:
            self.assertEqual(journey.render_journey(self.j, level),
                             journey.render_journey(self.j, level))

    def test_each_fact_renders_exactly_once_at_its_birth_node(self):
        text = journey.render_journey(self.j, "FULL")
        self.assertEqual(text.count("Tried env-override: failed"), 1)
        self.assertEqual(text.count("target db is postgres 14"), 1)

    def test_citation_skeleton_survives_every_level(self):
        for level in journey.LEVELS:
            text = journey.render_journey(self.j, level)
            for key in ("S0", "S1", "S2", "S3"):
                self.assertIn(key, text, f"{level} lost node key {key}")
            for r in self.ledger:
                self.assertIn(r["fp"], text, f"{level} lost fp {r['fp']}")
            self.assertIn("patch-loader", text)  # not_taken options survive too

    def test_mermaid_is_full_only_and_compact_collapses_micro_nodes(self):
        full = journey.render_journey(self.j, "FULL")
        compact = journey.render_journey(self.j, "COMPACT")
        spine = journey.render_journey(self.j, "SPINE")
        self.assertIn("```mermaid", full)
        self.assertNotIn("```mermaid", compact)
        self.assertNotIn("```mermaid", spine)
        self.assertIn("### S1", full)       # retry node: full block in FULL...
        self.assertNotIn("### S1", compact)  # ...one-liner in COMPACT
        self.assertLess(len(spine), len(compact))
        self.assertLess(len(compact), len(full))

    def test_abstract_first_and_recap_last(self):
        text = journey.render_journey(self.j, "FULL")
        self.assertTrue(text.startswith("# journey: probe — SUCCESS"))
        self.assertIn("RECAP: success", text)
        self.assertIn("## The path that worked", text)

    def test_path_that_worked_lists_worked_steps_not_learnings(self):
        text = journey.render_journey(self.j, "FULL")
        section = text.split("## The path that worked")[1].split("##")[0]
        self.assertIn("patch-loader", section)
        self.assertIn("verify", section)
        self.assertNotIn("loader reads /etc/app.yaml", section,
                         "a learning is not a completed step")
        # the learning still renders — at its birth node, tagged as a learning
        self.assertIn("(learning from S0:env-override)", text)

    def test_options_labeled_as_recorded_not_exhaustive(self):
        text = journey.render_journey(self.j, "FULL")
        self.assertIn("as recorded at the time", text)

    def test_free_text_is_capped(self):
        long = {"cycle": 0, "source": "clarify", "kind": "fact", "text": "x" * 5000,
                "fp": journey.fp("long"), "meta": {}}
        trace = [journey.plan_event("c0/plan", "plan", 0,
                                    {"disposition": "tasks", "rationale": "r",
                                     "tasks": [tk("t1", "m")]}, {}, 1)]
        j = journey.fold_journey("s", "success", "d", {"cycles": 1}, trace, [long])
        for level, cap in (("FULL", journey.TEXT_CAP), ("COMPACT",
                                                        journey.COMPACT_TEXT_CAP)):
            text = journey.render_journey(j, level)
            self.assertNotIn("x" * (cap + 2), text,
                             f"{level} render exceeded its text cap")

    def test_composed_option_and_evidence_lines_are_capped(self):
        long_method, long_why, long_text = "m" * 450, "w" * 450, "e" * 900
        trace = [journey.plan_event(
            "c0/plan", "plan", 0,
            {"disposition": "tasks", "rationale": long_why,
             "tasks": [tk("t1", long_method)]}, {}, 1)]
        ledger = [rec(0, "fact", long_text, source="clarify")]
        j = journey.fold_journey("caps", "failed", "stopped", {"cycles": 1},
                                 trace, ledger)
        for level, cap in (("FULL", journey.TEXT_CAP),
                           ("COMPACT", journey.COMPACT_TEXT_CAP)):
            lines = journey.render_journey(j, level).splitlines()
            option = next(line for line in lines if line.startswith("- ✓ "))
            evidence = next(line for line in lines if line.startswith("- [fact·fp "))
            self.assertLessEqual(len(option), cap)
            self.assertLessEqual(len(evidence), cap)

    def test_composed_hindsight_lines_are_capped(self):
        j = copy.deepcopy(self.j)
        j["hindsight"] = {
            "optimality": "acceptable",
            "hindsight_path": [{"method": "x" * 115} for _ in range(6)],
            "avoidable_branches": [{
                "node": "S0", "option": "o" * 115,
                "tier": "blind-spot", "why": "w" * 115,
            }],
            "promoted_learnings": [],
        }
        for level, cap in (("FULL", journey.TEXT_CAP),
                           ("COMPACT", journey.COMPACT_TEXT_CAP)):
            text = journey.render_journey(j, level)
            hindsight = text.split("## Hindsight", 1)[1].split("RECAP:", 1)[0]
            for line in hindsight.splitlines():
                self.assertLessEqual(len(line), cap, f"{level}: {line}")

    def test_rendered_free_text_cannot_inject_markdown_lines(self):
        hostile = "method\r\n## Heading\n```\ncode\n```"
        option = {"method": hostile, "taken": False, "why_not": hostile}
        evidence = {"kind": "fact", "fp": journey.fp("hostile"), "text": hostile}
        option_line = journey._option_line(option, journey.TEXT_CAP)
        evidence_line = journey._evidence_line(evidence, journey.TEXT_CAP)
        self.assertNotIn("\n", option_line)
        self.assertNotIn("\n", evidence_line)
        self.assertNotIn("\n## Heading", option_line + evidence_line)
        self.assertNotRegex(option_line + "\n" + evidence_line, r"(?m)^```$")
        j = copy.deepcopy(self.j)
        j["nodes"][0]["options"].append(option)
        j["nodes"][0]["evidence"].append(evidence)
        block = journey.render_journey(j, "COMPACT").split("## Hindsight", 1)[0]
        self.assertNotIn("\n## Heading", block)
        self.assertNotRegex(block, r"(?m)^```$")

    def test_hindsight_section_renders_verdict_or_skip_reason(self):
        withhs = dict(self.j)
        withhs["hindsight"] = {"optimality": "acceptable",
                               "avoidable_branches": [
                                   {"node": "S0", "option": "patch-loader",
                                    "tier": "genuinely-avoidable", "why": "w"}],
                               "promoted_learnings": ["L1"]}
        text = journey.render_journey(withhs, "FULL")
        self.assertIn("optimality: acceptable", text)
        self.assertIn("[genuinely-avoidable]", text)
        self.assertIn("learned: L1", text)
        skipped = dict(self.j)
        skipped["hindsight"] = {"skipped": "no leftover budget"}
        self.assertIn("hindsight unavailable — no leftover budget",
                      journey.render_journey(skipped, "FULL"))

    def test_long_method_labels_are_capped_at_every_render_tier(self):
        method = "long-method-" + "x" * 2200
        ledger = [rec(0, "fact", "short evidence", source="clarify")]
        trace = [journey.plan_event(
            "c0/plan", "plan", 0,
            {"disposition": "tasks", "rationale": "r",
             "tasks": [tk("t1", method)]}, {}, 1)]
        j = journey.fold_journey("caps", "failed", "stopped", {"cycles": 1},
                                 trace, ledger)
        allowances = {"FULL": journey.TEXT_CAP + 8,
                      "COMPACT": journey.COMPACT_TEXT_CAP + 8,
                      "SPINE": journey.COMPACT_TEXT_CAP}
        for level in journey.LEVELS:
            text = journey.render_journey(j, level)
            self.assertNotIn(method, text)
            self.assertLessEqual(max(map(len, text.splitlines())), allowances[level])
            for record in ledger:
                self.assertIn(record["fp"], text)

    def test_mermaid_labels_use_strict_whitelist(self):
        adversarial = 'bad" [node] --> next\n<script>'
        trace = [journey.plan_event(
            "c0/plan", "plan", 0,
            {"disposition": "tasks", "rationale": "r", "tasks": [tk("t1", "safe")],
             "alternatives": [{"method": adversarial, "why_not_now": "unsafe"}]},
            {}, 0)]
        j = journey.fold_journey("mermaid", "failed", "stopped", {"cycles": 1},
                                 trace, [])
        text = journey.render_journey(j, "FULL")
        block = text.split("```mermaid", 1)[1].split("```", 1)[0]
        labels = re.findall(r'"([^"]*)"', block)
        self.assertTrue(labels)
        for label in labels:
            self.assertRegex(label, r"\A[A-Za-z0-9 ·→:_./-]*\Z")
            self.assertNotIn("-->", label)
            self.assertNotIn("<script>", label)

    def test_compact_micro_nodes_retain_not_taken_options(self):
        task = tk("t1", "method")
        trace = [
            journey.retry_event("c0/t/t1/retry1", 0, task, 1,
                                {"verdict": "failed", "evidence": "no"}, 0),
            journey.delegate_event(
                "c0/t/t1/delegate", 0, task,
                {"attempted": True, "gate": {"why": "worth trying"},
                 "slug": "sub", "status": "failed"}, 0),
        ]
        j = journey.fold_journey("micro", "failed", "stopped", {"cycles": 1},
                                 trace, [])
        text = journey.render_journey(j, "COMPACT")
        self.assertIn("accept-dead-end", text)
        self.assertIn("delegate-to-method-explorer", text)
        self.assertIn("fold-dead-end", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
