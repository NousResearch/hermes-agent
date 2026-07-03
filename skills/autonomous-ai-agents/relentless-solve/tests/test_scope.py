#!/usr/bin/env python3
"""Unit tests for scope mode — the CLARIFY → PLAN, never-EXECUTE flow.

Covers: the scope_flow loop (FakeCtx, phases monkeypatched — the test_loop.py style),
the scope package (real write_scope_package into a tmp HERMES_HOME), the forced-read
capability, dod OPEN seeding, the --fact answer-back loop, next_questions passthrough,
dirty receipts, sibling-worktree lifecycle (real git in scratch repos), and cmd_scope
CLI wiring. Run: python3 tests/test_scope.py
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import types
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "scripts"))
sys.path.insert(0, _HERE)

import relentless  # noqa: E402
import knowledge  # noqa: E402
from test_loop import FakeCtx, ts, clar, tk, pl  # noqa: E402


def scope_inp(**over):
    base = {"prompt": "P", "slug": "s", "rounds": 2, "scope_budget": 10 ** 9,
            "k": 2, "inv_rounds": 1, "floor": 0.12, "capability": "act",  # act ON
            "answer_cwd": None, "knowledge": "off", "project": None,     # PURPOSE:
            "seed_facts": [], "plan_timeout": 60,                        # flow must
            "research_dir": None, "worktree_baseline": None}             # force read
    base.update(over)
    return base


class ScopeBase(unittest.TestCase):
    PATCHED = ("run_clarify", "request_plan", "persist")

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="scope-test-")
        self._home = relentless._HOME
        relentless._HOME = self.tmp
        self._orig = {n: getattr(relentless, n) for n in self.PATCHED}
        self.seeds_seen, self.caps_seen, self.plan_calls = [], [], []

    def tearDown(self):
        relentless._HOME = self._home
        for n, f in self._orig.items():
            setattr(relentless, n, f)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def wire(self, clarifies, plans):
        state = {"c": 0, "p": 0}

        def fake_clarify(problem, seeds, cfg, run_dir=None):
            self.seeds_seen.append(list(seeds))
            self.caps_seen.append(cfg["capability"])
            out = clarifies[min(state["c"], len(clarifies) - 1)]
            state["c"] += 1
            return out

        def fake_plan(slug_dir, slug, cycle, rendered, timeout, dodctx=None, dead_fps=()):
            self.plan_calls.append({"slug_dir": slug_dir, "cycle": cycle,
                                    "dead_fps": set(dead_fps), "dodctx": dodctx})
            out = plans[min(state["p"], len(plans) - 1)]
            state["p"] += 1
            return {**out, "slug": slug, "cycle": cycle}

        relentless.run_clarify = fake_clarify
        relentless.request_plan = fake_plan
        relentless.persist = lambda slug_dir, cycle, rendered, ledger: {"prompt_path": "x"}

    def scope_md(self, slug="s"):
        with open(os.path.join(self.tmp, "relentless", slug, "scope", "scope.md"),
                  encoding="utf-8") as fh:
            return fh.read()

    def scope_json(self, slug="s"):
        with open(os.path.join(self.tmp, "relentless", slug, "scope", "scope.json"),
                  encoding="utf-8") as fh:
            return json.load(fh)


class ScopeFlow(ScopeBase):
    def test_happy_path_scoped_first_round(self):
        self.wire([clar([ts("q1", "a1")])], [pl(tk("t1"), tk("t2"))])
        ctx = FakeCtx()
        out = relentless.scope_flow(ctx, scope_inp())
        self.assertEqual(out["outcome"], "scoped")
        self.assertEqual(out["rounds"], 1)
        self.assertEqual(ctx.keys, ["t0", "s0/clock", "s0/clarify", "s0/render",
                                    "s0/plan", "package"])
        md = self.scope_md()
        self.assertIn("VERDICT: scoped", md)
        self.assertIn("Proposed task breakdown", md)
        self.assertIn("method-t1", md)
        self.assertIn("q1 -> a1", md)  # facts learned
        self.assertEqual(self.scope_json()["verdict"], "scoped")

    def test_capability_is_forced_to_read(self):
        self.wire([clar([])], [pl(tk("t1"))])
        relentless.scope_flow(FakeCtx(), scope_inp(capability="act"))
        self.assertEqual(self.caps_seen, ["read"])

    def test_no_execute_keys_ever(self):
        self.wire([clar([ts("q1", "a1")])], [pl(tk("t1"))])
        ctx = FakeCtx()
        relentless.scope_flow(ctx, scope_inp())
        self.assertFalse([k for k in ctx.keys if "/t/" in k],
                         "scope must never issue task-execution steps")

    def test_dead_fps_always_empty(self):
        self.wire([clar([])], [pl(tk("t1"))])
        relentless.scope_flow(FakeCtx(), scope_inp())
        self.assertEqual(self.plan_calls[0]["dead_fps"], set())

    def test_needs_decision_targets_next_round_then_scopes(self):
        self.wire([clar([ts("q1", "a1")]), clar([ts("q2", "a2")])],
                  [pl(disposition="needs_decision", question="pg or sqlite?"),
                   pl(tk("t1"))])
        out = relentless.scope_flow(FakeCtx(), scope_inp())
        self.assertEqual(out["outcome"], "scoped")
        self.assertEqual(out["rounds"], 2)
        # round 1's clarify was seeded with the folded open decision
        self.assertTrue(any("pg or sqlite?" in s for s in self.seeds_seen[1]))
        self.assertIsNone(self.scope_json()["open_decision"])

    def test_needs_decision_persisting_is_scope_output(self):
        self.wire([clar([ts("q1", "a1")]), clar([ts("q2", "a2")])],
                  [pl(disposition="needs_decision", question="pg or sqlite?")])
        out = relentless.scope_flow(FakeCtx(), scope_inp())
        self.assertEqual(out["outcome"], "open-decisions")
        md = self.scope_md()
        self.assertIn("Open decisions", md)
        self.assertIn("pg or sqlite?", md)
        self.assertEqual(self.scope_json()["open_decision"], "pg or sqlite?")

    def test_exhausted_is_infeasible(self):
        self.wire([clar([ts("q1", "a1")])], [pl(disposition="exhausted")])
        out = relentless.scope_flow(FakeCtx(), scope_inp())
        self.assertEqual(out["outcome"], "infeasible")
        self.assertIn("VERDICT: infeasible", self.scope_md())

    def test_dry_stop_when_zero_fresh_and_converged(self):
        conv = "converged (no question above floor)"
        self.wire([clar([ts("q1", "a1")]), clar([], stop=conv)],
                  [pl(disposition="needs_decision", question="same q?")])
        out = relentless.scope_flow(FakeCtx(), scope_inp(rounds=3))
        self.assertEqual(out["outcome"], "dry")
        self.assertEqual(out["rounds"], 2)

    def test_budget_break_still_writes_package(self):
        self.wire([clar([])], [pl(tk("t1"))])
        ctx = FakeCtx(completed={"t0": 0})  # epoch t0 → any real clock busts the budget
        out = relentless.scope_flow(ctx, scope_inp(scope_budget=1))
        self.assertEqual(out["outcome"], "budget")
        self.assertEqual(out["rounds"], 0)
        self.assertIn("VERDICT: budget", self.scope_md())
        self.assertIsNone(self.scope_json()["plan_path"])

    def test_dod_open_line_seeds_round_zero(self):
        stub = types.SimpleNamespace(
            parse_dod=lambda text: {"state": "draft", "groups": [],
                                    "open": "which auth provider?"},
            unmet=lambda parsed: ["R1"], ids=lambda parsed: ["R1"])
        old = relentless._SPEC_MOD
        relentless._SPEC_MOD = stub
        try:
            self.wire([clar([])], [pl(tk("t1"))])
            relentless.scope_flow(FakeCtx(), scope_inp(dod="STATE: draft"))
        finally:
            relentless._SPEC_MOD = old
        self.assertTrue(any("OPEN (from the definition of done): which auth provider?"
                            in s for s in self.seeds_seen[0]))
        self.assertIn("What done means", self.scope_md())

    def test_human_facts_seed_round_zero(self):
        self.wire([clar([])], [pl(tk("t1"))])
        relentless.scope_flow(FakeCtx(), scope_inp(seed_facts=["the API is async"]))
        self.assertIn("the API is async", self.seeds_seen[0])
        self.assertIn("the API is async", self.scope_md())  # a fact in the package

    def test_next_questions_ranked_in_package(self):
        inv = {**clar([ts("q1", "a1")]),
               "next_questions": [{"question": "what about caching?", "value": 0.4},
                                  {"question": "which region?", "value": 0.2}]}
        self.wire([inv], [pl(tk("t1"))])
        relentless.scope_flow(FakeCtx(), scope_inp())
        md = self.scope_md()
        self.assertIn("Next questions", md)
        self.assertLess(md.index("what about caching?"), md.index("which region?"))
        self.assertEqual(len(self.scope_json()["next_questions"]), 2)

    def test_next_questions_absent_key_degrades(self):
        self.wire([clar([ts("q1", "a1")])], [pl(tk("t1"))])  # clar() has no such key
        relentless.scope_flow(FakeCtx(), scope_inp())
        self.assertNotIn("Next questions", self.scope_md())
        self.assertEqual(self.scope_json()["next_questions"], [])

    def test_dirty_receipts_fold_violation(self):
        baseline = {"porcelain": [], "head": "h", "diff_sha256": "d"}
        dirty = {"porcelain": [" M sneaky.py"], "head": "h", "diff_sha256": "d2"}
        old = (relentless.worktree_receipt, relentless.archive_violation_evidence,
               relentless.reset_research_worktree)
        relentless.worktree_receipt = lambda path, head: dirty
        relentless.archive_violation_evidence = lambda path, head, dest: {
            "ok": True, "manifest": []}
        relentless.reset_research_worktree = lambda path: {
            "ok": True, "steps": [], "receipt": baseline}
        try:
            self.wire([clar([])], [pl(tk("t1"))])
            ctx = FakeCtx()
            out = relentless.scope_flow(
                ctx, scope_inp(research_dir="/fake", worktree_baseline=baseline))
        finally:
            (relentless.worktree_receipt, relentless.archive_violation_evidence,
             relentless.reset_research_worktree) = old
        self.assertEqual(out["violations"], 1)
        self.assertEqual(ctx.keys, ["t0", "s0/clock", "s0/clarify", "s0/status",
                                    "s0/evidence", "s0/reset", "s0/render",
                                    "s0/plan", "package"])
        self.assertIn("WRITE-CONTRACT VIOLATION", self.scope_md())
        self.assertIn("sneaky.py", self.scope_md())

    def test_baseline_matching_dirt_is_not_a_violation(self):
        baseline = {"porcelain": [" M applied-diff.py"],
                    "head": "h", "diff_sha256": "d"}
        old = relentless.worktree_receipt
        relentless.worktree_receipt = lambda path, head: {**baseline, "ok": True}
        try:
            self.wire([clar([])], [pl(tk("t1"))])
            out = relentless.scope_flow(
                FakeCtx(),
                scope_inp(research_dir="/fake", worktree_baseline=baseline))
        finally:
            relentless.worktree_receipt = old
        self.assertEqual(out["violations"], 0)
        self.assertNotIn("WRITE-CONTRACT VIOLATION", self.scope_md())

    def test_replay_is_deterministic(self):
        self.wire([clar([ts("q1", "a1")])], [pl(tk("t1"))])
        live = FakeCtx()
        out1 = relentless.scope_flow(live, scope_inp())
        replay = FakeCtx(completed=dict(live.completed))
        out2 = relentless.scope_flow(replay, scope_inp())
        self.assertEqual(out1, out2)
        self.assertEqual(replay.executed, ["package"] if "package" in replay.executed
                         else [])  # nothing (or only the idempotent package) re-ran
        self.assertEqual(live.keys, replay.keys)

    def test_round_zero_violation_is_contained_before_clean_round_one(self):
        baseline = {"porcelain": [], "head": "h", "diff_sha256": "d"}
        dirty = {"porcelain": [" M sneaky.py"], "head": "h",
                 "diff_sha256": "changed", "ok": True}
        receipts = iter([dirty, {**baseline, "ok": True}])
        old = (relentless.worktree_receipt, relentless.archive_violation_evidence,
               relentless.reset_research_worktree)
        relentless.worktree_receipt = lambda path, head: next(receipts)
        relentless.archive_violation_evidence = lambda path, head, dest: {
            "ok": True, "manifest": []}
        relentless.reset_research_worktree = lambda path: {
            "ok": True, "steps": [], "receipt": baseline}
        try:
            self.wire([clar([ts("q0", "a0")]), clar([ts("q1", "a1")])],
                      [pl(disposition="needs_decision", question="more?"), pl(tk("t1"))])
            ctx = FakeCtx()
            out = relentless.scope_flow(
                ctx, scope_inp(research_dir="/fake", worktree_baseline=baseline))
        finally:
            (relentless.worktree_receipt, relentless.archive_violation_evidence,
             relentless.reset_research_worktree) = old
        self.assertEqual(out["outcome"], "scoped")
        self.assertEqual(out["tainted_rounds"], [0])
        self.assertEqual(ctx.keys, ["t0", "s0/clock", "s0/clarify", "s0/status",
                                    "s0/evidence", "s0/reset", "s0/render", "s0/plan",
                                    "s1/clock", "s1/clarify", "s1/status", "s1/render",
                                    "s1/plan", "package"])

    def test_failed_receipt_read_fails_closed_into_containment(self):
        baseline = {"porcelain": [], "head": "h", "diff_sha256": "d"}
        failed = {"porcelain": None, "head": "", "diff_sha256": "", "ok": False}
        calls = []
        old = (relentless.worktree_receipt, relentless.archive_violation_evidence,
               relentless.reset_research_worktree)
        relentless.worktree_receipt = lambda path, head: failed
        relentless.archive_violation_evidence = lambda path, head, dest: (
            calls.append("evidence") or {"ok": True, "manifest": []})
        relentless.reset_research_worktree = lambda path: (
            calls.append("reset") or {"ok": True, "steps": [], "receipt": baseline})
        try:
            self.wire([clar([])], [pl(tk("t1"))])
            ctx = FakeCtx()
            out = relentless.scope_flow(
                ctx, scope_inp(research_dir="/fake", worktree_baseline=baseline))
        finally:
            (relentless.worktree_receipt, relentless.archive_violation_evidence,
             relentless.reset_research_worktree) = old
        self.assertEqual(out["violations"], 1)
        self.assertEqual(calls, ["evidence", "reset"])
        self.assertIn("s0/evidence", ctx.keys)
        self.assertIn("s0/reset", ctx.keys)

    def test_archive_failure_stops_before_reset_but_writes_package(self):
        baseline = {"porcelain": [], "head": "h", "diff_sha256": "d"}
        dirty = {"porcelain": ["?? bad"], "head": "h", "diff_sha256": "x"}
        old = (relentless.worktree_receipt, relentless.archive_violation_evidence,
               relentless.reset_research_worktree)
        relentless.worktree_receipt = lambda path, head: dirty
        relentless.archive_violation_evidence = lambda path, head, dest: {
            "ok": False, "manifest": []}
        relentless.reset_research_worktree = lambda path: self.fail("reset must not run")
        try:
            self.wire([clar([ts("q", "a")])], [pl(tk("t1"))])
            ctx = FakeCtx()
            out = relentless.scope_flow(
                ctx, scope_inp(research_dir="/fake", worktree_baseline=baseline))
        finally:
            (relentless.worktree_receipt, relentless.archive_violation_evidence,
             relentless.reset_research_worktree) = old
        self.assertEqual(out["outcome"], "containment-failed")
        self.assertNotIn("s0/reset", ctx.keys)
        self.assertIn("package", ctx.keys)
        self.assertIn("VERDICT: containment-failed", self.scope_md())

    def test_reset_failure_stops_before_next_round_clarify(self):
        baseline = {"porcelain": [], "head": "h", "diff_sha256": "d"}
        dirty = {"porcelain": ["?? bad"], "head": "h", "diff_sha256": "x"}
        old = (relentless.worktree_receipt, relentless.archive_violation_evidence,
               relentless.reset_research_worktree)
        relentless.worktree_receipt = lambda path, head: dirty
        relentless.archive_violation_evidence = lambda path, head, dest: {
            "ok": True, "manifest": []}
        relentless.reset_research_worktree = lambda path: {
            "ok": False, "steps": [["reset", 1]], "receipt": dirty}
        try:
            self.wire([clar([ts("q0", "a0")]), clar([ts("q1", "a1")])],
                      [pl(disposition="needs_decision", question="more?")])
            ctx = FakeCtx()
            out = relentless.scope_flow(
                ctx, scope_inp(research_dir="/fake", worktree_baseline=baseline))
        finally:
            (relentless.worktree_receipt, relentless.archive_violation_evidence,
             relentless.reset_research_worktree) = old
        self.assertEqual(out["outcome"], "containment-failed")
        self.assertEqual(len(self.seeds_seen), 1)
        self.assertNotIn("s1/clarify", ctx.keys)

    def test_tainted_round_is_rendered_but_not_reused_or_promoted(self):
        baseline = {"porcelain": [], "head": "h", "diff_sha256": "d"}
        dirty = {"porcelain": [" M bad.py"], "head": "h", "diff_sha256": "x"}
        receipts = iter([dirty, {**baseline, "ok": True}])
        rendered = {}
        knowledge_path = os.path.join(self.tmp, "knowledge", "global.jsonl")
        old_path = knowledge.DEFAULT_PATH
        old_ctx = dict(relentless._KNOWLEDGE_CTX)
        old_funcs = (relentless.worktree_receipt,
                     relentless.archive_violation_evidence,
                     relentless.reset_research_worktree)
        relentless.worktree_receipt = lambda path, head: next(receipts)
        relentless.archive_violation_evidence = lambda path, head, dest: {
            "ok": True, "manifest": []}
        relentless.reset_research_worktree = lambda path: {
            "ok": True, "steps": [], "receipt": baseline}
        try:
            self.wire([clar([ts("tainted-question", "tainted-answer")]),
                       clar([ts("clean-question", "clean-answer")])],
                      [pl(disposition="needs_decision", question="more?"), pl(tk("t1"))])
            relentless.persist = lambda slug_dir, cycle, body, ledger: (
                rendered.__setitem__(cycle, body) or {"prompt_path": "x"})
            knowledge.DEFAULT_PATH = knowledge_path
            relentless.set_knowledge_ctx(True, "projX", "s")
            out = relentless.scope_flow(
                FakeCtx(), scope_inp(research_dir="/fake", worktree_baseline=baseline))
        finally:
            knowledge.DEFAULT_PATH = old_path
            relentless._KNOWLEDGE_CTX.update(old_ctx)
            (relentless.worktree_receipt,
             relentless.archive_violation_evidence,
             relentless.reset_research_worktree) = old_funcs
        tainted_text = "tainted-question -> tainted-answer"
        clean_text = "clean-question -> clean-answer"
        warning = "(violating round — re-verify)"
        md = self.scope_md()
        tainted_line = next(line for line in md.splitlines() if tainted_text in line)
        clean_line = next(line for line in md.splitlines() if clean_text in line)
        self.assertIn("⚠", tainted_line)
        self.assertIn(warning, tainted_line)
        self.assertNotIn("⚠", clean_line)
        self.assertNotIn(warning, clean_line)
        self.assertFalse(any(tainted_text in seed for seed in self.seeds_seen[1]))
        self.assertTrue(any("WRITE-CONTRACT VIOLATION" in seed
                            for seed in self.seeds_seen[1]))
        self.assertNotIn(tainted_text, rendered[1])
        self.assertEqual(out["tainted_rounds"], [0])
        self.assertEqual(self.scope_json()["tainted_rounds"], [0])
        with open(knowledge_path, encoding="utf-8") as fh:
            promoted = fh.read()
        self.assertNotIn(tainted_text, promoted)
        self.assertIn(clean_text, promoted)

    def test_return_counts_full_ledger_and_fact_records(self):
        baseline = {"porcelain": [], "head": "h", "diff_sha256": "d"}
        dirty = {"porcelain": ["?? bad"], "head": "h", "diff_sha256": "x"}
        old = (relentless.worktree_receipt, relentless.archive_violation_evidence,
               relentless.reset_research_worktree)
        relentless.worktree_receipt = lambda path, head: dirty
        relentless.archive_violation_evidence = lambda path, head, dest: {
            "ok": True, "manifest": []}
        relentless.reset_research_worktree = lambda path: {
            "ok": True, "steps": [], "receipt": baseline}
        try:
            self.wire([clar([ts("answered", "fact"),
                             ts("missing", "not found", status="NOT_FOUND")])],
                      [pl(tk("t1"))])
            out = relentless.scope_flow(
                FakeCtx(), scope_inp(research_dir="/fake", worktree_baseline=baseline))
        finally:
            (relentless.worktree_receipt, relentless.archive_violation_evidence,
             relentless.reset_research_worktree) = old
        self.assertEqual(out["n_records"], 3)  # clarify fact + clarify gap + isolation gap
        self.assertEqual(out["n_facts"], 1)
        self.assertIsInstance(out["tainted_rounds"], list)

    def test_replay_with_violation_is_deterministic(self):
        baseline = {"porcelain": [], "head": "h", "diff_sha256": "d"}
        dirty = {"porcelain": [" M sneaky.py"], "head": "h", "diff_sha256": "x"}
        old = (relentless.worktree_receipt, relentless.archive_violation_evidence,
               relentless.reset_research_worktree)
        relentless.worktree_receipt = lambda path, head: dirty
        relentless.archive_violation_evidence = lambda path, head, dest: {
            "ok": True, "manifest": []}
        relentless.reset_research_worktree = lambda path: {
            "ok": True, "steps": [], "receipt": baseline}
        try:
            self.wire([clar([ts("q", "a")])], [pl(tk("t1"))])
            live = FakeCtx()
            out1 = relentless.scope_flow(
                live, scope_inp(research_dir="/fake", worktree_baseline=baseline))
            replay = FakeCtx(completed=dict(live.completed))
            out2 = relentless.scope_flow(
                replay, scope_inp(research_dir="/fake", worktree_baseline=baseline))
        finally:
            (relentless.worktree_receipt, relentless.archive_violation_evidence,
             relentless.reset_research_worktree) = old
        self.assertEqual(out1, out2)
        self.assertEqual(live.keys, replay.keys)


def _run(cwd, *args, **kw):
    return subprocess.run(list(args), cwd=cwd, capture_output=True, text=True, **kw)


def _git(cwd, *args):
    return _run(cwd, "git", "-c", "user.email=t@t", "-c", "user.name=t", *args)


class WorktreeLifecycle(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="scope-wt-")
        self._home = relentless._HOME
        relentless._HOME = os.path.join(self.tmp, "home")
        self.repo = os.path.join(self.tmp, "repo")
        os.makedirs(self.repo)
        _git(self.repo, "init", "-q")
        with open(os.path.join(self.repo, "f.txt"), "w") as fh:
            fh.write("v1\n")
        _git(self.repo, "add", "."), _git(self.repo, "commit", "-qm", "c1")

    def tearDown(self):
        relentless._HOME = self._home
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_non_git_dir_degrades(self):
        plain = os.path.join(self.tmp, "plain")
        os.makedirs(plain)
        cwd, info = relentless.setup_research_worktree(plain, "s")
        self.assertEqual(cwd, plain)
        self.assertFalse(info["isolated"])

    def test_create_apply_and_clean_teardown(self):
        with open(os.path.join(self.repo, "f.txt"), "w") as fh:
            fh.write("v2-uncommitted\n")  # tracked, uncommitted
        cwd, info = relentless.setup_research_worktree(self.repo, "s")
        self.assertTrue(info["isolated"])
        self.assertNotEqual(os.path.realpath(cwd), os.path.realpath(self.repo))
        with open(os.path.join(cwd, "f.txt")) as fh:
            self.assertEqual(fh.read(), "v2-uncommitted\n")  # diff carried across
        self.assertTrue(info["diff_applied"])
        self.assertTrue(info["baseline"]["porcelain"])  # applied diff is baseline dirt
        td = relentless.teardown_research_worktree(self.repo, info)
        self.assertTrue(td["removed"])
        self.assertFalse(td["violation"])
        self.assertFalse(os.path.isdir(cwd))
        # the caller's own tree was never touched
        with open(os.path.join(self.repo, "f.txt")) as fh:
            self.assertEqual(fh.read(), "v2-uncommitted\n")

    def test_violation_keeps_worktree(self):
        cwd, info = relentless.setup_research_worktree(self.repo, "s")
        with open(os.path.join(cwd, "evil.txt"), "w") as fh:
            fh.write("x\n")
        td = relentless.teardown_research_worktree(self.repo, info)
        self.assertTrue(td["violation"])
        self.assertFalse(td["removed"])
        self.assertTrue(os.path.isdir(cwd))
        self.assertTrue(any("evil.txt" in ln for ln in td["new_dirt"]))

    def test_reuse_refreshes_to_current_head(self):
        cwd, _ = relentless.setup_research_worktree(self.repo, "s")
        with open(os.path.join(self.repo, "f.txt"), "w") as fh:
            fh.write("v2\n")
        _git(self.repo, "commit", "-aqm", "c2")
        cwd2, info2 = relentless.setup_research_worktree(self.repo, "s")
        self.assertEqual(os.path.realpath(cwd2), os.path.realpath(cwd))
        head_repo = _git(self.repo, "rev-parse", "HEAD").stdout.strip()
        head_wt = _git(cwd2, "rev-parse", "HEAD").stdout.strip()
        self.assertEqual(head_repo, head_wt)
        self.assertIsNone(info2["moved_aside"])

    def test_reuse_of_violated_worktree_moves_it_aside(self):
        cwd, _ = relentless.setup_research_worktree(self.repo, "s")
        with open(os.path.join(cwd, "evil.txt"), "w") as fh:
            fh.write("x\n")
        cwd2, info2 = relentless.setup_research_worktree(self.repo, "s")
        self.assertTrue(info2["isolated"])
        self.assertTrue(info2["moved_aside"].endswith("worktree.violated-1"))
        self.assertTrue(os.path.exists(os.path.join(info2["moved_aside"], "evil.txt")))
        self.assertFalse(os.path.exists(os.path.join(cwd2, "evil.txt")))

    def test_receipt_detects_same_porcelain_with_different_content(self):
        with open(os.path.join(self.repo, "f.txt"), "w") as fh:
            fh.write("caller edit\n")
        wt, info = relentless.setup_research_worktree(self.repo, "s")
        self.assertTrue(info["isolated"])
        with open(os.path.join(wt, "f.txt"), "w") as fh:
            fh.write("different worktree edit\n")
        receipt = relentless.worktree_receipt(wt, info["baseline"]["head"])
        self.assertEqual(receipt["porcelain"], info["baseline"]["porcelain"])
        self.assertNotEqual(receipt["diff_sha256"], info["baseline"]["diff_sha256"])
        self.assertFalse(relentless.receipt_matches(receipt, info["baseline"]))

    def test_hidden_commit_is_detected_and_reset_to_baseline(self):
        wt, info = relentless.setup_research_worktree(self.repo, "s")
        with open(os.path.join(wt, "f.txt"), "w") as fh:
            fh.write("committed illicit edit\n")
        _git(wt, "commit", "-am", "hidden")
        drifted = relentless.worktree_receipt(wt, info["baseline"]["head"])
        self.assertEqual(drifted["porcelain"], [])
        self.assertFalse(relentless.receipt_matches(drifted, info["baseline"]))
        reset = relentless.reset_research_worktree(wt)
        self.assertTrue(reset["ok"])
        self.assertEqual(_git(wt, "rev-parse", "HEAD").stdout.strip(),
                         info["baseline"]["head"])
        self.assertTrue(relentless.receipt_matches(
            relentless.worktree_receipt(wt, info["baseline"]["head"]),
            info["baseline"]))

    def test_full_reset_removes_tracked_untracked_ignored_and_commit_dirt(self):
        with open(os.path.join(self.repo, ".gitignore"), "w") as fh:
            fh.write("*.ignored\n")
        _git(self.repo, "add", ".gitignore")
        _git(self.repo, "commit", "-qm", "ignore files")
        wt, info = relentless.setup_research_worktree(self.repo, "s")
        with open(os.path.join(wt, "f.txt"), "w") as fh:
            fh.write("illicit commit content\n")
        _git(wt, "commit", "-am", "illicit")
        with open(os.path.join(wt, "f.txt"), "w") as fh:
            fh.write("tracked dirt after commit\n")
        with open(os.path.join(wt, "new.txt"), "w") as fh:
            fh.write("untracked\n")
        with open(os.path.join(wt, "foo.ignored"), "w") as fh:
            fh.write("ignored\n")
        reset = relentless.reset_research_worktree(wt)
        self.assertTrue(reset["ok"])
        self.assertFalse(os.path.exists(os.path.join(wt, "new.txt")))
        self.assertFalse(os.path.exists(os.path.join(wt, "foo.ignored")))
        self.assertTrue(relentless.receipt_matches(
            relentless.worktree_receipt(wt, info["baseline"]["head"]),
            info["baseline"]))

    def _assert_setup_failure_cleaned(self, result, reason):
        cwd, info = result
        self.assertEqual(cwd, self.repo)
        self.assertFalse(info["isolated"])
        self.assertIn(reason, info["reason"])
        wt = os.path.join(relentless._HOME, "relentless", "s", "scope", "worktree")
        listed = _git(self.repo, "worktree", "list").stdout
        self.assertNotIn(wt, listed)
        self.assertFalse(os.path.isdir(wt))

    def test_setup_fails_closed_when_worktree_add_fails(self):
        real_git = relentless._git

        def fake_git(cwd, *args, **kwargs):
            if args[:2] == ("worktree", "add"):
                return 1, "boom"
            return real_git(cwd, *args, **kwargs)

        relentless._git = fake_git
        try:
            result = relentless.setup_research_worktree(self.repo, "s")
        finally:
            relentless._git = real_git
        self._assert_setup_failure_cleaned(result, "worktree add failed")

    def test_setup_fails_closed_when_caller_diff_capture_fails(self):
        real_git = relentless._git

        def fake_git(cwd, *args, **kwargs):
            if args == ("diff", "--binary", "HEAD"):
                return 1, "boom"
            return real_git(cwd, *args, **kwargs)

        relentless._git = fake_git
        try:
            result = relentless.setup_research_worktree(self.repo, "s")
        finally:
            relentless._git = real_git
        self._assert_setup_failure_cleaned(result, "caller diff capture failed")

    def test_setup_fails_closed_when_caller_patch_apply_fails(self):
        with open(os.path.join(self.repo, "f.txt"), "w") as fh:
            fh.write("caller edit\n")
        real_run = relentless.subprocess.run

        def fake_run(argv, *args, **kwargs):
            if (isinstance(argv, list) and len(argv) >= 4 and argv[0] == "git"
                    and argv[-1] == "apply" and "input" in kwargs):
                return subprocess.CompletedProcess(argv, 1, stdout="", stderr="boom")
            return real_run(argv, *args, **kwargs)

        relentless.subprocess.run = fake_run
        try:
            result = relentless.setup_research_worktree(self.repo, "s")
        finally:
            relentless.subprocess.run = real_run
        self._assert_setup_failure_cleaned(result, "caller patch apply failed")

    def test_setup_fails_closed_when_caller_patch_write_fails(self):
        real_write = relentless._atomic_write

        def fake_write(path, content):
            if path.endswith("worktree-callerdiff.patch"):
                raise OSError("boom")
            return real_write(path, content)

        relentless._atomic_write = fake_write
        try:
            result = relentless.setup_research_worktree(self.repo, "s")
        finally:
            relentless._atomic_write = real_write
        self._assert_setup_failure_cleaned(result, "caller patch write failed")

    def test_evidence_archive_is_bounded_and_manifests_every_file(self):
        wt, info = relentless.setup_research_worktree(self.repo, "s")
        paths = []
        small = "00-small.txt"
        with open(os.path.join(wt, small), "w") as fh:
            fh.write("small\n")
        paths.append(small)
        symlink = "01-link"
        os.symlink("f.txt", os.path.join(wt, symlink))
        paths.append(symlink)
        large = "02-large.bin"
        with open(os.path.join(wt, large), "wb") as fh:
            fh.write(b"x" * (1024 * 1024 + 1))
        paths.append(large)
        for i in range(25):
            rel = f"small-{i:02d}.txt"
            with open(os.path.join(wt, rel), "w") as fh:
                fh.write(f"{i}\n")
            paths.append(rel)
        dest = os.path.join(self.tmp, "evidence")
        result = relentless.archive_violation_evidence(
            wt, info["baseline"]["head"], dest)
        self.assertTrue(result["ok"])
        with open(os.path.join(dest, "manifest.json"), encoding="utf-8") as fh:
            manifest = json.load(fh)
        by_path = {entry["path"]: entry for entry in manifest}
        self.assertEqual(set(paths), set(by_path) - {"violation.patch"})
        self.assertTrue(all(by_path[path].get("action") and
                            by_path[path].get("reason") for path in paths))
        self.assertEqual(by_path[symlink]["action"], "skipped")
        self.assertIn("not a regular file", by_path[symlink]["reason"])
        self.assertEqual(by_path[large]["action"], "skipped")
        self.assertIn("larger than 1 MiB", by_path[large]["reason"])
        small_entries = [by_path[path] for path in paths
                         if path not in (symlink, large)]
        self.assertEqual(sum(e["action"] == "copied" for e in small_entries), 20)
        skipped_for_cap = [e for e in small_entries if e["action"] == "skipped"]
        self.assertTrue(skipped_for_cap)
        self.assertTrue(all("copy limit reached" in e["reason"]
                            for e in skipped_for_cap))
        self.assertTrue(os.path.isfile(os.path.join(dest, "violation.patch")))

    def test_reset_then_teardown_removes_contained_worktree(self):
        wt, info = relentless.setup_research_worktree(self.repo, "s")
        with open(os.path.join(wt, "evil.txt"), "w") as fh:
            fh.write("dirt\n")
        self.assertTrue(relentless.reset_research_worktree(wt)["ok"])
        td = relentless.teardown_research_worktree(self.repo, info)
        self.assertEqual(td, {"removed": True, "violation": False})
        self.assertFalse(os.path.isdir(wt))


class ScopeCLI(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="scope-cli-")
        self._home = relentless._HOME
        relentless._HOME = self.tmp
        self._env = os.environ.pop("RELENTLESS_ACTIVE", None)
        self._kctx = dict(relentless._KNOWLEDGE_CTX)

    def tearDown(self):
        relentless._HOME = self._home
        os.environ.pop("RELENTLESS_ACTIVE", None)
        if self._env is not None:
            os.environ["RELENTLESS_ACTIVE"] = self._env
        relentless._KNOWLEDGE_CTX.update(self._kctx)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _args(self, **over):
        base = dict(prompt="scope this thing", prompt_file=None, slug=None, rounds=2,
                    budget=1800, k=6, inv_rounds=3, floor=0.12, answer_cwd=None,
                    dod=None, fact=["known fact"], evidence_file=None, plan_timeout=300,
                    knowledge="on", state_dir=None, accept_flow_change=False,
                    allow_nested=False)
        base.update(over)
        return argparse.Namespace(**base)

    def test_inp_wiring_forces_read_and_seeds(self):
        captured = {}

        def fake_engine(inp, slug, a):
            captured.update(inp=inp, slug=slug)
            return 0

        rc = relentless.cmd_scope(self._args(), fake_engine)
        self.assertEqual(rc, 0)
        inp = captured["inp"]
        self.assertEqual(inp["capability"], "read")
        self.assertEqual(inp["seed_facts"], ["known fact"])
        self.assertEqual(inp["knowledge"], "on")
        self.assertIsNone(inp["research_dir"])  # no answer_cwd → no isolation
        self.assertEqual(os.environ.get("RELENTLESS_ACTIVE"), captured["slug"])
        # isolation record still lands in scope.json
        p = os.path.join(self.tmp, "relentless", captured["slug"], "scope", "scope.json")
        with open(p, encoding="utf-8") as fh:
            self.assertFalse(json.load(fh)["isolation"]["isolated"])

    def test_hermetic_flag_reaches_knowledge_ctx(self):
        relentless.cmd_scope(self._args(knowledge="off"), lambda i, s, a: 0)
        self.assertFalse(relentless._KNOWLEDGE_CTX["enabled"])

    def test_isolation_reports_prior_violation_after_clean_teardown(self):
        repo = os.path.join(self.tmp, "repo")
        os.makedirs(repo)
        _git(repo, "init", "-q")
        with open(os.path.join(repo, "f.txt"), "w") as fh:
            fh.write("v1\n")
        _git(repo, "add", ".")
        _git(repo, "commit", "-qm", "c1")

        def fake_engine(inp, slug, args):
            with open(os.path.join(inp["research_dir"], "contained.txt"), "w") as fh:
                fh.write("temporary violation\n")
            self.assertTrue(
                relentless.reset_research_worktree(inp["research_dir"])["ok"])
            scope_dir = os.path.join(self.tmp, "relentless", slug, "scope")
            os.makedirs(scope_dir, exist_ok=True)
            with open(os.path.join(scope_dir, "scope.json"), "w", encoding="utf-8") as fh:
                json.dump({"tainted_rounds": [0]}, fh)
            return 0

        rc = relentless.cmd_scope(self._args(answer_cwd=repo), fake_engine)
        self.assertEqual(rc, 0)
        slug = relentless.derive_slug("scope this thing")
        with open(os.path.join(self.tmp, "relentless", slug, "scope", "scope.json"),
                  encoding="utf-8") as fh:
            isolation = json.load(fh)["isolation"]
        self.assertIs(isolation["had_violation"], True)
        self.assertIs(isolation["currently_diverged"], False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
