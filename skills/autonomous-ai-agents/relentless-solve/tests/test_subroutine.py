#!/usr/bin/env python3
"""Topology-B tests: the subroutine posture of relentless under a dev loop.

Covers: the B1 mechanical recursion guard (RELENTLESS_ACTIVE → exit 4, --allow-nested
escape), the B2 subroutine result contract (solve.json per route, pinned artifact
keys), and the B3 global knowledge tier (knowledge.py units, write_report promotion,
run_clarify seeding with the project-scoping and evidence-only poisoning guards).
Run: python3 tests/test_subroutine.py
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import types
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "scripts"))

import knowledge  # noqa: E402
import relentless  # noqa: E402


class EnvBase(unittest.TestCase):
    """Isolates RELENTLESS_ACTIVE, _HOME, the knowledge ctx, and the global path."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="subroutine-test-")
        self._home = relentless._HOME
        relentless._HOME = self.tmp
        self._env = os.environ.pop("RELENTLESS_ACTIVE", None)
        self._kctx = dict(relentless._KNOWLEDGE_CTX)
        self._kpath = knowledge.DEFAULT_PATH
        knowledge.DEFAULT_PATH = os.path.join(self.tmp, "knowledge", "global.jsonl")

    def tearDown(self):
        relentless._HOME = self._home
        os.environ.pop("RELENTLESS_ACTIVE", None)
        if self._env is not None:
            os.environ["RELENTLESS_ACTIVE"] = self._env
        relentless._KNOWLEDGE_CTX.update(self._kctx)
        knowledge.DEFAULT_PATH = self._kpath
        shutil.rmtree(self.tmp, ignore_errors=True)


class RecursionGuard(EnvBase):
    def test_nested_invocation_refuses_with_exit_4(self):
        os.environ["RELENTLESS_ACTIVE"] = "outer-slug"
        rc = relentless.main(["run", "--slug", "inner", "--prompt", "p"])
        self.assertEqual(rc, 4)

    def test_allow_nested_overrides(self):
        os.environ["RELENTLESS_ACTIVE"] = "outer-slug"
        old = relentless.classify
        relentless.classify = lambda intent, risk, oneshot=None: {
            "route": "trivial", "why": "test", "source": "model"}
        try:
            rc = relentless.main(["solve", "--prompt", "p", "--allow-nested",
                                  "--gate-only"])
        finally:
            relentless.classify = old
        self.assertEqual(rc, 0)

    def test_solve_marks_the_process_tree(self):
        active_seen = []

        def classify(intent, risk, oneshot=None):
            active_seen.append(os.environ.get("RELENTLESS_ACTIVE"))
            return {"route": "trivial", "why": "test", "source": "model"}

        old = relentless.classify
        relentless.classify = classify
        try:
            relentless.main(["solve", "--prompt", "probe task", "--gate-only"])
        finally:
            relentless.classify = old
        self.assertEqual(active_seen, [relentless.derive_slug("probe task")])
        self.assertIsNone(os.environ.get("RELENTLESS_ACTIVE"))

    def test_sequential_solve_calls_restore_active(self):
        old = relentless.classify
        relentless.classify = lambda intent, risk, oneshot=None: {
            "route": "trivial", "why": "test", "source": "model"}
        try:
            first = relentless.main(["solve", "--prompt", "p", "--gate-only"])
            self.assertEqual(first, 0)
            self.assertIsNone(os.environ.get("RELENTLESS_ACTIVE"))

            second = relentless.main(["solve", "--prompt", "p", "--gate-only"])
            self.assertEqual(second, 0)
            self.assertIsNone(os.environ.get("RELENTLESS_ACTIVE"))
        finally:
            relentless.classify = old

    def test_refused_nested_call_preserves_external_active(self):
        os.environ["RELENTLESS_ACTIVE"] = "some-external-value"
        rc = relentless.main(["run", "--slug", "inner", "--prompt", "p"])
        self.assertEqual(rc, 4)
        self.assertEqual(os.environ.get("RELENTLESS_ACTIVE"), "some-external-value")

    def test_allow_nested_restores_external_active(self):
        os.environ["RELENTLESS_ACTIVE"] = "external-value"
        old = relentless.classify
        relentless.classify = lambda intent, risk, oneshot=None: {
            "route": "trivial", "why": "test", "source": "model"}
        try:
            rc = relentless.main(["solve", "--prompt", "p", "--allow-nested",
                                  "--gate-only"])
        finally:
            relentless.classify = old
        self.assertEqual(rc, 0)
        self.assertEqual(os.environ.get("RELENTLESS_ACTIVE"), "external-value")

    def test_classify_exception_restores_active(self):
        def classify(intent, risk, oneshot=None):
            self.assertEqual(os.environ.get("RELENTLESS_ACTIVE"),
                             relentless.derive_slug(intent))
            raise RuntimeError("boom")

        old = relentless.classify
        relentless.classify = classify
        try:
            with self.assertRaisesRegex(RuntimeError, "boom"):
                relentless.main(["solve", "--prompt", "p"])
        finally:
            relentless.classify = old
        self.assertIsNone(os.environ.get("RELENTLESS_ACTIVE"))

    def test_engine_load_exception_restores_active(self):
        old = relentless._load_engine
        relentless._load_engine = lambda: (_ for _ in ()).throw(SystemExit("boom"))
        try:
            with self.assertRaisesRegex(SystemExit, "boom"):
                relentless.main(["run", "--slug", "x", "--prompt", "p"])
        finally:
            relentless._load_engine = old
        self.assertIsNone(os.environ.get("RELENTLESS_ACTIVE"))

    def test_resume_exposes_slug_then_restores_active(self):
        active_seen = []

        def flow(**kwargs):
            return lambda fn: fn

        def run_cli(flow_obj, argv):
            active_seen.append(os.environ.get("RELENTLESS_ACTIVE"))
            return 0

        old = relentless._load_engine
        relentless._load_engine = lambda: (flow, run_cli)
        try:
            rc = relentless.main(["resume", "--slug", "s", "--answer", "yes"])
        finally:
            relentless._load_engine = old
        self.assertEqual(rc, 0)
        self.assertEqual(active_seen, ["s"])
        self.assertIsNone(os.environ.get("RELENTLESS_ACTIVE"))


class SolveJsonContract(EnvBase):
    VERDICT = {"slug": "s", "route": "trivial", "why": "w", "source": "model",
               "risk": "act", "budget": {"total": 60, "splits": ""}}

    def _read(self, slug_dir):
        with open(os.path.join(slug_dir, "solve.json"), encoding="utf-8") as fh:
            return json.load(fh)

    def test_trivial_route_writes_answer_artifact(self):
        slug_dir = os.path.join(self.tmp, "relentless", "s")
        rc = relentless.solve_trivial("q", slug_dir, self.VERDICT, 60,
                                      oneshot=lambda p, timeout: "the answer")
        self.assertEqual(rc, 0)
        obj = self._read(slug_dir)
        self.assertEqual(
            sorted(obj), ["artifacts", "detail", "outcome", "report_path", "route",
                          "slug", "spent_s"])
        self.assertEqual(obj["outcome"], "answered")
        # round 3: EVERY route's artifacts also carry the journey pointer (B2 contract)
        self.assertEqual(sorted(obj["artifacts"]), ["answer", "journey"])
        self.assertEqual(obj["artifacts"]["answer"], "the answer")
        self.assertTrue(obj["artifacts"]["journey"].endswith("journey.json"))
        self.assertTrue(os.path.exists(obj["artifacts"]["journey"]),
                        "the degenerate journey must land BEFORE solve.json points at it")

    def test_single_route_writes_plan_tree_artifact(self):
        slug_dir = os.path.join(self.tmp, "relentless", "s")
        verdict = {**self.VERDICT, "route": "single_method"}
        rc = relentless.solve_single(
            "q", slug_dir, verdict, 600, "act",
            drive=lambda slug, ppath, dcfg: {"status": "SUCCESS", "detail": "done"})
        self.assertEqual(rc, 0)
        obj = self._read(slug_dir)
        self.assertEqual(obj["outcome"], "SUCCESS")
        self.assertTrue(obj["artifacts"]["plan_tree"].endswith("s-single/plan-tree.md"))

    def test_full_route_reads_engine_result(self):
        slug = "probe-task"
        slug_dir = os.path.join(self.tmp, "relentless", slug)
        state_dir = os.path.join(slug_dir, "flow")
        os.makedirs(state_dir)
        with open(os.path.join(state_dir, "state.json"), "w") as fh:
            json.dump({"result": {"outcome": "success", "cycles": 2, "detail": "d",
                                  "report": "/r.md"}}, fh)
        with open(os.path.join(slug_dir, "gate.json"), "w") as fh:  # skip classify
            json.dump({"route": "full", "why": "w", "source": "model"}, fh)
        args = types.SimpleNamespace(
            prompt="probe task", prompt_file=None, slug=slug, budget=60, risk="act",
            gate=False, route=None, gate_only=False, answer_cwd=None, dod=None,
            state_dir=None, knowledge="on", accept_flow_change=False,
            allow_nested=False)
        rc = relentless.cmd_solve(args, lambda inp, s, a: 0)
        self.assertEqual(rc, 0)
        obj = self._read(slug_dir)
        self.assertEqual(obj["outcome"], "success")
        self.assertEqual(obj["route"], "full")
        self.assertTrue(obj["artifacts"]["ledger"].endswith("ledger.jsonl"))
        self.assertTrue(obj["artifacts"]["last_plan"].endswith("c1/plan.json"))


def rec(fp, text, project="projA", kind="fact", slug="s0"):
    return {"fp": fp, "kind": kind, "text": text, "slug": slug, "cycle": 0,
            "ts": 1.0, "project": project}


class KnowledgeUnits(EnvBase):
    def test_append_dedups_by_fp(self):
        self.assertEqual(knowledge.append([rec("a", "x"), rec("a", "x-again")]), 1)
        self.assertEqual(knowledge.append([rec("a", "x")]), 0)
        self.assertEqual(len(knowledge._load(knowledge.DEFAULT_PATH)), 1)

    def test_read_recent_filters_project_and_caps(self):
        knowledge.append([rec(f"a{i}", f"t{i}") for i in range(5)]
                         + [rec("b1", "other", project="projB"),
                            rec("n1", "orphan", project=None)])
        got = knowledge.read_recent("projA", n=3)
        self.assertEqual([r["fp"] for r in got], ["a2", "a3", "a4"])
        self.assertEqual(len(knowledge.read_recent("projB")), 1)
        self.assertEqual(knowledge.read_recent(None), [])  # null project never seeds

    def test_torn_line_is_tolerated(self):
        knowledge.append([rec("a", "x")])
        with open(knowledge.DEFAULT_PATH, "a") as fh:
            fh.write('{"fp": "torn", "tex')  # a writer died mid-line
        self.assertEqual(len(knowledge.read_recent("projA")), 1)
        self.assertEqual(knowledge.append([rec("b", "y")]), 1)  # still writable

    def test_concurrent_writers_lose_nothing(self):
        def worker(tag):
            for i in range(25):
                knowledge.append([rec(f"{tag}{i}", f"text {tag}{i}")])
        threads = [threading.Thread(target=worker, args=(t,)) for t in ("x", "y")]
        [t.start() for t in threads]
        [t.join() for t in threads]
        recs = knowledge._load(knowledge.DEFAULT_PATH)
        self.assertEqual(len(recs), 50)
        self.assertEqual(len({r["fp"] for r in recs}), 50)

    def test_seed_texts_are_provenance_prefixed(self):
        knowledge.append([rec("a", "the db is postgres", slug="earlier-run")])
        self.assertEqual(knowledge.seed_texts("projA"),
                         ["PRIOR RUN earlier-run: the db is postgres"])

    def test_project_key_shared_across_worktrees(self):
        repo = os.path.join(self.tmp, "repo")
        os.makedirs(repo)
        def git(cwd, *a):
            return subprocess.run(
                ["git", "-C", cwd, "-c", "user.email=t@t", "-c", "user.name=t"] + list(a),
                capture_output=True, text=True)
        git(repo, "init", "-q")
        open(os.path.join(repo, "f"), "w").write("x")
        git(repo, "add", "."), git(repo, "commit", "-qm", "c")
        wt = os.path.join(self.tmp, "wt")
        git(repo, "worktree", "add", "--detach", wt, "HEAD")
        self.assertEqual(knowledge.project_key(repo), knowledge.project_key(wt))
        sub = os.path.join(self.tmp, "plain")
        os.makedirs(sub)
        self.assertEqual(knowledge.project_key(sub), os.path.realpath(sub))
        self.assertIsNone(knowledge.project_key(None))


LEDGER = [
    {"cycle": 0, "source": "clarify", "kind": "fact", "text": "F1", "fp": "f1", "meta": {}},
    {"cycle": 0, "source": "harvest", "kind": "dead-end", "text": "D1", "fp": "d1", "meta": {}},
    {"cycle": 0, "source": "assumption", "kind": "gap", "text": "G1", "fp": "g1", "meta": {}},
]


class Promotion(EnvBase):
    def test_write_report_promotes_facts_and_dead_ends_only(self):
        relentless.set_knowledge_ctx(True, "projA", "slugY")
        relentless.write_report(os.path.join(self.tmp, "sd"), "success", LEDGER, 1, "d")
        recs = knowledge._load(knowledge.DEFAULT_PATH)
        self.assertEqual({r["fp"] for r in recs}, {"f1", "d1"})  # the gap stayed local
        self.assertTrue(all(r["project"] == "projA" and r["slug"] == "slugY"
                            for r in recs))

    def test_promotion_is_idempotent(self):
        relentless.set_knowledge_ctx(True, "projA", "slugY")
        for _ in range(2):  # a forced step re-run must not duplicate
            relentless.write_report(os.path.join(self.tmp, "sd"), "success", LEDGER, 1, "d")
        self.assertEqual(len(knowledge._load(knowledge.DEFAULT_PATH)), 2)

    def test_hermetic_run_promotes_nothing(self):
        relentless.set_knowledge_ctx(False, "projA", "slugY")
        relentless.write_report(os.path.join(self.tmp, "sd"), "success", LEDGER, 1, "d")
        self.assertFalse(os.path.exists(knowledge.DEFAULT_PATH))

    def test_scope_package_promotes_too(self):
        relentless.set_knowledge_ctx(True, "projA", "scope-slug")
        relentless.write_scope_package(
            os.path.join(self.tmp, "sd"), {"prompt": "P", "slug": "scope-slug"},
            "scoped", "d", LEDGER, None, None, None, None, [], 0, [])
        self.assertEqual({r["fp"] for r in knowledge._load(knowledge.DEFAULT_PATH)},
                         {"f1", "d1"})


class Seeding(EnvBase):
    def _clarify(self, inp):
        """run_clarify against a stub investigator; returns the seeds iterate saw."""
        captured = {}
        stub = types.SimpleNamespace(
            apply_capability=lambda cfg, level: cfg,
            iterate=lambda problem, cfg, seed_evidence=None:
                captured.update(seeds=list(seed_evidence or [])) or
                {"tombstones": [], "stop_reason": "converged", "n_answered": 0,
                 "n_gaps": 0})
        old = relentless._INVESTIGATOR_MOD
        relentless._INVESTIGATOR_MOD = stub
        try:
            out = relentless.run_clarify("prob", ["local evidence"], inp)
        finally:
            relentless._INVESTIGATOR_MOD = old
        return captured["seeds"], out

    def test_same_project_records_seed_prefixed(self):
        knowledge.append([rec("a", "prior fact", slug="r1"),
                          rec("b", "other proj", project="projB"),
                          rec("c", "orphan", project=None)])
        seeds, _ = self._clarify({"k": 2, "inv_rounds": 1, "floor": 0.1,
                                  "answer_cwd": None, "capability": "read",
                                  "knowledge": "on", "project": "projA"})
        self.assertEqual(seeds, ["local evidence", "PRIOR RUN r1: prior fact"])

    def test_hermetic_and_null_project_seed_nothing(self):
        knowledge.append([rec("a", "prior fact")])
        for inp in ({"knowledge": "off", "project": "projA"},
                    {"knowledge": "on", "project": None}):
            seeds, _ = self._clarify({"k": 2, "inv_rounds": 1, "floor": 0.1,
                                      "answer_cwd": None, "capability": "read", **inp})
            self.assertEqual(seeds, ["local evidence"])

    def test_next_questions_passthrough_defaults_empty(self):
        _, out = self._clarify({"k": 2, "inv_rounds": 1, "floor": 0.1,
                                "answer_cwd": None, "capability": "read",
                                "knowledge": "off", "project": None})
        self.assertEqual(out["next_questions"], [])  # older investigators degrade


if __name__ == "__main__":
    unittest.main(verbosity=2)
