#!/usr/bin/env python3
"""Unit tests for the relentless_flow loop — no engine, no container, no network.

A FakeCtx stands in for the resumable-script engine (memoized step dict + suspend-raising
ask), and the module-level phase helpers (run_clarify/run_drive/run_harvest/persist/
write_report) are monkeypatched — the same DI-by-module-attribute style drive.py and
test_iterate.py use. Run:
    python3 tests/test_loop.py
"""

import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "scripts"))

import harvest  # noqa: E402
import relentless  # noqa: E402


class FakeSuspend(Exception):
    def __init__(self, key, question):
        super().__init__(key)
        self.key, self.question = key, question


class FakeCtx:
    """Engine stand-in: step memoizes by key; ask returns a scripted answer or suspends."""

    def __init__(self, completed=None, answers=None):
        self.completed = dict(completed or {})
        self.answers = dict(answers or {})
        self.keys, self.executed = [], []

    def step(self, key, fn, **kw):
        self.keys.append(key)
        if key not in self.completed:
            self.executed.append(key)
            self.completed[key] = fn()
        return self.completed[key]

    def ask(self, key, question, schema=None, **kw):
        self.keys.append(key)
        if key in self.answers:
            return self.answers[key]
        raise FakeSuspend(key, question)


def ts(q, fact, status="ANSWERED"):
    return {"question": q, "status": status, "fact": fact, "evidence": f"{q} -> {fact}"}


def clar(tombstones, stop="max_rounds reached"):
    return {"tombstones": tombstones, "stop_reason": stop,
            "n_answered": sum(1 for t in tombstones if t["status"] == "ANSWERED"),
            "n_gaps": sum(1 for t in tombstones if t["status"] == "NOT_FOUND")}


def dr(method, reason, cycle=0):
    return {"cycle": cycle, "source": "harvest", "kind": "dead-end",
            "text": f"Tried {method}: failed — {reason}", "fp": harvest.fp(method), "meta": {}}


def inp(**over):
    base = {"prompt": "P", "slug": "s", "k": 2, "inv_rounds": 1, "floor": 0.12,
            "capability": "act", "answer_cwd": None, "gate": False,
            "max_cycles": 5, "wallclock": 10 ** 9,
            "drive": {"max_ticks": 4, "per_tick_timeout": 60, "wallclock": 300}}
    base.update(over)
    return base


class LoopBase(unittest.TestCase):
    PATCHED = ("run_clarify", "run_drive", "run_harvest", "persist", "write_report")

    def setUp(self):
        self._orig = {n: getattr(relentless, n) for n in self.PATCHED}
        self.seeds_seen, self.rendered, self.reported = [], {}, {}

    def tearDown(self):
        for n, f in self._orig.items():
            setattr(relentless, n, f)

    def wire(self, clarifies, drives, harvests):
        """Script the three phase results (last element repeats); capture seeds + renders."""
        state = {"c": 0, "d": 0, "h": 0}

        def fake_clarify(problem, seeds, cfg):
            self.seeds_seen.append(list(seeds))
            out = clarifies[min(state["c"], len(clarifies) - 1)]
            state["c"] += 1
            return out

        def fake_drive(slug, prompt_path, dcfg):
            out = drives[min(state["d"], len(drives) - 1)]
            state["d"] += 1
            return {**out, "slug": slug}

        def fake_harvest(slug, cycle):
            out = harvests[min(state["h"], len(harvests) - 1)] if harvests else \
                {"records": [], "state": None, "fork": None}
            state["h"] += 1
            return out

        def fake_persist(slug_dir, cycle, rendered, ledger):
            self.rendered[cycle] = rendered
            return {"prompt_path": f"/fake/prompt-c{cycle}.md"}

        def fake_report(slug_dir, outcome, ledger, cycles, detail):
            self.reported.update(outcome=outcome, ledger=list(ledger), cycles=cycles)
            return "/fake/report.md"

        relentless.run_clarify = fake_clarify
        relentless.run_drive = fake_drive
        relentless.run_harvest = fake_harvest
        relentless.persist = fake_persist
        relentless.write_report = fake_report


class HappyAndRefine(LoopBase):
    def test_success_at_c0_key_sequence(self):
        self.wire([clar([ts("q1", "a1")])], [{"status": "SUCCESS", "detail": "done"}], [])
        ctx = FakeCtx()
        out = relentless.relentless_flow(ctx, inp())
        self.assertEqual(out["outcome"], "success")
        self.assertEqual(out["cycles"], 1)
        self.assertEqual(out["n_facts"], 1)
        self.assertEqual(ctx.keys, ["t0", "c0/clock", "c0/clarify", "c0/render",
                                    "c0/execute", "report"])

    def test_failure_feeds_refined_cycle(self):
        self.wire([clar([ts("q1", "a1")]), clar([], stop="converged (no question above floor)")],
                  [{"status": "EXHAUSTION", "detail": "dead"},
                   {"status": "SUCCESS", "detail": "done"}],
                  [{"records": [dr("alfa", "503")], "state": "EXHAUSTION-STOP", "fork": None}])
        out = relentless.relentless_flow(FakeCtx(), inp())
        self.assertEqual(out["outcome"], "success")
        self.assertEqual(out["cycles"], 2)
        # c1 clarify was seeded with both the c0 fact and the c0 dead-end
        self.assertIn("q1 -> a1", self.seeds_seen[1])
        self.assertTrue(any("Tried alfa" in s for s in self.seeds_seen[1]))
        # c1 rendered prompt carries the dead-end section; c0's does not
        self.assertIn("Dead ends — do NOT re-attempt", self.rendered[1])
        self.assertIn("Tried alfa", self.rendered[1])
        self.assertNotIn("Dead ends", self.rendered[0])
        self.assertIn("INTENT: P", self.rendered[1])  # intent verbatim inside the envelope
        # planner envelope pins this cycle's artifact slug
        self.assertIn("plans/s-c1/plan-tree.md", self.rendered[1])
        self.assertIn("plans/s-c0/plan-tree.md", self.rendered[0])


class StopConditions(LoopBase):
    def test_information_dry(self):
        self.wire([clar([ts("q1", "a1")]),
                   clar([ts("q1", "a1 reworded")], stop="converged (no question above floor)")],
                  [{"status": "EXHAUSTION", "detail": "dead"}],
                  [{"records": [dr("alfa", "503")], "state": "EXHAUSTION-STOP", "fork": None},
                   {"records": [dr("alfa", "502 fresh wording")], "state": "EXHAUSTION-STOP",
                    "fork": None}])
        out = relentless.relentless_flow(FakeCtx(), inp())
        self.assertEqual(out["outcome"], "information-dry")
        self.assertEqual(out["cycles"], 2)

    def test_not_dry_while_harvest_is_fresh(self):
        self.wire([clar([ts("q1", "a1")]),
                   clar([ts("q1", "a1")], stop="converged (no question above floor)"),
                   clar([ts("q1", "a1")], stop="converged (no question above floor)")],
                  [{"status": "EXHAUSTION", "detail": "dead"},
                   {"status": "EXHAUSTION", "detail": "dead"},
                   {"status": "SUCCESS", "detail": "done"}],
                  [{"records": [dr("alfa", "503")], "state": "EXHAUSTION-STOP", "fork": None},
                   {"records": [dr("bravo", "timeout")], "state": "EXHAUSTION-STOP",
                    "fork": None}])
        out = relentless.relentless_flow(FakeCtx(), inp())
        self.assertEqual(out["outcome"], "success")  # bravo was fresh info → kept going
        self.assertEqual(out["cycles"], 3)

    def test_max_cycles_cap(self):
        self.wire([clar([ts("q1", "a1")])], [{"status": "EXHAUSTION", "detail": "dead"}],
                  [{"records": [dr("alfa", "503")], "state": "EXHAUSTION-STOP", "fork": None}])
        out = relentless.relentless_flow(FakeCtx(), inp(max_cycles=2))
        self.assertEqual(out["outcome"], "max-cycles")
        self.assertEqual(out["cycles"], 2)

    def test_wallclock_stops_before_cycle(self):
        self.wire([clar([])], [{"status": "SUCCESS"}], [])
        ctx = FakeCtx(completed={"t0": 0.0, "c0/clock": 10.0 ** 12})
        out = relentless.relentless_flow(ctx, inp(wallclock=60))
        self.assertEqual(out["outcome"], "wallclock")
        self.assertEqual(out["cycles"], 0)
        self.assertEqual(ctx.keys, ["t0", "c0/clock", "report"])


class GuardHaltFork(LoopBase):
    GH = [{"records": [dr("source A", "503")], "state": "GUARD-HALT",
           "fork": "Which branch should be preferred?"}]

    def test_default_assume_and_note_continues(self):
        self.wire([clar([ts("q1", "a1")]), clar([], stop="converged (no question above floor)")],
                  [{"status": "GUARD_HALT", "detail": "guard"},
                   {"status": "SUCCESS", "detail": "done"}], self.GH)
        out = relentless.relentless_flow(FakeCtx(), inp())
        self.assertEqual(out["outcome"], "success")
        kinds = [(r["source"], r["kind"]) for r in self.reported["ledger"]]
        self.assertIn(("assumption", "gap"), kinds)
        assumed = [r for r in self.reported["ledger"] if r["source"] == "assumption"]
        self.assertIn("OPEN FORK", assumed[0]["text"])
        self.assertTrue(any("OPEN FORK" in s for s in self.seeds_seen[1]))  # next clarify sees it

    def test_gate_suspends_then_answer_continues(self):
        self.wire([clar([]), clar([], stop="converged (no question above floor)")],
                  [{"status": "GUARD_HALT", "detail": "guard"},
                   {"status": "SUCCESS", "detail": "done"}], self.GH)
        with self.assertRaises(FakeSuspend) as cm:
            relentless.relentless_flow(FakeCtx(), inp(gate=True))
        self.assertEqual(cm.exception.key, "c0/fork")

        self.setUp()  # fresh capture state
        self.wire([clar([]), clar([], stop="converged (no question above floor)")],
                  [{"status": "GUARD_HALT", "detail": "guard"},
                   {"status": "SUCCESS", "detail": "done"}], self.GH)
        out = relentless.relentless_flow(FakeCtx(answers={"c0/fork": "prefer source D"}),
                                         inp(gate=True))
        self.assertEqual(out["outcome"], "success")
        facts = [r for r in self.reported["ledger"] if r["kind"] == "fact"]
        self.assertTrue(any("prefer source D" in r["text"] for r in facts))


class ReplayDeterminism(LoopBase):
    def _wire_two_cycle(self):
        self.wire([clar([ts("q1", "a1")]), clar([], stop="converged (no question above floor)")],
                  [{"status": "EXHAUSTION", "detail": "dead"},
                   {"status": "SUCCESS", "detail": "done"}],
                  [{"records": [dr("alfa", "503")], "state": "EXHAUSTION-STOP", "fork": None}])

    def test_full_replay_executes_nothing(self):
        self._wire_two_cycle()
        ctx1 = FakeCtx()
        out1 = relentless.relentless_flow(ctx1, inp())

        def boom(*a, **kw):
            raise AssertionError("replay must not re-execute any step")
        for n in self.PATCHED:
            setattr(relentless, n, boom)
        ctx2 = FakeCtx(completed=dict(ctx1.completed))
        out2 = relentless.relentless_flow(ctx2, inp())
        self.assertEqual(ctx2.executed, [])
        self.assertEqual(ctx2.keys, ctx1.keys)
        self.assertEqual(out1, out2)

    def test_partial_replay_runs_only_the_tail(self):
        self._wire_two_cycle()
        ctx1 = FakeCtx()
        relentless.relentless_flow(ctx1, inp())
        partial = dict(ctx1.completed)
        del partial["report"]
        self._wire_two_cycle()  # fresh scripted fakes for the re-run
        ctx2 = FakeCtx(completed=partial)
        relentless.relentless_flow(ctx2, inp())
        self.assertEqual(ctx2.executed, ["report"])
        self.assertEqual(ctx2.keys, ctx1.keys)


class RenderAndFolds(unittest.TestCase):
    def test_render_sections_and_omission(self):
        ledger = [{"cycle": 0, "source": "clarify", "kind": "fact", "text": "F1", "fp": "1",
                   "meta": {}},
                  {"cycle": 0, "source": "harvest", "kind": "dead-end", "text": "Tried x",
                   "fp": "2", "meta": {}}]
        r = relentless.render("INTENT", ledger)
        self.assertTrue(r.startswith("INTENT"))
        self.assertIn("## Established facts", r)
        self.assertIn("- F1", r)
        self.assertIn("## Dead ends", r)
        self.assertNotIn("## Known gaps", r)  # empty section omitted
        self.assertNotIn("## ", relentless.render("INTENT", []))

    def test_fold_clarify_fp_on_question(self):
        ledger, seen = [], set()
        n1 = relentless.fold_clarify([ts("q1", "a1")], 0, ledger, seen)
        n2 = relentless.fold_clarify([ts("q1", "a1 reworded answer")], 1, ledger, seen)
        self.assertEqual((n1, n2), (1, 0))  # same question re-answered is not fresh
        self.assertEqual(len(ledger), 1)
        self.assertEqual(ledger[0]["kind"], "fact")

    def test_planner_envelope_frames_the_skill_and_artifacts(self):
        env = relentless.planner_envelope("INTENT-BODY", "x-c3")
        self.assertIn("resilient-planner skill", env)
        self.assertIn("skill_view", env)
        self.assertIn("INTENT: INTENT-BODY", env)
        self.assertIn("x-c3/plan-tree.md", env)
        self.assertIn("x-c3/journal.jsonl", env)

    def test_fold_gap_kind(self):
        ledger, seen = [], set()
        relentless.fold_clarify([ts("q2", "no creds", status="NOT_FOUND")], 0, ledger, seen)
        self.assertEqual(ledger[0]["kind"], "gap")


if __name__ == "__main__":
    unittest.main(verbosity=2)
