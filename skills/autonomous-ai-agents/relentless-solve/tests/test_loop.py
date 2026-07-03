#!/usr/bin/env python3
"""Unit tests for the relentless_flow loop — no engine, no container, no network.

A FakeCtx stands in for the resumable-script engine (memoized step dict + suspend-raising
ask), and the module-level phase helpers (run_clarify/request_plan/run_task/
write_plan_receipt/persist/write_report) are monkeypatched — the same DI-by-module-attribute
style drive.py and test_iterate.py use. Run:
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


def tk(tid, method=None, dep=None):
    m = method or f"method-{tid}"
    return {"id": tid, "method": m, "description": f"do {m}",
            "success_criterion": "observably done", "depends_on": dep or [],
            "status": "pending"}


def pl(*tasks, disposition="tasks", question=None):
    return {"schema": 1, "slug": "s", "cycle": 0, "disposition": disposition,
            "rationale": "r", "question": question, "tasks": list(tasks)}


def inp(**over):
    base = {"prompt": "P", "slug": "s", "k": 2, "inv_rounds": 1, "floor": 0.12,
            "capability": "act", "answer_cwd": None, "gate": False,
            "max_cycles": 5, "wallclock": 10 ** 9,
            "plan_timeout": 60, "task_timeout": 60,
            # 0 by default so existing/general tests keep today's single-attempt-then-
            # escalate semantics; LocalRetryLevel2/StalenessGateLevel1/PartialReplan
            # override this explicitly to exercise LEVEL 2/1.
            "local_retry_budget": 0, "local_k": 2, "local_inv_rounds": 1}
    base.update(over)
    return base


class LoopBase(unittest.TestCase):
    PATCHED = ("run_clarify", "request_plan", "run_task", "write_plan_receipt",
               "persist", "write_report", "_maybe_delegate_task", "build_journey",
               "run_hindsight")

    def setUp(self):
        self._orig = {n: getattr(relentless, n) for n in self.PATCHED}
        self.seeds_seen, self.rendered, self.reported, self.receipts = [], {}, {}, []
        self.clarify_run_dirs = []
        self.delegation_calls = []
        self.plan_dodctxs, self.plan_dead_fps, self.reports_saved = [], [], []
        self.journeys = []
        # Safe default: NEVER attempted, so no test exercises the real invoke_hermes/
        # run_drive path unless it explicitly overrides relentless._maybe_delegate_task
        # itself (matching how run_clarify/etc. are overridden per-test via wire()).
        relentless._maybe_delegate_task = (
            lambda slug, t, r, rm, cd, dcfg, risk, inp, oneshot=None, drive=None:
                self.delegation_calls.append(t["id"]) or
                {"attempted": False, "gate": {"alt_methods_plausible": False,
                                              "why": "test default: never attempt"}})

    def tearDown(self):
        for n, f in self._orig.items():
            setattr(relentless, n, f)

    def wire(self, clarifies, plans, verdicts=None):
        """Script the phase results (last element repeats); capture seeds + renders.

        plans: list of plan dicts (see pl()/tk()); verdicts: parallel list of
        {task_id: "failed"} overrides — unlisted tasks report "worked".
        """
        state = {"c": 0, "p": 0}
        verdicts = verdicts or []

        def fake_clarify(problem, seeds, cfg, run_dir=None):
            self.seeds_seen.append(list(seeds))
            self.clarify_run_dirs.append(run_dir)
            out = clarifies[min(state["c"], len(clarifies) - 1)]
            state["c"] += 1
            return out

        def fake_plan(slug_dir, slug, cycle, rendered, timeout, dodctx=None, dead_fps=()):
            self.plan_dodctxs.append(dodctx)
            self.plan_dead_fps.append(set(dead_fps))
            out = plans[min(state["p"], len(plans) - 1)]
            state["p"] += 1
            return {**out, "slug": slug, "cycle": cycle}

        def fake_task(task, cycle_dir, timeout, suffix=None, capability=None):
            over = verdicts[min(state["p"] - 1, len(verdicts) - 1)] if verdicts else {}
            v = over.get(task["id"], "worked")
            return {"id": task["id"], "method": task["method"], "verdict": v,
                    "evidence": f"{v} on {task['method']}"}

        def fake_receipt(cycle_dir, plan, results, report=None):
            self.receipts.append((cycle_dir, [r["verdict"] for r in results]))
            self.reports_saved.append(report)
            return "/fake/plan.json"

        def fake_persist(slug_dir, cycle, rendered, ledger):
            self.rendered[cycle] = rendered
            return {"prompt_path": f"/fake/prompt-c{cycle}.md"}

        def fake_report(slug_dir, outcome, ledger, cycles, detail, requirements=None,
                        journey_obj=None, hindsight=None):
            self.reported.update(outcome=outcome, ledger=list(ledger), cycles=cycles,
                                 requirements=requirements, journey=journey_obj,
                                 hindsight=hindsight)
            return "/fake/report.md"

        def fake_journey(slug_dir, slug, verdict, detail, receipts, trace, ledger):
            # JSON-serializable (the engine-contract suite journals it); the events
            # summary lets tests assert on WHAT decisions were traced without coupling
            # to journey.py's fold internals (test_journey.py covers those).
            self.journeys.append({"verdict": verdict, "receipts": dict(receipts),
                                  "events": [(e["at"], e["kind"]) for e in trace],
                                  "n_ledger": len(ledger)})
            return {"schema": 1, "slug": slug, "verdict": verdict, "detail": detail,
                    "receipts": dict(receipts), "nodes": [], "path": [],
                    "hindsight": None}

        def fake_hindsight(jobj, slug_dir, timeout):
            raise AssertionError(
                "run_hindsight must not fire without cascade+success+budget")

        relentless.run_clarify = fake_clarify
        relentless.request_plan = fake_plan
        relentless.run_task = fake_task
        relentless.write_plan_receipt = fake_receipt
        relentless.persist = fake_persist
        relentless.write_report = fake_report
        relentless.build_journey = fake_journey
        relentless.run_hindsight = fake_hindsight


class HappyAndRefine(LoopBase):
    def test_success_at_c0_key_sequence(self):
        self.wire([clar([ts("q1", "a1")])], [pl(tk("t1"), tk("t2"))])
        ctx = FakeCtx()
        out = relentless.relentless_flow(ctx, inp())
        self.assertEqual(out["outcome"], "success")
        self.assertEqual(out["cycles"], 1)
        self.assertEqual(out["n_facts"], 3)  # 1 clarify fact + 2 worked-task facts
        self.assertEqual(ctx.keys, ["t0", "c0/clock", "c0/clarify", "c0/render",
                                    "c0/plan", "c0/t/t1", "c0/t/t2", "c0/plan-out",
                                    "retro/journey", "report"])
        # no cascade → no hindsight steps; the journey still folds, with the plan
        # decision traced and the run's verdict/receipts on it
        self.assertEqual(self.journeys[0]["verdict"], "success")
        self.assertEqual(self.journeys[0]["events"], [("c0/plan", "plan")])
        self.assertEqual(self.journeys[0]["receipts"]["route"], "run")
        self.assertEqual(self.reported["hindsight"],
                         {"skipped": "hindsight runs only on a successful full route"})

    def test_failure_feeds_refined_cycle(self):
        self.wire([clar([ts("q1", "a1")]), clar([], stop="converged (no question above floor)")],
                  [pl(tk("t1", "alfa")), pl(tk("t1", "bravo"))],
                  [{"t1": "failed"}, {}])
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
        # the body IS the intent + ledger (the planner envelope wraps it downstream)
        self.assertTrue(self.rendered[0].startswith("P"))
        self.assertTrue(self.rendered[1].startswith("P"))
        # each cycle's clarify gets its own per-cycle journal dir
        self.assertTrue(self.clarify_run_dirs[0].endswith(os.path.join("s", "c0", "clarify")))
        self.assertTrue(self.clarify_run_dirs[1].endswith(os.path.join("s", "c1", "clarify")))

    def test_dependency_skip_after_failure(self):
        self.wire([clar([ts("q1", "a1")]), clar([], stop="converged (no question above floor)")],
                  [pl(tk("t1", "alfa"), tk("t2", "beta", dep=["t1"]), tk("t3", "gamma")),
                   pl(tk("t1", "delta"))],
                  [{"t1": "failed"}, {}])
        ctx = FakeCtx()
        out = relentless.relentless_flow(ctx, inp())
        self.assertEqual(out["outcome"], "success")
        # t2 was skipped (no step key), t3 still ran
        self.assertNotIn("c0/t/t2", ctx.keys)
        self.assertIn("c0/t/t3", ctx.keys)
        self.assertEqual(self.receipts[0][1], ["failed", "skipped", "worked"])
        # the skip produced NO ledger record; the blocker did
        texts = [r["text"] for r in self.reported["ledger"]]
        self.assertTrue(any("Tried alfa" in t for t in texts))
        self.assertFalse(any("beta" in t for t in texts))


class StopConditions(LoopBase):
    def test_information_dry(self):
        self.wire([clar([ts("q1", "a1")]),
                   clar([ts("q1", "a1 reworded")], stop="converged (no question above floor)")],
                  [pl(tk("t1", "alfa")), pl(tk("t1", "alfa"))],
                  [{"t1": "failed"}, {"t1": "failed"}])
        out = relentless.relentless_flow(FakeCtx(), inp())
        self.assertEqual(out["outcome"], "information-dry")
        self.assertEqual(out["cycles"], 2)

    def test_not_dry_while_harvest_is_fresh(self):
        self.wire([clar([ts("q1", "a1")]),
                   clar([ts("q1", "a1")], stop="converged (no question above floor)"),
                   clar([ts("q1", "a1")], stop="converged (no question above floor)")],
                  [pl(tk("t1", "alfa")), pl(tk("t1", "bravo")), pl(tk("t1", "charlie"))],
                  [{"t1": "failed"}, {"t1": "failed"}, {}])
        out = relentless.relentless_flow(FakeCtx(), inp())
        self.assertEqual(out["outcome"], "success")  # bravo was fresh info → kept going
        self.assertEqual(out["cycles"], 3)

    def test_max_cycles_cap(self):
        self.wire([clar([ts("q1", "a1")])], [pl(tk("t1", "alfa"))], [{"t1": "failed"}])
        out = relentless.relentless_flow(FakeCtx(), inp(max_cycles=2))
        self.assertEqual(out["outcome"], "max-cycles")
        self.assertEqual(out["cycles"], 2)

    def test_wallclock_stops_before_cycle(self):
        self.wire([clar([])], [pl(tk("t1"))])
        ctx = FakeCtx(completed={"t0": 0.0, "c0/clock": 10.0 ** 12})
        out = relentless.relentless_flow(ctx, inp(wallclock=60))
        self.assertEqual(out["outcome"], "wallclock")
        self.assertEqual(out["cycles"], 0)
        self.assertEqual(ctx.keys, ["t0", "c0/clock", "retro/journey", "report"])


class NeedsDecisionFork(LoopBase):
    Q = "Which branch should be preferred?"
    FORK = [pl(disposition="needs_decision", question=Q)]

    def test_default_assume_and_note_continues(self):
        self.wire([clar([ts("q1", "a1")]), clar([], stop="converged (no question above floor)")],
                  self.FORK + [pl(tk("t1"))])
        out = relentless.relentless_flow(FakeCtx(), inp())
        self.assertEqual(out["outcome"], "success")
        kinds = [(r["source"], r["kind"]) for r in self.reported["ledger"]]
        self.assertIn(("assumption", "gap"), kinds)
        assumed = [r for r in self.reported["ledger"] if r["source"] == "assumption"]
        self.assertIn("OPEN FORK", assumed[0]["text"])
        self.assertTrue(any("OPEN FORK" in s for s in self.seeds_seen[1]))  # next clarify sees it

    def test_gate_suspends_then_answer_continues(self):
        self.wire([clar([]), clar([], stop="converged (no question above floor)")],
                  self.FORK + [pl(tk("t1"))])
        with self.assertRaises(FakeSuspend) as cm:
            relentless.relentless_flow(FakeCtx(), inp(gate=True))
        self.assertEqual(cm.exception.key, "c0/fork")

        # Fresh capture state for the second run — NOT self.setUp() again: that would
        # re-capture self._orig from the ALREADY-patched (fake) values currently in
        # place, so tearDown() would restore those fakes instead of the true originals,
        # leaking a stale patch into whatever test runs next in the same process.
        self.seeds_seen, self.rendered, self.reported, self.receipts = [], {}, {}, []
        self.clarify_run_dirs = []
        self.wire([clar([]), clar([], stop="converged (no question above floor)")],
                  self.FORK + [pl(tk("t1"))])
        out = relentless.relentless_flow(FakeCtx(answers={"c0/fork": "prefer source D"}),
                                         inp(gate=True))
        self.assertEqual(out["outcome"], "success")
        facts = [r for r in self.reported["ledger"] if r["kind"] == "fact"]
        self.assertTrue(any("prefer source D" in r["text"] for r in facts))

    def test_repeated_exhaustion_goes_dry(self):
        self.wire([clar([ts("q1", "a1")]),
                   clar([], stop="converged (no question above floor)")],
                  [pl(disposition="exhausted")])
        out = relentless.relentless_flow(FakeCtx(), inp())
        # c0 exhaustion fact is fresh; c1's identical declaration is not → dry
        self.assertEqual(out["outcome"], "information-dry")
        self.assertEqual(out["cycles"], 2)
        texts = [r["text"] for r in self.reported["ledger"]]
        self.assertTrue(any("declared exhaustion" in t for t in texts))


class StalenessGateLevel1(unittest.TestCase):
    """stale_tail: pure code, no LLM, no ctx — direct unit tests (LEVEL 1's gate)."""

    def _meta(self, exhausted=False, scoped_texts=None, task_learnings=None):
        return {"attempts": 0, "fresh_local": 0, "exhausted": exhausted,
                "scoped_texts": scoped_texts or [], "task_learnings": task_learnings or []}

    def test_no_overlap_no_flag(self):
        result = {"evidence": "completely unrelated observation"}
        remaining = [tk("t2", "totally different approach")]
        stale, reason = relentless.stale_tail(result, self._meta(), remaining, [])
        self.assertFalse(stale)
        self.assertIsNone(reason)

    def test_dead_method_reuse_flags_stale(self):
        ledger = [{"cycle": 0, "source": "harvest", "kind": "dead-end",
                   "text": "Tried alfa: failed", "fp": harvest.fp("alfa"), "meta": {}}]
        remaining = [tk("t2", "alfa")]  # method == "alfa", already proven dead
        result = {"evidence": "unrelated"}
        stale, reason = relentless.stale_tail(result, self._meta(), remaining, ledger)
        self.assertTrue(stale)
        self.assertEqual(reason, "dead-method-reuse")

    def test_vocabulary_bleed_flags_stale(self):
        result = {"evidence": "the postgres migration requires a publication first"}
        remaining = [{"id": "t2", "method": "subscription", "description": "wire it up",
                      "success_criterion": "postgres publication exists and is active",
                      "depends_on": [], "status": "pending"}]
        stale, reason = relentless.stale_tail(result, self._meta(), remaining, [])
        self.assertTrue(stale)
        self.assertEqual(reason, "vocabulary-bleed")

    def test_stopwords_and_short_words_ignored(self):
        result = {"evidence": "it is on the way to do it"}  # all stopwords / len<=2
        remaining = [tk("t2", "beta")]
        stale, reason = relentless.stale_tail(result, self._meta(), remaining, [])
        self.assertFalse(stale)

    def test_learning_from_a_successful_task_can_still_flag_stale(self):
        # a task that WORKED can still surface a learning that invalidates a later task —
        # the gate reads retry_meta["task_learnings"], not just failure evidence.
        result = {"evidence": "worked as expected"}  # no overlap on its own
        remaining = [{"id": "t2", "method": "sync-call", "description": "read the "
                     "response body synchronously from the billing charges endpoint",
                     "success_criterion": "response body parsed", "depends_on": [],
                     "status": "pending"}]
        learning = ("the billing charges endpoint returns 202 and processes async via "
                   "a webhook, not a synchronous response body")
        stale, reason = relentless.stale_tail(
            result, self._meta(task_learnings=[learning]), remaining, [])
        self.assertTrue(stale)
        self.assertEqual(reason, "vocabulary-bleed")

    def test_intent_link_contributes_keywords(self):
        result = {"evidence": "checksum verification revealed a mismatch"}
        remaining = [{"id": "t2", "method": "resync", "description": "resync data",
                      "success_criterion": "counts match",
                      "intent_link": "checksum verification is required before done",
                      "depends_on": [], "status": "pending"}]
        stale, reason = relentless.stale_tail(result, self._meta(), remaining, [])
        self.assertTrue(stale)
        self.assertEqual(reason, "vocabulary-bleed")

    def test_retry_exhaustion_flags_stale_unconditionally(self):
        result = {"evidence": "still failing for unrelated reasons"}
        remaining = [tk("t2", "gamma")]
        stale, reason = relentless.stale_tail(result, self._meta(exhausted=True),
                                              remaining, [])
        self.assertTrue(stale)
        self.assertEqual(reason, "retry-exhaustion")

    def test_empty_remaining_degrades_gracefully(self):
        # run_intent_path only calls this when remaining is non-empty; confirm the gate
        # itself doesn't crash or false-positive if called with [] anyway.
        stale, reason = relentless.stale_tail({"evidence": "x"}, self._meta(), [], [])
        self.assertFalse(stale)


class LocalRetryLevel2(unittest.TestCase):
    """run_task_with_local_retry: direct unit tests via FakeCtx (LEVEL 2 in isolation)."""

    def setUp(self):
        self._orig_task = relentless.run_task
        self._orig_clarify = relentless.run_clarify

    def tearDown(self):
        relentless.run_task = self._orig_task
        relentless.run_clarify = self._orig_clarify

    def test_no_retry_on_first_success(self):
        calls = []
        relentless.run_task = lambda t, d, to, suffix=None, capability=None: (
            calls.append(suffix) or {"id": t["id"], "method": t["method"],
                                     "verdict": "worked", "evidence": "ok"})
        relentless.run_clarify = lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("must not clarify when the task succeeds first try"))
        ctx = FakeCtx()
        r, meta = relentless.run_task_with_local_retry(
            ctx, 0, tk("t1", "alfa"), "/fake/c0", 60, [], set(), inp(local_retry_budget=2))
        self.assertEqual(r["verdict"], "worked")
        self.assertEqual(calls, [None])  # exactly one attempt, the ORIGINAL step key
        self.assertEqual(ctx.keys, ["c0/t/t1"])
        self.assertEqual(meta, {"attempts": 0, "fresh_local": 0, "exhausted": False,
                                "scoped_texts": [], "task_learnings": [],
                                "delegated": False})

    def test_retry_recovers_after_scoped_clarify(self):
        attempts = {"n": 0}

        def fake_task(t, d, to, suffix=None, capability=None):
            attempts["n"] += 1
            if attempts["n"] == 1:
                return {"id": t["id"], "method": t["method"], "verdict": "failed",
                        "evidence": "boom"}
            return {"id": t["id"], "method": t["method"], "verdict": "worked",
                    "evidence": "fixed"}

        clarify_calls = []

        def fake_clarify(problem, seeds, cfg, run_dir=None):
            clarify_calls.append((problem, run_dir))
            return clar([ts("why did it fail", "needed a flag")])

        relentless.run_task = fake_task
        relentless.run_clarify = fake_clarify
        ctx = FakeCtx()
        ledger, seen = [], set()
        r, meta = relentless.run_task_with_local_retry(
            ctx, 0, tk("t1", "alfa"), "/fake/c0", 60, ledger, seen,
            inp(local_retry_budget=2))
        self.assertEqual(r["verdict"], "worked")
        self.assertEqual(ctx.keys,
                         ["c0/t/t1", "c0/t/t1/retry1/clarify", "c0/t/t1/retry1"])
        self.assertEqual(meta["attempts"], 1)
        self.assertEqual(meta["fresh_local"], 1)
        self.assertFalse(meta["exhausted"])
        self.assertEqual(len(clarify_calls), 1)
        self.assertIn("SPECIFIC cause of THIS failure", clarify_calls[0][0])
        self.assertTrue(clarify_calls[0][1].endswith(
            os.path.join("c0", "clarify-scoped", "t1-retry1")))
        # the scoped fact landed in the SHARED ledger, tagged distinctly from LEVEL 0
        self.assertEqual(len(ledger), 1)
        self.assertEqual(ledger[0]["source"], "scoped-clarify")
        self.assertIn("needed a flag", ledger[0]["text"])

    def test_learnings_accumulate_across_every_attempt_not_just_the_final_one(self):
        # a FAILED attempt can still surface something worth remembering — confirm it
        # isn't lost once the task eventually recovers on a later attempt.
        attempts = {"n": 0}

        def fake_task(t, d, to, suffix=None, capability=None):
            attempts["n"] += 1
            if attempts["n"] == 1:
                return {"id": t["id"], "method": t["method"], "verdict": "failed",
                        "evidence": "boom", "learnings": ["discovered during the failure"]}
            return {"id": t["id"], "method": t["method"], "verdict": "worked",
                    "evidence": "fixed", "learnings": ["discovered on the fix"]}

        relentless.run_task = fake_task
        relentless.run_clarify = lambda *a, **k: clar([])
        ctx = FakeCtx()
        r, meta = relentless.run_task_with_local_retry(
            ctx, 0, tk("t1", "alfa"), "/fake/c0", 60, [], set(),
            inp(local_retry_budget=1))
        self.assertEqual(meta["task_learnings"],
                         ["discovered during the failure", "discovered on the fix"])

    def test_budget_exhausted_escalation_matches_todays_single_attempt_shape(self):
        relentless.run_task = lambda t, d, to, suffix=None, capability=None: (
            {"id": t["id"], "method": t["method"], "verdict": "failed",
             "evidence": f"still broken (suffix={suffix})"})
        relentless.run_clarify = lambda *a, **k: clar([])
        ctx = FakeCtx()
        task = tk("t1", "alfa")
        r, meta = relentless.run_task_with_local_retry(
            ctx, 0, task, "/fake/c0", 60, [], set(), inp(local_retry_budget=2))
        self.assertEqual(r["verdict"], "failed")
        self.assertEqual(meta["attempts"], 2)
        self.assertTrue(meta["exhausted"])
        # byte-identical to what a single-attempt failure would harvest into today
        records = harvest.harvest_tasks({"tasks": [task]}, [r], 0)
        self.assertEqual(records[0]["fp"], harvest.fp("alfa"))
        self.assertTrue(records[0]["text"].startswith("Tried alfa: failed"))

    def test_zero_budget_matches_todays_single_attempt(self):
        calls = []
        relentless.run_task = lambda t, d, to, suffix=None, capability=None: (
            calls.append(suffix) or {"id": t["id"], "method": t["method"],
                                     "verdict": "failed", "evidence": "e"})
        relentless.run_clarify = lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("local_retry_budget=0 must never clarify"))
        ctx = FakeCtx()
        r, meta = relentless.run_task_with_local_retry(
            ctx, 0, tk("t1"), "/fake/c0", 60, [], set(), inp(local_retry_budget=0))
        self.assertEqual(calls, [None])
        self.assertEqual(ctx.keys, ["c0/t/t1"])
        self.assertFalse(meta["exhausted"])  # nothing to exhaust at budget=0


class TaskDelegationPrimitives(unittest.TestCase):
    """rp_delegation_gate / task_delegation_intent / run_task_delegation /
    _attempt_task_delegation — LEVEL 2's exhaustion-escalation building blocks, tested
    directly (no ctx, no engine)."""

    def _rm(self, scoped_texts=None, task_learnings=None):
        return {"scoped_texts": scoped_texts or [], "task_learnings": task_learnings or []}

    def test_intent_never_touches_the_top_level_prompt(self):
        task = tk("t1", "debezium-cdc")
        result = {"evidence": "target is a read-replica, cannot accept writes"}
        text = relentless.task_delegation_intent(task, result, self._rm())
        self.assertIn("debezium-cdc", text)
        self.assertIn("read-replica", text)
        self.assertNotIn("MIGRATE THE WHOLE INTENT", text)  # sanity: no leaked outer intent

    def test_intent_includes_local_learnings_when_present(self):
        task = tk("t1", "alfa")
        result = {"evidence": "boom"}
        rm = self._rm(scoped_texts=["scoped fact"], task_learnings=["a learning"])
        text = relentless.task_delegation_intent(task, result, rm)
        self.assertIn("scoped fact", text)
        self.assertIn("a learning", text)

    def test_gate_parses_plausible_true(self):
        oneshot = lambda p, t: '{"alt_methods_plausible": true, "why": "two vendors exist"}'
        gate = relentless.rp_delegation_gate(tk("t1"), {"evidence": "e"}, self._rm(),
                                             inp(), oneshot=oneshot)
        self.assertTrue(gate["alt_methods_plausible"])
        self.assertIn("two vendors", gate["why"])

    def test_gate_defaults_false_on_garbage(self):
        gate = relentless.rp_delegation_gate(tk("t1"), {"evidence": "e"}, self._rm(),
                                             inp(), oneshot=lambda p, t: "not json at all")
        self.assertFalse(gate["alt_methods_plausible"])

    def test_gate_defaults_false_on_exception(self):
        def boom(p, t):
            raise RuntimeError("network down")
        gate = relentless.rp_delegation_gate(tk("t1"), {"evidence": "e"}, self._rm(),
                                             inp(), oneshot=boom)
        self.assertFalse(gate["alt_methods_plausible"])
        self.assertIn("gate error", gate["why"])

    def test_run_task_delegation_builds_scoped_slug_and_writes_prompt(self):
        import tempfile
        calls = []

        def fake_drive(slug, ppath, dcfg):
            calls.append((slug, ppath, dcfg))
            with open(ppath, encoding="utf-8") as fh:
                calls.append(fh.read())
            return {"status": "SUCCESS", "detail": "found an alternative"}

        with tempfile.TemporaryDirectory() as tmp:
            cycle_dir = os.path.join(tmp, "c0")
            os.makedirs(cycle_dir, exist_ok=True)
            st = relentless.run_task_delegation(
                "billing-migration", tk("t1", "debezium-cdc"), {"evidence": "read-replica"},
                self._rm(), cycle_dir, {"max_ticks": 1}, "act", drive=fake_drive)
        self.assertEqual(st["status"], "SUCCESS")
        rp_slug, ppath, dcfg = calls[0]
        self.assertEqual(rp_slug, "billing-migration-c0-t1")
        self.assertTrue(ppath.endswith(os.path.join("rp-t1", "prompt.md")))
        self.assertEqual(dcfg, {"max_ticks": 1})
        self.assertIn("debezium-cdc", calls[1])  # the written prompt carries the intent

    def test_run_task_delegation_read_risk_adds_hard_constraint(self):
        import tempfile
        prompts = []

        def fake_drive(slug, ppath, dcfg):
            with open(ppath, encoding="utf-8") as fh:
                prompts.append(fh.read())
            return {"status": "SUCCESS"}

        with tempfile.TemporaryDirectory() as tmp:
            cycle_dir = os.path.join(tmp, "c0")
            os.makedirs(cycle_dir, exist_ok=True)
            relentless.run_task_delegation("s", tk("t1"), {"evidence": "e"}, self._rm(),
                                           cycle_dir, {}, "read", drive=fake_drive)
        self.assertIn("read-only", prompts[0])

    def test_attempt_delegation_reports_unavailable_when_sibling_missing(self):
        orig = relentless._envelope
        relentless._envelope = lambda: (_ for _ in ()).throw(
            SystemExit("method-explorer envelope.py not found"))
        try:
            out = relentless._attempt_task_delegation(
                "s", tk("t1"), {"evidence": "e"}, self._rm(), "/fake/c0", {}, "act")
        finally:
            relentless._envelope = orig
        self.assertEqual(out["disposition"], "delegation-unavailable")

    def test_attempt_delegation_reports_failed_on_runtime_error(self):
        import tempfile

        def boom_drive(slug, ppath, dcfg):
            raise RuntimeError("drive.py produced no parseable JSON")
        with tempfile.TemporaryDirectory() as tmp:
            cycle_dir = os.path.join(tmp, "c0")
            os.makedirs(cycle_dir, exist_ok=True)
            out = relentless._attempt_task_delegation(
                "s", tk("t1"), {"evidence": "e"}, self._rm(), cycle_dir, {}, "act",
                drive=boom_drive)
        self.assertEqual(out["disposition"], "delegation-failed")
        self.assertIn("RuntimeError", out["error"])

    def test_attempt_delegation_passes_through_success(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            cycle_dir = os.path.join(tmp, "c0")
            os.makedirs(cycle_dir, exist_ok=True)
            out = relentless._attempt_task_delegation(
                "s", tk("t1"), {"evidence": "e"}, self._rm(), cycle_dir, {}, "act",
                drive=lambda slug, ppath, dcfg: {"status": "SUCCESS", "detail": "d"})
        self.assertEqual(out["disposition"], "delegated")
        self.assertEqual(out["status"], "SUCCESS")

    def test_maybe_delegate_skips_attempt_when_gate_says_no(self):
        attempted = []
        relentless._attempt_task_delegation  # sanity import path exists
        out = relentless._maybe_delegate_task(
            "s", tk("t1"), {"evidence": "e"}, self._rm(), "/fake/c0", {}, "act", inp(),
            oneshot=lambda p, t: '{"alt_methods_plausible": false, "why": "env issue"}',
            drive=lambda *a: attempted.append(1))
        self.assertFalse(out["attempted"])
        self.assertEqual(attempted, [])

    def test_maybe_delegate_attempts_when_gate_says_yes(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            cycle_dir = os.path.join(tmp, "c0")
            os.makedirs(cycle_dir, exist_ok=True)
            out = relentless._maybe_delegate_task(
                "s", tk("t1"), {"evidence": "e"}, self._rm(), cycle_dir, {}, "act", inp(),
                oneshot=lambda p, t: '{"alt_methods_plausible": true, "why": "ok"}',
                drive=lambda slug, ppath, dcfg: {"status": "SUCCESS", "detail": "d"})
        self.assertTrue(out["attempted"])
        self.assertEqual(out["disposition"], "delegated")


class LocalRetryLevel2Delegation(unittest.TestCase):
    """run_task_with_local_retry's exhaustion -> delegation escalation, integrated."""

    def setUp(self):
        self._orig = {n: getattr(relentless, n)
                      for n in ("run_task", "run_clarify", "_maybe_delegate_task")}

    def tearDown(self):
        for n, f in self._orig.items():
            setattr(relentless, n, f)

    def test_delegation_not_attempted_when_allow_delegation_false(self):
        relentless.run_task = lambda t, d, to, suffix=None, capability=None: (
            {"id": t["id"], "method": t["method"], "verdict": "failed", "evidence": "e"})
        relentless.run_clarify = lambda *a, **k: clar([])
        relentless._maybe_delegate_task = lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("must not delegate when allow_delegation=False"))
        ctx = FakeCtx()
        r, meta = relentless.run_task_with_local_retry(
            ctx, 0, tk("t1", "alfa"), "/fake/c0", 60, [], set(),
            inp(local_retry_budget=1), allow_delegation=False)
        self.assertTrue(meta["exhausted"])
        self.assertFalse(meta["delegated"])
        self.assertNotIn("c0/t/t1/rp-delegate", ctx.keys)

    def test_successful_delegation_flips_result_to_worked(self):
        relentless.run_task = lambda t, d, to, suffix=None, capability=None: (
            {"id": t["id"], "method": t["method"], "verdict": "failed", "evidence": "boom"})
        relentless.run_clarify = lambda *a, **k: clar([])
        relentless._maybe_delegate_task = (
            lambda slug, t, r, rm, cd, dcfg, risk, i, oneshot=None, drive=None:
                {"attempted": True, "disposition": "delegated", "status": "SUCCESS",
                 "detail": "found the batch-copy workaround"})
        ctx = FakeCtx()
        r, meta = relentless.run_task_with_local_retry(
            ctx, 0, tk("t1", "alfa"), "/fake/c0", 60, [], set(),
            inp(local_retry_budget=1), allow_delegation=True)
        self.assertEqual(r["verdict"], "worked")
        self.assertIn("batch-copy workaround", r["evidence"])
        self.assertFalse(meta["exhausted"])  # a successful delegation un-exhausts the task
        self.assertTrue(meta["delegated"])
        self.assertIn("c0/t/t1/rp-delegate", ctx.keys)
        self.assertTrue(any("batch-copy workaround" in t for t in meta["task_learnings"]))

    def test_non_success_delegation_leaves_result_unchanged_and_folds_dead_end(self):
        relentless.run_task = lambda t, d, to, suffix=None, capability=None: (
            {"id": t["id"], "method": t["method"], "verdict": "failed", "evidence": "boom"})
        relentless.run_clarify = lambda *a, **k: clar([])
        relentless._maybe_delegate_task = (
            lambda slug, t, r, rm, cd, dcfg, risk, i, oneshot=None, drive=None:
                {"attempted": True, "disposition": "delegated",
                 "status": "EXHAUSTION-STOP", "detail": "no alternative worked either"})
        ctx = FakeCtx()
        ledger, seen = [], set()
        r, meta = relentless.run_task_with_local_retry(
            ctx, 0, tk("t1", "alfa"), "/fake/c0", 60, ledger, seen,
            inp(local_retry_budget=1), allow_delegation=True)
        self.assertEqual(r["verdict"], "failed")  # UNCHANGED
        self.assertTrue(meta["exhausted"])         # UNCHANGED — LEVEL 1 handles it as today
        self.assertFalse(meta["delegated"])
        dead_ends = [rec for rec in ledger if rec["kind"] == "dead-end"]
        self.assertTrue(any("method-explorer delegation" in rec["text"]
                            for rec in dead_ends))

    def test_delegation_skipped_when_deadline_already_hit(self):
        # local_retry_budget=1 so the retry loop actually runs and can hit the deadline.
        relentless.run_task = lambda t, d, to, suffix=None, capability=None: (
            {"id": t["id"], "method": t["method"], "verdict": "failed", "evidence": "e"})
        relentless.run_clarify = lambda *a, **k: clar([])
        relentless._maybe_delegate_task = lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("must not delegate when the deadline was already hit"))
        ctx = FakeCtx(completed={"c0/t/t1/retry1/clock": 10 ** 12})  # far in the future
        r, meta = relentless.run_task_with_local_retry(
            ctx, 0, tk("t1", "alfa"), "/fake/c0", 60, [], set(),
            inp(local_retry_budget=1), deadline=1000.0, allow_delegation=True)
        self.assertTrue(meta["exhausted"])
        self.assertFalse(meta["delegated"])


class RunIntentPathDelegationCap(unittest.TestCase):
    """run_intent_path's delegations_used counter — MAX_RP_DELEGATIONS_PER_CYCLE caps
    delegation ACROSS tasks in one cycle, mirroring MAX_REPLANS_PER_CYCLE's pattern."""

    def setUp(self):
        self._orig = {n: getattr(relentless, n)
                      for n in ("run_task", "run_clarify", "_maybe_delegate_task",
                                "_attempt_partial_replan", "stale_tail")}
        relentless.run_clarify = lambda *a, **k: clar([])
        # every task fails on every attempt -> every task is a delegation candidate
        relentless.run_task = lambda t, d, to, suffix=None, capability=None: (
            {"id": t["id"], "method": t["method"], "verdict": "failed", "evidence": "e"})
        # no staleness/replan noise — isolate the delegation cap specifically
        relentless.stale_tail = lambda *a, **k: (False, None)
        relentless._attempt_partial_replan = lambda *a, **k: {"disposition": "exhausted"}

    def tearDown(self):
        for n, f in self._orig.items():
            setattr(relentless, n, f)

    def _plan(self, *tasks):
        return {"schema": 2, "slug": "s", "cycle": 0, "disposition": "tasks",
                "rationale": "r", "question": None, "tasks": list(tasks)}

    def test_second_task_in_the_same_cycle_cannot_also_delegate(self):
        calls = []
        relentless._maybe_delegate_task = (
            lambda slug, t, r, rm, cd, dcfg, risk, i, oneshot=None, drive=None:
                calls.append(t["id"]) or
                {"attempted": True, "disposition": "delegated", "status": "SUCCESS",
                 "detail": f"fixed {t['id']}"})
        ctx = FakeCtx()
        plan = self._plan(tk("t1", "alfa", dep=[]), tk("t2", "beta", dep=[]))
        results, fresh = relentless.run_intent_path(
            ctx, 0, plan, "/fake/c0", 60, 60, [], set(),
            inp(local_retry_budget=1), "/fake/slug")
        self.assertEqual(calls, ["t1"])  # only the FIRST exhausted task got to delegate
        self.assertEqual(results[0]["verdict"], "worked")   # t1: delegation succeeded
        self.assertEqual(results[1]["verdict"], "failed")   # t2: cap already spent


class PartialReplan(unittest.TestCase):
    """run_intent_path's staleness-gate -> mid-cycle partial-replan wiring (LEVEL 1)."""

    def setUp(self):
        self._orig = {n: getattr(relentless, n)
                      for n in ("run_task", "run_clarify", "stale_tail",
                                "_attempt_partial_replan", "_maybe_delegate_task")}
        # No task fails in these tests, so LEVEL 2 never calls run_clarify — but a forced
        # staleness trigger DOES now fire LEVEL 1's pre-replan clarify (run_replan_clarify)
        # before every replan attempt; capture calls so replan-specific tests can assert
        # on the ordering.
        self.clarify_calls = []
        relentless.run_clarify = lambda problem, seeds, cfg, run_dir=None: (
            self.clarify_calls.append((problem, run_dir)) or clar([]))
        # Defensive default (no task fails here, so this shouldn't fire either — but
        # never let a future failing-task test reach the real invoke_hermes/run_drive).
        relentless._maybe_delegate_task = (
            lambda slug, t, r, rm, cd, dcfg, risk, inp, oneshot=None, drive=None:
                {"attempted": False, "gate": {"alt_methods_plausible": False,
                                              "why": "test default: never attempt"}})

    def tearDown(self):
        for n, f in self._orig.items():
            setattr(relentless, n, f)

    def _plan(self, *tasks):
        return {"schema": 2, "slug": "s", "cycle": 0, "disposition": "tasks",
                "rationale": "r", "question": None, "tasks": list(tasks)}

    def _always_worked(self):
        relentless.run_task = lambda t, d, to, suffix=None, capability=None: (
            {"id": t["id"], "method": t["method"], "verdict": "worked", "evidence": "e"})

    def test_gate_triggers_replan_swaps_remaining_tail(self):
        self._always_worked()
        relentless.stale_tail = lambda *a, **k: (True, "forced")
        calls = []

        def fake_attempt(slug_dir, slug, cycle, seq, body, done_ids, timeout, **kw):
            calls.append((seq, set(done_ids)))
            return {"schema": 2, "slug": slug, "cycle": cycle, "disposition": "tasks",
                    "rationale": "r", "question": None,
                    "tasks": [tk("t2-new", "delta")]}
        relentless._attempt_partial_replan = fake_attempt

        ctx = FakeCtx()
        plan = self._plan(tk("t1", "alfa"), tk("t2", "beta"))
        results, fresh = relentless.run_intent_path(
            ctx, 0, plan, "/fake/c0", 60, 60, [], set(), inp(), "/fake/slug")
        self.assertEqual([r["id"] for r in results], ["t1", "t2-new"])
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0], (1, {"t1"}))
        self.assertIn("c0/replan/after-t1", ctx.keys)
        self.assertNotIn("c0/t/t2", ctx.keys)  # the ORIGINAL t2 was replaced, never run
        # mapping intent to task (the replan) is preceded by a clarify pass, in that order
        clarify_i = ctx.keys.index("c0/replan/after-t1/clarify")
        replan_i = ctx.keys.index("c0/replan/after-t1")
        self.assertLess(clarify_i, replan_i)
        self.assertEqual(len(self.clarify_calls), 1)

    def test_pre_replan_clarify_problem_references_evidence_and_remaining_tasks(self):
        relentless.run_task = lambda t, d, to, suffix=None, capability=None: (
            {"id": t["id"], "method": t["method"], "verdict": "worked",
             "evidence": "discovered the blob column blocks replication",
             "learnings": ["exclude blob columns from CDC"]})
        relentless.stale_tail = lambda *a, **k: (True, "forced")
        relentless._attempt_partial_replan = lambda *a, **k: {"disposition": "exhausted"}
        ctx = FakeCtx()
        plan = self._plan(tk("t1", "alfa"), tk("t2", "beta"))
        relentless.run_intent_path(ctx, 0, plan, "/fake/c0", 60, 60, [], set(), inp(),
                                   "/fake/slug")
        problem, run_dir = self.clarify_calls[0]
        self.assertIn("discovered the blob column blocks replication", problem)
        self.assertIn("exclude blob columns from CDC", problem)
        self.assertIn("t2", problem)
        self.assertIn("beta", problem)
        self.assertTrue(run_dir.endswith(
            os.path.join("c0", "clarify-scoped", "replan1-after-t1")))

    def test_pre_replan_clarify_facts_fold_with_distinct_source(self):
        relentless.run_task = lambda t, d, to, suffix=None, capability=None: (
            {"id": t["id"], "method": t["method"], "verdict": "worked", "evidence": "e"})
        relentless.stale_tail = lambda *a, **k: (True, "forced")
        relentless.run_clarify = lambda *a, **k: clar([ts("worth asking?", "yes, X")])
        relentless._attempt_partial_replan = lambda *a, **k: {"disposition": "exhausted"}
        ctx = FakeCtx()
        ledger, seen = [], set()
        plan = self._plan(tk("t1", "alfa"), tk("t2", "beta"))
        results, fresh = relentless.run_intent_path(
            ctx, 0, plan, "/fake/c0", 60, 60, ledger, seen, inp(), "/fake/slug")
        pre_replan_facts = [r for r in ledger if r["source"] == "replan-clarify"]
        self.assertEqual(len(pre_replan_facts), 1)
        self.assertIn("yes, X", pre_replan_facts[0]["text"])
        self.assertGreaterEqual(fresh, 1)  # the clarify fact counts toward fresh_harv

    def test_max_replans_per_cycle_caps_calls(self):
        self._always_worked()
        relentless.stale_tail = lambda *a, **k: (True, "forced")
        calls = []
        relentless._attempt_partial_replan = (
            lambda sd, s, c, seq, body, done, to, **kw: calls.append(seq) or
            {"schema": 2, "slug": s, "cycle": c, "disposition": "tasks", "rationale": "r",
             "question": None,
             "tasks": [tk(f"r{seq}a", f"ma{seq}"), tk(f"r{seq}b", f"mb{seq}")]})
        ctx = FakeCtx()
        plan = self._plan(*[tk(f"t{i}", f"m{i}") for i in range(6)])
        relentless.run_intent_path(ctx, 0, plan, "/fake/c0", 60, 60, [], set(), inp(),
                                   "/fake/slug")
        self.assertEqual(calls, [1, 2, 3])  # capped at MAX_REPLANS_PER_CYCLE
        self.assertEqual(len(calls), relentless.MAX_REPLANS_PER_CYCLE)

    def test_replan_failure_is_non_fatal(self):
        self._always_worked()
        relentless.stale_tail = lambda *a, **k: (True, "forced")
        relentless._attempt_partial_replan = (
            lambda *a, **k: {"disposition": "replan-failed", "error": "boom"})
        ctx = FakeCtx()
        plan = self._plan(tk("t1", "alfa"), tk("t2", "beta"))
        ledger, seen = [], set()
        results, fresh = relentless.run_intent_path(
            ctx, 0, plan, "/fake/c0", 60, 60, ledger, seen, inp(), "/fake/slug")
        self.assertEqual([r["id"] for r in results], ["t1", "t2"])  # ORIGINAL tail ran
        self.assertTrue(any("failed technically" in r["text"] for r in ledger))

    def test_needs_decision_from_replan_never_asks_even_under_gate(self):
        self._always_worked()
        relentless.stale_tail = lambda *a, **k: (True, "forced")
        relentless._attempt_partial_replan = (
            lambda *a, **k: {"disposition": "needs_decision", "question": "which way?"})

        class BoomCtx(FakeCtx):
            def ask(self, *a, **k):
                raise AssertionError("LEVEL 1 must never suspend, even under --gate")
        ctx = BoomCtx()
        plan = self._plan(tk("t1", "alfa"), tk("t2", "beta"))
        ledger, seen = [], set()
        results, fresh = relentless.run_intent_path(
            ctx, 0, plan, "/fake/c0", 60, 60, ledger, seen, inp(gate=True), "/fake/slug")
        self.assertEqual([r["id"] for r in results], ["t1", "t2"])  # tail UNCHANGED
        self.assertTrue(any("OPEN FORK (partial replan" in r["text"] for r in ledger))

    def test_exhausted_from_replan_stops_the_tail(self):
        self._always_worked()
        relentless.stale_tail = lambda *a, **k: (True, "forced")
        relentless._attempt_partial_replan = lambda *a, **k: {"disposition": "exhausted"}
        ctx = FakeCtx()
        plan = self._plan(tk("t1", "alfa"), tk("t2", "beta"))
        results, fresh = relentless.run_intent_path(
            ctx, 0, plan, "/fake/c0", 60, 60, [], set(), inp(), "/fake/slug")
        self.assertEqual([r["id"] for r in results], ["t1"])  # t2 never attempted


class JourneyTrace(LoopBase):
    """The decision TRACE the flow hands the journey fold — flow-level assertions on
    WHICH decisions get traced and in what order; the fold/render internals live in
    test_journey.py. Every plan-of-record change is a traced event: cycle plans,
    LEVEL 2 retries, delegation gates, LEVEL 1 partial replans."""

    def test_mid_cycle_decisions_are_traced_in_order(self):
        self.wire([clar([ts("q1", "a1")]),
                   clar([], stop="converged (no question above floor)")],
                  [pl(tk("t1", "alfa"), tk("t2", "beta")), pl(tk("t3", "gamma"))],
                  [{"t1": "failed"}, {}])
        orig_attempt = relentless._attempt_partial_replan
        relentless._attempt_partial_replan = lambda *a, **k: {
            "schema": 2, "slug": "s", "cycle": 0, "disposition": "tasks",
            "rationale": "new tail", "question": None, "tasks": [tk("t2b", "delta")]}
        try:
            out = relentless.relentless_flow(FakeCtx(), inp(local_retry_budget=1))
        finally:
            relentless._attempt_partial_replan = orig_attempt
        self.assertEqual(out["outcome"], "success")
        # t1 fails → one local retry (traced) → exhaustion → delegation gate (traced,
        # default fake declines) → retry-exhaustion staleness → partial replan (traced)
        # → new tail worked but t1 stays failed → cycle 1 plans fresh (traced) → success.
        self.assertEqual(self.journeys[0]["events"],
                         [("c0/plan", "plan"),
                          ("c0/t/t1/retry1", "retry"),
                          ("c0/t/t1/rp-delegate", "delegate"),
                          ("c0/replan/after-t1", "replan"),
                          ("c1/plan", "plan")])


class ReplayDeterminism(LoopBase):
    def _wire_two_cycle(self):
        self.wire([clar([ts("q1", "a1")]), clar([], stop="converged (no question above floor)")],
                  [pl(tk("t1", "alfa")), pl(tk("t1", "bravo"))],
                  [{"t1": "failed"}, {}])

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

    def test_replay_determinism_with_local_retry(self):
        # Single task per cycle (remaining always empty) so ONLY LEVEL 2's retry path is
        # exercised, not LEVEL 1's replan — that combination is covered by PartialReplan
        # above (which tests run_intent_path directly) plus this class's existing
        # coverage of a plain multi-cycle replay.
        self.wire([clar([ts("q1", "a1")]), clar([ts("q2", "a2")]),
                   clar([], stop="converged (no question above floor)")],
                  [pl(tk("t1", "alfa")), pl(tk("t1", "bravo"))],
                  [{"t1": "failed"}, {}])
        ctx1 = FakeCtx()
        out1 = relentless.relentless_flow(ctx1, inp(local_retry_budget=1))
        self.assertIn("c0/t/t1/retry1/clarify", ctx1.keys)
        self.assertIn("c0/t/t1/retry1", ctx1.keys)

        def boom(*a, **kw):
            raise AssertionError("replay must not re-execute any step")
        for n in self.PATCHED:
            setattr(relentless, n, boom)
        ctx2 = FakeCtx(completed=dict(ctx1.completed))
        out2 = relentless.relentless_flow(ctx2, inp(local_retry_budget=1))
        self.assertEqual(ctx2.executed, [])
        self.assertEqual(ctx2.keys, ctx1.keys)
        self.assertEqual(out1, out2)


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

    def test_fold_gap_kind(self):
        ledger, seen = [], set()
        relentless.fold_clarify([ts("q2", "no creds", status="NOT_FOUND")], 0, ledger, seen)
        self.assertEqual(ledger[0]["kind"], "gap")


class StopVocab(unittest.TestCase):
    """stop_is_converged: constant from the loaded investigator wins; substring is the
    fallback for replays (module never imported) and older investigators."""

    def tearDown(self):
        relentless._INVESTIGATOR_MOD = None

    def test_fallback_substring_when_module_absent(self):
        relentless._INVESTIGATOR_MOD = None
        self.assertTrue(relentless.stop_is_converged("converged (no question above floor)"))
        self.assertFalse(relentless.stop_is_converged("max_rounds reached"))
        self.assertFalse(relentless.stop_is_converged(None))

    def test_module_constant_takes_precedence(self):
        class FakeIterate:
            STOP_CONVERGED = "EVSI-converged"
        relentless._INVESTIGATOR_MOD = FakeIterate
        self.assertTrue(relentless.stop_is_converged("EVSI-converged: floor 0.12"))
        # bare substring no longer suffices once the module pins its own wording
        self.assertFalse(relentless.stop_is_converged("converged for other reasons"))


_DOD_TEXT = """# DoD: s   STATE: agreed
INTENT: both things hold at the end
REQUIREMENTS   (markers: ○ unmet · ✓ met (receipt) · ~ waived (receipted reason))
- R1   the outcome group                          [after: —]
  - R1.1  thing one holds                          ○
  - R1.2  thing two holds                          ○
"""

_HAVE_DD = os.path.isdir(os.path.abspath(os.path.join(_HERE, "..", "..",
                                                      "define-done", "scripts")))


@unittest.skipUnless(_HAVE_DD, "define-done skill not on disk (sibling)")
class DodWiring(LoopBase):
    """--dod end to end at the flow level: the dod travels as TEXT in the immutable
    input, plans get the dodctx, each executed cycle lands a completion report, and
    the final report carries the requirements rollup. No new step keys either way."""

    def _tk(self, tid, serves, method=None):
        return {**tk(tid, method), "serves": serves}

    def test_dod_cycle_reports_and_rolls_up(self):
        self.wire([clar([ts("q1", "a1")])],
                  [pl(self._tk("t1", ["R1.1"]), self._tk("t2", ["R1.2"]))])
        ctx = FakeCtx()
        out = relentless.relentless_flow(ctx, inp(dod=_DOD_TEXT))
        self.assertEqual(out["outcome"], "success")
        # same key sequence as the no-dod success test — no new step keys
        self.assertEqual(ctx.keys, ["t0", "c0/clock", "c0/clarify", "c0/render",
                                    "c0/plan", "c0/t/t1", "c0/t/t2", "c0/plan-out",
                                    "retro/journey", "report"])
        self.assertEqual(self.plan_dodctxs[0]["unmet"], ["R1.1", "R1.2"])
        self.assertIn("## Requirements (definition of done)", self.rendered[0])
        rep = self.reports_saved[0]
        self.assertEqual(rep["status"], "complete")
        self.assertEqual(rep["requirements"], {"R1.1": "met", "R1.2": "met"})
        self.assertEqual(self.reported["requirements"], {"R1.1": "met", "R1.2": "met"})

    def test_failed_task_blocks_its_requirement(self):
        self.wire([clar([]), clar([], stop="converged (no question above floor)")],
                  [pl(self._tk("t1", ["R1.1"], "alfa"), self._tk("t2", ["R1.2"]))],
                  [{"t1": "failed"}])
        relentless.relentless_flow(FakeCtx(), inp(dod=_DOD_TEXT, max_cycles=1))
        rep = self.reports_saved[0]
        self.assertEqual(rep["status"], "partial")
        self.assertEqual(rep["requirements"], {"R1.1": "blocked", "R1.2": "met"})
        # the report's delta is exactly the cycle's fresh knowledge (nothing passed in)
        self.assertEqual({r["kind"] for r in rep["delta"]}, {"dead-end", "fact"})

    def test_dead_fps_reach_the_next_cycles_plan_request(self):
        self.wire([clar([])],
                  [pl(self._tk("t1", ["R1.1", "R1.2"], "alfa")),
                   pl(self._tk("t1", ["R1.1", "R1.2"], "bravo"))],
                  [{"t1": "failed"}, {}])
        relentless.relentless_flow(FakeCtx(), inp(dod=_DOD_TEXT))
        self.assertEqual(self.plan_dead_fps[0], set())
        self.assertIn(harvest.fp("alfa"), self.plan_dead_fps[1])

    def test_no_dod_stays_dark(self):
        self.wire([clar([ts("q1", "a1")])], [pl(tk("t1"))])
        relentless.relentless_flow(FakeCtx(), inp())
        self.assertEqual(self.plan_dodctxs, [None])
        self.assertEqual(self.reports_saved, [None])
        self.assertIsNone(self.reported["requirements"])
        self.assertNotIn("Requirements", self.rendered[0])


class NeedsSplit(LoopBase):
    """The executor-declared granularity escalation: no local retry, no delegation,
    straight to a forced partial replan with the split hint in the ledger."""

    def _wire_split(self, replans):
        self.wire([clar([ts("q1", "a1")])], [pl(tk("t1", "alfa"), tk("t2", "beta"))])
        calls = {"n": 0}

        def fake_task(task, cycle_dir, timeout, suffix=None, capability=None):
            if task["id"] == "t1" and calls["n"] == 0:
                calls["n"] += 1
                return {"id": "t1", "method": "alfa", "verdict": "needs_split",
                        "evidence": "two systems in one task", "learnings": [],
                        "split": ["do the db half", "do the api half"]}
            return {"id": task["id"], "method": task["method"], "verdict": "worked",
                    "evidence": "ok", "learnings": [], "split": []}

        relentless.run_task = fake_task
        self.replan_bodies = []
        self._orig_apr = relentless._attempt_partial_replan

        def fake_replan(slug_dir, slug, cycle, seq, body, done_ids, timeout,
                        dodctx=None, dead_fps=()):
            self.replan_bodies.append(body)
            return replans[min(seq - 1, len(replans) - 1)]

        relentless._attempt_partial_replan = fake_replan

    def tearDown(self):
        relentless._attempt_partial_replan = getattr(self, "_orig_apr",
                                                     relentless._attempt_partial_replan)
        super().tearDown()

    def test_short_circuits_retry_and_delegation_and_forces_replan(self):
        self._wire_split([pl(tk("t1a", "db-half"), tk("t1b", "api-half"))])
        ctx = FakeCtx()
        out = relentless.relentless_flow(ctx, inp(local_retry_budget=2, max_cycles=1))
        # no LEVEL 2 machinery fired for the needs_split verdict...
        self.assertFalse([k for k in ctx.keys if "t/t1/retry" in k])
        self.assertFalse([k for k in ctx.keys if "rp-delegate" in k])
        self.assertEqual(self.delegation_calls, [])
        # ...but LEVEL 1's partial replan did, immediately after t1
        self.assertIn("c0/replan/after-t1/clarify", ctx.keys)
        self.assertIn("c0/replan/after-t1", ctx.keys)
        # the spliced tail ran (t2 was replaced by the split tasks)
        self.assertIn("c0/t/t1a", ctx.keys)
        self.assertIn("c0/t/t1b", ctx.keys)
        self.assertNotIn("c0/t/t2", ctx.keys)
        self.assertEqual(out["outcome"], "max-cycles")  # t1 itself never "worked"

    def test_split_hint_is_a_fact_and_reaches_the_replan_prompt(self):
        self._wire_split([pl(tk("t1a", "db-half"))])
        relentless.relentless_flow(FakeCtx(), inp(max_cycles=1))
        hints = [r for r in self.reported["ledger"]
                 if r["kind"] == "fact" and "SPLIT HINT" in r["text"]]
        self.assertEqual(len(hints), 1)
        self.assertEqual(hints[0]["fp"], harvest.fp("split alfa"))
        self.assertIn("do the db half", hints[0]["text"])
        self.assertTrue(self.replan_bodies)
        self.assertIn("SPLIT HINT", self.replan_bodies[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
