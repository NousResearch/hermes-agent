#!/usr/bin/env python3
"""Unit tests for solve's routing, budget cascade, and receipts — no container, no engine.

FakeCtx + module-attribute monkeypatching, same style as test_loop.py. Run:
    python3 tests/test_solve.py
"""

import json
import os
import shutil
import sys
import tempfile
import types
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "scripts"))

import relentless  # noqa: E402
from test_loop import FakeCtx, clar  # noqa: E402 — reuse the established fakes

# solve_single builds its prompt via the method-explorer (fka resilient-planner)
# envelope; that seam is only testable where the sibling (or an env override) is on
# disk — skip, don't fail, like the contract suites. (Routing.setUp patches _HOME,
# so the deployed fallback is moot.)
_HAVE_RP = (bool(os.environ.get("METHOD_EXPLORER_DIR"))
            or bool(os.environ.get("RESILIENT_ENVELOPE_DIR"))
            or any(os.path.isdir(os.path.abspath(
                os.path.join(_HERE, "..", "..", skill, "scripts")))
                for skill in ("method-explorer", "resilient-planner")))


def solve_args(**kw):
    base = dict(prompt="probe task", prompt_file=None, budget=1800, risk="act",
                gate=False, slug=None, route=None, gate_only=False, answer_cwd=None,
                state_dir=None, accept_flow_change=False)
    base.update(kw)
    return types.SimpleNamespace(**base)


class CascadeMath(unittest.TestCase):
    """The cycle's share is recomputed from `remaining` at every cycle boundary and split
    between the planning oneshot (20%) and the per-task oneshots (80% / n_tasks)."""

    PATCHED = ("run_clarify", "request_plan", "run_task", "write_plan_receipt",
               "persist", "write_report", "_maybe_delegate_task", "build_journey",
               "run_hindsight")

    def _run_flow(self, *, cascade, wallclock, clocks, cycles, n_tasks=2, max_cycles=None,
                  local_retry_budget=0, extra_completed=None):
        """Drive relentless_flow with scripted clocks; cycles: 'fail'|'success' per cycle.
        local_retry_budget=0 by default so these math-only scenarios never actually
        exercise LEVEL 2's retry loop (fake_task/fake_clarify below don't script retries).
        extra_completed: additional FakeCtx.completed entries (e.g. a mid-cycle
        replan-boundary clock key) for tests that DO force retry/replan paths under
        cascade=True — deadline checks there issue a fresh, un-scripted time.time() read
        unless pre-seeded in the same synthetic clock domain as `clocks`.
        Returns ([plan timeout per cycle], [task timeout per run_task call])."""
        cap_plan, cap_task = [], []
        orig = {n: getattr(relentless, n) for n in self.PATCHED}
        state = {"p": 0}

        def fake_plan(slug_dir, slug, cycle, rendered, timeout, dodctx=None,
                      dead_fps=()):
            cap_plan.append(timeout)
            state["p"] += 1
            return {"schema": 1, "slug": slug, "cycle": cycle, "disposition": "tasks",
                    "rationale": "r", "question": None,
                    "tasks": [{"id": f"t{i}", "method": f"m{cycle}-{i}", "description": "d",
                               "success_criterion": "c", "depends_on": [],
                               "status": "pending"}
                              for i in range(n_tasks)]}

        def fake_task(task, cycle_dir, timeout, capability=None):
            cap_task.append(timeout)
            v = "worked" if cycles[state["p"] - 1] == "success" else "failed"
            return {"id": task["id"], "method": task["method"], "verdict": v,
                    "evidence": "e"}

        relentless.run_clarify = lambda p, s, i, run_dir=None: clar([])
        relentless.request_plan = fake_plan
        relentless.run_task = fake_task
        relentless.write_plan_receipt = lambda d, p, r, report=None: "/dev/null/plan.json"
        relentless.persist = lambda d, c, r, l: {"prompt_path": f"/dev/null/p{c}"}
        relentless.write_report = lambda d, o, l, c, dt, **kw: "/dev/null/report"
        relentless.build_journey = lambda d, s, v, dt, rc, tr, l: {
            "schema": 1, "slug": s, "verdict": v, "nodes": [], "path": [],
            "hindsight": None}
        # cascade=True + success reaches retro/clock; the REAL time.time() there is far
        # past these synthetic clocks, so `remaining` is negative and the judge is
        # budget-skipped — this fake exists to fail loudly if that ever changes.
        relentless.run_hindsight = lambda j, d, to: (_ for _ in ()).throw(
            AssertionError("hindsight must be budget-skipped under synthetic clocks"))
        # Defensive default — these math-only scenarios don't intend to reach LEVEL 2's
        # delegation escalation, but never let a future test accidentally hit the real
        # invoke_hermes/run_drive path if it does.
        relentless._maybe_delegate_task = (
            lambda slug, t, r, rm, cd, dcfg, risk, inp, oneshot=None, drive=None:
                {"attempted": False, "gate": {"alt_methods_plausible": False,
                                              "why": "test default: never attempt"}})
        completed = {"t0": 1000.0}
        for i, t in enumerate(clocks):
            completed[f"c{i}/clock"] = 1000.0 + t
        completed.update(extra_completed or {})
        try:
            inp = {"prompt": "x", "slug": "s", "wallclock": wallclock, "cascade": cascade,
                   "max_cycles": max_cycles or len(clocks),
                   "local_retry_budget": local_retry_budget}
            relentless.relentless_flow(FakeCtx(completed=completed), inp)
        finally:
            for n, f in orig.items():
                setattr(relentless, n, f)
        return cap_plan, cap_task

    def test_shares_recompute_and_flow_back(self):
        # 10000s over 8 cycles, 2 tasks each, local_retry_budget=0 (attempts_per_task=1);
        # cycle 0 at t=0 → share 10000/8 = 1250: plan min(300,250)=250,
        # task 1250*0.7/(2*1) = 437.5 -> 437. Cycle 1 at t=1000 → share 9000/7 ≈ 1285.7:
        # plan 257, task 1285.7*0.7/2 ≈ 449.99 -> 450 — the cheap cycle flowed back.
        plans, tasks = self._run_flow(cascade=True, wallclock=10000, max_cycles=8,
                                      clocks=[0, 1000], cycles=["fail", "success"])
        self.assertEqual(plans, [250, 257])
        self.assertEqual(tasks, [437, 437, 450, 450])

    def test_task_timeout_divides_by_attempts_per_task(self):
        # Same share as test_shares_recompute_and_flow_back's cycle 0 (1250), but
        # local_retry_budget=2 -> attempts_per_task=3: task_to = 1250*0.7/(2*3) ≈ 145.8
        # -> 145, versus 437 with attempts_per_task=1 — the per-attempt share shrinks so
        # a cycle's total task-attempt budget stays bounded regardless of retry depth.
        # cycles=["success"] so no task ever actually fails/retries in this scenario —
        # only the TIMEOUT FORMULA is under test here, not the retry loop itself.
        _, tasks = self._run_flow(cascade=True, wallclock=10000, max_cycles=8,
                                  clocks=[0], cycles=["success"], local_retry_budget=2)
        self.assertEqual(tasks, [145, 145])

    def test_floor_and_ceiling(self):
        # Tiny share → the floors; huge budget → the caps.
        lo = self._run_flow(cascade=True, wallclock=600, max_cycles=2, clocks=[0],
                            cycles=["success"], n_tasks=4)  # share 300: plan 60, task 60→120
        hi = self._run_flow(cascade=True, wallclock=10 ** 6, max_cycles=2, clocks=[0],
                            cycles=["success"])
        self.assertEqual((lo[0][0], lo[1][0]),
                         (relentless.PLAN_TO_FLOOR, relentless.TASK_TO_FLOOR))
        self.assertEqual((hi[0][0], hi[1][0]),
                         (relentless.PLAN_TO_CAP, relentless.TASK_TO_CAP))

    def test_no_cascade_is_legacy_static(self):
        plans, tasks = self._run_flow(cascade=False, wallclock=10000, clocks=[0],
                                      cycles=["success"])
        self.assertEqual(plans[0], relentless.DEFAULTS["plan_timeout"])
        self.assertEqual(tasks[0], relentless.DEFAULTS["task_timeout"])

    def test_replan_share_floor_and_ceiling(self):
        # Force LEVEL 1's staleness gate to fire on the first task (so a partial replan
        # is requested) and capture request_partial_replan's timeout via a fake that
        # returns "exhausted" (a valid, non-splicing disposition) so the flow ends
        # cleanly. n_tasks=2 so `remaining` is non-empty after the first task.
        cap_replan = []
        orig_stale, orig_attempt = relentless.stale_tail, relentless._attempt_partial_replan
        relentless.stale_tail = lambda *a, **k: (True, "forced-for-test")
        relentless._attempt_partial_replan = (
            lambda sd, s, c, seq, body, done, timeout, **kw:
                cap_replan.append(timeout) or {"disposition": "exhausted"})
        # cycle 0's "now" is 1000.0 (t0 + clocks[0]); pre-seed the replan-boundary clock
        # to the SAME synthetic instant so the deadline check reads "no time elapsed"
        # rather than a real (huge) time.time() value.
        seed = {"c0/replan/after-t0/clock": 1000.0}
        try:
            self._run_flow(cascade=True, wallclock=600, max_cycles=2, clocks=[0],
                           cycles=["success"], n_tasks=2,
                           extra_completed=seed)  # tiny share → the floor
            self._run_flow(cascade=True, wallclock=10 ** 6, max_cycles=2, clocks=[0],
                           cycles=["success"], n_tasks=2,
                           extra_completed=seed)  # huge budget → the cap
        finally:
            relentless.stale_tail, relentless._attempt_partial_replan = (
                orig_stale, orig_attempt)
        self.assertEqual(cap_replan, [relentless.REPLAN_TO_FLOOR, relentless.REPLAN_TO_CAP])


class GateNoteReceipt(unittest.TestCase):
    def test_gate_note_is_first_ledger_row(self):
        seen_ledger = []
        names = ("run_clarify", "request_plan", "run_task", "write_plan_receipt",
                 "persist", "write_report", "build_journey", "run_hindsight")
        orig = {n: getattr(relentless, n) for n in names}
        relentless.run_clarify = lambda p, s, i, run_dir=None: clar([])
        relentless.request_plan = lambda d, s, c, r, t, **kw: {
            "schema": 1, "slug": s, "cycle": c, "disposition": "tasks", "rationale": "r",
            "question": None,
            "tasks": [{"id": "t1", "method": "m", "description": "d",
                       "success_criterion": "c", "depends_on": [], "status": "pending"}]}
        relentless.run_task = lambda t, d, to, capability=None: {"id": t["id"], "method": t["method"],
                                                "verdict": "worked", "evidence": "e"}
        relentless.write_plan_receipt = lambda d, p, r, report=None: "/dev/null/plan.json"
        relentless.persist = lambda d, c, r, l: {"prompt_path": "/dev/null/p"}
        relentless.write_report = (lambda d, o, ledger, c, dt, **kw:
                                   seen_ledger.extend(ledger))
        relentless.build_journey = lambda d, s, v, dt, rc, tr, l: {
            "schema": 1, "slug": s, "verdict": v, "nodes": [], "path": [],
            "hindsight": None}
        relentless.run_hindsight = lambda j, d, to: {"skipped": "test"}
        try:
            relentless.relentless_flow(
                FakeCtx(completed={"t0": 0.0, "c0/clock": 0.0}),
                {"prompt": "x", "slug": "s", "max_cycles": 1,
                 "gate_note": "GATE: route=full (model) — multi-method"})
        finally:
            for n, f in orig.items():
                setattr(relentless, n, f)
        self.assertTrue(seen_ledger and seen_ledger[0]["source"] == "gate"
                        and seen_ledger[0]["text"].startswith("GATE:"),
                        f"gate verdict must be the ledger's first receipt: {seen_ledger[:1]}")


class Routing(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="solve-test-")
        self._home, self._plans = relentless._HOME, relentless.PLANS_DIR
        relentless._HOME = self.tmp
        relentless.PLANS_DIR = os.path.join(self.tmp, "plans")
        self.ran = []

    def tearDown(self):
        relentless._HOME, relentless.PLANS_DIR = self._home, self._plans
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_trivial_short_circuits_all_machinery(self):
        drives = []
        orig_d, orig_o = relentless.run_drive, relentless.run_oneshot
        relentless.run_drive = lambda *a, **k: drives.append(a) or {"status": "SUCCESS"}
        relentless.run_oneshot = lambda p, timeout=0: "42."
        try:
            rc = relentless.cmd_solve(solve_args(route="trivial"),
                                      engine_run=lambda *a: self.ran.append("engine"))
        finally:
            relentless.run_drive, relentless.run_oneshot = orig_d, orig_o
        self.assertEqual(rc, 0)
        self.assertEqual((self.ran, drives), ([], []),
                         "trivial route must touch neither the engine nor drive")
        report = open(os.path.join(self.tmp, "relentless",
                                   relentless.derive_slug("probe task"), "report.md")).read()
        for token in ("SLUG:", "ROUTE: trivial", "BUDGET: total=1800s", "RISK: act",
                      "STOP: answered", "42."):
            self.assertIn(token, report, f"receipt header missing {token!r}")
        # one journey schema for EVERY route: the loopless routes write a degenerate
        # one-decision chain (plan node + terminal), never a special case downstream
        with open(os.path.join(self.tmp, "relentless",
                               relentless.derive_slug("probe task"),
                               "journey.json")) as fh:
            j = json.load(fh)
        self.assertEqual((j["schema"], j["verdict"], j["receipts"]["route"]),
                         (1, "answered", "trivial"))
        self.assertEqual([n["kind"] for n in j["nodes"]], ["plan", "terminal"])
        self.assertIn("42.", j["nodes"][0]["evidence"][0]["text"])
        with open(os.path.join(self.tmp, "relentless",
                               relentless.derive_slug("probe task"), "solve.json")) as fh:
            solved = json.load(fh)
        self.assertEqual(solved["artifacts"]["journey"],
                         os.path.join(self.tmp, "relentless",
                                      relentless.derive_slug("probe task"), "journey.json"))
        self.assertTrue(os.path.exists(solved["artifacts"]["journey"]))

    @unittest.skipUnless(_HAVE_RP, "method-explorer envelope not on disk")
    def test_single_method_drives_once_no_engine(self):
        drives = []
        orig_d = relentless.run_drive
        relentless.run_drive = lambda slug, pp, dcfg: (
            drives.append((slug, dcfg)) or {"status": "SUCCESS", "detail": "done"})
        try:
            rc = relentless.cmd_solve(solve_args(route="single_method", risk="read",
                                                 budget=900),
                                      engine_run=lambda *a: self.ran.append("engine"))
        finally:
            relentless.run_drive = orig_d
        self.assertEqual((rc, self.ran, len(drives)), (0, [], 1))
        slug, dcfg = drives[0]
        self.assertEqual(dcfg["wallclock"], 840, "single_method gets all-but-60s of budget")
        prompt = open(os.path.join(self.tmp, "relentless",
                                   relentless.derive_slug("probe task"),
                                   "prompt-single.md")).read()
        self.assertIn("method-explorer", prompt)
        self.assertIn("read-only", prompt, "risk=read must pin a read-only HARD constraint")
        slug_dir = os.path.join(self.tmp, "relentless",
                                relentless.derive_slug("probe task"))
        with open(os.path.join(slug_dir, "solve.json")) as fh:
            solved = json.load(fh)
        self.assertEqual(solved["artifacts"]["journey"],
                         os.path.join(slug_dir, "journey.json"))
        self.assertTrue(os.path.exists(solved["artifacts"]["journey"]))

    def test_full_route_maps_risk_budget_cascade_into_engine_input(self):
        got = {}

        def engine_run(inp, slug, args):
            got.update(inp)
            return 0
        rc = relentless.cmd_solve(solve_args(route="full", risk="read", budget=2400),
                                  engine_run=engine_run)
        self.assertEqual(rc, 0)
        self.assertEqual(got["capability"], "read", "risk must map to clarify capability")
        self.assertEqual(got["wallclock"], 2400, "budget is the outer wallclock pool")
        self.assertTrue(got["cascade"], "full route must enable the budget cascade")
        self.assertTrue(got["gate_note"].startswith("GATE: route=full"),
                        "gate verdict must ride into the ledger")

    def test_full_route_does_not_surface_stale_journey_without_terminal_result(self):
        slug = relentless.derive_slug("probe task")
        slug_dir = os.path.join(self.tmp, "relentless", slug)
        os.makedirs(slug_dir, exist_ok=True)
        stale = os.path.join(slug_dir, "journey.json")
        with open(stale, "w", encoding="utf-8") as fh:
            json.dump({"stale": True}, fh)
        rc = relentless.cmd_solve(solve_args(route="full"),
                                  engine_run=lambda *args: 0)
        self.assertEqual(rc, 0)
        with open(os.path.join(slug_dir, "solve.json")) as fh:
            solved = json.load(fh)
        self.assertTrue(os.path.exists(stale))
        self.assertIsNone(solved["artifacts"]["journey"])

    def test_resume_refresh_uses_terminal_result_and_gates_journey(self):
        slug = "resume-probe"
        slug_dir = os.path.join(self.tmp, "relentless", slug)
        state_dir = os.path.join(slug_dir, "flow")
        os.makedirs(state_dir, exist_ok=True)
        verdict = {"slug": slug, "route": "full", "risk": "act", "source": "flag",
                   "why": "test", "budget": {"total": 100}}
        with open(os.path.join(slug_dir, "gate.json"), "w", encoding="utf-8") as fh:
            json.dump(verdict, fh)
        journey_path = os.path.join(slug_dir, "journey.json")
        with open(journey_path, "w", encoding="utf-8") as fh:
            json.dump({"schema": 1}, fh)
        result = {"outcome": "success", "detail": "done", "cycles": 2,
                  "report": os.path.join(slug_dir, "report.md")}
        with open(os.path.join(state_dir, "state.json"), "w", encoding="utf-8") as fh:
            json.dump({"result": result}, fh)
        relentless._refresh_solve_json_after_resume(slug, state_dir)
        with open(os.path.join(slug_dir, "solve.json")) as fh:
            solved = json.load(fh)
        self.assertEqual(solved["outcome"], "success")
        self.assertEqual(solved["artifacts"]["journey"], journey_path)

    def test_resume_refresh_ignores_trivial_gate(self):
        slug = "trivial-resume"
        slug_dir = os.path.join(self.tmp, "relentless", slug)
        os.makedirs(slug_dir, exist_ok=True)
        with open(os.path.join(slug_dir, "gate.json"), "w", encoding="utf-8") as fh:
            json.dump({"slug": slug, "route": "trivial"}, fh)
        relentless._refresh_solve_json_after_resume(slug, os.path.join(slug_dir, "flow"))
        self.assertFalse(os.path.exists(os.path.join(slug_dir, "solve.json")))


if __name__ == "__main__":
    unittest.main(verbosity=2)
