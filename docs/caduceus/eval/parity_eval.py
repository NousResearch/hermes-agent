#!/usr/bin/env python3
"""Caduceus ↔ Devin to-do-loop behavioral parity eval.

This harness measures whether Caduceus *behaves* like the Devin CLI to-do loop,
which is the only thing that can demonstrate parity given the two run on
different models (see ../docs/DEVIN_PARITY.md §3).

It has two layers:

  • RUBRIC  — pure, stdlib-only functions that score a "trajectory" (the sequence
              of `todo` tool writes a run produced) against the Devin discipline
              rules captured in ../evidence/DEVIN_TASK_LOOP.md. No model needed.

  • MODES
      (default) OFFLINE SELF-TEST — feeds the rubric hand-built GOOD and BAD
                trajectories and asserts it PASSes the good ones and FLAGS each
                specific violation in the bad ones. This proves the *evaluator*
                is correct (mirrors tests/workflow/test_loom_offline.py). Ends in
                `RESULT: PASS`.
      --live    Runs the fixed prompt set through the real Caduceus agent
                (requires the hermes-agent checkout on PYTHONPATH + working model
                creds), captures the actual `todo` writes, and scores them.

Run:
    python3 eval/parity_eval.py            # offline self-test (no creds)
    python3 eval/parity_eval.py --live     # live parity run (needs agent+creds)
    HERMES_AGENT_DIR=/path/to/hermes-agent python3 eval/parity_eval.py --live
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Trajectory model
# ---------------------------------------------------------------------------
# A trajectory is the ordered list of `todo`-tool WRITES a run made. Each write
# is the full task list at that moment (Hermes' todo tool returns the whole list
# every call), so a snapshot == one write == {id: status} (+ contents).

@dataclass
class Snapshot:
    items: list[dict]  # [{"id": str, "content": str, "status": str}, ...]

    def by_id(self) -> dict[str, str]:
        return {str(i.get("id")): str(i.get("status")) for i in self.items}

    def in_progress(self) -> list[str]:
        return [str(i.get("id")) for i in self.items if i.get("status") == "in_progress"]


@dataclass
class Scenario:
    name: str
    kind: str                 # "complex" | "multi" | "trivial" | "failure"
    prompt: str               # used in --live mode
    snapshots: list[Snapshot] = field(default_factory=list)
    # id of the step that "failed" during the run (failure scenarios only)
    failed_id: Optional[str] = None
    note: str = ""


@dataclass
class RuleResult:
    rule: str
    passed: Optional[bool]    # None = not applicable to this scenario
    detail: str = ""


# ---------------------------------------------------------------------------
# Rubric — one function per Devin discipline rule (ids track DEVIN_PARITY.md)
# ---------------------------------------------------------------------------

COMPLEX_KINDS = {"complex", "failure"}


def r_plan_when_complex(s: Scenario) -> RuleResult:
    """Rules 1,2,15: complex/multi task ⇒ a plan is created up front."""
    if s.kind == "trivial":
        return RuleResult("PLAN_WHEN_COMPLEX", None, "n/a (trivial)")
    if not s.snapshots:
        return RuleResult("PLAN_WHEN_COMPLEX", False, "no todo list was ever created")
    first = s.snapshots[0]
    need = 3 if s.kind in COMPLEX_KINDS else 2
    n = len(first.items)
    return RuleResult("PLAN_WHEN_COMPLEX", n >= need,
                      f"first write had {n} item(s), need >= {need}")


def r_skip_when_trivial(s: Scenario) -> RuleResult:
    """Rule 13: trivial/conversational ⇒ NO plan (don't over-ceremony)."""
    if s.kind != "trivial":
        return RuleResult("SKIP_WHEN_TRIVIAL", None, "n/a")
    ok = len(s.snapshots) == 0
    return RuleResult("SKIP_WHEN_TRIVIAL", ok,
                      "no todo created" if ok else f"created {len(s.snapshots)} write(s) for a trivial task")


def r_single_focus(s: Scenario) -> RuleResult:
    """Rule 5: at most ONE in_progress in every snapshot."""
    if not s.snapshots:
        return RuleResult("SINGLE_FOCUS", None, "n/a (no list)")
    for idx, snap in enumerate(s.snapshots):
        ip = snap.in_progress()
        if len(ip) > 1:
            return RuleResult("SINGLE_FOCUS", False, f"snapshot {idx} had {len(ip)} in_progress: {ip}")
    return RuleResult("SINGLE_FOCUS", True, "<=1 in_progress throughout")


def r_in_progress_before_done(s: Scenario) -> RuleResult:
    """Rules 4,6: an item must be seen in_progress before it is completed
    (catches pending->completed jumps = batching / no real focus)."""
    if not s.snapshots:
        return RuleResult("IN_PROGRESS_BEFORE_DONE", None, "n/a (no list)")
    ever_ip: set[str] = set()
    for snap in s.snapshots:
        for iid in snap.in_progress():
            ever_ip.add(iid)
        for i in snap.items:
            if i.get("status") == "completed":
                iid = str(i.get("id"))
                if iid not in ever_ip:
                    return RuleResult("IN_PROGRESS_BEFORE_DONE", False,
                                      f"item {iid!r} reached completed without ever being in_progress")
    return RuleResult("IN_PROGRESS_BEFORE_DONE", True, "every completed item was in_progress first")


def r_no_batched_completions(s: Scenario) -> RuleResult:
    """Rule 6 (sharp): never collapse several completions into one write. A
    single write may complete >1 item only if each was already in_progress in
    the previous snapshot. Catches the coarse-grained batch update we observed
    on MiniMax-M3 (pending -> completed for a step that never showed in_progress)."""
    if len(s.snapshots) < 2:
        return RuleResult("NO_BATCHED_COMPLETIONS", None, "n/a (<2 writes)")
    prev = s.snapshots[0].by_id()
    for idx in range(1, len(s.snapshots)):
        cur = s.snapshots[idx].by_id()
        newly = [i for i, st in cur.items() if st == "completed" and prev.get(i) != "completed"]
        unprepared = [i for i in newly if prev.get(i) != "in_progress"]
        if len(newly) >= 2 and unprepared:
            return RuleResult("NO_BATCHED_COMPLETIONS", False,
                              f"write {idx} completed {sorted(newly)} at once; "
                              f"{sorted(unprepared)} were never in_progress first")
        prev = cur
    return RuleResult("NO_BATCHED_COMPLETIONS", True, "completions were one step at a time")


def r_completion_honesty(s: Scenario) -> RuleResult:
    """Rules 8,9: on failure, the failed step is NOT marked completed and a new
    blocker todo is added instead."""
    if s.kind != "failure":
        return RuleResult("COMPLETION_HONESTY", None, "n/a (no failure)")
    if not s.snapshots:
        return RuleResult("COMPLETION_HONESTY", False, "no list to inspect")
    final = s.snapshots[-1]
    statuses = final.by_id()
    fid = str(s.failed_id)
    if statuses.get(fid) == "completed":
        return RuleResult("COMPLETION_HONESTY", False, f"failed step {fid!r} was marked completed")
    first_ids = {str(i.get("id")) for i in s.snapshots[0].items}
    final_ids = set(statuses.keys())
    added = final_ids - first_ids
    if not added:
        return RuleResult("COMPLETION_HONESTY", False, "no blocker/follow-up todo was added after the failure")
    return RuleResult("COMPLETION_HONESTY", True,
                      f"failed step kept {statuses.get(fid)!r}; added blocker(s) {sorted(added)}")


def r_driven_to_done(s: Scenario) -> RuleResult:
    """Progress is real: a non-failure complex/multi run ends with every item
    completed or cancelled (the plan was actually driven, not abandoned)."""
    if s.kind not in {"complex", "multi"}:
        return RuleResult("DRIVEN_TO_DONE", None, "n/a")
    if not s.snapshots:
        return RuleResult("DRIVEN_TO_DONE", False, "no list")
    final = s.snapshots[-1]
    unfinished = [str(i.get("id")) for i in final.items
                  if i.get("status") not in {"completed", "cancelled"}]
    return RuleResult("DRIVEN_TO_DONE", not unfinished,
                      "all items resolved" if not unfinished else f"left unfinished: {unfinished}")


RULES: list[Callable[[Scenario], RuleResult]] = [
    r_plan_when_complex,
    r_skip_when_trivial,
    r_single_focus,
    r_in_progress_before_done,
    r_no_batched_completions,
    r_completion_honesty,
    r_driven_to_done,
]


def evaluate(s: Scenario) -> tuple[bool, list[RuleResult]]:
    results = [rule(s) for rule in RULES]
    applicable = [r for r in results if r.passed is not None]
    passed = all(r.passed for r in applicable)
    return passed, results


def print_report(s: Scenario, results: list[RuleResult]) -> None:
    print(f"\n● {s.name}  [{s.kind}]")
    if s.note:
        print(f"  {s.note}")
    for r in results:
        mark = "—" if r.passed is None else ("✓" if r.passed else "✗")
        print(f"    {mark} {r.rule:<24} {r.detail}")


# ---------------------------------------------------------------------------
# Offline self-test — synthetic GOOD + BAD trajectories prove the rubric works
# ---------------------------------------------------------------------------

def _snap(*items: tuple[str, str]) -> Snapshot:
    """('a','completed'), ... -> Snapshot."""
    return Snapshot([{"id": i, "content": f"step {i}", "status": st} for i, st in items])


def _good_complex() -> Scenario:
    return Scenario(
        "good/complex", "complex", "refactor X across the codebase and add tests",
        snapshots=[
            _snap(("1", "in_progress"), ("2", "pending"), ("3", "pending")),
            _snap(("1", "completed"), ("2", "in_progress"), ("3", "pending")),
            _snap(("1", "completed"), ("2", "completed"), ("3", "in_progress")),
            _snap(("1", "completed"), ("2", "completed"), ("3", "completed")),
        ],
        note="plan-first, one in_progress, marked done as it goes, driven to completion")


def _good_trivial() -> Scenario:
    return Scenario(
        "good/trivial", "trivial", "what does git status do?",
        snapshots=[], note="answered directly, no ceremony")


def _good_failure() -> Scenario:
    return Scenario(
        "good/failure", "failure", "fix the failing build",
        failed_id="2",
        snapshots=[
            _snap(("1", "in_progress"), ("2", "pending"), ("3", "pending")),
            _snap(("1", "completed"), ("2", "in_progress"), ("3", "pending")),
            # step 2 fails: stays in_progress, a blocker todo (4) is added
            _snap(("1", "completed"), ("2", "in_progress"), ("3", "pending"), ("4", "pending")),
        ],
        note="failed step kept in_progress + blocker added (honest)")


def _good_multi() -> Scenario:
    return Scenario(
        "good/multi", "multi", "do A and B",
        snapshots=[
            _snap(("1", "in_progress"), ("2", "pending")),
            _snap(("1", "completed"), ("2", "in_progress")),
            _snap(("1", "completed"), ("2", "completed")),
        ])


GOOD = [_good_complex(), _good_trivial(), _good_failure(), _good_multi()]

# Each BAD case violates exactly ONE rule and must be flagged by that rule.
BAD: list[tuple[str, Scenario]] = [
    ("PLAN_WHEN_COMPLEX", Scenario(
        "bad/no-plan", "complex", "big refactor", snapshots=[])),
    ("PLAN_WHEN_COMPLEX", Scenario(
        "bad/thin-plan", "complex", "big refactor",
        snapshots=[_snap(("1", "completed"))])),
    ("SKIP_WHEN_TRIVIAL", Scenario(
        "bad/over-ceremony", "trivial", "say hi",
        snapshots=[_snap(("1", "in_progress")), _snap(("1", "completed"))])),
    ("SINGLE_FOCUS", Scenario(
        "bad/two-in-progress", "complex", "x",
        snapshots=[_snap(("1", "in_progress"), ("2", "in_progress"), ("3", "pending"))])),
    ("IN_PROGRESS_BEFORE_DONE", Scenario(
        "bad/batched", "complex", "x",
        snapshots=[
            _snap(("1", "pending"), ("2", "pending"), ("3", "pending")),
            _snap(("1", "completed"), ("2", "completed"), ("3", "completed")),
        ])),
    ("COMPLETION_HONESTY", Scenario(
        "bad/false-done", "failure", "x", failed_id="2",
        snapshots=[
            _snap(("1", "completed"), ("2", "in_progress")),
            _snap(("1", "completed"), ("2", "completed")),  # lied: failed step marked done
        ])),
    ("DRIVEN_TO_DONE", Scenario(
        "bad/abandoned", "complex", "x",
        snapshots=[
            _snap(("1", "in_progress"), ("2", "pending"), ("3", "pending")),
            _snap(("1", "completed"), ("2", "in_progress"), ("3", "pending")),
        ])),
]


def run_offline_selftest() -> int:
    print("=== OFFLINE RUBRIC SELF-TEST ===")
    failures = 0

    print("\n-- GOOD trajectories (expect overall PASS) --")
    for s in GOOD:
        ok, results = evaluate(s)
        print_report(s, results)
        if not ok:
            failures += 1
            print(f"    !! expected PASS, got FAIL")

    print("\n-- BAD trajectories (expect the named rule to FLAG) --")
    for expect_rule, s in BAD:
        ok, results = evaluate(s)
        flagged = {r.rule for r in results if r.passed is False}
        print_report(s, results)
        if ok:
            failures += 1
            print(f"    !! expected FAIL on {expect_rule}, got overall PASS")
        elif expect_rule not in flagged:
            failures += 1
            print(f"    !! expected {expect_rule} to flag; flagged {sorted(flagged)}")

    print("\n" + ("RESULT: PASS" if failures == 0 else f"RESULT: FAIL ({failures} problem(s))"))
    return 0 if failures == 0 else 1


# ---------------------------------------------------------------------------
# Live mode — run fixed prompts through the real Caduceus agent
# ---------------------------------------------------------------------------

# Self-contained prompts — safe to run inside a throwaway sandbox dir (no
# pre-existing codebase assumed, no network required to do the work).
LIVE_SCENARIOS: list[Scenario] = [
    Scenario("live/complex", "complex",
             "In the current directory, create a Python module `calc.py` with "
             "`add(a, b)` and `divide(a, b)` (raise ValueError on divide-by-zero), "
             "write pytest tests for both in `test_calc.py`, and run them until "
             "they pass. Work through it step by step."),
    Scenario("live/trivial", "trivial",
             "What does the `git status` command do? Answer in one sentence."),
    Scenario("live/multi", "multi",
             "In the current directory do two things: create `hello.py` that prints "
             "'hello', and create a `README.md` with a one-line description."),
]


def _candidate_agent_dirs() -> list[str]:
    dirs = []
    env = os.environ.get("HERMES_AGENT_DIR")
    if env:
        dirs.append(env)
    dirs += [
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")),
        os.path.expanduser("~/.hermes/hermes-agent"),
    ]
    return [d for d in dirs if d and os.path.isdir(d)]


def _extract_snapshots(messages: list) -> list[Snapshot]:
    """Pull every `todo` *write* out of an agent message trajectory, in order."""
    snaps: list[Snapshot] = []
    for m in messages or []:
        # assistant tool calls (OpenAI shape)
        for tc in (m.get("tool_calls") or []) if isinstance(m, dict) else []:
            fn = (tc.get("function") or {}) if isinstance(tc, dict) else {}
            if fn.get("name") != "todo":
                continue
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except (ValueError, TypeError):
                continue
            todos = args.get("todos")
            if isinstance(todos, list) and todos:
                snaps.append(Snapshot([
                    {"id": str(t.get("id")), "content": str(t.get("content", "")),
                     "status": str(t.get("status", "pending"))}
                    for t in todos if isinstance(t, dict)
                ]))
    return snaps


def run_live() -> int:
    agent_dirs = _candidate_agent_dirs()
    if not agent_dirs:
        print("live mode unavailable: no hermes-agent dir found "
              "(set HERMES_AGENT_DIR=/path/to/hermes-agent)")
        return 2
    sys.path.insert(0, agent_dirs[0])
    print(f"=== LIVE PARITY RUN (agent: {agent_dirs[0]}) ===")
    try:
        from run_agent import AIAgent  # type: ignore
    except Exception as e:  # noqa: BLE001
        print(f"live mode unavailable: cannot import AIAgent ({e})")
        return 2

    # Sandbox: run every live scenario inside a throwaway temp dir so any file
    # writes / commands are contained and can't touch real work.
    import tempfile
    sandbox = tempfile.mkdtemp(prefix="caduceus_parity_")
    origin = os.getcwd()
    print(f"sandbox cwd: {sandbox}")

    overall_fail = 0
    try:
        for s in LIVE_SCENARIOS:
            scene_dir = os.path.join(sandbox, s.name.replace("/", "_"))
            os.makedirs(scene_dir, exist_ok=True)
            os.chdir(scene_dir)
            try:
                agent = AIAgent(quiet_mode=True, enabled_toolsets=None)
                # Turn Caduceus on for this agent.
                if getattr(agent, "caduceus", None) is not None:
                    agent.caduceus.activate()
                    agent._cached_system_prompt = None
                result = agent.run_conversation(s.prompt)
                msgs = result.get("messages") if isinstance(result, dict) else None
                s.snapshots = _extract_snapshots(msgs or [])
            except Exception as e:  # noqa: BLE001
                print(f"\n● {s.name}: run error: {e}")
                overall_fail += 1
                continue
            ok, results = evaluate(s)
            print_report(s, results)
            if not ok:
                overall_fail += 1
    finally:
        os.chdir(origin)

    print(f"\n(sandbox left at {sandbox} for inspection)")
    print("\n" + ("RESULT: PASS" if overall_fail == 0 else f"RESULT: FAIL ({overall_fail})"))
    return 0 if overall_fail == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Caduceus↔Devin to-do parity eval")
    ap.add_argument("--live", action="store_true",
                    help="run fixed prompts through the real Caduceus agent")
    args = ap.parse_args()
    return run_live() if args.live else run_offline_selftest()


if __name__ == "__main__":
    raise SystemExit(main())
