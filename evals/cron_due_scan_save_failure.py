#!/usr/bin/env python3
"""Repro: a store that cannot be written takes down the whole cron tick.

`cron/jobs.py::_get_due_jobs_locked` runs scan-time self-heals (half-paused jobs self-disable, a
completed-one-shot retention sweep, record normalization) and then persists them with a bare
`save_jobs(...)` on the way out, before returning the due list.  Any write failure there — ENOSPC
on a full disk, a read-only mount, EACCES, a locked store — raises out of `get_due_jobs()`, so
`cron_tick()` aborts and NO job on that profile runs until a write succeeds.  The ironic part: the
normalization directly above it exists so a malformed store cannot abort the scan (its docstring:
"used to abort the whole scan before save_jobs(), freezing the scheduler in a fast-forward loop"),
yet the save it performs has no such containment.

Observed live on v0.21.5 with /opt/data full (2026-10-04):

    ERROR cron.scheduler_provider: Cron tick error for profile at /opt/data: [Errno 28] No space left on device
      File "/opt/hermes/cron/jobs.py", line 3240, in _get_due_jobs_locked
        save_jobs(raw_jobs, removed_ids=scan.removed or None)
      ...
      File "/opt/hermes/cron/jobs.py", line 1542, in save_jobs
        _save_jobs_unlocked(jobs, removed_ids=removed_ids, replace=replace)
    OSError: [Errno 28] No space left on device

repeating on every tick (18:54, 19:00, 19:09, 19:14 …) with every job on the profile reporting
"missed its scheduled time" and re-anchoring.

THE SCAN IS ONLY HALF THE TICK.  `tick()` then calls `advance_next_runs()` to take the recurring
occurrence off the schedule BEFORE dispatch (`cron/scheduler_tick.py` line 83), and that persist
raises under the same conditions — so even with the scan contained, the tick still aborts before
`_submit_with_guard()` and the profile stops firing anyway.  Reviewing PR #132933, maintainer
ehz0ah reproduced exactly that at head 76676fb0084f06890f76280f50387325ad2d59c7:

    scan_returned=['due-job']
    tick_raised=28:No space left on device
    submitted=[]

The repair/advance is already applied in memory — that is what the scan and the same-process dedupe
key off — so the persist is a side effect a later tick can retry.  This eval drives the full tick
(`cron.scheduler.tick`, `sync=True`) so a fix for the scan alone cannot pass it.

Run against a checkout (or the installed tree):

    HERMES_REPO=<repo-or-/opt/hermes> python3 evals/cron_due_scan_save_failure.py

Legs, each in its OWN subprocess (module state is fixed at import):

  baseline      writable store  → the due occurrence is dispatched.
  advance-fail  ONLY the schedule-advance persist fails (the reviewer's repro, isolated from the
                scan's persist): before → tick raises, nothing submitted; after → dispatched once,
                and a second tick does not double-fire it.
  full-disk     RLIMIT_FSIZE caps every store write (a real ENOSPC-class condition): before → tick
                raises; after → the tick no longer raises and the profile keeps ticking. Dispatched
                may still be empty here — correctly so: the run's durable fire claim fails CLOSED
                while the store is unwritable, which is what keeps at-most-once intact.
  claim-fail    ONLY the fire claim's persist fails, with a real (unwritable) execution ledger in
                play. The occurrence must NOT execute (fail-closed) — and, on the fixed tree, the
                receipt the dispatch created must reach a terminal state instead of sitting
                `claimed` forever, which is the leak this leg guards.

The "before" leg is a genuinely pristine tree: this script resets every file this PR fixes
(`cron/jobs.py`, `cron/scheduler.py`) to the merge-base with `origin/main` and refuses the leg
unless NONE of the fix markers are present.

Exit code: 0 when the fixed tree degrades instead of aborting, 1 when the tick still raises.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

sys.path.insert(0, os.environ.get("HERMES_REPO", "/opt/hermes"))

# One marker per file this PR fixes; the "before" leg reverts all of them together.
_FIX_MARKERS = {
    "cron/jobs.py": "cron.schedule_advance.persist_failed",
    "cron/scheduler.py": "_settle_unstarted_execution",
}
_FIXED_PATHS = tuple(_FIX_MARKERS)
# The recurring job the seeded store makes due; the child script seeds the same id.
adv_job_id = "due000000001"

# Well above a couple of SQLite pages (so what fails is the jobs.json payload and the store
# rewrites, not the execution ledger) and well below the 60 KB prompt the seed writes.
_FSIZE_LIMIT = 32768

_CHILD = r'''
import errno, json, os, resource, signal, sys, tempfile
from datetime import datetime
from pathlib import Path

repo, mode = sys.argv[1], sys.argv[2]
sys.path.insert(0, repo)
home = Path(tempfile.mkdtemp(prefix="cron-tick-ab-"))
os.environ["HERMES_HOME"] = str(home)

from cron import jobs as cronjobs
import cron.scheduler as sched

# ON the schedule's grid (whole minute, local offset): a next_run_at off the grid or in another
# offset is treated as stale / a timezone shift and re-anchored instead of fired.  This minute's
# instant is inside the 120s grace a "* * * * *" job gets.
due_at = datetime.now().astimezone().replace(second=0, microsecond=0).isoformat()
DUE, HALF = "due000000001", "half00000001"
cron_dir = home / "cron"
(cron_dir / "output").mkdir(parents=True, exist_ok=True)
store = cronjobs._current_cron_store().jobs_file
base = {
    "schedule": {"kind": "cron", "expr": "* * * * *", "display": "every minute"},
    "deliver": "local", "script": None, "repeat": None, "state": "active",
    "created_at": due_at, "last_run_at": None, "prompt": "noop " + "x" * 60000,
}
store.write_text(json.dumps({"jobs": [
    # The job the operator needs to fire this tick.
    dict(base, id=DUE, name="due job", enabled=True, next_run_at=due_at),
    # enabled=true with pause markers: _self_disable_half_paused() repairs it mid-scan and sets
    # needs_save, which is what makes the scan write the store on its way out.
    dict(base, id=HALF, name="half paused", enabled=True, next_run_at=due_at,
         paused_at=due_at, paused_reason="seed"),
]}, indent=2), encoding="utf-8")

# ── Fault injection ────────────────────────────────────────────────────────────────────────────
# The tick's run boundary is stubbed so a full tick runs without a model or a delivery.  The REAL
# create_execution is kept (wrapped only to record it): a recorded call proves the tick reached
# dispatch, and the real ledger row is what lets this eval assert the receipt reached a terminal
# state.  claim_job_for_fire is NOT stubbed — the real fire claim is the at-most-once gate this
# eval is about.
from cron.executions import list_executions
dispatched = []
executed = []
_real_create_execution = sched.create_execution
def _record_execution(job_id, **_kw):
    dispatched.append(job_id)
    return _real_create_execution(job_id, **_kw)
sched.create_execution = _record_execution
sched.run_one_job = lambda job, **_kw: executed.append(job["id"]) or True
sched._running_job_ids.clear()


def _open_receipts():
    try:
        rows = list_executions(limit=100)
    except Exception as exc:
        return ["unreadable: %s: %s" % (type(exc).__name__, exc)]
    return [{"id": r["id"], "job_id": r["job_id"], "status": r["status"]}
            for r in rows if r["status"] in ("claimed", "running")]

previous = None
real_save = cronjobs.save_jobs
if mode == "advance-fail":
    # Fails ONLY the tick's schedule-advance persist: the store is otherwise perfectly writable.
    # The advance is distinguishable from the scan's repair persist by its payload — the due job's
    # next_run_at has moved off the seeded instant.
    fired = {"done": False}
    def _advance_only(jobs, *a, **kw):
        due = next((j for j in jobs if j.get("id") == DUE), None)
        if due is not None and due.get("next_run_at") != due_at and not fired["done"]:
            fired["done"] = True
            raise OSError(errno.ENOSPC, "No space left on device")
        return real_save(jobs, *a, **kw)
    cronjobs.save_jobs = _advance_only
elif mode == "claim-fail":
    # Fails ONLY the fire claim's persist — the one store write whose payload carries ``fire_claim``
    # (the scan's repair save and the schedule advance do not). The claim is what fails CLOSED while
    # the store is unwritable, so the occurrence must NOT execute; the point of this leg is that its
    # execution receipt still reaches a terminal state instead of sitting ``claimed`` forever.
    def _claim_only(jobs, *a, **kw):
        if any(isinstance(j.get("fire_claim"), dict) for j in jobs):
            raise OSError(errno.ENOSPC, "No space left on device")
        return real_save(jobs, *a, **kw)
    cronjobs.save_jobs = _claim_only
elif mode == "full-disk":
    if os.name != "posix":
        print(json.dumps({"skip": "RLIMIT_FSIZE is POSIX-only"})); raise SystemExit(0)
    soft, hard = resource.getrlimit(resource.RLIMIT_FSIZE)
    signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
    # hard == -1 is RLIM_INFINITY: cap the SOFT limit below it rather than min()-ing against -1.
    cap = 32768 if hard < 0 else min(32768, hard)
    resource.setrlimit(resource.RLIMIT_FSIZE, (cap, hard))
    previous = (soft, hard)

out = {"mode": mode}
try:
    # The reviewer's sequence: the scan first (its repairs persist), then the production tick.
    try:
        out["scan_returned"] = [j.get("name") for j in cronjobs.get_due_jobs()]
    except Exception as exc:
        out["scan_returned"] = None
        out["scan_error"] = f"{type(exc).__name__}: {exc}"

    try:
        sched.tick(verbose=False, sync=True)
        out["tick_raised"] = None
    except BaseException as exc:
        out["tick_raised"] = f"{getattr(exc, 'errno', None)}:{exc}"
    out["dispatched"] = list(dispatched)
    out["open_receipts"] = _open_receipts()

    try:
        sched.tick(verbose=False, sync=True)
        out["second_tick_raised"] = None
    except BaseException as exc:
        out["second_tick_raised"] = f"{getattr(exc, 'errno', None)}:{exc}"
    out["dispatched_after_second_tick"] = list(dispatched)
    out["open_receipts_after_second_tick"] = _open_receipts()

    if mode == "full-disk":
        # The store becomes writable again: the occurrence must run, and only once, because the
        # fire claim (which fails closed while the store is broken) is the real at-most-once gate.
        out["executed_while_unwritable"] = list(executed)
        resource.setrlimit(resource.RLIMIT_FSIZE, previous)
        previous = None
        try:
            sched.tick(verbose=False, sync=True)
            out["recovery_tick_raised"] = None
        except BaseException as exc:
            out["recovery_tick_raised"] = f"{getattr(exc, 'errno', None)}:{exc}"
        out["executed_after_recovery"] = list(executed)
        out["open_receipts_after_recovery"] = _open_receipts()
        try:
            sched.tick(verbose=False, sync=True)
            out["post_recovery_tick_raised"] = None
        except BaseException as exc:
            out["post_recovery_tick_raised"] = f"{getattr(exc, 'errno', None)}:{exc}"
        out["executed_after_post_recovery_tick"] = list(executed)
    out["executed"] = list(executed)
out["open_receipts_final"] = _open_receipts()
finally:
    if previous is not None:
        resource.setrlimit(resource.RLIMIT_FSIZE, previous)

print(json.dumps(out))
'''


def _tick_leg(repo_root: str, mode: str) -> dict:
    proc = subprocess.run([sys.executable, "-c", _CHILD, repo_root, mode],
                          capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        print(proc.stderr[-2000:])
        raise SystemExit(f"REPRO INVALID: tick child failed (rc={proc.returncode})")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _marker_present(repo_root: str) -> bool:
    """True when ANY file this PR fixes still carries its fix marker."""
    for rel, marker in _FIX_MARKERS.items():
        try:
            with open(os.path.join(repo_root, *rel.split("/")), encoding="utf-8") as fh:
                if marker in fh.read():
                    return True
        except OSError:
            return False
    return False


def _render(label: str, leg: dict) -> str:
    parts = [f"scan_returned={leg.get('scan_returned')!r}"]
    if leg.get("scan_error"):
        parts.append(f"scan_raised={leg['scan_error']}")
    parts.append(f"tick_raised={leg.get('tick_raised')!r}")
    parts.append(f"dispatched={leg.get('dispatched')!r}")
    parts.append(f"dispatched_after_second_tick={leg.get('dispatched_after_second_tick')!r}")
    parts.append(f"executed={leg.get('executed')!r}")
    parts.append(f"open_receipts={leg.get('open_receipts')!r}")
    if "open_receipts_final" in leg:
        parts.append(f"open_receipts_final={leg.get('open_receipts_final')!r}")
    if "executed_while_unwritable" in leg:
        parts.append(f"executed_while_unwritable={leg.get('executed_while_unwritable')!r}")
    if "executed_after_recovery" in leg:
        parts.append(f"executed_after_recovery={leg.get('executed_after_recovery')!r}")
        parts.append(
            f"executed_after_post_recovery_tick={leg.get('executed_after_post_recovery_tick')!r}")
    return f"  {label:<8} " + "  ".join(parts)


def main() -> int:
    repo_root = os.environ.get("HERMES_REPO", "/opt/hermes")
    print(f"repo: {repo_root}")
    print(f"fix marker present at start: {_marker_present(repo_root)}\n")

    legs = ("baseline", "advance-fail", "claim-fail", "full-disk")
    results: dict = {}
    exit_code = 0

    for phase in ("before", "after"):
        if phase == "before":
            # Only meaningful for a committed fix; with the working tree already patched, git
            # checkout would discard the uncommitted change, so the caller drives that leg by hand.
            proc = subprocess.run(
                ["git", "-C", repo_root, "diff", "--quiet", "--", *_FIXED_PATHS])
            if proc.returncode != 0:
                print("before   SKIPPED — the tree has uncommitted changes to files this PR fixes; "
                      "run this script against a committed fix (it takes the merge-base itself), "
                      "or take the pristine revision by hand.\n")
                continue
            subprocess.run(
                ["git", "-C", repo_root, "checkout", "origin/main", "--", *_FIXED_PATHS],
                check=True)
            if _marker_present(repo_root):
                subprocess.run(
                    ["git", "-C", repo_root, "checkout", "HEAD", "--", *_FIXED_PATHS],
                    check=True)
                print("REPRO INVALID: the pristine revision already carries a fix marker")
                return 2
            try:
                for mode in legs:
                    results[("before", mode)] = _tick_leg(repo_root, mode)
            finally:
                subprocess.run(
                    ["git", "-C", repo_root, "checkout", "HEAD", "--", *_FIXED_PATHS],
                    check=True)
        else:
            if not _marker_present(repo_root):
                print("REPRO INVALID: the fixed tree does not carry the fix marker")
                return 2
            for mode in legs:
                results[("after", mode)] = _tick_leg(repo_root, mode)

    for mode in legs:
        before, after = results.get(("before", mode)), results.get(("after", mode))
        print(f"{mode}:")
        if before:
            print(_render("before", before))
        if after:
            print(_render("after", after))
        print()

    # Verdict: the fixed tree must not abort a leg the pristine tree aborted, and the isolated
    # advance failure must go from "nothing dispatched" to "dispatched".
    verdict_bad = []
    adv_before, adv_after = results.get(("before", "advance-fail")), results.get(("after", "advance-fail"))
    if adv_before and adv_after:
        if not adv_before.get("tick_raised"):
            verdict_bad.append("advance-fail(before) did not raise — the repro is not genuine")
        if adv_after.get("tick_raised"):
            verdict_bad.append(f"advance-fail(after) still raised {adv_after['tick_raised']}")
        if not adv_after.get("dispatched"):
            verdict_bad.append("advance-fail(after) dispatched nothing")
        if adv_after.get("dispatched_after_second_tick") != adv_after.get("dispatched"):
            verdict_bad.append("advance-fail(after) double-fired the occurrence on the next tick")
    cf_before, cf_after = results.get(("before", "claim-fail")), results.get(("after", "claim-fail"))
    if cf_before and cf_after:
        # On the pristine tree the tick aborts at the scan's persist BEFORE any receipt is created,
        # so `open_receipts` there is legitimately empty — this leg passes trivially on old code
        # (that is exactly what the reviewer described), while the unit regression in
        # tests/cron/test_due_scan_save_failure.py fails on 5742a7cd19 and passes on this head.
        # The assertions below are therefore all on the fixed side:
        if cf_after.get("open_receipts"):
            verdict_bad.append(
                f"claim-fail(after) left an execution receipt non-terminal: {cf_after['open_receipts']}")
        if cf_after.get("open_receipts_final"):
            verdict_bad.append(
                "claim-fail(after) left a receipt open after the next tick: "
                f"{cf_after['open_receipts_final']}")
        if cf_after.get("executed"):
            verdict_bad.append(
                "claim-fail(after) EXECUTED the occurrence although the fire claim could not be "
                "written — fail-closed at-most-once broken")
    fd_before, fd_after = results.get(("before", "full-disk")), results.get(("after", "full-disk"))
    if fd_before and fd_after:
        if not fd_before.get("tick_raised") and not fd_before.get("scan_error"):
            verdict_bad.append("full-disk(before) did not raise — the repro is not genuine")
        if fd_after.get("tick_raised"):
            verdict_bad.append(f"full-disk(after) still raised {fd_after['tick_raised']}")
        if fd_after.get("executed_while_unwritable"):
            verdict_bad.append("full-disk(after) EXECUTED an occurrence while the store was still "
                               "unwritable — the fire claim did not fail closed")
        if fd_after.get("executed_after_recovery") != [adv_job_id]:
            verdict_bad.append("full-disk(after) did not execute the occurrence exactly once "
                               "after the store recovered")
        if fd_after.get("executed_after_post_recovery_tick") != [adv_job_id]:
            verdict_bad.append("full-disk(after) re-fired the occurrence on a later tick — "
                               "at-most-once broken")
    for mode in legs:
        after = results.get(("after", mode))
        if after and after.get("skip"):
            print(f"SKIP: {after['skip']} — cannot simulate the write failure")
            return 2

    if verdict_bad:
        for reason in verdict_bad:
            print(f"BUG: {reason}")
        return 1 if any("after" in r for r in verdict_bad) else 2
    if adv_after:
        print("OK: the tick survives the failed persist and still dispatches the occurrence exactly "
              "once; a full disk degrades the profile instead of stopping every job on it.")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
