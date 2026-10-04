#!/usr/bin/env python3
"""Repro: a failed save inside the due scan takes down the whole cron tick.

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

Run against a checkout (or the installed tree):

    PYTHONPATH=<repo-or-/opt/hermes> python3 evals/cron_due_scan_save_failure.py

Exit code: 0 when the due scan still returns its due jobs (fixed), 1 when it raises (bug present).

Each scan runs in its own child process: the cron store path resolves per process, and a scan
repairs (and saves) the store, so a second scan in the same process would have nothing left to
save and would prove nothing.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

sys.path.insert(0, os.environ.get("HERMES_REPO", "/opt/hermes"))

# A store write above this fails.  Comfortably above a SQLite page (4096) so what fails is the
# jobs.json payload and nothing else, and well below the 20 KB prompt the seed writes.
_FSIZE_LIMIT = 8192

_CHILD = r'''
import json, os, resource, signal, sys, tempfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, sys.argv[1])
home = Path(tempfile.mkdtemp(prefix="cron-due-scan-"))
os.environ["HERMES_HOME"] = str(home)

from cron import jobs as cronjobs

# ON the schedule's grid (whole minute, local offset): a next_run_at off the grid or in another
# offset is treated as stale / a timezone shift and re-anchored instead of fired.  This minute's
# instant is inside the 120s grace a "* * * * *" job gets.
due_at = datetime.now().astimezone().replace(second=0, microsecond=0).isoformat()
cron_dir = home / "cron"
(cron_dir / "output").mkdir(parents=True, exist_ok=True)
base = {
    "schedule": {"kind": "cron", "expr": "* * * * *", "display": "every minute"},
    "deliver": "local", "script": None, "repeat": None, "state": "active",
    "created_at": due_at, "last_run_at": None, "prompt": "noop " + "x" * 20000,
}
store = cronjobs._current_cron_store().jobs_file
store.write_text(json.dumps({"jobs": [
    # The job the operator needs to fire this tick.
    dict(base, id="due000000001", name="due job", enabled=True, next_run_at=due_at),
    # enabled=true with pause markers: _self_disable_half_paused() repairs it mid-scan and sets
    # needs_save, which is what makes the scan write the store on its way out.
    dict(base, id="half00000001", name="half paused", enabled=True, next_run_at=due_at,
         paused_at=due_at, paused_reason="seed"),
]}, indent=2), encoding="utf-8")

previous = None
if sys.argv[2] == "break":
    if os.name != "posix":
        print(json.dumps({"skip": "RLIMIT_FSIZE is POSIX-only"})); raise SystemExit(0)
    soft, hard = resource.getrlimit(resource.RLIMIT_FSIZE)
    signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
    # hard == -1 is RLIM_INFINITY: cap the SOFT limit below it rather than min()-ing against -1.
    cap = 8192 if hard < 0 else min(8192, hard)
    resource.setrlimit(resource.RLIMIT_FSIZE, (cap, hard))
    previous = (soft, hard)

try:
    due = cronjobs.get_due_jobs()
except Exception as exc:
    print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))
else:
    print(json.dumps({"due": [j.get("name") for j in due]}))
finally:
    if previous is not None:
        resource.setrlimit(resource.RLIMIT_FSIZE, previous)
'''


def _scan(repo_root: str, mode: str) -> dict:
    proc = subprocess.run([sys.executable, "-c", _CHILD, repo_root, mode],
                          capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        print(proc.stderr[-2000:])
        raise SystemExit(f"REPRO INVALID: scan child failed (rc={proc.returncode})")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def main() -> int:
    repo_root = os.environ.get("HERMES_REPO", "/opt/hermes")

    # Pass 1 — baseline: with a writable store the due job comes back, proving the seed is genuine.
    baseline = _scan(repo_root, "baseline")
    if baseline.get("error") or "due job" not in (baseline.get("due") or []):
        print(f"REPRO INVALID: baseline={baseline} — the seed produced no due job, so a failure "
              f"below would prove nothing")
        return 2
    print(f"baseline (writable store): scan returned {len(baseline['due'])} due job(s) "
          f"{baseline['due']}")

    # Pass 2 — the store write fails the way a full disk makes it fail.
    broken = _scan(repo_root, "break")
    if broken.get("skip"):
        print(f"SKIP: {broken['skip']} — cannot simulate the write failure")
        return 2
    if broken.get("error"):
        print(f"BUG: due scan raised {broken['error']}")
        print("     the tick aborts, so NO job on this profile runs until the store is writable")
        return 1
    names = broken.get("due") or []
    print(f"OK: due scan returned {len(names)} job(s) despite the failed save: {names}")
    if "due job" not in names:
        print("BUG: the due job was dropped instead of returned")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())