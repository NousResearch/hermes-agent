#!/usr/bin/env python3
"""Run the post-mortem harness against one or two checkouts and print a comparison table.

    python -m evals.postmortem.run --repo /path/to/checkout            # one ref: pass/fail per probe
    python -m evals.postmortem.run --repo A --compare B                  # two refs: side by side
    python -m evals.postmortem.run --repo A --live                       # also the probes that spend money

Each probe is a standalone script run in a fresh interpreter with the target checkout on sys.path and a
temp HERMES_HOME (probes that need real credentials say so and are only run with --live). A probe
"passes" when its process exits 0 AND its stdout contains the expected marker documented in
PROBES below; the marker is the behaviour the corresponding PR fixed. Run the forensics lanes
separately (they need a state.db copy): see forensics/README section in ../README.md.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

# (script, args-template, expected stdout substring, live?, PR)
# Two probes pass on main as well: scanner_bypass_probe (main blocked the witness too, as "malformed") and
# notice_delivery_probe (main has no interim notice to mis-deliver). They guard against regressing INTO the
# round-1 defects, which is why they are here.
PROBES = [
    ("live_ab/hardline_scanner_matrix.py", ["{repo}"], "ALL OK", False, "#103492"),
    ("review_probes/scanner_bypass_probe.py", ["{repo}"], '"hardline": true', False, "#103492"),
    ("live_ab/subagent_context_cap.py", ["{repo}"], "trigger=200,000", False, "#103513"),
    ("live_ab/batch_failure_notice.py", ["{repo}"], "TASK_FAILURE_NOTICE", False, "#103549"),
    ("review_probes/notice_delivery_probe.py", ["{repo}"], "PROBE_COMPLETE", False, "#103549"),
    ("review_probes/cache_estimator_probe.py", ["{repo}"], '"preflight_should_compress": true', False, "#103476"),
    ("review_probes/rewrite_hint_probe.py", ["{repo}"], "remote_fifo", False, "#103551"),
    ("live_ab/auth_stampede.py", ["{repo}", "12"], "server_401=0", False, "#103526"),
    ("review_probes/credential_identity_probe.py", ["{repo}", "pr"], '"after_sub": "account-A"', False, "#103526"),
    # live (real provider calls, cents each)
    ("live_ab/goal_judge_wait.py", ["{repo}", "3"], "('wait', ", True, "#103534"),
    ("live_ab/cache_prefix_wire.py", ["{repo}", "B"], "", True, "#103476"),
]


def run_probe(script: str, args: list[str], repo: str, timeout: int = 240) -> tuple[int, str]:
    env = dict(os.environ)
    env.setdefault("HERMES_HOME", tempfile.mkdtemp(prefix="pm-probe-"))
    env["PYTHONPATH"] = repo + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [sys.executable, str(HERE / script), *[a.format(repo=repo) for a in args]]
    try:
        p = subprocess.run(cmd, cwd=repo, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, env=env)
        return p.returncode, (p.stdout + "\n" + p.stderr)
    except subprocess.TimeoutExpired:
        return 124, "TIMEOUT"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--repo", required=True)
    ap.add_argument("--compare", default=None, help="second checkout to run side by side (e.g. main vs branch)")
    ap.add_argument("--live", action="store_true", help="also run probes that make real provider calls")
    ap.add_argument("--only", default=None, help="substring filter on script path or PR number")
    a = ap.parse_args(argv)
    repos = [a.repo] + ([a.compare] if a.compare else [])
    rows = []
    for script, args, marker, live, pr in PROBES:
        if live and not a.live:
            continue
        if a.only and a.only not in script and a.only not in pr:
            continue
        cells = []
        for repo in repos:
            rc, out = run_probe(script, args, os.path.abspath(repo))
            ok = rc == 0 and (marker in out if marker else True)
            cells.append("PASS" if ok else f"FAIL(rc={rc})")
        rows.append((pr, script, *cells))
    width = max(len(r[1]) for r in rows) if rows else 20
    head = f"{'PR':<9} {'probe':<{width}} " + " ".join(f"{os.path.basename(os.path.normpath(r)):<14}" for r in repos)
    print(head); print("-" * len(head))
    for r in rows:
        print(f"{r[0]:<9} {r[1]:<{width}} " + " ".join(f"{c:<14}" for c in r[2:]))
    failed = any("FAIL" in c for r in rows for c in r[2 + (1 if a.compare else 0):])  # only the LAST column gates
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
