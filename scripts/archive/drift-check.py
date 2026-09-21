#!/usr/bin/env python3
"""Hermes Agent fork drift checker — alerts when drift exceeds thresholds.
Configured as a cron job via cronjob(action='create', script=...).
Outputs nothing when drift is acceptable, alert text when threshold exceeded.
"""

import subprocess
import os
import sys

# Vanilla upstream mirror. Relocated 2026-06-02 out of ~/.hermes (runtime dir)
# to ~/repos/hermes-agent-upstream. Override with HERMES_AGENT_PATH if needed.
HERMES_AGENT_PATH = os.path.expanduser(
    os.environ.get("HERMES_AGENT_PATH", "~/repos/hermes-agent-upstream")
)
BEHIND_THRESHOLD = 5  # Alert if more than this many commits behind
AHEAD_THRESHOLD = 20  # Alert if more than this many non-merge commits ahead
MERGE_COMMIT_PREFIXES = ("4e685277d",)  # Known merge commit SHAs (can grow)

def run(cmd, timeout=30):
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, cwd=HERMES_AGENT_PATH
    )
    return result.stdout.strip(), result.stderr.strip(), result.returncode

def main():
    if not os.path.isdir(os.path.join(HERMES_AGENT_PATH, ".git")):
        print(f"SKIP: {HERMES_AGENT_PATH} is not a git repo")
        return

    # Fetch to get latest upstream
    subprocess.run(
        ["git", "fetch", "origin", "--quiet"],
        cwd=HERMES_AGENT_PATH, capture_output=True, timeout=60
    )

    # Count behind
    behind_out, _, _ = run(["git", "rev-list", "--count", "HEAD..origin/main", "--no-merges"])
    behind = int(behind_out) if behind_out.strip().isdigit() else 0

    # Count ahead (our commits not in upstream)
    ahead_out, _, _ = run(["git", "rev-list", "--count", "origin/main..HEAD", "--no-merges"])
    ahead = int(ahead_out) if ahead_out.strip().isdigit() else 0

    # Check dirty files
    dirty_out, _, _ = run(["git", "diff", "--name-only", "HEAD"])
    dirty_count = len([l for l in dirty_out.split("\n") if l.strip()]) if dirty_out else 0

    issues = []
    if behind > BEHIND_THRESHOLD:
        issues.append(f"⚠️ {behind} commits behind upstream (threshold: {BEHIND_THRESHOLD})")
    if ahead > AHEAD_THRESHOLD:
        issues.append(f"⚠️ {ahead} commits ahead of upstream (threshold: {AHEAD_THRESHOLD})")
    if dirty_count > 0:
        issues.append(f"⚠️ {dirty_count} dirty file(s) uncommitted — live on running gateway")

    if not issues:
        return  # Silent — no news is good news

    # Format alert
    print(f"🤖 HERMES AGENT FORK DRIFT")
    for issue in issues:
        print(f"  {issue}")
    print(f"")
    print(f"  Runbook: ~/.hermes/kanban/workspaces/t_17b9655e/FORK_DRIFT_RUNBOOK.md")
    print(f"  To sync: cd ~/repos/hermes-agent-upstream && git merge origin/main")

if __name__ == "__main__":
    main()
