#!/usr/bin/env python3
"""
Moss Conflict Watcher — detects open Sahil-SS9 PRs on NousResearch/hermes-agent
that have gone CONFLICTING against origin/main.

Detection only. Never touches git or GitHub state. Queues newly-detected
conflicts into moss-conflict-queue.json for moss-conflict-resolver (an
agent-driven cron) to investigate. Runs as a no_agent cron script.

Root cause this exists for (2026-07-09 audit): the pipeline had no mechanism
at all for noticing an already-opened PR had drifted into conflict with
origin/main. Three PRs sat CONFLICTING with zero automation ever looking at
them — two of them recreated from stale SHAs during the 2026-07-08 branch-
deletion incident, one just aged out over two weeks.
"""
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

# P13 isolation: HERMES_HOME parameterises the hermes root.
_HERMES_HOME = os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))
STATE_FILE = os.path.join(_HERMES_HOME, "data/moss-conflict-queue.json")
TARGET_REPOS = [
    "NousResearch/hermes-agent",
]
TRACKED_AUTHORS = [
    "Sahil-SS9",
]

# P13 dry-run: --dry-run prevents GitHub calls (gh pr list) and state
# writes. The script reports what it WOULD do but never touches the
# network or the state file.
_DRY_RUN = "--dry-run" in sys.argv

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"updated_at": None, "tracked": []}

def save_state(state):
    if _DRY_RUN:
        return  # dry-run: never write the state file
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

def get_open_prs(repo, author):
    if _DRY_RUN:
        return []  # dry-run: no GitHub API call
    result = subprocess.run(
        ["gh", "pr", "list", "--repo", repo, "--author", author, "--state", "open",
         "--json", "number,title,headRefName,mergeable,url,updatedAt", "--limit", "100"],
        capture_output=True, text=True, timeout=30
    )
    if result.returncode != 0:
        print(f"ERROR: gh pr list failed for {repo}: {result.stderr}", file=sys.stderr)
        return []
    return json.loads(result.stdout)

def main():
    state = load_state()
    now = datetime.now(timezone.utc).isoformat()
    newly_detected = []

    for repo in TARGET_REPOS:
        for author in TRACKED_AUTHORS:
            prs = get_open_prs(repo, author)
            # mergeable is only reliably computed by GitHub a short while after
            # a push/rebase — UNKNOWN means "not yet computed", not "clean".
            conflicting = {p["number"]: p for p in prs if p.get("mergeable") == "CONFLICTING"}
            tracked_by_number = {t["number"]: t for t in state["tracked"] if t["repo"] == repo}

            for num, pr in conflicting.items():
                if num not in tracked_by_number:
                    entry = {
                        "repo": repo, "number": num, "title": pr["title"],
                        "branch": pr["headRefName"], "url": pr["url"],
                        "detected_at": now, "status": "pending",
                    }
                    state["tracked"].append(entry)
                    newly_detected.append(entry)

            # Drop anything no longer conflicting (fixed, closed, merged, or
            # GitHub's mergeable check has since resolved it clean) — resolver
            # updates status in place for items it's still working, so only
            # prune entries whose PR has genuinely left the conflicting set.
            state["tracked"] = [
                t for t in state["tracked"]
                if t["repo"] != repo or t["number"] in conflicting
            ]

    state["updated_at"] = now
    save_state(state)

    if newly_detected:
        lines = [f"⚠️ {len(newly_detected)} PR(s) newly CONFLICTING against main:"]
        for e in newly_detected:
            lines.append(f"#{e['number']} — {e['title']}\n{e['url']}")
        print("\n".join(lines))
    # Silent otherwise — no_agent convention.

if __name__ == "__main__":
    main()
