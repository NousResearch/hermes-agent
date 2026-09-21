#!/usr/bin/env python3
"""
Mossy branch cleanup — scans the upstream repo for fix/ branches whose
associated PRs have been closed/merged or whose issues are resolved,
and deletes them (both local and fork remote).

Runs daily, delivers to #build-log.
Silent when no cleanup candidates exist.

Auto-deletes branches where:
- PR was closed (duplicate, superseded, implemented_on_main, cannot_reproduce)
- PR was merged (via cherry-pick or direct merge)
- Branch is merged into main
- Issue is closed AND branch is stale (>14 days)

Preserves branches where:
- PR is still OPEN (active contribution)
- No PR exists and issue is still open (unsubmitted work)
"""
import subprocess, json, sys, os, re
from pathlib import Path
from datetime import datetime, timedelta

REPO = Path(os.environ.get("MOSS_BRANCH_REPO", "/home/kensei/repos/hermes-agent-vanilla"))
DAYS_STALE = 14
DRY_RUN = os.environ.get("MOSS_BRANCH_DRY_RUN", "") == "1"

def run(cmd, timeout=30):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout, cwd=REPO)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None

def get_branches():
    """Get all local fix/ branches with their last commit date."""
    raw = run("git for-each-ref --format='%(refname:short)|%(committerdate:short)' refs/heads/fix/")
    if not raw:
        return []
    branches = []
    for line in raw.strip().split("\n"):
        parts = line.split("|")
        if len(parts) == 2:
            date_parts = parts[1].split("-")
            if len(date_parts) == 3:
                uk_date = f"{date_parts[2]}/{date_parts[1]}/{date_parts[0]}"
            else:
                uk_date = parts[1]
            branches.append({"name": parts[0], "last_commit": uk_date, "last_commit_raw": parts[1]})
    return branches

def get_issue_status(issue_num):
    """Check if an issue is closed and whether it has closing PRs."""
    state = run(f"gh issue view {issue_num} --repo NousResearch/hermes-agent --json state --jq '.state' 2>/dev/null")
    closing_prs = run(f"gh issue view {issue_num} --repo NousResearch/hermes-agent --json closedByPullRequestsReferences --jq '.closedByPullRequestsReferences.totalCount' 2>/dev/null")
    return {
        "state": state or "UNKNOWN",
        "closing_prs": int(closing_prs) if closing_prs and closing_prs.isdigit() else 0
    }

def check_branch_merged(branch):
    """Check if the branch is merged into main."""
    result = run(f"git branch --merged main --format='%(refname:short)' | grep -Fx '{branch}'")
    return result is not None and result.strip() == branch

def get_pr_status(branch):
    """Check PR state on upstream for this branch. Returns (state, pr_number) or (None, None)."""
    # Search upstream PRs by head branch name (no owner prefix — gh pr list
    # resolves the fork head automatically).
    result = run(f"gh pr list --repo NousResearch/hermes-agent --state all --head '{branch}' --json number,state --limit 1 2>/dev/null", timeout=15)
    if result:
        try:
            prs = json.loads(result)
            if prs:
                return prs[0].get("state"), prs[0].get("number")
        except (json.JSONDecodeError, IndexError, KeyError):
            pass
    return None, None

def delete_branch(branch):
    """Delete branch locally and on fork remote."""
    if DRY_RUN:
        print(f"[dry-run] would delete branch: {branch}")
        return []
    errors = []
    # Delete local
    r1 = run(f"git branch -D {branch}", timeout=15)
    if r1 is None:
        errors.append("local delete failed")
    # Delete from fork
    r2 = run(f"git push fork --delete {branch}", timeout=30)
    if r2 is None:
        errors.append("fork delete failed")
    return errors

def main():
    if not REPO.exists():
        return 0

    # Fetch latest main
    run("git fetch origin main --quiet", timeout=30)

    branches = get_branches()
    if not branches:
        return 0

    candidates = []
    deleted = []
    preserved = []
    stale_cutoff = (datetime.now() - timedelta(days=DAYS_STALE)).strftime("%Y-%m-%d")

    for branch in branches:
        name = branch["name"]
        last_commit = branch["last_commit"]
        last_commit_raw = branch["last_commit_raw"]

        match = re.search(r'issue-(\d+)', name)
        if not match:
            continue
        issue_num = match.group(1)

        is_merged = check_branch_merged(name)
        issue = get_issue_status(issue_num)
        pr_state, pr_num = get_pr_status(name)

        # Determine cleanup action
        reasons = []
        should_delete = False

        if is_merged:
            reasons.append("branch merged into main")
            should_delete = True

        if pr_state in ("CLOSED", "MERGED"):
            reasons.append(f"PR #{pr_num} {pr_state.lower()}")
            should_delete = True

        if issue["state"] == "CLOSED":
            reasons.append(f"issue #{issue_num} closed")
            if issue["closing_prs"] > 0:
                reasons.append(f"closed by {issue['closing_prs']} PR(s)")
            # Auto-delete if issue is closed, regardless of staleness.
            # If no PR was opened from this branch and the issue is closed,
            # the work is orphaned — someone else fixed it or it was resolved
            # differently. Keeping the branch serves no purpose.
            if pr_state is None:
                reasons.append("no PR opened from this branch — orphaned work")
                should_delete = True
            elif last_commit_raw < stale_cutoff:
                reasons.append(f"stale (last commit {last_commit})")
                should_delete = True

        if should_delete:
            candidates.append({
                "branch": name,
                "issue": issue_num,
                "issue_state": issue["state"],
                "merged": is_merged,
                "pr_state": pr_state,
                "pr_num": pr_num,
                "last_commit": last_commit,
                "reasons": reasons
            })
        elif reasons:
            # Has reasons but not enough to auto-delete (e.g. issue closed but not stale)
            preserved.append({
                "branch": name,
                "reasons": reasons,
                "last_commit": last_commit
            })

    if not candidates:
        return 0  # silent — nothing to clean up

    # Auto-delete
    for c in candidates:
        errors = delete_branch(c["branch"])
        if errors:
            c["delete_errors"] = errors
            preserved.append(c)  # Couldn't delete, keep in preserved for reporting
        else:
            deleted.append(c)

    # Output report
    print(f"Branch cleanup: {len(deleted)} deleted, {len(preserved)} preserved")
    if deleted:
        print("\nDeleted:")
        for c in deleted:
            status = "MERGED" if c["merged"] else f"PR #{c['pr_num']} {c['pr_state']}" if c['pr_state'] else f"issue {c['issue_state']}"
            print(f"  ✅ {c['branch']} ({status}, last commit {c['last_commit']})")
            for r in c["reasons"]:
                print(f"     — {r}")
    if preserved:
        print("\nPreserved (issue closed but not stale enough):")
        for c in preserved:
            print(f"  ⏳ {c['branch']} (last commit {c['last_commit']})")
            for r in c["reasons"]:
                print(f"     — {r}")

    return 0

if __name__ == "__main__":
    sys.exit(main())