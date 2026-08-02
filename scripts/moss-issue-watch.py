#!/usr/bin/env python3
"""
Moss Issue & PR Watcher — polls GitHub for new issues AND PR activity
on target repos. Tracks submissions, comments, merges, and status changes.

Runs as a no_agent cron script. Outputs new activity for delivery.
"""

# P13: disabled-staging guard — exit early when cron is disabled
import os as _os, sys as _sys
if _os.environ.get("DRY_RUN") == "1":
    print(f"[DRY_RUN] {_os.path.basename(__file__)}")
    _sys.exit(0)

import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone, timedelta

STATE_FILE = os.path.expanduser("~/.hermes/data/moss-issue-watch.json")
FIX_QUEUE_FILE = os.path.expanduser("~/.hermes/data/moss-fix-queue.json")
MAX_QUEUE_SIZE = 20
MIN_ISSUE_AGE_HOURS = 4  # Don't queue issues younger than this — avoids racing other contributors
MERGED_PR_LOOKBACK_DAYS = 7  # Check for recently merged PRs referencing the same issue
TARGET_REPOS = [
    "NousResearch/hermes-agent",
]
TRACKED_AUTHORS = [
    "Sahil-SS9",
]

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            state = json.load(f)
            # Migrate from old format (no seen_prs)
            if "seen_prs" not in state:
                state["seen_prs"] = {}
            return state
    return {"last_checked": None, "seen_issues": {}, "seen_prs": {}}

def save_state(state):
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

def get_open_issues(repo):
    result = subprocess.run(
        ["gh", "issue", "list", "--repo", repo, "--state", "open",
         "--json", "number,title,labels,createdAt,url,body",
         "--limit", "50"],
        capture_output=True, text=True, timeout=30
    )
    if result.returncode != 0:
        print(f"ERROR: gh issue list failed for {repo}: {result.stderr}", file=sys.stderr)
        return []
    return json.loads(result.stdout)

def classify_issue(issue):
    labels = [l["name"] for l in issue.get("labels", [])]
    body = issue.get("body", "").lower()
    title = issue.get("title", "").lower()
    if "type/feature" in labels and len(body) < 100:
        return None
    if "type/bug" in labels:
        return "bug"
    if "type/security" in labels:
        return "security"
    # Detect [Bug] in title even without type/bug label — many upstream issues
    # use the [Bug] prefix but don't have the type/bug label applied yet.
    # This was the root cause of 304 bug issues being silently skipped.
    if title.startswith("[bug]") or "[bug]:" in title:
        return "bug"
    if "needs-repro" in labels:
        if "steps to reproduce" in body or "reproduction" in body:
            return "bug"
        return None
    return None

def has_linked_pr(repo, issue_number):
    """Check if an issue already has a closing PR reference (linked, open, or recently merged).

    Returns (True, reason) if a PR exists, (False, None) otherwise.
    """
    # 1. Check via gh issue view for closing PR references
    result = subprocess.run(
        ["gh", "issue", "view", str(issue_number), "--repo", repo,
         "--json", "closedByPullRequestsReferences"],
        capture_output=True, text=True, timeout=15
    )
    if result.returncode == 0:
        data = json.loads(result.stdout)
        refs = data.get("closedByPullRequestsReferences", {})
        # gh currently returns a list; older GraphQL-shaped output used a
        # mapping with totalCount. Accept both representations.
        if isinstance(refs, list):
            has_refs = bool(refs)
        elif isinstance(refs, dict):
            has_refs = refs.get("totalCount", 0) > 0 or bool(refs.get("nodes"))
        else:
            has_refs = False
        if has_refs:
            return (True, "has closing PR reference")

    # 2. Search for open PRs that reference this issue number (title + body match)
    search_result = subprocess.run(
        ["gh", "search", "prs", "--repo", repo, "--state", "open", "--match", "title,body",
         f"\"#{issue_number}\"",
         "--json", "number,title", "--limit", "10"],
        capture_output=True, text=True, timeout=15
    )
    if search_result.returncode == 0:
        prs = json.loads(search_result.stdout)
        if prs:  # Any PR matching the issue number search is a potential duplicate
            pr_list = ", ".join(f"#{p['number']}" for p in prs)
            return (True, f"open PR(s) {pr_list} reference this issue")

    # 3. Search for recently MERGED PRs that reference this issue — catches
    #    fixes that landed on main but where the issue hasn't been closed yet.
    #    This was the root cause of #54500 (duplicate of already-merged #55579)
    #    and #56228 (subsumed by merged #57006).
    since = (datetime.now(timezone.utc) - timedelta(days=MERGED_PR_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    merged_result = subprocess.run(
        ["gh", "search", "prs", "--repo", repo, "--merged", "--match", "title,body", "--updated", f">={since}",
         f"\"#{issue_number}\"",
         "--json", "number,title,closedAt", "--limit", "10"],
        capture_output=True, text=True, timeout=15
    )
    if merged_result.returncode == 0:
        merged_prs = json.loads(merged_result.stdout)
        if merged_prs:  # Any merged PR matching the issue number is a duplicate
            pr_list = ", ".join(f"#{p['number']}" for p in merged_prs)
            return (True, f"merged PR(s) {pr_list} already fixed this issue")

    return (False, None)


def has_semantic_duplicate(repo, issue):
    """Search for open or recently merged PRs that touch the same file/symbol,
    even if they reference a different issue number.

    This catches the Category A duplicates: same bug, different issue number.
    Root cause of #58148 (dup of #50397) and #57959 (dup of #54887).

    Returns (True, reason) if a likely duplicate PR exists, (False, None) otherwise.
    """
    body = issue.get("body", "") or ""
    title = issue.get("title", "") or ""

    # Extract potential source file references from the issue body
    # Look for patterns like agent/foo.py, hermes_cli/bar.py, etc.
    file_refs = set(re.findall(r'(?<!\w)((?:agent|hermes_cli|gateway|tests|plugins|cli|run_agent|dashboard)/[\w/]+\.py)', body))

    # Extract key symbols mentioned (function/class names near "function", "method", "in")
    # Look for backtick-quoted identifiers that look like Python symbols
    symbol_refs = set(re.findall(r'`([a-z_][a-z0-9_]*(?:\.[a-z_][a-z0-9_]*)+)`', body, re.IGNORECASE))
    # Filter to likely code symbols (at least 6 chars, contains underscore or dot)
    symbol_refs = {s for s in symbol_refs if len(s) >= 6 and ('_' in s or '.' in s)}

    # Build search queries — search by filename, symbol, or title keyword
    search_terms = set()
    for f in file_refs:
        # Use the basename without extension as a search term
        basename = os.path.basename(f).replace('.py', '')
        if len(basename) >= 4:
            search_terms.add(basename)
    for s in symbol_refs:
        # Use the last component of a dotted symbol
        parts = s.split('.')
        last = parts[-1]
        if len(last) >= 4:
            search_terms.add(last)

    # Also extract meaningful keywords from the issue title itself.
    # This catches cases where the duplicate PR title shares keywords with
    # the issue title but doesn't mention the same source file.
    # Root cause of #58148: issue title had "reasoning" + "empty" + "content",
    # PR #50397 title had the same words, but no shared file reference.
    title_words = set(re.findall(r'\b[a-z]{4,}\b', title.lower()))
    generic_words = {"fix", "error", "crash", "when", "that", "this", "from", "with", "handle",
                      "been", "have", "does", "will", "also", "into", "some", "issue", "bug"}
    meaningful_title_words = {w for w in title_words if w not in generic_words and len(w) >= 5}
    search_terms.update(meaningful_title_words)

    if not search_terms:
        return (False, None)

    # Search for open PRs whose titles contain these terms
    for term in list(search_terms)[:6]:  # Limit to 6 searches to avoid rate limits
        result = subprocess.run(
            ["gh", "search", "prs", "--repo", repo, "--state", "open", "--match", "title,body",
             f"\"{term}\"",
             "--json", "number,title", "--limit", "5"],
            capture_output=True, text=True, timeout=15
        )
        if result.returncode != 0:
            continue
        prs = json.loads(result.stdout)
        for pr in prs:
            pr_title = pr.get("title", "").lower()
            # Only flag as duplicate if the PR title shares the search term
            # AND the issue title is semantically similar (shares a keyword)
            if term.lower() in pr_title:
                # Check for shared keywords between issue title and PR title
                issue_words = set(re.findall(r'\b[a-z]{4,}\b', title.lower()))
                pr_words = set(re.findall(r'\b[a-z]{4,}\b', pr_title))
                overlap = issue_words & pr_words
                # Exclude generic words that match everything
                generic = {"fix", "error", "crash", "when", "that", "this", "from", "with", "handle"}
                meaningful_overlap = overlap - generic
                if len(meaningful_overlap) >= 2:
                    return (True, f"open PR #{pr['number']} '{pr.get('title','')}' touches same code (term: {term})")

        # Also check recently merged PRs
        since = (datetime.now(timezone.utc) - timedelta(days=MERGED_PR_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
        merged_result = subprocess.run(
            ["gh", "search", "prs", "--repo", repo, "--merged", "--match", "title,body", "--updated", f">={since}",
             f"\"{term}\"",
             "--json", "number,title,closedAt", "--limit", "5"],
            capture_output=True, text=True, timeout=15
        )
        if merged_result.returncode != 0:
            continue
        merged_prs = json.loads(merged_result.stdout)
        for pr in merged_prs:
            pr_title = pr.get("title", "").lower()
            if term.lower() in pr_title:
                issue_words = set(re.findall(r'\b[a-z]{4,}\b', title.lower()))
                pr_words = set(re.findall(r'\b[a-z]{4,}\b', pr_title))
                overlap = issue_words & pr_words
                generic = {"fix", "error", "crash", "when", "that", "this", "from", "with", "handle"}
                meaningful_overlap = overlap - generic
                if len(meaningful_overlap) >= 2:
                    return (True, f"merged PR #{pr['number']} '{pr.get('title','')}' already fixed same area (term: {term})")

    return (False, None)

def get_recent_prs(repo):
    """Fetch recent PRs by tracked authors only. Uses --author filter at the API level."""
    tracked = []
    for author in TRACKED_AUTHORS:
        result = subprocess.run(
            ["gh", "pr", "list", "--repo", repo, "--state", "all", "--author", author,
             "--json", "number,title,state,author,createdAt,updatedAt,mergedAt,closedAt,url,labels,comments,reviews",
             "--limit", "50"],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            print(f"ERROR: gh pr list failed for {repo} author {author}: {result.stderr}", file=sys.stderr)
            continue
        tracked.extend(json.loads(result.stdout))
    return tracked

def get_all_open_prs(repo):
    """Return all currently OPEN PRs by tracked authors (no state tracking, just snapshot)."""
    open_prs = []
    for author in TRACKED_AUTHORS:
        result = subprocess.run(
            ["gh", "pr", "list", "--repo", repo, "--state", "open", "--author", author,
             "--json", "number,title,createdAt,url,labels,headRefName",
             "--limit", "50"],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            open_prs.extend(json.loads(result.stdout))
    return open_prs

def check_pr_activity(repo, state):
    """Check for new or updated PRs and return activity report."""
    prs = get_recent_prs(repo)
    seen = state["seen_prs"].get(repo, {})
    now = datetime.now(timezone.utc).isoformat()
    activity = []

    for pr in prs:
        num = pr["number"]
        key = str(num)
        prev = seen.get(key)

        if prev is None:
            # New PR — never seen before
            if pr["state"] == "OPEN":
                labels = [l["name"] for l in pr.get("labels", [])]
                priority = "P1" if "P1" in labels else ("P2" if "P2" in labels else "P3")
                activity.append({
                    "type": "new_pr",
                    "repo": repo,
                    "number": num,
                    "title": pr["title"],
                    "author": pr["author"]["login"],
                    "priority": priority,
                    "url": pr["url"],
                })
                seen[key] = {
                    "number": num,
                    "title": pr["title"],
                    "state": pr["state"],
                    "last_comment_count": len(pr.get("comments", [])),
                    "last_review_count": len(pr.get("reviews", [])),
                    "seen_at": now,
                }
            elif pr["state"] == "MERGED":
                activity.append({
                    "type": "merged_pr",
                    "repo": repo,
                    "number": num,
                    "title": pr["title"],
                    "author": pr["author"]["login"],
                    "url": pr["url"],
                })
                seen[key] = {
                    "number": num,
                    "title": pr["title"],
                    "state": "MERGED",
                    "seen_at": now,
                }
            else:
                seen[key] = {
                    "number": num,
                    "title": pr["title"],
                    "state": pr["state"],
                    "seen_at": now,
                }
        elif prev["state"] != pr["state"]:
            # State change — e.g. OPEN → MERGED
            if pr["state"] == "MERGED":
                activity.append({
                    "type": "merged_pr",
                    "repo": repo,
                    "number": num,
                    "title": pr["title"],
                    "author": pr["author"]["login"],
                    "url": pr["url"],
                })
            elif pr["state"] == "CLOSED":
                activity.append({
                    "type": "closed_pr",
                    "repo": repo,
                    "number": num,
                    "title": pr["title"],
                    "url": pr["url"],
                })
            seen[key]["state"] = pr["state"]
            seen[key]["seen_at"] = now
        elif pr["state"] == "OPEN" and prev["state"] == "OPEN":
            # Check for new comments or reviews using list response counts
            current_comments = len(pr.get("comments", []))
            current_reviews = len(pr.get("reviews", []))
            prev_comments = prev.get("last_comment_count", 0)
            prev_reviews = prev.get("last_review_count", 0)

            new_comments = current_comments - prev_comments
            new_reviews = current_reviews - prev_reviews

            if new_comments > 0 or new_reviews > 0:
                activity.append({
                    "type": "pr_activity",
                    "repo": repo,
                    "number": num,
                    "title": pr["title"],
                    "new_comments": new_comments,
                    "new_reviews": new_reviews,
                    "url": pr["url"],
                })
                seen[key]["last_comment_count"] = current_comments
                seen[key]["last_review_count"] = current_reviews
                seen[key]["seen_at"] = now

    state["seen_prs"][repo] = seen
    return activity

def main():
    state = load_state()
    now = datetime.now(timezone.utc).isoformat()
    new_issues = []
    pr_activity = []
    all_open_prs = []

    for repo in TARGET_REPOS:
        # Check issues
        issues = get_open_issues(repo)
        seen = state["seen_issues"].get(repo, {})

        for issue in issues:
            num = issue["number"]
            key = str(num)
            existing = seen.get(key)
            # Every classification is terminal EXCEPT too_new — a too_new issue
            # must be re-evaluated on later ticks once it clears the age gate,
            # otherwise it is skipped forever ("if key in seen: continue" ran
            # before the age gate ever got a chance to re-check). Root cause of
            # the 2026-07-04 to 2026-07-09 queue starvation: 145/152 issues
            # marked too_new were never promoted.
            if existing and existing.get("classification") != "too_new":
                continue
            classification = classify_issue(issue)
            if classification is None:
                seen[key] = {"number": num, "title": issue["title"], "classification": "skipped", "seen_at": now}
                continue

            # MINIMUM AGE GATE — don't queue issues younger than MIN_ISSUE_AGE_HOURS.
            # This gives the community time to self-organize and prevents racing
            # other contributors who may already be working on a fix.
            # Root cause of #58202 (raced #58201 by 45 seconds).
            created_at = issue.get("createdAt", "")
            if created_at:
                try:
                    created_dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                    age_hours = (datetime.now(timezone.utc) - created_dt).total_seconds() / 3600
                    if age_hours < MIN_ISSUE_AGE_HOURS:
                        seen[key] = {"number": num, "title": issue["title"], "classification": "too_new",
                                     "age_hours": round(age_hours, 1), "seen_at": now}
                        continue
                except (ValueError, TypeError):
                    pass  # If we can't parse the date, don't block on age

            # Skip issues that already have a linked or open PR — prevents
            # the fix-pipeline from queuing issues that someone already
            # addressed. Now also checks recently MERGED PRs to catch fixes
            # that landed on main but where the issue hasn't been closed yet.
            has_pr, pr_reason = has_linked_pr(repo, num)
            if has_pr:
                seen[key] = {"number": num, "title": issue["title"], "classification": "has_pr",
                             "skip_reason": pr_reason, "seen_at": now}
                continue

            # SEMANTIC DUPLICATE CHECK — search for open or recently merged PRs
            # that touch the same file/symbol even if they reference a different
            # issue number. Catches same-bug-different-issue duplicates.
            # Root cause of #58148 (dup of #50397) and #57959 (dup of #54887).
            has_dup, dup_reason = has_semantic_duplicate(repo, issue)
            if has_dup:
                seen[key] = {"number": num, "title": issue["title"], "classification": "has_dup",
                             "skip_reason": dup_reason, "seen_at": now}
                continue

            labels = [l["name"] for l in issue.get("labels", [])]
            priority = "P1" if "P1" in labels else ("P2" if "P2" in labels else "P3")
            seen[key] = {"number": num, "title": issue["title"], "classification": classification, "priority": priority, "url": issue["url"], "seen_at": now}
            new_issues.append({"repo": repo, "number": num, "title": issue["title"], "classification": classification, "priority": priority, "url": issue["url"]})

        state["seen_issues"][repo] = seen

        # Check PR activity
        pr_activity.extend(check_pr_activity(repo, state))

        # Collect all currently open PRs (for the report)
        all_open_prs.extend(get_all_open_prs(repo))

    state["last_checked"] = now
    save_state(state)

    # Output — short Discord body + HTML attachment
    output_lines = []
    html_parts = []

    # Write fixable bugs to pipeline queue.
    # Accept P1, P2, and P3 bugs — most upstream issues default to P3 when
    # no priority label is applied. Filtering P3 out caused the pipeline to
    # starve (0 pending items for 4+ days).
    #
    # Sourced from two places, not just this run's fresh classifications:
    # 1. `new_issues` — issues classified bug/security in THIS run (includes
    #    issues just promoted out of too_new).
    # 2. A full backfill scan of `state["seen_issues"]` for any bug/security
    #    issue that never made it into the queue — e.g. it was classified in
    #    a run where MAX_QUEUE_SIZE was already full. Without this backfill, a
    #    capped-out issue is skipped forever on future ticks (classification is
    #    terminal, so the per-issue loop above never revisits it).
    fixable = [iss for iss in new_issues if iss["classification"] in ("bug", "security")]

    # Start from the existing file and preserve every key untouched (e.g.
    # `completed`/`skipped`/`prs_opened` — whatever the agent-driven
    # moss-fix-pipeline cron tracks on its side). Only `updated_at` and
    # `pending` are ours to own. Previously this rebuilt the dict from
    # scratch with just {updated_at, pending}, silently deleting the fix
    # pipeline's own bookkeeping every 15 minutes.
    queue_state = {"updated_at": now, "pending": []}
    known_keys = set()
    if os.path.exists(FIX_QUEUE_FILE):
        try:
            with open(FIX_QUEUE_FILE) as f:
                existing = json.load(f)
                if isinstance(existing, dict):
                    queue_state = dict(existing)
                    queue_state["updated_at"] = now
                    queue_state["pending"] = existing.get("pending", [])
                for bucket_name, bucket_val in (existing.items() if isinstance(existing, dict) else []):
                    if not isinstance(bucket_val, list):
                        continue
                    for i in bucket_val:
                        if isinstance(i, dict) and "repo" in i and "number" in i:
                            known_keys.add(f"{i['repo']}#{i['number']}")
        except (json.JSONDecodeError, OSError):
            pass

    for iss in fixable:
        key = f"{iss['repo']}#{iss['number']}"
        if key not in known_keys and len(queue_state["pending"]) < MAX_QUEUE_SIZE:
            queue_state["pending"].append(iss)
            known_keys.add(key)

    for repo_name, repo_seen in state["seen_issues"].items():
        if len(queue_state["pending"]) >= MAX_QUEUE_SIZE:
            break
        for rec in repo_seen.values():
            if rec.get("classification") not in ("bug", "security"):
                continue
            qkey = f"{repo_name}#{rec['number']}"
            if qkey in known_keys:
                continue
            if len(queue_state["pending"]) >= MAX_QUEUE_SIZE:
                break
            queue_state["pending"].append({
                "repo": repo_name, "number": rec["number"], "title": rec["title"],
                "classification": rec["classification"], "priority": rec.get("priority", "P3"),
                "url": rec.get("url", ""),
            })
            known_keys.add(qkey)

    with open(FIX_QUEUE_FILE, "w") as f:
        json.dump(queue_state, f, indent=2)

    # Only output on PR activity — new issues silently populate the fix queue
    if pr_activity:
        # Discord body — PR activity only, no issue counts
        parts = []
        pr_count = len([a for a in pr_activity if a["type"] in ("new_pr", "merged_pr", "closed_pr")])
        if pr_count:
            parts.append(f"{pr_count} PR updates")
        output_lines.append(" · ".join(parts))

        # Build HTML report
        def esc(s):
            return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\"", "&quot;")

        html_parts.append(f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Mossy Issue Watch · {now[:10]}</title>
<style>
:root {{ color-scheme:dark; --bg:#0f1115; --card:#181b22; --line:#2a2f3a; --text:#e7e3d8; --muted:#8e96a6; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--text); font:14px/1.55 system-ui,-apple-system,sans-serif; }}
main {{ width:min(900px,calc(100%-32px)); margin:0 auto; padding:28px 0 56px; }}
h1 {{ margin:0 0 2px; font-size:24px; letter-spacing:-.02em; }}
.sub {{ color:var(--muted); margin:0 0 24px; }}
.pills {{ display:flex; gap:10px; flex-wrap:wrap; margin:14px 0 20px; }}
.pill {{ background:var(--card); border:1px solid var(--line); border-radius:999px; padding:5px 11px; font-size:13px; font-variant-numeric:tabular-nums; }}
section {{ background:var(--card); border:1px solid var(--line); border-radius:12px; margin:16px 0; overflow:hidden; }}
h2 {{ font-size:16px; margin:0; padding:14px 16px; border-bottom:1px solid var(--line); }}
table {{ width:100%; border-collapse:collapse; }}
th {{ text-align:left; padding:10px 16px; font-size:12px; color:var(--muted); text-transform:uppercase; letter-spacing:.05em; border-bottom:1px solid var(--line); }}
td {{ padding:10px 16px; border-bottom:1px solid var(--line); vertical-align:top; font-size:13px; }}
td:last-child, th:last-child {{ text-align:right; }}
a {{ color:#5b8def; text-decoration:none; }}
a:hover {{ text-decoration:underline; }}
.tag {{ display:inline-block; padding:1px 7px; border-radius:4px; font-size:11px; font-weight:600; }}
.tag-bug {{ background:#401510; color:#e0584c; }}
.tag-sec {{ background:#152040; color:#5b8def; }}
.tag-P1 {{ background:#401510; color:#e0584c; }}
.tag-P2 {{ background:#402e10; color:#e0982e; }}
.tag-P3 {{ background:#152040; color:#5b8def; }}
.tag-new {{ background:#0c3d2e; color:#46b39a; }}
.tag-merged {{ background:#0c3d2e; color:#46b39a; }}
.tag-closed {{ background:#2a2f3a; color:#8e96a6; }}
</style></head><body>
<main>
<h1>Mossy Issue Watch</h1>
<p class="sub">{now[:19].replace("T", " ")}</p>
<div class="pills">""")

        if new_issues:
            p1 = sum(1 for iss in new_issues if iss["priority"] == "P1")
            p2 = sum(1 for iss in new_issues if iss["priority"] == "P2")
            p3 = sum(1 for iss in new_issues if iss["priority"] == "P3")
            html_parts.append(f"""<span class="pill">📋 <b>{len(new_issues)}</b> new issues</span>""")
            if p1: html_parts.append(f"""<span class="pill">🔴 P1 <b>{p1}</b></span>""")
            if p2: html_parts.append(f"""<span class="pill">🟡 P2 <b>{p2}</b></span>""")
            if p3: html_parts.append(f"""<span class="pill">⚪ P3 <b>{p3}</b></span>""")

        if pr_activity:
            new_prs = sum(1 for a in pr_activity if a["type"] == "new_pr")
            merged = sum(1 for a in pr_activity if a["type"] == "merged_pr")
            closed = sum(1 for a in pr_activity if a["type"] == "closed_pr")
            activity_count = sum(1 for a in pr_activity if a["type"] == "pr_activity")
            if new_prs: html_parts.append(f"""<span class="pill">🆕 <b>{new_prs}</b> new PRs</span>""")
            if merged: html_parts.append(f"""<span class="pill">✅ <b>{merged}</b> merged</span>""")
            if closed: html_parts.append(f"""<span class="pill">❌ <b>{closed}</b> closed</span>""")
            if activity_count: html_parts.append(f"""<span class="pill">💬 <b>{activity_count}</b> with activity</span>""")

        html_parts.append("</div>")

        # Issues table
        if new_issues:
            html_parts.append("""<section><h2>New Issues</h2><table><thead><tr><th>Issue</th><th>Title</th><th></th></tr></thead><tbody>""")
            for iss in new_issues:
                cls = "bug" if iss["classification"] == "bug" else "sec"
                tag_line = f"<span class='tag tag-{cls}'>{iss['classification']}</span> <span class='tag tag-{iss['priority']}'>{iss['priority']}</span>"
                html_parts.append(f"""<tr><td><a href="{esc(iss['url'])}">{esc(iss['repo'])}#{iss['number']}</a></td><td>{esc(iss['title'][:80])}</td><td>{tag_line}</td></tr>""")
            html_parts.append("</tbody></table></section>")

        # PR activity table
        if pr_activity:
            html_parts.append("""<section><h2>PR Activity</h2><table><thead><tr><th>PR</th><th>Title</th><th></th></tr></thead><tbody>""")
            for a in pr_activity:
                if a["type"] == "new_pr":
                    tag = "<span class='tag tag-new'>NEW</span>"
                    suffix = f"by {esc(a.get('author',''))}"
                elif a["type"] == "merged_pr":
                    tag = "<span class='tag tag-merged'>MERGED</span>"
                    suffix = f"by {esc(a.get('author',''))}"
                elif a["type"] == "closed_pr":
                    tag = "<span class='tag tag-closed'>CLOSED</span>"
                    suffix = ""
                elif a["type"] == "pr_activity":
                    parts = []
                    if a["new_comments"] > 0: parts.append(f"{a['new_comments']} comments")
                    if a["new_reviews"] > 0: parts.append(f"{a['new_reviews']} reviews")
                    tag = "<span class='tag tag-new'>💬</span>"
                    suffix = " · ".join(parts)
                else:
                    tag, suffix = "", ""
                html_parts.append(f"""<tr><td><a href="{esc(a['url'])}">{esc(a['repo'])}#{a['number']}</a></td><td>{esc(a['title'][:70])}</td><td>{tag} {esc(suffix)}</td></tr>""")
            html_parts.append("</tbody></table></section>")

        # All Open PRs section (always shown when PRs exist)
        if all_open_prs:
            html_parts.append("""<section><h2>All Open PRs</h2><table><thead><tr><th>PR</th><th>Title</th><th>Opened</th></tr></thead><tbody>""")
            for pr in sorted(all_open_prs, key=lambda p: p.get("createdAt", "")):
                created = pr.get("createdAt", "")[:10]
                labels = [l["name"] for l in pr.get("labels", [])]
                priority = "P1" if "P1" in labels else ("P2" if "P2" in labels else "P3")
                pri_tag = f"<span class='tag tag-{priority}'>{priority}</span>"
                html_parts.append(f"""<tr><td><a href="{esc(pr['url'])}">{esc(pr.get('repo','NousResearch/hermes-agent'))}#{pr['number']}</a></td><td>{esc(pr['title'][:80])}</td><td>{esc(created)} {pri_tag}</td></tr>""")
            html_parts.append("</tbody></table></section>")

        html_parts.append("</main></body></html>")

        # Write HTML report
        report_dir = os.path.expanduser("~/.hermes/cron/output/moss-watch-reports")
        os.makedirs(report_dir, exist_ok=True)
        report_path = os.path.join(report_dir, f"moss-watch-{now[:10]}.html")
        with open(report_path, "w") as f:
            f.write("\n".join(html_parts))

        # Final output: short Discord body + MEDIA tag
        print("\n".join(output_lines))
        print(f"MEDIA:{report_path}")

if __name__ == "__main__":
    main()
