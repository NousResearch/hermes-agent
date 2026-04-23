#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KIT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$KIT_DIR/../.." && pwd)"

python3 - "$REPO_DIR" "$KIT_DIR" <<'PY'
import json
import os
import re
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path

repo_dir = Path(sys.argv[1])
kit_dir = Path(sys.argv[2])
artifacts_dir = kit_dir / "artifacts"
artifacts_dir.mkdir(parents=True, exist_ok=True)

OWNER = "NousResearch"
REPO = "hermes-agent"
PR_NUMBER = 14297
BRANCH_REF = "fork/hermes/delegation-readiness-doctor-clean"
API = f"https://api.github.com/repos/{OWNER}/{REPO}"


def run(cmd, cwd=None, check=True):
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if check and proc.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd)}\nstdout={proc.stdout}\nstderr={proc.stderr}")
    return proc


def git_credential_token():
    cred_path = Path.home() / ".git-credentials"
    if not cred_path.exists():
        return None
    for line in cred_path.read_text().splitlines():
        if "github.com" not in line:
            continue
        parsed = urllib.parse.urlparse(line.strip())
        if parsed.username and parsed.password:
            return urllib.parse.unquote(parsed.password)
        if "@" in parsed.netloc and ":" in parsed.netloc.split("@")[0]:
            return urllib.parse.unquote(parsed.netloc.split("@")[0].split(":", 1)[1])
    return None


def gh_get(path):
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "hermes-starter-kit-refresh",
    }
    token = git_credential_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(f"{API}{path}", headers=headers)
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def now(fmt):
    return run(["date", fmt]).stdout.strip()


def write_artifact(stem, body):
    stamp = now("+%Y-%m-%dT%H-%M-%S%z")
    timestamped = artifacts_dir / f"{stem}-{stamp}.md"
    latest = artifacts_dir / f"latest-{stem}.md"
    previous = latest.read_text() if latest.exists() else None
    timestamped.write_text(body)
    latest.write_text(body)
    return latest, timestamped, previous


def extract_prior_signature(text):
    if not text:
        return None
    match = re.search(r"State signature: `(.*?)`", text)
    return match.group(1) if match else None


def ahead_behind():
    run(["git", "fetch", "--all", "--prune"], cwd=repo_dir)
    counts = run(["git", "rev-list", "--left-right", "--count", f"{BRANCH_REF}...origin/main"], cwd=repo_dir).stdout.strip()
    ahead_str, behind_str = counts.split()
    origin_main_sha = run(["git", "rev-parse", "origin/main"], cwd=repo_dir).stdout.strip()
    return int(ahead_str), int(behind_str), origin_main_sha


pr = gh_get(f"/pulls/{PR_NUMBER}")
head_sha = pr["head"]["sha"]
base_sha = pr["base"]["sha"]
reviews = gh_get(f"/pulls/{PR_NUMBER}/reviews?per_page=100")
issue_comments = gh_get(f"/issues/{PR_NUMBER}/comments?per_page=100")
status = gh_get(f"/commits/{head_sha}/status")
check_runs = gh_get(f"/commits/{head_sha}/check-runs?per_page=100")
check_suites = gh_get(f"/commits/{head_sha}/check-suites?per_page=100")

review_count = len(reviews)
issue_comment_count = len(issue_comments)
check_run_items = check_runs.get("check_runs", [])
check_suite_items = check_suites.get("check_suites", [])
action_required_suites = [s for s in check_suite_items if s.get("conclusion") == "action_required"]
ahead_count, behind_count, origin_main_sha = ahead_behind()
ahead_behind_value = f"{ahead_count} / {behind_count}"
created = now("+%Y-%m-%d %H:%M %Z")

maintainer_request = None
for comment in issue_comments:
    body = comment.get("body", "")
    author = (comment.get("user") or {}).get("login", "")
    if "Maintainer unblock request for PR #14297" in body or (author == "NplusM420" and "action_required" in body):
        maintainer_request = comment
        break

historical_candidates = [
    "artifacts/latest-readiness-proof.md",
    "artifacts/latest-clean-commit-surface.md",
    "artifacts/latest-broken-state-roundtrip.md",
    "artifacts/latest-reviewer-handoff.md",
    "artifacts/latest-ship-review.md",
]
missing_historical = [rel for rel in historical_candidates if not (kit_dir / rel).exists()]

branch_current = behind_count == 0 and base_sha == origin_main_sha
approval_blocked = branch_current and len(action_required_suites) > 0 and not check_run_items

if behind_count > 0:
    blocker_call = (
        f"The approval-only model is stale. The PR branch is {behind_count} commit(s) behind live origin/main "
        f"({origin_main_sha[:12]}) and needs a fresh replay/branch refresh before workflow approval is the real blocker again."
    )
    exact_next_move = (
        f"Create a fresh worktree from origin/main ({origin_main_sha[:12]}), replay the MVP surface, rerun the focused proof suite, "
        f"and refresh PR #{PR_NUMBER} before resuming the workflow-approval wait loop."
    )
elif approval_blocked:
    blocker_call = "The PR is current on live upstream base. The only blocker is maintainer workflow approval / first real CI movement."
    exact_next_move = (
        f"Watch PR #{PR_NUMBER} for workflow approval, check-run start, or review activity on head {head_sha}. "
        f"On the next state change, rerun this script and answer that exact signal immediately."
    )
else:
    blocker_call = "The blocker surface has changed; inspect the regenerated review and CI artifacts now."
    exact_next_move = "Route the changed review/CI surface through the regenerated packet immediately."

pr_monitor_next_move = (
    "Do not repost the maintainer nudge. Wait for workflow approval, a real check run, or a review event, then rerun this packet and answer that exact signal."
    if maintainer_request and approval_blocked
    else exact_next_move
)

pr_monitor = f"""# Delegation Readiness Doctor — PR Review Monitor

Generated: {created}

## PR identity
- Title: {pr['title']}
- URL: {pr['html_url']}
- State: {pr['state']}
- Draft: {pr['draft']}
- Mergeable: {pr['mergeable']}
- Mergeable state: {pr['mergeable_state']}
- Base ← Head: `main <- {pr['head']['label']}`
- Head SHA: `{head_sha}`
- Base SHA: `{base_sha}`
- Live `origin/main` SHA: `{origin_main_sha}`
- Commits / files: `{pr['commits']} commits`, `{pr['changed_files']} files`
- Additions / deletions: `{pr['additions']} / {pr['deletions']}`
- Ahead / behind vs `origin/main`: `{ahead_behind_value}`

## Review surface
- Review count: {review_count}
- Issue comment count: {issue_comment_count}
- Review comment count: {pr['review_comments']}

## Automation surface
- Combined statuses: {len(status.get('statuses', []))}
- Combined status state: {status.get('state')}
- Check runs: {len(check_run_items)}
- Check suites: {len(check_suite_items)}
- Action-required suites: {len(action_required_suites)}

## Live blocker
{blocker_call}

## Exact next move
{pr_monitor_next_move}
"""

ci_verdict = (
    "STALE_BASE_DRIFT"
    if behind_count > 0
    else "WAITING_FOR_WORKFLOW_APPROVAL"
    if approval_blocked
    else "CHECK_RUNS_PRESENT"
    if check_run_items
    else "NO_ACTION_REQUIRED_SUITES"
)
ci_lines = []
for run_item in check_run_items[:10]:
    ci_lines.append(f"- {run_item['name']} — {run_item.get('status')} / {run_item.get('conclusion')}")
if not ci_lines:
    ci_lines.append("- none yet")

historical_lines = [f"- {item}" for item in missing_historical] or ["- none"]
ci_interpreter = f"""# Delegation Readiness Doctor — CI Result Interpreter

Generated: {created}
PR: {pr['html_url']}
Head SHA: `{head_sha}`
Verdict: **{ci_verdict}**

## Current CI surface
- Combined status state: {status.get('state')}
- Check runs: {len(check_run_items)}
- Check suites: {len(check_suite_items)}
- Action-required suites: {len(action_required_suites)}
- Ahead / behind vs `origin/main`: {ahead_behind_value}

### Check runs
{os.linesep.join(ci_lines)}

## Historical proof pointers still missing in this checkout
{os.linesep.join(historical_lines)}

## Exact next move
{exact_next_move}
"""

suite_lines = []
for suite in action_required_suites[:10]:
    suite_lines.append(
        f"- Suite `{suite['id']}` — {suite.get('status')} / {suite.get('conclusion')} | created {suite.get('created_at')} | updated {suite.get('updated_at')}"
    )
if not suite_lines:
    suite_lines.append("- none")

workflow_brief = f"""# Delegation Readiness Doctor — Workflow Approval Brief

Generated: {created}
PR: {pr['html_url']}
Head SHA: `{head_sha}`
Base SHA: `{base_sha}`
Live `origin/main` SHA: `{origin_main_sha}`

## Live signature
- Combined status state: {status.get('state')}
- Combined status contexts: {len(status.get('statuses', []))}
- Check runs: {len(check_run_items)}
- Check suites: {len(check_suite_items)}
- Action-required suites: {len(action_required_suites)}
- Ahead / behind vs `origin/main`: {ahead_behind_value}

## Action-required suites
{os.linesep.join(suite_lines)}

## Exact maintainer move
{'A maintainer with repo permissions needs to approve and run the PR workflows for this forked branch/head commit.' if approval_blocked else 'Workflow approval is not the only active blocker; inspect branch freshness or live CI first.'}

## Verification after approval
1. Refresh `latest-pr-review-monitor.md`.
2. Confirm at least one real check run or status context exists for head `{head_sha}`.
3. If a failing run appears, answer that exact failure instead of treating the PR as approval-blocked.
"""

trigger_state = "ALREADY_POSTED_REFERENCE_ONLY" if maintainer_request else "READY_TO_POST"
trigger_body = maintainer_request.get('body', '').strip() if maintainer_request else (
    "Maintainer unblock request for PR #14297:\n\nThe Delegation Readiness Doctor PR is ready for review, but GitHub has the fork workflows stuck at `action_required` with 0 real check runs."
)
workflow_trigger = f"""# Delegation Readiness Doctor — Workflow Approval Trigger

Generated: {created}
PR: {pr['html_url']}
Head SHA: `{head_sha}`
Trigger state: **{trigger_state}**

## Current blocker
- Action-required suites: {len(action_required_suites)}
- Real check runs: {len(check_run_items)}
- Existing maintainer request comment: {'yes' if maintainer_request else 'no'}
- Ahead / behind vs `origin/main`: {ahead_behind_value}

## Maintainer nudge text
{trigger_body}

## Exact next move
{'Do not repost unless the blocker signature changes materially.' if maintainer_request and approval_blocked else exact_next_move}
"""

state_signature = {
    "head_sha": head_sha,
    "base_sha": base_sha,
    "origin_main_sha": origin_main_sha,
    "ahead": ahead_count,
    "behind": behind_count,
    "action_required_suites": len(action_required_suites),
    "check_runs": len(check_run_items),
    "reviews": review_count,
    "issue_comments": issue_comment_count,
    "maintainer_request_posted": bool(maintainer_request),
}
state_signature_json = json.dumps(state_signature, sort_keys=True)
state_change = f"""# Delegation Readiness Doctor — Workflow Approval State Change

Generated: {created}
State signature: `{state_signature_json}`

## Verdict
{'BLOCKER_PERSISTS' if approval_blocked else 'BLOCKER_CHANGED'}

## Exact next move
{'Wait for real upstream movement; do not repost the existing maintainer request.' if maintainer_request and approval_blocked else exact_next_move}
"""

refresh_body = f"""# Delegation Readiness Doctor — Upstream Blocker Refresh

Generated: {created}
PR: {pr['html_url']}
State signature: `{state_signature_json}`

## Current live state
- Head SHA: `{head_sha}`
- Base SHA: `{base_sha}`
- Live `origin/main` SHA: `{origin_main_sha}`
- Mergeable: `{pr['mergeable']}`
- Mergeable state: `{pr['mergeable_state']}`
- Ahead / behind vs `origin/main`: `{ahead_behind_value}`
- GitHub check suites: `{len(check_suite_items)}` total / `{len(action_required_suites)}` action_required
- GitHub check runs: `{len(check_run_items)}`
- Reviews: `{review_count}`
- Issue comments: `{issue_comment_count}`

## Blocker call
{blocker_call}

## Durable packet restored in this checkout
- `artifacts/latest-pr-review-monitor.md`
- `artifacts/latest-ci-result-interpreter.md`
- `artifacts/latest-workflow-approval-brief.md`
- `artifacts/latest-workflow-approval-trigger.md`
- `artifacts/latest-workflow-approval-state-change.md`

## Historical proof pointers still missing in this checkout
{os.linesep.join(historical_lines)}

## Exact next move
{exact_next_move}
"""

write_artifact("pr-review-monitor", pr_monitor)
write_artifact("ci-result-interpreter", ci_interpreter)
write_artifact("workflow-approval-brief", workflow_brief)
write_artifact("workflow-approval-trigger", workflow_trigger)
write_artifact("workflow-approval-state-change", state_change)
latest_refresh, timestamped_refresh, prev_refresh_packet = write_artifact("upstream-blocker-refresh", refresh_body)

previous_signature = extract_prior_signature(prev_refresh_packet)
change_vs_previous = "unchanged" if previous_signature == state_signature_json else "changed"
refresh_with_change = refresh_body + f"\n## Change vs previous packet\n- {change_vs_previous}\n"
latest_refresh.write_text(refresh_with_change)
timestamped_refresh.write_text(refresh_with_change)

print("UPSTREAM_BLOCKER_PACKET_UNCHANGED" if previous_signature == state_signature_json else "UPSTREAM_BLOCKER_PACKET_REFRESHED")
print(str(latest_refresh))
print(str(timestamped_refresh))
PY
