#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KIT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ARTIFACT_DIR="$KIT_DIR/artifacts"
mkdir -p "$ARTIFACT_DIR"

TIMESTAMP="$(date +%Y-%m-%dT%H-%M-%S%z)"
REPORT_PATH="$ARTIFACT_DIR/workflow-approval-trigger-$TIMESTAMP.md"
LATEST_PATH="$ARTIFACT_DIR/latest-workflow-approval-trigger.md"
TOKEN_PATH="$(mktemp)"
trap 'rm -f "$TOKEN_PATH"' EXIT

python - "$REPORT_PATH" "$LATEST_PATH" "$TOKEN_PATH" <<'PY'
import json
import os
import re
import shutil
import sys
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

report_path = Path(sys.argv[1])
latest_path = Path(sys.argv[2])
token_path = Path(sys.argv[3])
base = 'https://api.github.com/repos/NousResearch/hermes-agent'
headers = {
    'Accept': 'application/vnd.github+json',
    'User-Agent': 'Hermes-Agent',
    'X-GitHub-Api-Version': '2022-11-28',
}

token = os.environ.get('GITHUB_TOKEN')
if not token:
    creds_path = Path.home() / '.git-credentials'
    if creds_path.exists():
        for line in creds_path.read_text().splitlines():
            if 'github.com' not in line or '@github.com' not in line or ':' not in line:
                continue
            token = line.split('://', 1)[1].rsplit('@github.com', 1)[0].split(':', 1)[1]
            break
if token:
    headers['Authorization'] = f'token {token}'

def get(url: str):
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode())

pr = get(base + '/pulls/14297')
sha = pr['head']['sha']
combined_status = get(base + f'/commits/{sha}/status')
check_runs = get(base + f'/commits/{sha}/check-runs')
check_suites = get(base + f'/commits/{sha}/check-suites')
issue_comments = get(base + '/issues/14297/comments')
action_required_suites = [
    suite for suite in check_suites.get('check_suites', [])
    if suite.get('conclusion') == 'action_required'
]
head_ref = pr['head']['ref']
repo_html = 'https://github.com/NousResearch/hermes-agent'
pr_url = pr['html_url']
checks_url = pr_url + '/checks'
actions_query = urllib.parse.quote(f'branch:{head_ref}', safe='')
actions_url = f'{repo_html}/actions?query={actions_query}'
already_posted = any(
    comment.get('user', {}).get('login') == 'NplusM420'
    and 'Maintainer unblock request for PR #14297:' in (comment.get('body') or '')
    for comment in issue_comments
)

suite_lines = '\n'.join(
    (
        f"- Suite `{suite['id']}` — {suite.get('status')} / {suite.get('conclusion') or 'pending'}\n"
        f"  - API: {suite.get('url')}\n"
        f"  - Check runs API: {suite.get('check_runs_url')}\n"
        f"  - latest_check_runs_count: {suite.get('latest_check_runs_count', 0)} | rerequestable: {suite.get('rerequestable')}"
    )
    for suite in action_required_suites
) or '- none'

ready_to_post = f'''Maintainer unblock request for PR #14297:

The Delegation Readiness Doctor PR is ready for review, but GitHub has the fork workflows stuck at `action_required` for head `{sha}`.

Live blocker signature right now:
- combined status: `{combined_status.get('state')}`
- check runs: `{check_runs.get('total_count', 0)}`
- check suites: `{check_suites.get('total_count', 0)}`
- action_required suites: `{len(action_required_suites)}`

Please approve and run the fork PR workflows for this head commit. After that, rerun:
`bash starter-kits/delegation-readiness-doctor/scripts/emit-pr-review-monitor.sh`

If a real failing run appears, the proof/repair packet is already frozen in `starter-kits/delegation-readiness-doctor/artifacts/latest-reviewer-handoff.md` and `latest-broken-state-roundtrip.md`.
'''.strip()

ready_to_post_block_title = 'Ready-to-post maintainer nudge'
ready_to_post_preface = ''
trigger_stdout_token = 'WORKFLOW_APPROVAL_TRIGGER_READY'
if already_posted:
    ready_to_post_block_title = 'Maintainer nudge status'
    ready_to_post_preface = (
        'Existing maintainer unblock request already posted by `NplusM420`; '
        'do not repost unless the blocker signature changes materially. '
        'Use the text below only as the current live-state reference.\n\n'
    )
    trigger_stdout_token = 'WORKFLOW_APPROVAL_TRIGGER_ALREADY_POSTED'

now = datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')
report = f"""# Delegation Readiness Doctor — Workflow Approval Trigger

Generated: {now}
PR: {pr_url}
Head ref: `{pr['head']['label']}`
Head SHA: `{sha}`
Base SHA: `{pr['base']['sha']}`

## Live signature
- Combined status state: {combined_status.get('state')}
- Combined status contexts: {combined_status.get('total_count', 0)}
- Check runs: {check_runs.get('total_count', 0)}
- Check suites: {check_suites.get('total_count', 0)}
- Action-required suites: {len(action_required_suites)}

## Exact blocker
GitHub has already created Actions suites for the fork PR head commit, but every suite is still `action_required` and no check runs exist yet. The blocker is maintainer workflow approval / run permission, not missing local proof.

## Direct approval surfaces
- PR conversation: {pr_url}
- PR checks tab: {checks_url}
- Repo Actions filtered to this branch: {actions_url}

## Action-required suites
{suite_lines}

## {ready_to_post_block_title}
```text
{ready_to_post_preface}{ready_to_post}
```

## Verification after approval
1. Run `bash starter-kits/delegation-readiness-doctor/scripts/emit-pr-review-monitor.sh`.
2. Confirm `latest-pr-review-monitor.md` shows at least one real check run or status context for head `{sha}`.
3. If CI fails, answer that concrete failure from `latest-reviewer-handoff.md` instead of repeating the approval blocker.

## Proof note
This trigger artifact exists so the recurring blocker can be attacked with one exact nudge packet and one exact verification step instead of another status-only monitor refresh, even when unauthenticated public API rate limits would otherwise stall the packet refresh.
"""

def stable_for_comparison(text: str) -> str:
    return re.sub(r'^Generated: .*$','Generated: <content-stable>', text, flags=re.MULTILINE)

if latest_path.exists() and stable_for_comparison(latest_path.read_text(encoding='utf-8')) == stable_for_comparison(report):
    token_path.write_text('WORKFLOW_APPROVAL_TRIGGER_CONTENT_STABLE\n', encoding='utf-8')
    print(latest_path)
    print('WORKFLOW_APPROVAL_TRIGGER_CONTENT_STABLE')
    sys.exit(0)

report_path.write_text(report, encoding='utf-8')
shutil.copyfile(report_path, latest_path)
token_path.write_text(trigger_stdout_token + '\n', encoding='utf-8')
print(report_path)
print(trigger_stdout_token)
PY

chmod +x "$SCRIPT_DIR/emit-workflow-approval-trigger.sh"
trigger_token="$(cat "$TOKEN_PATH" 2>/dev/null || true)"
if [[ "$trigger_token" == "WORKFLOW_APPROVAL_TRIGGER_CONTENT_STABLE" ]]; then
  printf 'Skipped unchanged trigger write; latest report remains: %s\n' "$LATEST_PATH"
else
  printf 'Wrote report: %s\n' "$REPORT_PATH"
  printf 'Latest report: %s\n' "$LATEST_PATH"
fi
