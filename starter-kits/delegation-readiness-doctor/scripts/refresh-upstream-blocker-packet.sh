#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KIT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ARTIFACT_DIR="$KIT_DIR/artifacts"
mkdir -p "$ARTIFACT_DIR"

TIMESTAMP="$(date +%Y-%m-%dT%H-%M-%S%z)"
REPORT_PATH="$ARTIFACT_DIR/upstream-blocker-refresh-$TIMESTAMP.md"
LATEST_PATH="$ARTIFACT_DIR/latest-upstream-blocker-refresh.md"

# Keep the maintainer handoff baseline current before running the state-change
# detector. The detector intentionally reads latest-reviewer-handoff.md as its
# branch-refresh baseline; if this baseline is stale immediately after a
# GitHub-side branch refresh, the detector will falsely report BASE_BRANCH_ADVANCED.
bash "$SCRIPT_DIR/sync-reviewer-handoff-baseline.sh"
bash "$SCRIPT_DIR/emit-workflow-approval-state-change.sh"
bash "$SCRIPT_DIR/emit-pr-review-monitor.sh"
bash "$SCRIPT_DIR/emit-ci-result-interpreter.sh"
bash "$SCRIPT_DIR/emit-workflow-approval-trigger.sh"
bash "$SCRIPT_DIR/emit-workflow-approval-brief.sh"

python - "$ARTIFACT_DIR" "$REPORT_PATH" "$LATEST_PATH" <<'PY'
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path


def extract_latest_signature(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    text = path.read_text(encoding='utf-8')

    def match_field(pattern: str, default: str = 'unknown', flags: int = re.MULTILINE) -> str:
        m = re.search(pattern, text, flags)
        return m.group(1).strip() if m else default

    return {
        'head_sha': match_field(r'^- Head SHA: `(.*?)`$'),
        'base_sha': match_field(r'^- Base SHA: `(.*?)`$'),
        'mergeable': match_field(r'^- Mergeable: `(.*?)`$'),
        'mergeable_state': match_field(r'^- Mergeable state: `(.*?)`$'),
        'review_triplet': match_field(r'^- Review / issue comment / review comment counts: `(.*?)`$'),
        'combined_status': match_field(r'^- Combined status: `(.*?)`$'),
        'check_runs': match_field(r'^- Check runs: `(.*?)`$'),
        'action_required': match_field(r'^- Action-required suites: `(.*?)`$'),
        'state_change_verdict': match_field(r'^- State-change verdict: `(.*?)`$'),
        'ci_verdict': match_field(r'^- CI interpreter verdict: `(.*?)`$'),
        'trigger_mode': match_field(r'^- Maintainer trigger mode: `(.*?)`$'),
        'artifact_consistency': match_field(r'^- Artifact consistency: `(.*?)`$'),
        'blocker': match_field(r'^## Live blocker\n(.*?)(?:\n## |\Z)', default='Unknown', flags=re.MULTILINE | re.DOTALL).strip(),
        'next_move': match_field(r'^## Exact next move\n(.*?)(?:\n## |\Z)', default='Unknown', flags=re.MULTILINE | re.DOTALL).strip(),
    }

artifacts_dir = Path(sys.argv[1])
report_path = Path(sys.argv[2])
latest_path = Path(sys.argv[3])

previous_signature = extract_latest_signature(latest_path)

state_change = (artifacts_dir / 'latest-workflow-approval-state-change.md').read_text(encoding='utf-8')
pr_monitor = (artifacts_dir / 'latest-pr-review-monitor.md').read_text(encoding='utf-8')
ci_interp = (artifacts_dir / 'latest-ci-result-interpreter.md').read_text(encoding='utf-8')
trigger = (artifacts_dir / 'latest-workflow-approval-trigger.md').read_text(encoding='utf-8')
approval_brief = (artifacts_dir / 'latest-workflow-approval-brief.md').read_text(encoding='utf-8')


def match(text: str, pattern: str, default: str = 'unknown', flags: int = re.MULTILINE) -> str:
    m = re.search(pattern, text, flags)
    return m.group(1).strip() if m else default

head_sha = match(pr_monitor, r'^- Head SHA: `(.*?)`$')
base_sha = match(trigger, r'^Base SHA: `(.*?)`$')
brief_head_sha = match(approval_brief, r'^Head SHA: `(.*?)`$')
brief_base_sha = match(approval_brief, r'^Base SHA: `(.*?)`$')
mergeable = match(pr_monitor, r'^- Mergeable: (.*?)$')
mergeable_state = match(pr_monitor, r'^- Mergeable state: (.*?)$')
review_count = match(pr_monitor, r'^- Review count: (.*?)$')
issue_comment_count = match(pr_monitor, r'^- Issue comment count: (.*?)$')
review_comment_count = match(pr_monitor, r'^- Review comment count: (.*?)$')
combined_status = match(ci_interp, r'^- Combined status state: (.*?)$')
check_runs = match(ci_interp, r'^- Check runs: (.*?)$')
action_required = match(ci_interp, r'^- Action-required suites: (.*?)$')
state_change_verdict = match(state_change, r'^\*\*(.*?)\*\*$')
ci_verdict = match(ci_interp, r'^Verdict: \*\*(.*?)\*\*$')
next_move = match(state_change, r'^## Exact next move\n(.*?)(?:\n## |\Z)', default='Refresh the blocker packet and answer the first real upstream signal immediately.', flags=re.MULTILINE | re.DOTALL).strip()
blocker = match(pr_monitor, r'^## Live blocker\n(.*?)(?:\n## |\Z)', default='Unknown', flags=re.MULTILINE | re.DOTALL).strip()
trigger_mode = 'already-posted reference only' if 'Existing maintainer unblock request already posted' in trigger else 'ready-to-post nudge'
artifact_consistency = 'consistent' if head_sha == brief_head_sha and base_sha == brief_base_sha else f'mismatch: trigger/pr `{head_sha}`/`{base_sha}` vs brief `{brief_head_sha}`/`{brief_base_sha}`'
current_signature = {
    'head_sha': head_sha,
    'base_sha': base_sha,
    'mergeable': mergeable,
    'mergeable_state': mergeable_state,
    'review_triplet': f'{review_count} / {issue_comment_count} / {review_comment_count}',
    'combined_status': combined_status,
    'check_runs': check_runs,
    'action_required': action_required,
    'state_change_verdict': state_change_verdict,
    'ci_verdict': ci_verdict,
    'trigger_mode': trigger_mode,
    'artifact_consistency': artifact_consistency,
    'blocker': blocker,
    'next_move': next_move,
}
material_change = previous_signature is None or any(
    previous_signature.get(key) != value for key, value in current_signature.items()
)
change_summary = (
    'No material blocker-state change since the previous `latest-upstream-blocker-refresh.md` snapshot; this run refreshed the packet and confirmed the blocker is unchanged.'
    if not material_change
    else 'Material blocker-state change detected versus the previous `latest-upstream-blocker-refresh.md` snapshot. Treat this packet as the new canonical blocker surface.'
)
refresh_token = 'UPSTREAM_BLOCKER_PACKET_REFRESHED' if material_change else 'UPSTREAM_BLOCKER_PACKET_UNCHANGED'

now = datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')
report = f"""# Delegation Readiness Doctor — Upstream Blocker Refresh

Generated: {now}

## Why this artifact exists
One-command refresh of the live upstream blocker packet so a cron pass can update every approval/CI artifact together and make one honest blocker call from the same head SHA.

## Refreshed surfaces
- `latest-reviewer-handoff.md`
- `latest-workflow-approval-state-change.md`
- `latest-pr-review-monitor.md`
- `latest-ci-result-interpreter.md`
- `latest-workflow-approval-trigger.md`
- `latest-workflow-approval-brief.md`

## Live summary
- Head SHA: `{head_sha}`
- Base SHA: `{base_sha}`
- Mergeable: `{mergeable}`
- Mergeable state: `{mergeable_state}`
- Review / issue comment / review comment counts: `{review_count} / {issue_comment_count} / {review_comment_count}`
- Combined status: `{combined_status}`
- Check runs: `{check_runs}`
- Action-required suites: `{action_required}`
- State-change verdict: `{state_change_verdict}`
- CI interpreter verdict: `{ci_verdict}`
- Maintainer trigger mode: `{trigger_mode}`
- Artifact consistency: `{artifact_consistency}`

## Live blocker
{blocker}

## Exact next move
{next_move}

## Change vs previous packet
{change_summary}

## Verification note
This packet is only honest if the five component artifacts above were refreshed in the same run and agree on the live head/base SHA pair. Re-run this script instead of refreshing those files piecemeal when the next cron pass needs a current blocker packet.
"""
report_path.write_text(report, encoding='utf-8')
shutil.copyfile(report_path, latest_path)
print(report_path)
print(refresh_token)
PY

bash "$SCRIPT_DIR/validate-artifact-consistency.sh"

chmod +x "$SCRIPT_DIR/refresh-upstream-blocker-packet.sh"
printf 'Wrote report: %s\n' "$REPORT_PATH"
printf 'Latest report: %s\n' "$LATEST_PATH"
