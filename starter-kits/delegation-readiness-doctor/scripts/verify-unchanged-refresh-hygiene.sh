#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KIT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ARTIFACT_DIR="$KIT_DIR/artifacts"
mkdir -p "$ARTIFACT_DIR"

TIMESTAMP="$(date +%Y-%m-%dT%H-%M-%S%z)"
REPORT_PATH="$ARTIFACT_DIR/unchanged-refresh-hygiene-proof-$TIMESTAMP.md"
LATEST_PATH="$ARTIFACT_DIR/latest-unchanged-refresh-hygiene-proof.md"
BEFORE_PATH="$(mktemp)"
AFTER_PATH="$(mktemp)"
REFRESH_LOG="$(mktemp)"

cleanup() {
  rm -f "$BEFORE_PATH" "$AFTER_PATH" "$REFRESH_LOG"
}
trap cleanup EXIT

snapshot_state() {
  local output_path="$1"
  python - "$ARTIFACT_DIR" "$output_path" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

artifact_dir = Path(sys.argv[1])
output_path = Path(sys.argv[2])
latest_names = sorted(path.name for path in artifact_dir.glob('latest-*.md'))
timestamped_names = sorted(
    path.name
    for path in artifact_dir.glob('*.md')
    if path.name not in latest_names and '-20' in path.name
)
latest_hashes = {}
for name in latest_names:
    data = (artifact_dir / name).read_bytes()
    latest_hashes[name] = hashlib.sha256(data).hexdigest()
output_path.write_text(json.dumps({
    'latest_hashes': latest_hashes,
    'timestamped_names': timestamped_names,
}, indent=2, sort_keys=True) + '\n', encoding='utf-8')
PY
}

snapshot_state "$BEFORE_PATH"

set +e
bash "$SCRIPT_DIR/refresh-upstream-blocker-packet.sh" >"$REFRESH_LOG" 2>&1
refresh_status=$?
set -e

snapshot_state "$AFTER_PATH"

python - "$BEFORE_PATH" "$AFTER_PATH" "$REFRESH_LOG" "$REPORT_PATH" "$LATEST_PATH" <<'PY'
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

before_path = Path(sys.argv[1])
after_path = Path(sys.argv[2])
refresh_log_path = Path(sys.argv[3])
report_path = Path(sys.argv[4])
latest_path = Path(sys.argv[5])

before = json.loads(before_path.read_text(encoding='utf-8'))
after = json.loads(after_path.read_text(encoding='utf-8'))
refresh_log = refresh_log_path.read_text(encoding='utf-8')
unchanged_token = 'UPSTREAM_BLOCKER_PACKET_UNCHANGED' in refresh_log
hygiene_token = 'UNCHANGED_PACKET_HYGIENE' in refresh_log
latest_unchanged = before['latest_hashes'] == after['latest_hashes']
timestamped_unchanged = before['timestamped_names'] == after['timestamped_names']

missing = []
if not unchanged_token:
    missing.append('refresh did not emit UPSTREAM_BLOCKER_PACKET_UNCHANGED')
if not hygiene_token:
    missing.append('refresh did not emit UNCHANGED_PACKET_HYGIENE')
if not latest_unchanged:
    missing.append('latest-* artifact hashes changed during an unchanged refresh')
if not timestamped_unchanged:
    before_set = set(before['timestamped_names'])
    after_set = set(after['timestamped_names'])
    added = sorted(after_set - before_set)
    removed = sorted(before_set - after_set)
    missing.append(f'timestamped artifact set changed; added={added}, removed={removed}')

verdict = 'UNCHANGED_REFRESH_HYGIENE_PROVED' if not missing else 'UNCHANGED_REFRESH_HYGIENE_FAILED'
now = datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')
log_excerpt = '\n'.join(refresh_log.splitlines()[-30:])
report = f"""# Delegation Readiness Doctor — Unchanged Refresh Hygiene Proof

Generated: {now}

## Verdict
{verdict}

## What this proves
This verifier guards the external-wait loop breaker: when the upstream blocker packet is materially unchanged, rerunning the one-command refresh must not rewrite canonical `latest-*` artifacts or leave fresh timestamped component artifacts behind.

## Checks
- Refresh emitted `UPSTREAM_BLOCKER_PACKET_UNCHANGED`: `{unchanged_token}`
- Refresh emitted `UNCHANGED_PACKET_HYGIENE`: `{hygiene_token}`
- Canonical `latest-*` artifact hashes unchanged: `{latest_unchanged}`
- Timestamped artifact set unchanged: `{timestamped_unchanged}`
- Latest artifact count checked: `{len(after['latest_hashes'])}`
- Timestamped artifact count checked: `{len(after['timestamped_names'])}`

## Refresh log excerpt
```text
{log_excerpt}
```

## Failure notes
{chr(10).join(f'- {item}' for item in missing) if missing else '- none'}
"""
report_path.write_text(report, encoding='utf-8')
shutil.copyfile(report_path, latest_path)
print(report_path)
print(verdict)
raise SystemExit(0 if not missing else 1)
PY

printf 'Latest proof: %s\n' "$LATEST_PATH"
