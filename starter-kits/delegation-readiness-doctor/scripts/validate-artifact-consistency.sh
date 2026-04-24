#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KIT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ARTIFACT_DIR="$KIT_DIR/artifacts"

python - "$ARTIFACT_DIR" <<'PY'
import re
import sys
from pathlib import Path

artifacts_dir = Path(sys.argv[1])
artifacts = [
    'latest-upstream-blocker-refresh.md',
    'latest-workflow-approval-state-change.md',
    'latest-pr-review-monitor.md',
    'latest-ci-result-interpreter.md',
    'latest-workflow-approval-trigger.md',
    'latest-workflow-approval-brief.md',
]

patterns = {
    'head': [
        re.compile(r'^- Head SHA: `(.*?)`$', re.MULTILINE),
        re.compile(r'^Head SHA: `(.*?)`$', re.MULTILINE),
        re.compile(r'^- Current head SHA: `(.*?)`$', re.MULTILINE),
        re.compile(r'^- Previous head SHA: `(.*?)`$', re.MULTILINE),
    ],
    'base': [
        re.compile(r'^- Base SHA: `(.*?)`$', re.MULTILINE),
        re.compile(r'^Base SHA: `(.*?)`$', re.MULTILINE),
        re.compile(r'^- Base SHA: \*\*`(.*?)`\*\*$', re.MULTILINE),
        re.compile(r'^- Current PR base SHA: `(.*?)`$', re.MULTILINE),
        re.compile(r'^- Previous base SHA: `(.*?)`$', re.MULTILINE),
    ],
}


def extract(text: str, field: str) -> str:
    for pattern in patterns[field]:
        match = pattern.search(text)
        if match:
            return match.group(1).strip().strip('`*')
    return 'missing'

rows = []
for name in artifacts:
    path = artifacts_dir / name
    if not path.exists():
        rows.append((name, 'missing-file', 'missing-file'))
        continue
    text = path.read_text(encoding='utf-8')
    rows.append((name, extract(text, 'head'), extract(text, 'base')))

heads = {row[1] for row in rows}
known_bases = {row[2] for row in rows if row[2] not in {'missing', 'missing-file'}}
missing_head_rows = [row for row in rows if row[1] in {'missing', 'missing-file'}]
missing_base_rows = [row for row in rows if row[2] in {'missing', 'missing-file'}]

print('# Delegation Readiness Doctor — Artifact Consistency Check')
print()
for name, head, base in rows:
    print(f'- {name}: head={head} | base={base}')

if (
    len(heads) == 1
    and 'missing' not in heads
    and 'missing-file' not in heads
    and len(known_bases) == 1
    and not missing_head_rows
    and not missing_base_rows
):
    base_summary = next(iter(known_bases))
    print()
    print(f'CONSISTENT: head={next(iter(heads))} | base={base_summary}')
    sys.exit(0)

print()
print('DRIFT_DETECTED')
sys.exit(1)
PY

chmod +x "$SCRIPT_DIR/validate-artifact-consistency.sh"
