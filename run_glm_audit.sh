#!/usr/bin/env bash
# Read-only independent audit. It deliberately receives no Claude findings.
set -euo pipefail
ROOT=/home/lfdm/worktrees/hermes-routing-audit-20260710
OUT="$ROOT/audit-results"
mkdir -p "$OUT"
cd "$ROOT"
set +e
glm-code -p 'Read AUDIT_PACKET.md and independently conduct the required W0/D-advisory source audit. Use only Read, Grep, and Glob. Return the six requested markdown sections. Do not modify anything or invoke terminal/service/cron/Kanban/delegation actions. Do not assume any competing audit exists.' \
  --allowedTools 'Read,Grep,Glob' \
  --max-turns 12 \
  --output-format json \
  > "$OUT/glm-5.2-audit.json" 2>&1
rc=$?
set -e
printf 'exit_code=%s\ncompleted_at=%s\n' "$rc" "$(date --iso-8601=seconds)" > "$OUT/glm-5.2-audit.status"
exit "$rc"
