#!/usr/bin/env bash
# Read-only external audit. Run only inside the isolated routing-audit worktree.
set -euo pipefail
ROOT=/home/lfdm/worktrees/hermes-routing-audit-20260710
OUT="$ROOT/audit-results"
mkdir -p "$OUT"
cd "$ROOT"
set +e
/home/lfdm/.hermes/scripts/claude-delegate \
  --task 'Read AUDIT_PACKET.md and conduct the required W0/D-advisory source audit. Use only Read, Grep, and Glob. Return the six requested markdown sections. Do not modify anything or invoke terminal/service/cron/Kanban/delegation actions.' \
  --packet "$ROOT/AUDIT_PACKET.md" \
  --tools 'Read,Grep,Glob' \
  --max-turns 12 \
  --timeout 600 \
  > "$OUT/claude-max-audit.json" 2>&1
rc=$?
set -e
printf 'exit_code=%s\ncompleted_at=%s\n' "$rc" "$(date --iso-8601=seconds)" > "$OUT/claude-max-audit.status"
exit "$rc"
