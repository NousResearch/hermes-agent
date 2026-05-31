#!/usr/bin/env bash
# Write .sync-failure.json for the sync-upstream workflow failure handler.
# Usage: record-sync-failure.sh <failed_step> [failed_branch]
set -euo pipefail

FAILED_STEP="${1:?failed_step required}"
FAILED_BRANCH="${2:-}"
UPSTREAM_SHA="${upstream_sha:-unknown}"

CONFLICT_FILES="[]"
if git rev-parse --git-dir >/dev/null 2>&1; then
  mapfile -t files < <(git diff --name-only --diff-filter=U 2>/dev/null || true)
  if ((${#files[@]})); then
    CONFLICT_FILES="$(printf '%s\n' "${files[@]}" | jq -R . | jq -s .)"
  fi
fi

jq -n \
  --arg failed_step "$FAILED_STEP" \
  --arg failed_branch "$FAILED_BRANCH" \
  --arg upstream_sha "$UPSTREAM_SHA" \
  --argjson conflict_files "$CONFLICT_FILES" \
  '{
    failed_step: $failed_step,
    failed_branch: $failed_branch,
    upstream_sha: $upstream_sha,
    conflict_files: $conflict_files
  }' > .sync-failure.json

echo "Recorded sync failure: step=$FAILED_STEP branch=$FAILED_BRANCH" >&2
