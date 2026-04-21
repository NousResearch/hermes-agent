#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "Refusing to update from a dirty worktree." >&2
  echo "Commit, stash, or clean changes first." >&2
  exit 1
fi

current_branch="$(git branch --show-current)"
if [[ -z "$current_branch" ]]; then
  echo "Could not determine current branch." >&2
  exit 1
fi

ts="$(date -u +%Y%m%dT%H%M%SZ)"
backup_branch="backup/pre-update-${ts}"
worktree_dir="$(dirname "$ROOT")/hermes-update-${ts}"

printf 'Creating backup branch: %s
' "$backup_branch"
git branch "$backup_branch" HEAD

printf 'Fetching upstream main...
'
git fetch origin main --tags
printf 'Refreshing local upstream-main mirror...
'
if git show-ref --verify --quiet refs/heads/upstream-main; then
  git branch -f upstream-main origin/main
else
  git branch upstream-main origin/main
fi

printf 'Creating temporary worktree: %s
' "$worktree_dir"
git worktree add "$worktree_dir" "$current_branch"

cleanup() {
  if [[ -d "$worktree_dir" ]]; then
    git -C "$ROOT" worktree remove --force "$worktree_dir" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

cd "$worktree_dir"
printf 'Rebasing %s onto upstream-main...
' "$current_branch"
git rebase upstream-main

cat <<'EOF'

Rebase complete in temporary worktree.
Run verification before pushing:

  /home/samade10/.hermes/venv/bin/python -m pytest -q -o addopts=''     tests/cli/test_run_agent_omx_handoff.py     tests/gateway/test_omx_executor.py     tests/gateway/test_meta_router_execution_result.py     tests/gateway/test_meta_router_memory_plan.py

  /home/samade10/.npm-global/bin/omx doctor
  cd /home/samade10/.openclaw/workspace
  python3 skills/maintainer/meta-router/scripts/model_learner.py
  python3 skills/maintainer/meta-router/scripts/skill_learner.py
  python3 skills/maintainer/meta-router/scripts/plugin_sync.py --report

If satisfied, push the rebased branch from the main repo.
The backup branch remains available for rollback.
EOF
