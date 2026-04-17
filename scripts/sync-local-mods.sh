#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=./lib/git-sync-common.sh
source "$SCRIPT_DIR/lib/git-sync-common.sh"

PATCH_BRANCH="henry/patches"
BASE_BRANCH="main"
SNAPSHOT_DIRTY=0
DRY_RUN=0
SMOKE_TEST=""

usage() {
  cat <<'EOF'
Usage: scripts/sync-local-mods.sh [options]

Refresh from upstream while preserving Henry-local patch commits.

Options:
  --patch-branch <name>   Branch containing local customizations (default: henry/patches)
  --base-branch <name>    Clean upstream-tracking branch to refresh (default: main)
  --snapshot-dirty        Snapshot a dirty working tree before sync work begins
  --smoke-test <command>  Command to run after replay in the disposable sync worktree
  --dry-run               Print the planned actions without mutating refs/worktrees
  --help                  Show this help text

Planned flow:
  1. fetch origin --prune
  2. reject or snapshot dirty local work
  3. hard-reset the base branch to origin/<base>
  4. create a disposable sync branch/worktree
  5. replay local patch commits from <base>..<patch-branch>
  6. optionally run a smoke test command in the sync worktree
EOF
}

log() {
  printf '[sync-local-mods] %s\n' "$*"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --patch-branch)
      [[ $# -ge 2 ]] || die "--patch-branch requires a value"
      PATCH_BRANCH="$2"
      shift 2
      ;;
    --base-branch)
      [[ $# -ge 2 ]] || die "--base-branch requires a value"
      BASE_BRANCH="$2"
      shift 2
      ;;
    --snapshot-dirty)
      SNAPSHOT_DIRTY=1
      shift
      ;;
    --smoke-test)
      [[ $# -ge 2 ]] || die "--smoke-test requires a value"
      SMOKE_TEST="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

ROOT=$(repo_root)
cd "$ROOT"

require_clean_or_snapshot_flag "$SNAPSHOT_DIRTY"

SYNC_ID="sync-$(timestamp)"
SYNC_BRANCH="sync/$SYNC_ID"
SYNC_PATH="$ROOT/.worktrees/$SYNC_ID"

log "repo root: $ROOT"
log "base branch: $BASE_BRANCH"
log "patch branch: $PATCH_BRANCH"

if working_tree_dirty; then
  log "dirty working tree detected; snapshot behavior will be implemented in a later phase"
fi

PLANNED_COMMANDS=(
  "git fetch origin --prune"
  "git checkout $BASE_BRANCH"
  "git reset --hard origin/$BASE_BRANCH"
  "git worktree add -b $SYNC_BRANCH $SYNC_PATH $BASE_BRANCH"
  "git rev-list --reverse $BASE_BRANCH..$PATCH_BRANCH"
)

if [[ -n "$SMOKE_TEST" ]]; then
  PLANNED_COMMANDS+=("git -C $SYNC_PATH sh -lc $(printf '%q' "$SMOKE_TEST")")
fi

for cmd in "${PLANNED_COMMANDS[@]}"; do
  log "plan: $cmd"
done

if [[ "$DRY_RUN" == "1" ]]; then
  log "dry-run requested; no changes applied"
  exit 0
fi

log "phase-1 skeleton only: command execution is not implemented yet"
log "next phase will add snapshot, reset, worktree creation, replay, and smoke-test execution"
