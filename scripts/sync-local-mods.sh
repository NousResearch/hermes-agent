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
SYNC_ID=""
SYNC_BRANCH=""
SYNC_PATH=""
RESCUE_BRANCH=""
ORIGINAL_BRANCH=""
ROOT=""

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

Flow:
  1. fetch origin --prune
  2. reject or snapshot dirty local work into rescue/<timestamp>
  3. hard-reset the base branch to origin/<base>
  4. create a disposable sync branch/worktree
  5. replay local patch commits from <base>..<patch-branch>
  6. optionally run a smoke test command in the sync worktree
EOF
}

log() {
  printf '[sync-local-mods] %s\n' "$*"
}

run_cmd() {
  if [[ "$DRY_RUN" == "1" ]]; then
    log "plan: $*"
    return 0
  fi
  log "run: $*"
  "$@"
}

run_in_sync_worktree() {
  if [[ "$DRY_RUN" == "1" ]]; then
    log "plan: (cd $SYNC_PATH && sh -lc $*)"
    return 0
  fi
  log "run in sync worktree: $*"
  (
    cd "$SYNC_PATH"
    sh -lc "$*"
  )
}

snapshot_dirty_work() {
  [[ -n "$RESCUE_BRANCH" ]] || RESCUE_BRANCH="rescue/$(timestamp)"

  if [[ "$DRY_RUN" == "1" ]]; then
    log "dirty working tree detected"
    log "plan: git switch -c $RESCUE_BRANCH"
    log "plan: git add -A"
    log "plan: git commit -m wip: pre-sync snapshot before upstream refresh"
    return 0
  fi

  log "dirty working tree detected"
  run_cmd git switch -c "$RESCUE_BRANCH"
  run_cmd git add -A
  if git diff --cached --quiet; then
    die "expected staged changes for rescue snapshot, but index is empty"
  fi
  run_cmd git commit -m "wip: pre-sync snapshot before upstream refresh"
  log "created rescue snapshot: $RESCUE_BRANCH"
}

verify_prerequisites() {
  if [[ "$DRY_RUN" == "1" ]]; then
    log "plan: verify refs origin/$BASE_BRANCH and $PATCH_BRANCH"
    return 0
  fi

  remote_branch_exists "origin/$BASE_BRANCH" || die "missing remote branch origin/$BASE_BRANCH"
  branch_exists "$PATCH_BRANCH" || die "missing local patch branch $PATCH_BRANCH"
}

create_sync_worktree() {
  ensure_dir_missing "$SYNC_PATH"
  if branch_exists "$SYNC_BRANCH"; then
    die "sync branch already exists: $SYNC_BRANCH"
  fi
  run_cmd git worktree add -b "$SYNC_BRANCH" "$SYNC_PATH" "$BASE_BRANCH"
}

replay_patch_stack() {
  if [[ "$DRY_RUN" == "1" ]]; then
    log "plan: git rev-list --reverse ${BASE_BRANCH}..${PATCH_BRANCH}"
    log "plan: replay each listed commit into $SYNC_PATH with cherry-pick"
    return 0
  fi

  local -a patch_commits
  mapfile -t patch_commits < <(git rev-list --reverse "${BASE_BRANCH}..${PATCH_BRANCH}")

  if [[ "${#patch_commits[@]}" -eq 0 ]]; then
    log "no local patch commits to replay from ${BASE_BRANCH}..${PATCH_BRANCH}"
    return 0
  fi

  local sha subject
  for sha in "${patch_commits[@]}"; do
    subject=$(git log -1 --format=%s "$sha")
    log "replaying $sha $subject"
    if ! git -C "$SYNC_PATH" cherry-pick "$sha"; then
      log "cherry-pick failed for $sha $subject"
      log "resolve conflicts in $SYNC_PATH"
      exit 1
    fi
  done
}

run_smoke_test_if_configured() {
  if [[ -z "$SMOKE_TEST" ]]; then
    log "no smoke test command configured"
    return 0
  fi

  run_in_sync_worktree "$SMOKE_TEST"
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
ORIGINAL_BRANCH=$(current_branch)
SYNC_ID="sync-$(timestamp)"
SYNC_BRANCH="sync/$SYNC_ID"
SYNC_PATH="$ROOT/.worktrees/$SYNC_ID"
RESCUE_BRANCH="rescue/$(timestamp)"

log "repo root: $ROOT"
log "starting branch: $ORIGINAL_BRANCH"
log "base branch: $BASE_BRANCH"
log "patch branch: $PATCH_BRANCH"
log "sync branch: $SYNC_BRANCH"
log "sync path: $SYNC_PATH"

require_clean_or_snapshot_flag "$SNAPSHOT_DIRTY"

if working_tree_dirty; then
  snapshot_dirty_work
fi

run_cmd git fetch origin --prune
verify_prerequisites
run_cmd git checkout "$BASE_BRANCH"
run_cmd git reset --hard "origin/$BASE_BRANCH"
create_sync_worktree
replay_patch_stack
run_smoke_test_if_configured

if [[ -n "$RESCUE_BRANCH" && "$DRY_RUN" == "0" ]]; then
  log "rescue snapshot available at: $RESCUE_BRANCH"
fi

log "sync worktree ready at: $SYNC_PATH"
log "review and test there before promoting changes"
