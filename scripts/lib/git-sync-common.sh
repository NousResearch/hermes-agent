#!/usr/bin/env bash
set -euo pipefail

repo_root() {
  git rev-parse --show-toplevel
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

current_branch() {
  git rev-parse --abbrev-ref HEAD
}

timestamp() {
  date +%Y%m%d-%H%M%S
}

working_tree_dirty() {
  [[ -n "$(git status --porcelain)" ]]
}

require_clean_or_snapshot_flag() {
  local snapshot_dirty="${1:-0}"
  if working_tree_dirty && [[ "$snapshot_dirty" != "1" ]]; then
    die "working tree is dirty; commit it first or rerun with --snapshot-dirty"
  fi
}

branch_exists() {
  local branch_name="$1"
  git show-ref --verify --quiet "refs/heads/${branch_name}"
}

remote_branch_exists() {
  local remote_branch="$1"
  git show-ref --verify --quiet "refs/remotes/${remote_branch}"
}

ensure_dir_missing() {
  local path="$1"
  [[ ! -e "$path" ]] || die "path already exists: $path"
}
