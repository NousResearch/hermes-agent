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
