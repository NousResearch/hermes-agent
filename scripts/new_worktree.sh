#!/usr/bin/env bash
# new_worktree.sh — spin up an isolated git worktree for parallel coding sessions
# on the sibling C# repos, so two sessions never collide in one working tree.
#
# Worktrees share the repo's object store (cheap, no full re-clone) but give each
# session its own directory on its own branch off a freshly-fetched default branch.
# This is the right tool when multiple sessions / agents edit the same repo at once:
# edits are isolated, history is tracked, and you merge back through a normal PR.
#
# Scope: sibling C# repos ONLY (yallaplay-services, SpadesUnity, GinRummyUnity).
# This Hermes pilot repo intentionally does not use this helper; its project
# context defines its own local commit cadence.
#
# Usage:
#   bash scripts/new_worktree.sh <repo> <name> [base]
#   bash scripts/new_worktree.sh --list
#   bash scripts/new_worktree.sh --remove <repo> <name>
#
#   <repo>  one of: backend|services|yallaplay-services , spades|SpadesUnity , rummy|GinRummyUnity
#   <name>  short slug for the branch/dir (e.g. fix-tarneeb). Branch = <name>, dir = <repo>-wt-<name>.
#   [base]  base ref to branch from (default: the repo's own origin/HEAD, freshly
#           fetched — origin/master for backend, origin/development for the Unity clients).
#
# Examples:
#   bash scripts/new_worktree.sh backend fix-tarneeb
#   bash scripts/new_worktree.sh spades crash-guard
#   bash scripts/new_worktree.sh --remove backend fix-tarneeb
#   bash scripts/new_worktree.sh --list
#
# Worktrees live under /mnt/ephemeral/git/ (the fast ephemeral volume where the
# sibling clones live). They are EPHEMERAL — a VM stop can wipe uncommitted edits,
# so push your branch early when the work matters.

set -euo pipefail

EPHEMERAL_GIT="/mnt/ephemeral/git"

die() { echo "error: $*" >&2; exit 1; }

resolve_repo() {
  case "$1" in
    backend|services|yallaplay-services) echo "yallaplay-services" ;;
    spades|SpadesUnity)                  echo "SpadesUnity" ;;
    rummy|GinRummyUnity)                 echo "GinRummyUnity" ;;
    yallaplay-hermes-agent|hermes-agent|agent|local)
      die "this Hermes pilot repo has its own local commit policy — worktrees are for sibling C# repos only" ;;
    *) die "unknown repo '$1' (expected: backend|spades|rummy)" ;;
  esac
}

repo_path() {
  local p="$EPHEMERAL_GIT/$1"
  [ -d "$p/.git" ] || [ -f "$p/.git" ] || die "repo not found at $p — clone or restore the sibling checkout first"
  echo "$p"
}

cmd_list() {
  for r in yallaplay-services SpadesUnity GinRummyUnity; do
    local p="$EPHEMERAL_GIT/$r"
    [ -e "$p/.git" ] || continue
    echo "== $r =="
    git -C "$p" worktree list
    echo
  done
}

cmd_remove() {
  local repo name path wt
  repo="$(resolve_repo "$1")"
  name="$2"
  path="$(repo_path "$repo")"
  wt="$EPHEMERAL_GIT/${repo}-wt-${name}"
  git -C "$path" worktree remove "$wt"
  echo "removed worktree $wt"
  echo "note: branch '$name' still exists — delete with: git -C $path branch -D $name"
}

default_base() {
  # the repo's own default branch (origin/master for backend, origin/development for Unity)
  local path="$1" head
  head="$(git -C "$path" symbolic-ref -q refs/remotes/origin/HEAD 2>/dev/null || true)"
  if [ -n "$head" ]; then
    echo "origin/${head##*/}"
  else
    echo "origin/master"   # fallback if origin/HEAD isn't set locally
  fi
}

cmd_add() {
  local repo name base path wt
  repo="$(resolve_repo "$1")"
  name="$2"
  path="$(repo_path "$repo")"
  wt="$EPHEMERAL_GIT/${repo}-wt-${name}"

  [ -e "$wt" ] && die "worktree dir already exists: $wt (remove it first: bash scripts/new_worktree.sh --remove $repo $name)"

  echo "fetching origin in $repo ..."
  git -C "$path" fetch origin --prune

  base="${3:-$(default_base "$path")}"

  echo "creating worktree $wt on new branch '$name' off $base ..."
  git -C "$path" worktree add -b "$name" "$wt" "$base"

  echo
  echo "worktree ready:"
  echo "  cd $wt"
  echo "  # edit, commit, push -u origin $name, open a PR"
  echo "  # when done: bash scripts/new_worktree.sh --remove $repo $name"
}

[ $# -ge 1 ] || die "usage: new_worktree.sh <repo> <name> [base] | --list | --remove <repo> <name>"

case "$1" in
  --list)   cmd_list ;;
  --remove) shift; [ $# -eq 2 ] || die "usage: --remove <repo> <name>"; cmd_remove "$@" ;;
  -h|--help) sed -n '2,32p' "$0" ;;
  *)        [ $# -ge 2 ] || die "usage: new_worktree.sh <repo> <name> [base]"; cmd_add "$@" ;;
esac
