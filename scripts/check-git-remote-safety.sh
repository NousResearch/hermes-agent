#!/bin/zsh
set -eu

repo_root="$(git rev-parse --show-toplevel 2>/dev/null)"
cd "$repo_root"

branch="$(git branch --show-current)"
upstream_push="$(git remote get-url --push upstream 2>/dev/null || true)"
tracking="$(git rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>/dev/null || true)"
worktree_status="$(git status --porcelain --untracked-files=all)"

print "REPO=$repo_root"
print "BRANCH=${branch:-detached}"
print "TRACKING=${tracking:-none}"
print "UPSTREAM_PUSH=${upstream_push:-missing}"

if [[ -z "$branch" ]]; then
  print "FAIL: detached HEAD; choose a branch before push/PR" >&2
  exit 2
fi

if [[ "$upstream_push" != "DISABLED" ]]; then
  print "FAIL: upstream push URL must be DISABLED before push/PR" >&2
  exit 3
fi

if [[ "$branch" == "main" && "${ALLOW_MAIN_PUSH:-0}" != "1" ]]; then
  print "FAIL: main push requires ALLOW_MAIN_PUSH=1 and explicit human intent" >&2
  exit 4
fi

if [[ "$branch" != codex/* && "${ALLOW_NON_CODEX_BRANCH:-0}" != "1" ]]; then
  print "FAIL: feature branch should use codex/* or set ALLOW_NON_CODEX_BRANCH=1" >&2
  exit 5
fi

if [[ -n "$worktree_status" && "${ALLOW_DIRTY:-0}" != "1" ]]; then
  print "FAIL: working tree is not clean; commit or stash before push/PR" >&2
  git status --short --untracked-files=all >&2
  exit 6
fi

if [[ "$tracking" != fork/* && "$tracking" != origin/* ]]; then
  print "FAIL: tracking branch should be on Kevin-owned fork/origin before push/PR" >&2
  exit 7
fi

print "OK: git remote safety check passed"
