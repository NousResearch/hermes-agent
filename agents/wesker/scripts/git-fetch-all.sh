#!/usr/bin/env bash
# git-fetch-all: fetch all tracked repos, report behind counts
# Designed for cron usage — stdout captured by Hermes cron delivery.
set -euo pipefail

repos=(
  /home/kensei/apps/hermes-workspace
  /home/kensei/apps/postiz-docker
  /home/kensei/repos/hermes-ACII-Skins
  /home/kensei/repos/KenseiAgent
)

all_clean=true
any_auth_fail=false

for repo in "${repos[@]}"; do
  name=$(basename "$repo")

  if ! cd "$repo" 2>/dev/null; then
    echo "[WARN] $name: repo dir missing"
    all_clean=false
    continue
  fi

  if ! git remote get-url origin &>/dev/null; then
    echo "[WARN] $name: no 'origin' remote"
    continue
  fi

  fetch_out=$(GIT_TERMINAL_PROMPT=0 git fetch origin 2>&1) || true
  fetch_rc=$?

  if echo "$fetch_out" | grep -qi "could not read\|authentication\|credentials\|denied\|403\|401"; then
    echo "[AUTH] $name: needs auth credentials (HTTPS without cached token)"
    any_auth_fail=true
    continue
  fi

  if [ $fetch_rc -ne 0 ]; then
    echo "[FAIL] $name: fetch error: $(echo "$fetch_out" | head -1)"
    all_clean=false
    continue
  fi

  if [ -n "$fetch_out" ]; then
    echo "$fetch_out" | sed "s/^/$name: /"
  fi

  behind=$(git rev-list --count HEAD..origin/main 2>/dev/null || git rev-list --count HEAD..origin/master 2>/dev/null || echo "0")
  if [ "$behind" -gt 0 ]; then
    echo "[BEHIND] $name: $behind commit(s) behind origin"
    all_clean=false
  fi
done

if $all_clean && ! $any_auth_fail; then
  echo "All repos up to date."
fi
if $any_auth_fail; then
  echo ""
  echo "Auth note: some private repos need HTTPS credentials (gh auth login or SSH)."
fi

$all_clean
