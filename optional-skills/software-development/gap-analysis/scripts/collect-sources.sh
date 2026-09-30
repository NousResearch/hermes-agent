#!/usr/bin/env bash
# collect-sources.sh — read-only source inventory for the gap-analysis skill.
# Prints presence/absence of the standard audit sources, spec files, git state,
# GitHub CLI auth, and build/test manifests. Writes nothing.
#
# Usage: scripts/collect-sources.sh [repo-root]
set -euo pipefail

ROOT="${1:-.}"
cd "$ROOT"

check() {
  if [ -e "$2" ]; then echo "PRESENT  $1"; else echo "ABSENT   $1"; fi
}

echo "== Source inventory ($ROOT) =="
check ".specs/" ".specs"
check "docs/" "docs"
check "docs/architecture/" "docs/architecture"
check ".claude/CONTEXT.md" ".claude/CONTEXT.md"
check ".claude/MEMORY.md" ".claude/MEMORY.md"
check ".claude/memory/" ".claude/memory"
check ".claude/rules/" ".claude/rules"
check ".claude/agents/" ".claude/agents"
check "CLAUDE.md" "CLAUDE.md"
check "AGENTS.md" "AGENTS.md"
check "README.md" "README.md"
check "ORCHESTRATOR-ROADMAP.md" "ORCHESTRATOR-ROADMAP.md"

echo
echo "== Specs =="
if [ -d ".specs" ]; then
  ls -1 .specs/SPEC-*.md 2>/dev/null || echo "(no SPEC files)"
else
  echo "(no .specs dir)"
fi

echo
echo "== Git =="
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "branch:  $(git branch --show-current 2>/dev/null || echo detached)"
  echo "commit:  $(git rev-parse --short HEAD 2>/dev/null || echo none)"
  echo "dirty:   $(git status --porcelain 2>/dev/null | wc -l | tr -d ' ') changed files"
  echo "remote:  $(git remote get-url origin 2>/dev/null || echo none)"
  subs="$(git submodule status 2>/dev/null || true)"
  if [ -n "$subs" ]; then echo "submodules:"; echo "$subs"; else echo "submodules: none"; fi
else
  echo "not a git repository"
fi

echo
echo "== GitHub CLI =="
if command -v gh >/dev/null 2>&1; then
  gh auth status 2>&1 | head -4 || true
  gh repo view --json nameWithOwner -q .nameWithOwner 2>/dev/null || echo "no repo access via gh"
else
  echo "gh not installed — Issues phase will be blocked"
fi

echo
echo "== Build/test manifests =="
for f in package.json pom.xml build.gradle build.gradle.kts settings.gradle settings.gradle.kts \
         go.mod Cargo.toml pyproject.toml requirements.txt Makefile Dockerfile \
         docker-compose.yml docker-compose.yaml; do
  [ -f "$f" ] && echo "$f"
done
find . -maxdepth 3 \( -name "*.sln" -o -name "*.csproj" \) 2>/dev/null | head -10
find .github/workflows -maxdepth 1 \( -name "*.yml" -o -name "*.yaml" \) 2>/dev/null | head -10 || true
