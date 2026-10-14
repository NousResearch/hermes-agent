#!/usr/bin/env bash
# architecture-doctor.sh — read-only preflight for the architecture skill.
# Inventories docs/architecture/ (ADRs, design docs, diagram artifacts),
# detects the diagram engines this skill routes to (mmdc, drawio CLI, archify),
# and checks the sibling skills it delegates to (including gap-analysis).
# Writes nothing.
#
# Usage: scripts/architecture-doctor.sh [repo-root]
set -euo pipefail

ROOT="${1:-.}"
cd "$ROOT"

# Skills roots scanned for delegated/third-party skills. Extend with
# SKILL_ROOTS env var if your runtime installs elsewhere.
SKILL_ROOTS="${SKILL_ROOTS:-./.agents/skills $HOME/.agents/skills $HOME/.claude/skills $HOME/.devin/skills $HOME/.config/devin/skills $HOME/.opencode/skills $HOME/.config/opencode/skills $HOME/.cursor/skills $HOME/.copilot/skills $HOME/.gemini/skills $HOME/.codex/skills $HOME/.codeium/windsurf/skills}"

find_skill() {
  local root
  for root in $SKILL_ROOTS; do
    [ -f "$root/$1/SKILL.md" ] && echo "$root/$1"
  done
  return 0
}

bin() {
  if command -v "$1" >/dev/null 2>&1; then echo "PRESENT  $1"; else echo "ABSENT   $1"; fi
}

echo "== docs/architecture/ inventory ($ROOT) =="
if [ -d docs/architecture ]; then
  echo "PRESENT  docs/architecture/"
  for ext in md mmd drawio png svg pdf json html; do
    n="$(find docs/architecture -maxdepth 1 -type f -name "*.$ext" | wc -l | tr -d ' ')"
    printf '         .%-7s %s file(s)\n' "$ext" "$n"
  done
  last_ad="$(ls -1 docs/architecture/AD-*.md 2>/dev/null | sort | tail -1 || true)"
  if [ -n "$last_ad" ]; then
    num="$(basename "$last_ad" | sed -n 's/^AD-\([0-9]*\).*/\1/p')"
    num="${num:-0}"
    printf '         last ADR: %s | next: AD-%04d\n' "$(basename "$last_ad")" "$((10#$num + 1))"
  else
    echo "         no ADRs yet | next: AD-0001"
  fi
else
  echo "ABSENT   docs/architecture/ — create with: mkdir -p docs/architecture"
fi

if [ -d docs/adr ]; then
  echo "PRESENT  docs/adr/ (legacy ADR dir — keep consistent with the AD index)"
fi

echo
echo "== Binaries / engines =="
bin mmdc
bin drawio
bin draw.io
bin node
bin npx

echo
echo "== Delegated skills =="
for s in mermaid-architecture drawio-architecture gap-analysis archify; do
  hits="$(find_skill "$s")"
  if [ -n "$hits" ]; then
    echo "FOUND    $s"
    echo "$hits" | sed 's/^/         /'
  else
    echo "MISSING  $s"
  fi
done

echo
echo "== Routing hints =="
if [ ! -d docs/architecture ]; then
  echo "- create docs/architecture/ first (mkdir -p docs/architecture)"
fi
if ! command -v mmdc >/dev/null 2>&1; then
  echo "- mmdc absent: mermaid-architecture falls back to 'npx -y @mermaid-js/mermaid-cli'"
fi
if ! command -v drawio >/dev/null 2>&1 && ! command -v draw.io >/dev/null 2>&1; then
  echo "- drawio CLI absent: use drawio-architecture Path A (MCP) or fall back to mermaid-architecture"
fi
if [ -z "$(find_skill archify)" ]; then
  echo "- archify not installed: optional engine skipped; falls back to mermaid-architecture (never install external skills at runtime)"
fi
if [ -z "$(find_skill gap-analysis)" ]; then
  echo "- gap-analysis missing: skip the audit handoff and report it to the user"
fi
