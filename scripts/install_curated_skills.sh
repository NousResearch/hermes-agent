#!/usr/bin/env bash
# Install curated, safer skills for Hermes (finance + research + devops).
# Sources: official optional-skills (★ trusted), skills.sh (community — inspect first).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HERMES="${HERMES:-$ROOT/venv/bin/hermes}"
YES="${SKILLS_INSTALL_YES:--y}"

install_official() {
  local id="$1"
  echo ">> official: $id"
  "$HERMES" skills install "official/$id" $YES || echo "WARN: failed $id"
}

echo "=== Tier 1: Official finance (NousResearch repo, audit built-in) ==="
FINANCE=(
  finance/3-statement-model
  finance/comps-analysis
  finance/dcf-model
  finance/excel-author
  finance/lbo-model
  finance/merger-model
  finance/pptx-author
  finance/stocks
)
for id in "${FINANCE[@]}"; do
  install_official "$id"
done

echo "=== Tier 2: Official research / productivity (finance workflow) ==="
EXTRA=(
  research/duckduckgo-search
  research/searxng-search
  research/domain-intel
  productivity/watchers
  devops/kanban
)
for id in "${EXTRA[@]}"; do
  install_official "$id" 2>/dev/null || true
done

echo "=== Tier 3: skills.sh community (inspect + install if clean) ==="
# High install count on skills.sh; still community trust — run audit on install.
COMMUNITY=(
  "skills-sh/sugarforever/01coder-agent-skills/china-stock-analysis"
)
for id in "${COMMUNITY[@]}"; do
  echo ">> inspect: $id"
  "$HERMES" skills inspect "$id" >/dev/null 2>&1 || { echo "SKIP inspect failed: $id"; continue; }
  echo ">> install: $id"
  "$HERMES" skills install "$id" $YES || echo "WARN: blocked or failed: $id"
done

echo "=== Done. Run: hermes skills list && hermes skills audit ==="
"$HERMES" skills list 2>&1 | tail -5
