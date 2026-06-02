#!/usr/bin/env bash
# Install the 4 categories of "god-tier" agent skills (user request May 2026).
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HERMES="${HERMES:-$ROOT/venv/bin/hermes}"
YES="${SKILLS_INSTALL_YES:--y}"
LOG="${SKILLS_INSTALL_LOG:-/tmp/hermes-god-tier-skills.log}"
exec > >(tee -a "$LOG") 2>&1

install_id() {
  local id="$1"
  echo ">> install: $id"
  "$HERMES" skills install "$id" $YES || echo "WARN: $id"
}

echo "=== 1) Code & system: anthropics/skills + vercel-labs/agent-skills ==="
ANTHROPICS=(
  "skills-sh/anthropics/skills/skills/skill-creator"
  "skills-sh/anthropics/skills/skills/mcp-builder"
  "skills-sh/anthropics/skills/skills/pdf"
  "skills-sh/anthropics/skills/skills/docx"
  "skills-sh/anthropics/skills/skills/xlsx"
  "skills-sh/anthropics/skills/skills/pptx"
  "skills-sh/anthropics/skills/skills/webapp-testing"
  "skills-sh/anthropics/skills/skills/frontend-design"
  "skills-sh/anthropics/skills/skills/claude-api"
  "skills-sh/anthropics/skills/skills/doc-coauthoring"
)
VERCEL=(
  "skills-sh/vercel-labs/agent-skills/deploy-to-vercel"
  "skills-sh/vercel-labs/agent-skills/web-design-guidelines"
  "skills-sh/vercel-labs/agent-skills/react-best-practices"
  "skills-sh/vercel-labs/agent-skills/composition-patterns"
  "skills-sh/vercel-labs/agent-skills/vercel-optimize"
  "skills-sh/vercel-labs/agent-skills/agent-browser"
)
for id in "${ANTHROPICS[@]}" "${VERCEL[@]}"; do
  install_id "$id"
done
install_id "official/devops/docker-management"

echo "=== 2) Data & memory: MCP memory + database skills ==="
install_id "skills-sh/modelcontextprotocol/servers/src/memory"
install_id "skills-sh/planetscale/database-skills/postgres"
install_id "skills-sh/planetscale/database-skills/mysql"
install_id "skills-sh/supercent-io/skills/database-schema-design"

echo "=== 3) Web search & browser automation ==="
install_id "skills-sh/brave/brave-search/web-search"
install_id "skills-sh/tavily-ai/skills/search"
install_id "skills-sh/tavily-ai/skills/tavily-research"
install_id "skills-sh/microsoft/playwright/playwright-dev"
install_id "skills-sh/openai/skills/playwright"
install_id "skills-sh/github/awesome-copilot/playwright-automation"

echo "=== 4) K-Dense scientific skills (git clone → ~/.hermes/skills/research) ==="
CACHE="${TMPDIR:-/tmp}/hermes-kdense-skills"
DEST="${HOME}/.hermes/skills/research"
if [[ ! -d "$CACHE/claude-scientific-skills/.git" ]]; then
  rm -rf "$CACHE"
  git clone --depth 1 https://github.com/K-Dense-AI/claude-scientific-skills.git "$CACHE/claude-scientific-skills"
fi
count=0
while IFS= read -r skill_md; do
  name="$(basename "$(dirname "$skill_md")")"
  target="$DEST/kdense-$name"
  if [[ -d "$target" ]]; then
    continue
  fi
  cp -a "$(dirname "$skill_md")" "$target"
  count=$((count + 1))
done < <(find "$CACHE/claude-scientific-skills/scientific-skills" -name SKILL.md 2>/dev/null)
echo "Copied $count K-Dense scientific skills to $DEST/kdense-*"

echo "=== Git fallbacks (when skills.sh hits GitHub rate limit) ==="
clone_copy() {
  local url="$1" subpath="$2" dest="$3"
  local cache="${TMPDIR:-/tmp}/hermes-skill-clone-$(basename "$url" .git)"
  [[ -d "$cache/.git" ]] || git clone --depth 1 "$url" "$cache"
  [[ -d "$cache/$subpath" ]] || return 0
  [[ -d "$dest" ]] || cp -a "$cache/$subpath" "$dest"
}
clone_copy "https://github.com/planetscale/database-skills.git" "skills/postgres" "${HOME}/.hermes/skills/postgres"
clone_copy "https://github.com/openai/skills.git" "skills/.curated/playwright" "${HOME}/.hermes/skills/devops/openai-playwright"
clone_copy "https://github.com/openai/skills.git" "skills/.curated/playwright-interactive" "${HOME}/.hermes/skills/devops/openai-playwright-interactive"
# Brave may need --force (audit caution on pip install hints)
"$HERMES" skills install skills-sh/brave/brave-search/web-search -y --force 2>/dev/null || true

echo "=== Done. Log: $LOG ==="
find "${HOME}/.hermes/skills" -name SKILL.md 2>/dev/null | wc -l | xargs echo "Total SKILL.md:"
echo "Enable subset: hermes skills config  (do NOT load all 800+ at once)"
