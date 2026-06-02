#!/usr/bin/env bash
# Bulk-install useful Hermes skills: all official optional-skills + OpenClaw sync + vetted skills.sh.
# Safe defaults: official first; community only after inspect; OpenClaw skills use rename on conflict.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HERMES="${HERMES:-$ROOT/venv/bin/hermes}"
YES="${SKILLS_INSTALL_YES:--y}"
LOG="${SKILLS_INSTALL_LOG:-/tmp/hermes-skills-install.log}"

exec > >(tee -a "$LOG") 2>&1
echo "=== Hermes power skills install $(date -Iseconds) ==="

install_official_path() {
  local rel="$1"  # e.g. finance/dcf-model
  echo ">> official: $rel"
  "$HERMES" skills install "official/$rel" $YES || echo "WARN: official/$rel"
}

echo "=== Tier 1: All official optional-skills from repo ==="
while IFS= read -r skill_md; do
  rel="${skill_md#optional-skills/}"
  rel="${rel%/SKILL.md}"
  install_official_path "$rel"
done < <(find "$ROOT/optional-skills" -name SKILL.md | sort)

echo "=== Tier 2: OpenClaw user skills (skills only; avoids soul/model conflicts) ==="
MIG_SCRIPT="${HOME}/.hermes/skills/migration/openclaw-migration/scripts/openclaw_to_hermes.py"
if [[ -d "${HOME}/.openclaw/skills" && -f "$MIG_SCRIPT" ]]; then
  python3 "$MIG_SCRIPT" --source "${HOME}/.openclaw" --target "${HOME}/.hermes" \
    --execute --skill-conflict rename --include skills,shared-skills \
    || echo "WARN: openclaw skills migrate had issues"
else
  echo "SKIP: no ~/.openclaw/skills or migration script (install official/migration/openclaw-migration first)"
fi

echo "=== Tier 3: skills.sh community (inspect then install) ==="
COMMUNITY=(
  "skills-sh/sugarforever/01coder-agent-skills/china-stock-analysis"
  "skills-sh/anthropics/skills/skill-creator"
  "skills-sh/anthropics/skills/mcp-builder"
)
for id in "${COMMUNITY[@]}"; do
  echo ">> inspect: $id"
  if ! "$HERMES" skills inspect "$id" >/dev/null 2>&1; then
    echo "SKIP inspect failed: $id"
    continue
  fi
  echo ">> install: $id"
  "$HERMES" skills install "$id" $YES || echo "WARN: $id"
done

echo "=== Summary ==="
echo "Search more community skills: hermes skills search <query> --source skills-sh --limit 10"
find "${HOME}/.hermes/skills" -name SKILL.md 2>/dev/null | wc -l | xargs echo "SKILL.md count:"
"$HERMES" skills audit 2>&1 | tail -15 || true
echo "Log: $LOG"
