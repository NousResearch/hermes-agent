#!/usr/bin/env bash
# antseed-smart-delegate/test.sh — Verify skill integrity
set -uo pipefail

SKILL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PASS=0
FAIL=0
TOTAL=0

check() {
  local desc="$1"
  local result="$2"
  TOTAL=$((TOTAL + 1))
  if [[ "$result" == "ok" ]]; then
    PASS=$((PASS + 1))
    echo "  ✅ $desc"
  else
    FAIL=$((FAIL + 1))
    echo "  ❌ $desc — $result"
  fi
}

echo "🧪 AntSeed Smart Delegate — Skill Integrity Checks"
echo ""

# --- File structure ---
echo "📂 File structure"
for f in SKILL.md references/setup.md scripts/models.sh scripts/best-peer.sh; do
  [[ -f "$SKILL_DIR/$f" ]] && check "$f exists" "ok" || check "$f exists" "MISSING"
done

# model-catalog.md should NOT exist (replaced by dynamic models.sh)
[[ ! -f "$SKILL_DIR/references/model-catalog.md" ]] && check "model-catalog.md removed (dynamic)" "ok" || check "model-catalog.md should be removed" "still exists"

# --- SKILL.md frontmatter ---
echo ""
echo "📋 SKILL.md frontmatter"
for field in name description version prerequisites; do
  grep -q "^${field}:" "$SKILL_DIR/SKILL.md" && check "frontmatter: $field" "ok" || check "frontmatter: $field" "MISSING"
done

# --- Scripts executable ---
echo ""
echo "⚙️ Script permissions"
for s in scripts/models.sh scripts/best-peer.sh; do
  [[ -x "$SKILL_DIR/$s" ]] && check "$s executable" "ok" || check "$s executable" "not executable"
done

# --- No hardcoded model names in scripts ---
echo ""
echo "🔍 Hardcoded model check (should be 0)"
for s in scripts/models.sh scripts/best-peer.sh; do
  count=$(grep -cE '(deepseek|claude|gpt|minimax|qwen|llama|gemini|glm)' "$SKILL_DIR/$s" 2>/dev/null | head -1 || echo 0)
  count=$(echo "$count" | tr -d '[:space:]' | grep -oE '^[0-9]+' || echo 0)
  if [[ "$count" -eq 0 ]]; then
    check "$s: no hardcoded model names" "ok"
  else
    check "$s: hardcoded model names found" "$count occurrences"
  fi
done

# --- SKILL.md references ---
echo ""
echo "🔗 SKILL.md reference integrity"
# Should NOT reference model-catalog.md
if grep -q 'model-catalog' "$SKILL_DIR/SKILL.md"; then
  check "SKILL.md: no model-catalog.md reference" "still referenced"
else
  check "SKILL.md: no model-catalog.md reference" "ok"
fi

# Should reference models.sh and best-peer.sh
grep -q 'models.sh' "$SKILL_DIR/SKILL.md" && check "SKILL.md: references models.sh" "ok" || check "SKILL.md: references models.sh" "MISSING"
grep -q 'best-peer.sh' "$SKILL_DIR/SKILL.md" && check "SKILL.md: references best-peer.sh" "ok" || check "SKILL.md: references best-peer.sh" "MISSING"

# --- Dynamic tag-driven design ---
echo ""
echo "🏷️ Tag-driven design"
grep -q 'TASK_TAGS\|desired.*tags\|tag_match' "$SKILL_DIR/scripts/best-peer.sh" && check "best-peer.sh: uses tag-based scoring" "ok" || check "best-peer.sh: uses tag-based scoring" "MISSING"
grep -q 'categorize\|TAG_CATEGORIES\|detect_category' "$SKILL_DIR/scripts/models.sh" && check "models.sh: tag-based categorization" "ok" || check "models.sh: tag-based categorization" "MISSING"

# --- Live network queries ---
echo ""
echo "🌐 Live network queries"
grep -q "network.*browse\|'network'.*'browse'" "$SKILL_DIR/scripts/best-peer.sh" && check "best-peer.sh: queries network" "ok" || check "best-peer.sh: queries network" "MISSING"
grep -q "network.*peer\|'network'.*'peer'" "$SKILL_DIR/scripts/best-peer.sh" && check "best-peer.sh: fetches peer details" "ok" || check "best-peer.sh: fetches peer details" "MISSING"
grep -q 'v1/models' "$SKILL_DIR/scripts/models.sh" && check "models.sh: proxy fallback" "ok" || check "models.sh: proxy fallback" "MISSING"

# --- Summary ---
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Results: $PASS/$TOTAL passed, $FAIL failed"
if [[ $FAIL -gt 0 ]]; then
  echo "❌ SOME CHECKS FAILED"
  exit 1
else
  echo "✅ ALL CHECKS PASSED"
  exit 0
fi
