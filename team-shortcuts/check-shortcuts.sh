#!/usr/bin/env bash
#
# check-shortcuts.sh — ตรวจเครื่องพนักงานว่า Prompt Shortcut ต่อครบหรือไม่
#
# ใช้ได้โดยไม่ต้องมี repo Hermes Agent:
#   curl -fsSL https://raw.githubusercontent.com/rattanasak-ops/hermes-agent/main/team-shortcuts/check-shortcuts.sh | bash
#
set -euo pipefail

ROOT="$HOME/ObsidianVault/HermesAgent"
REGISTRY="$ROOT/ai-context/prompt-shortcut-registry.md"
SKILL="$ROOT/skills/prompt-shortcuts/SKILL.md"
INDEX="$ROOT/skills/prompt-shortcuts/Prompt Shortcuts.md"
REFS="$ROOT/skills/prompt-shortcuts/references"
CODEX="$HOME/.codex/skills/prompt-shortcuts"
CLAUDE="$HOME/.claude/CLAUDE.md"

pass=true

count_table() {
  local path="$1"
  [ -f "$path" ] || { echo 0; return; }
  grep -c '^| `' "$path" 2>/dev/null || true
}

count_skill_map() {
  local path="$1"
  [ -f "$path" ] || { echo 0; return; }
  awk '
    /^## Shortcut Map/ { on=1; next }
    on && /^## / { on=0 }
    on && /^\| `/ { n++ }
    END { print n+0 }
  ' "$path"
}

print_check() {
  local label="$1"
  local value="$2"
  local expected="$3"
  if [ "$value" = "$expected" ]; then
    printf 'PASS %-28s %s\n' "$label" "$value"
  else
    printf 'FAIL %-28s %s (ควรเป็น %s)\n' "$label" "$value" "$expected"
    pass=false
  fi
}

exists_check() {
  local label="$1"
  local path="$2"
  if [ -e "$path" ]; then
    printf 'PASS %-28s %s\n' "$label" "$path"
  else
    printf 'FAIL %-28s ไม่พบ %s\n' "$label" "$path"
    pass=false
  fi
}

echo "══ ตรวจ Prompt Shortcut บนเครื่องนี้ ══"
exists_check "registry_exists" "$REGISTRY"
exists_check "skill_exists" "$SKILL"
exists_check "index_exists" "$INDEX"
exists_check "codex_link_exists" "$CODEX"

if [ -f "$CLAUDE" ] && grep -q 'HERMES_SHORTCUTS_START' "$CLAUDE"; then
  printf 'PASS %-28s %s\n' "claude_bridge_exists" "$CLAUDE"
else
  printf 'FAIL %-28s ไม่พบตัวชี้ Shortcut ใน %s\n' "claude_bridge_exists" "$CLAUDE"
  pass=false
fi

print_check "registry_count" "$(count_table "$REGISTRY")" "29"
print_check "skill_map_count" "$(count_skill_map "$SKILL")" "29"
print_check "index_count" "$(count_table "$INDEX")" "29"
prompt_count=0
if [ -d "$REFS" ]; then
  prompt_count="$(find "$REFS" -maxdepth 1 -type f -name '*.md' | wc -l | tr -d ' ')"
fi
print_check "prompt_md_count" "$prompt_count" "33"

echo ""
if [ "$pass" = true ]; then
  echo "RESULT: PASS"
  echo "เครื่องนี้ต่อ Prompt Shortcut พร้อมใช้ 29/29"
else
  echo "RESULT: FAIL"
  echo "ให้รันตัวติดตั้งใหม่:"
  echo "curl -fsSL https://raw.githubusercontent.com/rattanasak-ops/hermes-agent/main/team-shortcuts/install-from-github.sh | bash"
  exit 1
fi
