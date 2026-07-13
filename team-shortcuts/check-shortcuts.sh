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
INSTALLED_VERSION="$ROOT/.shortcut-version"
HOOK_DOCTOR="$HOME/.local/bin/hermes-hook-doctor"
WRITE_PERMIT="$HOME/.local/bin/hermes-write-permit"

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

match_check() {
  local label="$1"
  local left="$2"
  local right="$3"
  if [ "$left" = "$right" ]; then
    printf 'PASS %-28s %s\n' "$label" "$left"
  else
    printf 'FAIL %-28s %s ไม่ตรงกับ %s\n' "$label" "$left" "$right"
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
exists_check "hook_doctor_exists" "$HOOK_DOCTOR"
exists_check "write_permit_exists" "$WRITE_PERMIT"
exists_check "installed_version_exists" "$INSTALLED_VERSION"

if [ -f "$CLAUDE" ] && grep -q 'HERMES_SHORTCUTS_START' "$CLAUDE"; then
  printf 'PASS %-28s %s\n' "claude_bridge_exists" "$CLAUDE"
else
  printf 'FAIL %-28s ไม่พบตัวชี้ Shortcut ใน %s\n' "claude_bridge_exists" "$CLAUDE"
  pass=false
fi

if [ -x "$HOOK_DOCTOR" ] && "$HOOK_DOCTOR" >/dev/null 2>&1; then
  printf 'PASS %-28s %s\n' "hook_health" "3/3"
else
  printf 'FAIL %-28s %s\n' "hook_health" "ด่านจริงไม่ผ่าน"
  pass=false
fi

registry_count="$(count_table "$REGISTRY")"
skill_count="$(count_skill_map "$SKILL")"
index_count="$(count_table "$INDEX")"
match_check "registry_vs_skill" "$registry_count" "$skill_count"
match_check "registry_vs_index" "$registry_count" "$index_count"
prompt_count=0
if [ -d "$REFS" ]; then
  prompt_count="$(find "$REFS" -maxdepth 1 -type f -name '*.md' | wc -l | tr -d ' ')"
fi
if [ "$prompt_count" -ge "$registry_count" ] 2>/dev/null; then
  printf 'PASS %-28s %s (คำสั่งลัด %s)\n' "prompt_file_coverage" "$prompt_count" "$registry_count"
else
  printf 'FAIL %-28s %s น้อยกว่าคำสั่งลัด %s\n' "prompt_file_coverage" "$prompt_count" "$registry_count"
  pass=false
fi

expected_version="${HERMES_SHORTCUT_EXPECTED_VERSION:-}"
if [ -z "$expected_version" ] && command -v curl >/dev/null 2>&1; then
  # ใช้ GitHub Contents API แทน raw URL เพื่อไม่อ่านรุ่นเก่าจาก CDN cache หลังเพิ่งอัปเดต
  expected_version="$(curl -fsSL -H 'Accept: application/vnd.github.raw+json' \
    'https://api.github.com/repos/rattanasak-ops/hermes-agent/contents/team-shortcuts/VERSION?ref=main' \
    2>/dev/null | tr -d '[:space:]' || true)"
fi
installed_version=""
if [ -f "$INSTALLED_VERSION" ]; then
  installed_version="$(tr -d '[:space:]' < "$INSTALLED_VERSION")"
fi
if [ -n "$expected_version" ]; then
  print_check "shortcut_version" "$installed_version" "$expected_version"
else
  printf 'WARN %-28s ตรวจรุ่นล่าสุดจาก GitHub ไม่ได้\n' "shortcut_version"
fi

if [ -f "$REFS/use-new-chat.md" ]; then
  grep -q 'version: "2.6"' "$REFS/use-new-chat.md" || pass=false
  grep -q 'ห้ามเรียกรอบที่ 3' "$REFS/use-new-chat.md" || pass=false
fi

echo ""
if [ "$pass" = true ]; then
  echo "RESULT: PASS"
  echo "เครื่องนี้ต่อ Prompt Shortcut พร้อมใช้ $registry_count/$registry_count · รุ่น $installed_version"
else
  echo "RESULT: FAIL"
  echo "ให้รันตัวติดตั้งใหม่:"
  echo "curl -fsSL https://raw.githubusercontent.com/rattanasak-ops/hermes-agent/main/team-shortcuts/install-from-github.sh | bash"
  exit 1
fi
