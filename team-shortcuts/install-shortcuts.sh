#!/usr/bin/env bash
#
# install-shortcuts.sh — ตัวติดตั้ง Prompt Shortcut สำหรับพนักงาน (ทำครั้งเดียวต่อเครื่อง)
#
# ทำอะไร (ภาษาคน):
#   1. คัดชุด Shortcut (ทะเบียน + prompt 22 ไฟล์) ไปไว้ในโฟลเดอร์บ้านของผู้ใช้คนนี้
#   2. ต่อให้ Claude Code มองเห็น Shortcut ทุกโปรเจกต์ (ผ่าน ~/.claude/CLAUDE.md)
#   3. ต่อให้ Codex มองเห็น Shortcut (ผ่านทางลัด ~/.codex/skills/prompt-shortcuts)
#   4. ต่อให้ Cursor มองเห็น Shortcut (ผ่านทางลัดชดเชยที่อยู่เดิมของเจ้าของระบบ)
#
# วิธีใช้:
#   bash install-shortcuts.sh          # ติดตั้ง Claude Code + Codex (ไม่ต้องใช้สิทธิ์ผู้ดูแล)
#   bash install-shortcuts.sh --cursor # ติดตั้งเพิ่มทางลัดให้ Cursor ด้วย (อาจขอรหัสผู้ดูแล 1 ครั้ง)
#
set -euo pipefail

# --- ที่อยู่มาตรฐานบนเครื่องพนักงาน (อิงโฟลเดอร์บ้าน ใช้ได้ทุกชื่อบัญชี) ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAYLOAD="$SCRIPT_DIR/payload"
DEST_ROOT="$HOME/ObsidianVault/HermesAgent"
REGISTRY="$DEST_ROOT/ai-context/prompt-shortcut-registry.md"
SKILL_SRC="$DEST_ROOT/skills/prompt-shortcuts"

# --- ที่อยู่เดิมที่ไฟล์ตัวเชื่อมทุกตัวในโปรเจกต์ชี้ถึง (ใช้ทำทางลัดชดเชยให้ Cursor) ---
OWNER_PATH="/Users/rattanasak/ObsidianVault/HermesAgent"

WANT_CURSOR=0
[ "${1:-}" = "--cursor" ] && WANT_CURSOR=1

say() { printf '%s\n' "$*"; }

# --- ตรวจ payload ก่อน ---
if [ ! -f "$PAYLOAD/ai-context/prompt-shortcut-registry.md" ]; then
  say "ผิดพลาด: ไม่พบ payload ที่ $PAYLOAD — รันสคริปต์นี้จากในโฟลเดอร์ team-shortcuts"
  exit 1
fi

# --- 1) คัดชุด Shortcut เข้าโฟลเดอร์บ้าน ---
say "[1/4] คัดชุด Shortcut ไป $DEST_ROOT"
mkdir -p "$DEST_ROOT/ai-context" "$DEST_ROOT/skills"
cp "$PAYLOAD/ai-context/prompt-shortcut-registry.md" "$DEST_ROOT/ai-context/"
rm -rf "$SKILL_SRC"
cp -R "$PAYLOAD/skills/prompt-shortcuts" "$DEST_ROOT/skills/"
REF_COUNT="$(ls -1 "$SKILL_SRC/references/"*.md 2>/dev/null | wc -l | tr -d ' ')"
say "      สำเร็จ: ทะเบียน 1 ไฟล์ + prompt $REF_COUNT ไฟล์"

# --- 2) ต่อ Claude Code (ทุกโปรเจกต์ผ่าน global memory) ---
say "[2/4] ต่อ Claude Code ผ่าน ~/.claude/CLAUDE.md"
mkdir -p "$HOME/.claude"
CLAUDE_MD="$HOME/.claude/CLAUDE.md"
touch "$CLAUDE_MD"
MARK_START="<!-- HERMES_SHORTCUTS_START -->"
MARK_END="<!-- HERMES_SHORTCUTS_END -->"
# ลบบล็อกเดิม (ถ้ามี) เพื่อให้รันซ้ำได้ไม่พัง
if grep -qF "$MARK_START" "$CLAUDE_MD"; then
  awk -v s="$MARK_START" -v e="$MARK_END" '
    $0==s{skip=1} !skip{print} $0==e{skip=0}' "$CLAUDE_MD" > "$CLAUDE_MD.tmp"
  mv "$CLAUDE_MD.tmp" "$CLAUDE_MD"
fi
{
  printf '\n%s\n' "$MARK_START"
  printf '## Prompt Shortcuts (ติดตั้งโดย install-shortcuts.sh)\n\n'
  printf 'เมื่อผู้ใช้เรียก Shortcut เช่น `Use Act-As`, `Use Comply`, `Use Continue`, `Review Chat` หรือชื่อย่อใกล้เคียง\n'
  printf 'ให้เปิดอ่านทะเบียนนี้ก่อนเสมอ แล้วเปิดไฟล์ prompt ที่แมปไว้ ห้ามเดาจากความจำ:\n\n'
  printf -- '- `%s`\n' "$REGISTRY"
  printf '%s\n' "$MARK_END"
} >> "$CLAUDE_MD"
say "      สำเร็จ: เพิ่มตัวชี้ทะเบียนใน $CLAUDE_MD"

# --- 3) ต่อ Codex (ทางลัด skill) ---
say "[3/4] ต่อ Codex ผ่าน ~/.codex/skills/prompt-shortcuts"
mkdir -p "$HOME/.codex/skills"
CODEX_LINK="$HOME/.codex/skills/prompt-shortcuts"
rm -rf "$CODEX_LINK"
ln -s "$SKILL_SRC" "$CODEX_LINK"
say "      สำเร็จ: $CODEX_LINK -> $SKILL_SRC"

# --- 4) ต่อ Cursor (ทางลัดชดเชยที่อยู่เดิมของเจ้าของระบบ) ---
say "[4/4] ต่อ Cursor"
if [ "$WANT_CURSOR" -eq 0 ]; then
  say "      ข้าม (ไม่ได้ใส่ --cursor) — ถ้าพนักงานใช้ Cursor ให้รันใหม่ด้วย: bash install-shortcuts.sh --cursor"
elif [ -e "$OWNER_PATH" ]; then
  say "      ที่อยู่ $OWNER_PATH มีอยู่แล้วบนเครื่องนี้ — ไม่ต้องทำทางลัด (น่าจะเป็นเครื่องเจ้าของระบบ)"
else
  OWNER_PARENT="$(dirname "$OWNER_PATH")"
  if mkdir -p "$OWNER_PARENT" 2>/dev/null && ln -snf "$DEST_ROOT" "$OWNER_PATH" 2>/dev/null; then
    say "      สำเร็จ: $OWNER_PATH -> $DEST_ROOT"
  else
    say "      ต้องใช้สิทธิ์ผู้ดูแล 1 ครั้ง สำหรับ Cursor — รันคำสั่งนี้:"
    say "        sudo mkdir -p \"$OWNER_PARENT\" && sudo ln -snf \"$DEST_ROOT\" \"$OWNER_PATH\""
  fi
fi

say ""
say "เสร็จสิ้น. ปิดแล้วเปิดโปรแกรม AI ใหม่ 1 รอบ แล้วลองพิมพ์ Shortcut เช่น  Use Comply"
