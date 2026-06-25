#!/usr/bin/env bash
#
# sync-from-vault.sh — สำหรับเจ้าของระบบเท่านั้น
# ดึงชุด Shortcut ล่าสุดจาก Obsidian vault มาอัปเดต payload ในรีโปก่อนแจกให้ทีม
#
# ใช้เมื่อ: เพิ่ม/แก้ Shortcut ใน vault แล้วต้องการให้พนักงานได้ของใหม่
# วิธีใช้:  bash sync-from-vault.sh   แล้ว git commit + push
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VAULT="${HERMES_VAULT:-$HOME/ObsidianVault/HermesAgent}"
PAYLOAD="$SCRIPT_DIR/payload"

if [ ! -f "$VAULT/ai-context/prompt-shortcut-registry.md" ]; then
  echo "ผิดพลาด: ไม่พบ vault ที่ $VAULT (ตั้ง HERMES_VAULT ชี้ vault ได้)"
  exit 1
fi

mkdir -p "$PAYLOAD/ai-context" "$PAYLOAD/skills"
cp "$VAULT/ai-context/prompt-shortcut-registry.md" "$PAYLOAD/ai-context/"
rm -rf "$PAYLOAD/skills/prompt-shortcuts"
cp -R "$VAULT/skills/prompt-shortcuts" "$PAYLOAD/skills/"

REF_COUNT="$(ls -1 "$PAYLOAD/skills/prompt-shortcuts/references/"*.md 2>/dev/null | wc -l | tr -d ' ')"
echo "อัปเดต payload แล้ว: ทะเบียน 1 ไฟล์ + prompt $REF_COUNT ไฟล์"
echo "ขั้นต่อไป: git add team-shortcuts/payload && git commit && git push"
