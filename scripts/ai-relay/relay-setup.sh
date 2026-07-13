#!/usr/bin/env bash
# relay-setup.sh — พนักงานรันตัวเดียวจบ: โหลดเครื่องมือ + ติดตั้ง + ตรวจ AI Relay
# ปลอดภัย: ไม่ฝัง token/รหัสลับ · ไม่ต้องมี repo Hermes Agent ในเครื่อง
# ปรับได้ด้วย env: RELAY_ARCHIVE_URL (ไฟล์ GitHub archive) · RELAY_DIR (โฟลเดอร์ cache)
set -euo pipefail

ARCHIVE_URL="${RELAY_ARCHIVE_URL:-https://github.com/rattanasak-ops/hermes-agent/archive/refs/heads/main.tar.gz}"
TARGET_DIR="${RELAY_DIR:-$HOME/.hermes/ai-relay-tools}"
TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/ai-relay-setup.XXXXXX")"

cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

echo "══ AI Relay setup ══"
echo "source:  $ARCHIVE_URL"
echo "cache:   $TARGET_DIR"
echo ""

if ! command -v curl >/dev/null 2>&1; then
  echo "ผิดพลาด: ไม่พบ curl — ต้องมี curl เพื่อโหลด AI Relay จาก GitHub"
  exit 1
fi
if ! command -v tar >/dev/null 2>&1; then
  echo "ผิดพลาด: ไม่พบ tar — ต้องมี tar เพื่อแตกไฟล์ชุดติดตั้ง"
  exit 1
fi
if ! command -v rsync >/dev/null 2>&1; then
  echo "ผิดพลาด: ไม่พบ rsync — ต้องมี rsync เพื่อคัดไฟล์เครื่องมือให้ตรงกัน"
  exit 1
fi

# 1) โหลดเฉพาะชุดเครื่องมือจาก GitHub archive
echo "→ โหลด AI Relay จาก GitHub ..."
curl -fsSL "$ARCHIVE_URL" -o "$TMP_DIR/hermes-agent.tar.gz"
tar -xzf "$TMP_DIR/hermes-agent.tar.gz" -C "$TMP_DIR"

SRC_DIR="$(find "$TMP_DIR" -maxdepth 3 -type d -path '*/scripts/ai-relay' | head -n 1)"
if [ -z "$SRC_DIR" ] || [ ! -f "$SRC_DIR/install-local.sh" ]; then
  echo "ผิดพลาด: โหลดแล้ว แต่ไม่พบ scripts/ai-relay/install-local.sh"
  exit 1
fi

mkdir -p "$TARGET_DIR/scripts/ai-relay"
rsync -a --delete "$SRC_DIR/" "$TARGET_DIR/scripts/ai-relay/"

# 2) ติดตั้งคำสั่ง relay (สร้าง symlink ที่ ~/.local/bin · ไม่ล็อกอินแทน ไม่แตะรหัสลับ)
echo "→ ติดตั้งคำสั่ง relay ..."
bash "$TARGET_DIR/scripts/ai-relay/install-local.sh"

# 3) ตรวจความพร้อมเครื่อง
echo "→ ตรวจความพร้อม (relay-doctor) ..."
NEW_RELAY_DOCTOR="$HOME/.local/bin/relay-doctor"
if [ -x "$NEW_RELAY_DOCTOR" ]; then
  "$NEW_RELAY_DOCTOR" || true
elif command -v relay-doctor >/dev/null 2>&1; then
  relay-doctor || true
else
  echo "  (ยังเรียก relay-doctor ไม่ได้ — เพิ่ม ~/.local/bin เข้า PATH ก่อน)"
fi

cat <<MSG

══ เสร็จแล้ว ══ ขั้นต่อไป (ทำครั้งเดียว):
  1) วางไฟล์ token ที่แอดมินส่งให้:
       ~/.hermes/.env
     ต้องมี AI_PORTAL_URL, AI_PORTAL_CLAUDE_TOKEN, AI_PORTAL_CODEX_TOKEN, AI_PORTAL_GROK_TOKEN
     หมายเหตุ: Claude/Codex/Grok ผ่าน AI Portal ไม่ต้อง login local
  2) ถ้าจะใช้ Gemini local ค่อยล็อกอินเพิ่ม:
       gemini auth login
  3) เช็คว่าตอนนี้ตัวไหนพร้อมจริง:
       relay-doctor
  4) ถามว่างานแบบนี้ควรใช้ AI ตัวไหน:
       relay-suggest --task-type backend --cwd "$TARGET_DIR"

คราวหน้าจะอัปเดต = รันคำสั่ง curl relay-setup.sh ซ้ำได้เลย
กติกาการทำงานทั้งหมดอ่านได้ที่: $TARGET_DIR/scripts/ai-relay/RELAY-RULES.md
MSG
