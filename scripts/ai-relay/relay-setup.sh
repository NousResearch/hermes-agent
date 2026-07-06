#!/usr/bin/env bash
# relay-setup.sh — พนักงานรันตัวเดียวจบ: โหลด(ครั้งแรก) / อัปเดต + ติดตั้ง + ตรวจ AI Relay
# ปลอดภัย: ไม่ฝัง token/รหัสลับ · ใช้สิทธิ์ git ที่พนักงานตั้งไว้เองบนเครื่อง
# ปรับได้ด้วย env:  RELAY_REPO_URL (ที่อยู่ repo)  ·  RELAY_DIR (โฟลเดอร์ปลายทาง)
set -euo pipefail

REPO_URL="${RELAY_REPO_URL:-git@github.com:rattanasak-ops/hermes-agent.git}"
TARGET_DIR="${RELAY_DIR:-$HOME/hermes-agent}"

echo "══ AI Relay setup ══"
echo "repo:    $REPO_URL"
echo "โฟลเดอร์: $TARGET_DIR"
echo ""

# 1) โหลดครั้งแรก หรือ อัปเดตของเดิม (ff-only = ไม่ทับงานค้าง ไม่สร้าง merge มั่ว)
if [ -d "$TARGET_DIR/.git" ]; then
  echo "→ อัปเดตของเดิม (git pull) ..."
  git -C "$TARGET_DIR" pull --ff-only
else
  echo "→ โหลดครั้งแรก (git clone) ..."
  git clone "$REPO_URL" "$TARGET_DIR"
fi

# 2) ติดตั้งคำสั่ง relay (สร้าง symlink ที่ ~/.local/bin · ไม่ล็อกอินแทน ไม่แตะรหัสลับ)
echo "→ ติดตั้งคำสั่ง relay ..."
bash "$TARGET_DIR/scripts/ai-relay/install-local.sh"

# 3) ตรวจความพร้อมเครื่อง
echo "→ ตรวจความพร้อม (relay-doctor) ..."
if command -v relay-doctor >/dev/null 2>&1; then
  relay-doctor || true
else
  echo "  (ยังเรียก relay-doctor ไม่ได้ — เพิ่ม ~/.local/bin เข้า PATH ก่อน)"
fi

cat <<MSG

══ เสร็จแล้ว ══ ขั้นต่อไป (ทำครั้งเดียว):
  1) ล็อกอิน AI ที่จะใช้เขียนโค้ด (เลือกเท่าที่ใช้):
       codex login
      grok login --oauth
       gemini auth login
  2) เช็คว่าตอนนี้ตัวไหนพร้อมจริง:
       relay-status --probe --cwd "$TARGET_DIR"
  3) ถามว่างานแบบนี้ควรใช้ AI ตัวไหน:
       relay-suggest --task-type backend --cwd "$TARGET_DIR"

คราวหน้าจะอัปเดต = รัน relay-setup.sh ซ้ำได้เลย (มันจะ git pull + ติดตั้งใหม่ให้เอง)
กติกาการทำงานทั้งหมดอ่านได้ที่: $TARGET_DIR/scripts/ai-relay/RELAY-RULES.md
MSG
