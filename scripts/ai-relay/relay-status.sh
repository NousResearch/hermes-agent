#!/usr/bin/env bash
# relay-status — แสดงว่า "ตอนนี้ใช้ AI ตัวไหนทำอะไร" จากของที่มีอยู่แล้วในระบบ
#   อ่าน: งานที่กำลังทำสด (now.json) · ใบมอบหมาย (.relay-active) · รุ่น AI (hermes status) ·
#         สมุดบันทึก (.hermes/ai-relay/ledger.md) · log ข้ามค่าย (calls.jsonl)
# ใช้:  relay-status.sh          ดูครั้งเดียว
#       relay-status.sh --watch  ดูสด (รีเฟรชทุก 2 วินาที · กด Ctrl+C ออก)
# อ่านอย่างเดียว ไม่เปลืองเครดิต AI
set -uo pipefail
ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

show_status() {
  echo "═══════════ AI Relay · สถานะตอนนี้ ═══════════"
  echo "โปรเจกต์: $(basename "$ROOT")  ·  เครื่อง: $(hostname -s)  ·  $(date '+%H:%M:%S')"
  echo

  # 0) งานที่กำลังทำอยู่ตอนนี้ (สด)
  now="$ROOT/.hermes/ai-relay/now.json"
  echo "[กำลังทำอยู่ตอนนี้]"
  if [ -f "$now" ]; then
    t=$(grep -o '"tool":"[^"]*"'       "$now" | cut -d'"' -f4)
    k=$(grep -o '"task":"[^"]*"'       "$now" | cut -d'"' -f4)
    p=$(grep -o '"phase":"[^"]*"'      "$now" | cut -d'"' -f4)
    s=$(grep -o '"started_at":"[^"]*"' "$now" | cut -d'"' -f4)
    echo "  ⏳ ${t:-?}  กำลัง: ${p:-?}   (งาน ${k:-?} · เริ่ม ${s:-?})"
  else
    echo "  — ว่าง ไม่มีงานกำลังทำ"
  fi
  echo

  # 1) ใครถูกมอบหมายให้เขียน / ตรวจ
  relay="$ROOT/.relay-active"
  [ -f "$relay" ] || relay="$HOME/.claude/relay-active"
  if [ -f "$relay" ]; then
    w=$(grep -m1 '^writer='   "$relay" | cut -d= -f2-)
    r=$(grep -m1 '^reviewer=' "$relay" | cut -d= -f2-)
    src=$([ "$relay" = "$ROOT/.relay-active" ] && echo "ใบของโปรเจกต์นี้" || echo "ใบกลางทั้งเครื่อง")
    echo "[ใครทำงาน] ($src)"
    echo "  คนเขียนโค้ด (writer)  : ${w:-ไม่ระบุ}"
    echo "  คนตรวจ (reviewer)     : ${r:-ไม่ระบุ}"
  else
    echo "[ใครทำงาน] ยังไม่เปิดโหมด Relay (ไม่มีใบมอบหมาย)"
  fi
  echo

  # 2) รุ่น AI ที่ตั้งไว้ + ตัวไหนล็อกอินไม่ผ่าน
  echo "[รุ่น AI ที่ตั้งไว้ · สถานะล็อกอิน]"
  hermes status 2>/dev/null | grep -iE 'provider|model|logged in|not logged' | head -6 | sed 's/^/  /' \
    || echo "  (เรียก hermes status ไม่ได้)"
  echo

  # 3) งานล่าสุดในสมุดบันทึก
  echo "[งานล่าสุดในสมุดบันทึก]"
  if [ -f "$ROOT/.hermes/ai-relay/ledger.md" ]; then
    grep -vE '^\s*$' "$ROOT/.hermes/ai-relay/ledger.md" | tail -2 | cut -c1-120 | sed 's/^/  /'
  else
    echo "  (ยังไม่มีสมุดบันทึก)"
  fi
  echo

  # 4) การเรียก AI ข้ามค่ายล่าสุด
  cc="$HOME/.cursor/mcp-servers/hermes-cross-check/calls.jsonl"
  if [ -f "$cc" ]; then
    echo "[เรียก AI ข้ามค่ายล่าสุด]"
    tail -2 "$cc" | sed 's/^/  /'
  fi
  echo "═══════════════════════════════════════════════"
}

if [ "${1:-}" = "--watch" ]; then
  while true; do
    clear 2>/dev/null || true
    show_status
    echo "(ดูสด · รีเฟรชทุก 2 วิ · Ctrl+C ออก)"
    sleep 2
  done
else
  show_status
fi
