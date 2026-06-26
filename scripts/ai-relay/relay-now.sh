#!/usr/bin/env bash
# relay-now — บันทึก/ล้าง "งานที่ AI กำลังทำอยู่ตอนนี้"
#   ให้ตัวเรียก AI (relay-call) เรียกตอนเริ่มงานและตอนจบงาน เพื่อให้ตัวแสดงสถานะเห็นแบบสด
# ใช้:
#   relay-now.sh set --tool grok --task P1-I2 --phase "กำลังเขียนโค้ด"
#   relay-now.sh clear
#   relay-now.sh show
set -uo pipefail
ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
NOW="$ROOT/.hermes/ai-relay/now.json"
mkdir -p "$(dirname "$NOW")"

cmd="${1:-show}"; shift || true
tool=""; task=""; phase=""
while [ $# -gt 0 ]; do
  case "$1" in
    --tool)  tool="$2";  shift 2;;
    --task)  task="$2";  shift 2;;
    --phase) phase="$2"; shift 2;;
    *) shift;;
  esac
done

case "$cmd" in
  set)
    ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf '{"tool":"%s","task":"%s","phase":"%s","started_at":"%s","staff":"%s"}\n' \
      "$tool" "$task" "$phase" "$ts" "${HERMES_STAFF:-${USER:-unknown}}" > "$NOW"
    echo "บันทึกแล้ว: $tool กำลัง $phase (งาน $task)"
    ;;
  clear)
    rm -f "$NOW"
    echo "ล้างสถานะแล้ว (ไม่มีงานกำลังทำ)"
    ;;
  show)
    [ -f "$NOW" ] && cat "$NOW" || echo '{}'
    ;;
  *)
    echo "ใช้: relay-now.sh set|clear|show"; exit 1;;
esac
