#!/usr/bin/env bash
# relay-grok — เรียก Grok ผ่าน AI Relay จากที่ไหนก็ได้ในเครื่อง
# ใช้:
#   relay-grok "สรุปงานนี้ให้หน่อย"
#   relay-grok path/to/brief.md
set -euo pipefail

if [ "$#" -eq 0 ]; then
  echo 'ใช้: relay-grok "ข้อความที่ต้องการส่งให้ Grok"'
  echo "หรือ: relay-grok path/to/brief.md"
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELAY_CALL="${RELAY_CALL_BIN:-}"
if [ -z "${RELAY_CALL}" ]; then
  RELAY_CALL="$(command -v relay-call 2>/dev/null || true)"
fi
if [ -z "${RELAY_CALL}" ] && [ -x "${SCRIPT_DIR}/relay-call" ]; then
  RELAY_CALL="${SCRIPT_DIR}/relay-call"
fi
if [ -z "${RELAY_CALL}" ]; then
  echo "ไม่พบคำสั่ง relay-call"
  exit 10
fi

ROOT="${AI_RELAY_ROOT:-$HOME}"
TASK_ID="${AI_RELAY_TASK_ID:-quick-$(date -u +%Y%m%dT%H%M%SZ)}"

RELAY_ADD_GROK="$(command -v relay-add-grok 2>/dev/null || true)"
if [ -n "${RELAY_ADD_GROK}" ]; then
  "${RELAY_ADD_GROK}" --cwd "${ROOT}" >/dev/null 2>&1 || true
fi

if [ "$#" -eq 1 ] && [ -f "$1" ]; then
  PROMPT_FILE="$1"
else
  PROMPT_FILE="$*"
fi

exec "${RELAY_CALL}" --tool grok --task-id "${TASK_ID}" --prompt-file "${PROMPT_FILE}" --cwd "${ROOT}"
