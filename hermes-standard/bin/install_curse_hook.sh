#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

SOURCE_HOOK="$REPO_ROOT/hermes-standard/learning/hooks/ai-fail-stats-v2.py"
SOURCE_KEYWORDS="$REPO_ROOT/hermes-standard/learning/curse-keywords.json"

CLAUDE_DIR="${HOME:?HOME ไม่ถูกตั้งค่า}/.claude"
HOOK_DIR="$CLAUDE_DIR/hooks"
STATS_DIR="$CLAUDE_DIR/ai-fail-stats"
DEST_HOOK="$HOOK_DIR/ai-fail-stats.py"
DEST_KEYWORDS="$STATS_DIR/curse-keywords.json"

STAMP="$(date +%Y%m%d-%H%M%S)"
HOOK_BACKUP=""
KEYWORDS_BACKUP=""
HOOK_EXISTED=0
KEYWORDS_EXISTED=0
ROLLBACK_NEEDED=0
TEMP_STATS_DIR=""

say() {
  printf '%s\n' "$*"
}

err() {
  printf '%s\n' "$*" >&2
}

backup_label() {
  local path="$1"
  if [[ -n "$path" ]]; then
    printf '%s' "$path"
  else
    printf 'ไม่มีของเดิมให้สำรอง'
  fi
}

restore_backup() {
  local reason="$1"
  set +e
  err "ติดตั้งล้มเหลว: $reason"

  if [[ "$HOOK_EXISTED" == "1" && -n "$HOOK_BACKUP" && -f "$HOOK_BACKUP" ]]; then
    cp -p "$HOOK_BACKUP" "$DEST_HOOK"
    err "กู้ hook เดิมกลับแล้ว: $DEST_HOOK"
  else
    rm -f "$DEST_HOOK"
    err "ลบ hook ที่ติดตั้งใหม่แล้ว เพราะก่อนติดตั้งไม่มีไฟล์เดิม"
  fi

  if [[ "$KEYWORDS_EXISTED" == "1" && -n "$KEYWORDS_BACKUP" && -f "$KEYWORDS_BACKUP" ]]; then
    cp -p "$KEYWORDS_BACKUP" "$DEST_KEYWORDS"
    err "กู้ keyword เดิมกลับแล้ว: $DEST_KEYWORDS"
  else
    rm -f "$DEST_KEYWORDS"
    err "ลบ keyword ที่ติดตั้งใหม่แล้ว เพราะก่อนติดตั้งไม่มีไฟล์เดิม"
  fi

  if [[ -n "$TEMP_STATS_DIR" ]]; then
    rm -rf "$TEMP_STATS_DIR"
  fi

  err "backup hook: $(backup_label "$HOOK_BACKUP")"
  err "backup keyword: $(backup_label "$KEYWORDS_BACKUP")"
  err "วิธีถอนกลับเอง: คัดลอกไฟล์ .bak กลับไปทับไฟล์ปลายทาง เช่น cp <ไฟล์.bak> <ไฟล์จริง>"
}

fail_after_install() {
  local message="$1"
  restore_backup "$message"
  exit 1
}

on_error() {
  local line="$1"
  if [[ "$ROLLBACK_NEEDED" == "1" ]]; then
    restore_backup "เกิด error ที่บรรทัด $line"
  else
    err "ติดตั้งล้มเหลว: เกิด error ที่บรรทัด $line"
  fi
  exit 1
}

trap 'on_error "$LINENO"' ERR

[[ -f "$SOURCE_HOOK" ]] || {
  err "ติดตั้งล้มเหลว: ไม่พบ hook ต้นทาง $SOURCE_HOOK"
  exit 1
}
[[ -f "$SOURCE_KEYWORDS" ]] || {
  err "ติดตั้งล้มเหลว: ไม่พบ keyword ต้นทาง $SOURCE_KEYWORDS"
  exit 1
}

mkdir -p "$HOOK_DIR" "$STATS_DIR"

if [[ -f "$DEST_HOOK" ]]; then
  HOOK_EXISTED=1
  HOOK_BACKUP="$DEST_HOOK.bak-$STAMP"
  cp -p "$DEST_HOOK" "$HOOK_BACKUP"
fi

if [[ -f "$DEST_KEYWORDS" ]]; then
  KEYWORDS_EXISTED=1
  KEYWORDS_BACKUP="$DEST_KEYWORDS.bak-$STAMP"
  cp -p "$DEST_KEYWORDS" "$KEYWORDS_BACKUP"
fi

ROLLBACK_NEEDED=1

cp "$SOURCE_HOOK" "$DEST_HOOK"
cp "$SOURCE_KEYWORDS" "$DEST_KEYWORDS"
chmod +x "$DEST_HOOK"

TEMP_STATS_DIR="$(mktemp -d)"
cp "$SOURCE_KEYWORDS" "$TEMP_STATS_DIR/curse-keywords.json"

HOOK_OUTPUT="$TEMP_STATS_DIR/hook-output.json"
HOOK_ERROR="$TEMP_STATS_DIR/hook-error.txt"
if ! printf '{"prompt":"fuck you opus","cwd":"/tmp"}\n' |
  AI_FAIL_STATS_DIR="$TEMP_STATS_DIR" python3 "$DEST_HOOK" >"$HOOK_OUTPUT" 2>"$HOOK_ERROR"; then
  fail_after_install "ยิงทดสอบ hook ไม่ผ่าน"
fi

if [[ -s "$HOOK_ERROR" ]]; then
  fail_after_install "ยิงทดสอบ hook มีข้อความ error: $(cat "$HOOK_ERROR")"
fi

if ! python3 - "$HOOK_OUTPUT" <<'PY'
import json
import sys

path = sys.argv[1]
text = open(path, encoding="utf-8").read().strip()
if not text:
    raise SystemExit("ไม่มี JSON ตอบกลับ")
data = json.loads(text)
if "systemMessage" not in data or "hookSpecificOutput" not in data:
    raise SystemExit("JSON ตอบกลับไม่ครบ key หลัก")
if "target:opus" not in json.dumps(data, ensure_ascii=False):
    raise SystemExit("JSON ตอบกลับไม่มี target:opus")
PY
then
  fail_after_install "JSON ตอบกลับจาก hook ไม่ถูกต้อง"
fi

if [[ ! -f "$TEMP_STATS_DIR/log.jsonl" ]]; then
  fail_after_install "ไม่พบ log ชั่วคราวจาก hook"
fi

if ! grep -q '"category": "target:opus"' "$TEMP_STATS_DIR/log.jsonl"; then
  fail_after_install "log ชั่วคราวไม่มีหมวด target:opus"
fi

rm -rf "$TEMP_STATS_DIR"
TEMP_STATS_DIR=""
ROLLBACK_NEEDED=0
trap - ERR

say "ติดตั้งสำเร็จ"
say "hook: $DEST_HOOK"
say "keyword: $DEST_KEYWORDS"
say "backup hook: $(backup_label "$HOOK_BACKUP")"
say "backup keyword: $(backup_label "$KEYWORDS_BACKUP")"
say "ทดสอบแล้ว: hook ตอบ JSON และ log ชั่วคราวมีหมวด target:opus"
say "วิธีถอนกลับ: คัดลอกไฟล์ .bak กลับไปทับไฟล์จริง เช่น cp <ไฟล์.bak> <ไฟล์จริง>"
