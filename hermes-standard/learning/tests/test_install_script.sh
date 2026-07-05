#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
INSTALLER="$ROOT/hermes-standard/bin/install_curse_hook.sh"
SOURCE_HOOK="$ROOT/hermes-standard/learning/hooks/ai-fail-stats-v2.py"
SOURCE_KEYWORDS="$ROOT/hermes-standard/learning/curse-keywords.json"

fail() {
  printf 'FAIL: %s\n' "$*" >&2
  exit 1
}

assert_file() {
  [[ -f "$1" ]] || fail "ไม่พบไฟล์ $1"
}

assert_executable() {
  [[ -x "$1" ]] || fail "ไฟล์ไม่ executable: $1"
}

assert_same_file() {
  cmp -s "$1" "$2" || fail "ไฟล์ไม่ตรงกัน: $1 != $2"
}

make_fake_home() {
  local home="$1"
  mkdir -p "$home/.claude/hooks" "$home/.claude/ai-fail-stats"
  printf '#!/usr/bin/env python3\nprint("old hook")\n' > "$home/.claude/hooks/ai-fail-stats.py"
  chmod +x "$home/.claude/hooks/ai-fail-stats.py"
  printf '{"old": true}\n' > "$home/.claude/ai-fail-stats/curse-keywords.json"
}

single_backup_for() {
  local dir="$1"
  local pattern="$2"
  local count
  count="$(find "$dir" -maxdepth 1 -name "$pattern" -print | wc -l | tr -d ' ')"
  [[ "$count" == "1" ]] || fail "จำนวน backup ของ $pattern ต้องเป็น 1 แต่ได้ $count"
  find "$dir" -maxdepth 1 -name "$pattern" -print
}

backup_count_for() {
  local dir="$1"
  local pattern="$2"
  find "$dir" -maxdepth 1 -name "$pattern" -print | wc -l | tr -d ' '
}

test_install_with_fake_home_creates_backups_and_validates_hook() {
  assert_file "$INSTALLER"
  assert_file "$SOURCE_HOOK"
  assert_file "$SOURCE_KEYWORDS"

  local tmp home out err hook_bak keywords_bak hook_stamp keywords_stamp stats_dir proc_out
  tmp="$(mktemp -d)"
  home="$tmp/home"
  make_fake_home "$home"

  out="$tmp/stdout.txt"
  err="$tmp/stderr.txt"
  HOME="$home" bash "$INSTALLER" >"$out" 2>"$err"

  assert_same_file "$SOURCE_HOOK" "$home/.claude/hooks/ai-fail-stats.py"
  assert_same_file "$SOURCE_KEYWORDS" "$home/.claude/ai-fail-stats/curse-keywords.json"
  assert_executable "$home/.claude/hooks/ai-fail-stats.py"

  hook_bak="$(single_backup_for "$home/.claude/hooks" 'ai-fail-stats.py.bak-*')"
  keywords_bak="$(single_backup_for "$home/.claude/ai-fail-stats" 'curse-keywords.json.bak-*')"
  grep -q 'old hook' "$hook_bak" || fail "backup hook ไม่มีข้อมูลเดิม"
  grep -q '"old": true' "$keywords_bak" || fail "backup keyword ไม่มีข้อมูลเดิม"

  hook_stamp="${hook_bak##*.bak-}"
  keywords_stamp="${keywords_bak##*.bak-}"
  [[ "$hook_stamp" == "$keywords_stamp" ]] || fail "timestamp backup ไม่ตรงกัน"

  grep -q 'ติดตั้งสำเร็จ' "$out" || fail "stdout ไม่สรุปว่าติดตั้งสำเร็จ"
  grep -q 'วิธีถอนกลับ' "$out" || fail "stdout ไม่มีวิธีถอนกลับ"
  [[ ! -s "$err" ]] || fail "stderr ควรว่าง แต่ได้: $(cat "$err")"

  sleep 1
  HOME="$home" bash "$INSTALLER" >"$tmp/stdout-rerun.txt" 2>"$tmp/stderr-rerun.txt"
  [[ "$(backup_count_for "$home/.claude/hooks" 'ai-fail-stats.py.bak-*')" == "2" ]] ||
    fail "รันซ้ำแล้ว backup hook ไม่เพิ่มเป็น 2"
  [[ "$(backup_count_for "$home/.claude/ai-fail-stats" 'curse-keywords.json.bak-*')" == "2" ]] ||
    fail "รันซ้ำแล้ว backup keyword ไม่เพิ่มเป็น 2"
  assert_same_file "$SOURCE_HOOK" "$home/.claude/hooks/ai-fail-stats.py"
  assert_same_file "$SOURCE_KEYWORDS" "$home/.claude/ai-fail-stats/curse-keywords.json"
  [[ ! -s "$tmp/stderr-rerun.txt" ]] || fail "stderr รอบรันซ้ำควรว่าง แต่ได้: $(cat "$tmp/stderr-rerun.txt")"

  stats_dir="$tmp/stats"
  mkdir -p "$stats_dir"
  cp "$SOURCE_KEYWORDS" "$stats_dir/curse-keywords.json"
  proc_out="$tmp/hook-output.json"
  printf '{"prompt":"fuck you opus","cwd":"/tmp"}\n' |
    AI_FAIL_STATS_DIR="$stats_dir" HOME="$home" python3 "$home/.claude/hooks/ai-fail-stats.py" >"$proc_out"
  python3 - "$proc_out" <<'PY'
import json
import sys
text = open(sys.argv[1], encoding="utf-8").read().strip()
data = json.loads(text)
assert "target:opus" in data["systemMessage"]
PY
  grep -q '"category": "target:opus"' "$stats_dir/log.jsonl" || fail "log ชั่วคราวไม่มี target:opus"
}

test_failed_validation_restores_existing_files() {
  assert_file "$INSTALLER"
  assert_file "$SOURCE_HOOK"
  assert_file "$SOURCE_KEYWORDS"

  local tmp fake_repo home out err hook_bak keywords_bak
  tmp="$(mktemp -d)"
  fake_repo="$tmp/fake-repo"
  mkdir -p "$fake_repo/hermes-standard/bin" "$fake_repo/hermes-standard/learning/hooks"
  cp "$INSTALLER" "$fake_repo/hermes-standard/bin/install_curse_hook.sh"
  cp "$SOURCE_HOOK" "$fake_repo/hermes-standard/learning/hooks/ai-fail-stats-v2.py"
  printf '{ broken json\n' > "$fake_repo/hermes-standard/learning/curse-keywords.json"

  home="$tmp/home"
  make_fake_home "$home"

  out="$tmp/stdout.txt"
  err="$tmp/stderr.txt"
  if HOME="$home" bash "$fake_repo/hermes-standard/bin/install_curse_hook.sh" >"$out" 2>"$err"; then
    fail "installer ต้อง exit ไม่เท่ากับ 0 เมื่อ keyword ต้นทางเสีย"
  fi

  grep -q 'old hook' "$home/.claude/hooks/ai-fail-stats.py" || fail "ไม่ได้กู้ hook เดิมกลับ"
  grep -q '"old": true' "$home/.claude/ai-fail-stats/curse-keywords.json" || fail "ไม่ได้กู้ keyword เดิมกลับ"
  hook_bak="$(single_backup_for "$home/.claude/hooks" 'ai-fail-stats.py.bak-*')"
  keywords_bak="$(single_backup_for "$home/.claude/ai-fail-stats" 'curse-keywords.json.bak-*')"
  grep -q 'old hook' "$hook_bak" || fail "backup hook หลัง fail ไม่มีข้อมูลเดิม"
  grep -q '"old": true' "$keywords_bak" || fail "backup keyword หลัง fail ไม่มีข้อมูลเดิม"
  grep -q 'ติดตั้งล้มเหลว' "$err" || fail "stderr ไม่บอกว่าติดตั้งล้มเหลว"
}

test_install_with_fake_home_creates_backups_and_validates_hook
test_failed_validation_restores_existing_files
printf 'test_install_script.sh: ผ่าน 2/2\n'
