#!/usr/bin/env bash
#
# install-from-github.sh — ติดตั้ง Prompt Shortcut + AI Relay จาก GitHub ในคำสั่งเดียว
#
# วิธีใช้สำหรับพนักงาน:
#   curl -fsSL https://raw.githubusercontent.com/rattanasak-ops/hermes-agent/main/team-shortcuts/install-from-github.sh | bash
#   curl -fsSL https://raw.githubusercontent.com/rattanasak-ops/hermes-agent/main/team-shortcuts/install-from-github.sh | bash -s -- --cursor
#
set -euo pipefail

ARCHIVE_URL="${HERMES_SHORTCUT_ARCHIVE_URL:-https://github.com/rattanasak-ops/hermes-agent/archive/refs/heads/main.tar.gz}"
TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/hermes-shortcuts.XXXXXX")"
RELAY_DIR="${RELAY_DIR:-$HOME/.hermes/ai-relay-tools}"

cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

say() { printf '%s\n' "$*"; }

if ! command -v curl >/dev/null 2>&1; then
  say "ผิดพลาด: ไม่พบ curl — ต้องมี curl เพื่อโหลดชุดติดตั้งจาก GitHub"
  exit 1
fi
if ! command -v tar >/dev/null 2>&1; then
  say "ผิดพลาด: ไม่พบ tar — ต้องมี tar เพื่อแตกไฟล์ชุดติดตั้ง"
  exit 1
fi

say "══ ติดตั้งระบบคำสั่งลัดของทีมจาก GitHub ══"
say "ไม่ต้องมี repo Hermes Agent ในเครื่องพนักงาน"
say ""

curl -fsSL "$ARCHIVE_URL" -o "$TMP_DIR/hermes-agent.tar.gz"
tar -xzf "$TMP_DIR/hermes-agent.tar.gz" -C "$TMP_DIR"

TEAM_DIR="$(find "$TMP_DIR" -maxdepth 2 -type d -path '*/team-shortcuts' | head -n 1)"
if [ -z "$TEAM_DIR" ] || [ ! -f "$TEAM_DIR/install-shortcuts.sh" ]; then
  say "ผิดพลาด: โหลดชุดติดตั้งแล้ว แต่ไม่พบ team-shortcuts/install-shortcuts.sh"
  exit 1
fi
ARCHIVE_ROOT="$(dirname "$TEAM_DIR")"
RELAY_SRC="$ARCHIVE_ROOT/scripts/ai-relay"
if [ ! -f "$RELAY_SRC/install-local.sh" ]; then
  say "ผิดพลาด: โหลดชุดติดตั้งแล้ว แต่ไม่พบ scripts/ai-relay/install-local.sh"
  exit 1
fi

bash "$TEAM_DIR/install-shortcuts.sh" "$@"

say ""
say "══ ติดตั้ง AI Relay ══"
mkdir -p "$RELAY_DIR/scripts/ai-relay"
rsync -a --delete "$RELAY_SRC/" "$RELAY_DIR/scripts/ai-relay/"
bash "$RELAY_DIR/scripts/ai-relay/install-local.sh"

ensure_local_bin_path() {
  local rc_file="$1"
  local start="# HERMES_LOCAL_BIN_START"
  local end="# HERMES_LOCAL_BIN_END"
  touch "$rc_file"
  if grep -qF "$start" "$rc_file"; then
    awk -v s="$start" -v e="$end" '
      $0==s{skip=1} !skip{print} $0==e{skip=0}' "$rc_file" > "$rc_file.tmp"
    mv "$rc_file.tmp" "$rc_file"
  fi
  {
    printf '\n%s\n' "$start"
    printf 'export PATH="$HOME/.local/bin:$PATH"\n'
    printf '%s\n' "$end"
  } >> "$rc_file"
}

ensure_local_bin_path "$HOME/.zshrc"
ensure_local_bin_path "$HOME/.bashrc"
export PATH="$HOME/.local/bin:$PATH"
say "      สำเร็จ: เพิ่ม ~/.local/bin ให้ zsh และ bash"

if [ -x "$HOME/.local/bin/relay-doctor" ]; then
  "$HOME/.local/bin/relay-doctor" || true
else
  say "ผิดพลาด: ติดตั้งแล้วแต่ไม่พบ relay-doctor"
  exit 1
fi

say ""
say "ติดตั้งระบบแล้ว · ขั้นที่พนักงานต้องทำเองหนึ่งครั้งต่อ Notebook/บัญชี VPS:"
say "  1. รับไฟล์ ~/.hermes/.env ส่วนตัวจากแอดมิน (ห้ามส่ง token ในแชทกลุ่ม)"
say "  2. รัน: chmod 600 ~/.hermes/.env"
say "Claude/Opus, Codex และ Grok ใช้สิทธิ์ผ่าน AI Portal · ไม่ต้อง login แยกแต่ละโปรแกรม"
say "Gemini เป็นตัวเสริมแบบ local เท่านั้น จึงค่อย login เมื่องานกำหนดให้ใช้"
say "หลังวางไฟล์สิทธิ์ให้ปิดแล้วเปิดโปรแกรม AI ใหม่ แล้วตรวจด้วย:"
say "  relay-doctor"
say "  relay-status --probe --cwd \"$HOME\""
say ""
say "ตรวจคำสั่งลัดซ้ำได้ทุกเมื่อด้วยคำสั่ง:"
say "  curl -fsSL https://raw.githubusercontent.com/rattanasak-ops/hermes-agent/main/team-shortcuts/check-shortcuts.sh | bash"
