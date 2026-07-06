#!/usr/bin/env bash
#
# install-from-github.sh — ติดตั้ง Prompt Shortcut จาก GitHub โดยไม่ต้องมี repo Hermes Agent
#
# วิธีใช้สำหรับพนักงาน:
#   curl -fsSL https://raw.githubusercontent.com/rattanasak-ops/hermes-agent/main/team-shortcuts/install-from-github.sh | bash
#   curl -fsSL https://raw.githubusercontent.com/rattanasak-ops/hermes-agent/main/team-shortcuts/install-from-github.sh | bash -s -- --cursor
#
set -euo pipefail

ARCHIVE_URL="${HERMES_SHORTCUT_ARCHIVE_URL:-https://github.com/rattanasak-ops/hermes-agent/archive/refs/heads/main.tar.gz}"
TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/hermes-shortcuts.XXXXXX")"

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

say "══ ติดตั้ง Prompt Shortcut จาก GitHub ══"
say "ไม่ต้องมี repo Hermes Agent ในเครื่องพนักงาน"
say ""

curl -fsSL "$ARCHIVE_URL" -o "$TMP_DIR/hermes-agent.tar.gz"
tar -xzf "$TMP_DIR/hermes-agent.tar.gz" -C "$TMP_DIR"

TEAM_DIR="$(find "$TMP_DIR" -maxdepth 2 -type d -path '*/team-shortcuts' | head -n 1)"
if [ -z "$TEAM_DIR" ] || [ ! -f "$TEAM_DIR/install-shortcuts.sh" ]; then
  say "ผิดพลาด: โหลดชุดติดตั้งแล้ว แต่ไม่พบ team-shortcuts/install-shortcuts.sh"
  exit 1
fi

bash "$TEAM_DIR/install-shortcuts.sh" "$@"

say ""
say "ตรวจซ้ำได้ทุกเมื่อด้วยคำสั่ง:"
say "  curl -fsSL https://raw.githubusercontent.com/rattanasak-ops/hermes-agent/main/team-shortcuts/check-shortcuts.sh | bash"
