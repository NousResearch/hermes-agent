#!/usr/bin/env bash
# ติดตั้ง violation-audit ให้เครื่องนี้:
# 1. ทำคำสั่ง violation-audit เรียกได้จากทุกที่ (~/.local/bin)
# 2. ตั้งเวลารันอัตโนมัติทุกวันจันทร์ 09:00 ผ่าน launchd + เด้งแจ้งเตือน macOS
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT="${ROOT}/scripts/violation-audit/violation_audit.py"
BIN_DIR="${HOME}/.local/bin"
PLIST="${HOME}/Library/LaunchAgents/com.hermes.violation-audit.plist"
LOG_DIR="${HOME}/.claude/ai-fail-stats"

chmod +x "${SCRIPT}"
mkdir -p "${BIN_DIR}" "${LOG_DIR}"
ln -sf "${SCRIPT}" "${BIN_DIR}/violation-audit"

PYTHON_BIN="$(command -v python3 || true)"
if [ -z "${PYTHON_BIN}" ]; then
  echo "ERROR: ไม่พบ python3 บนเครื่องนี้ — ติดตั้ง python3 ก่อนแล้วรันใหม่" >&2
  exit 1
fi

if [ "$(uname -s)" = "Darwin" ]; then
  cat > "${PLIST}" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.hermes.violation-audit</string>
  <key>ProgramArguments</key>
  <array>
    <string>${PYTHON_BIN}</string>
    <string>${SCRIPT}</string>
    <string>--notify</string>
  </array>
  <key>StartCalendarInterval</key>
  <dict>
    <key>Weekday</key><integer>1</integer>
    <key>Hour</key><integer>9</integer>
    <key>Minute</key><integer>0</integer>
  </dict>
  <key>StandardOutPath</key><string>${LOG_DIR}/violation-audit.launchd.log</string>
  <key>StandardErrorPath</key><string>${LOG_DIR}/violation-audit.launchd.err</string>
</dict>
</plist>
PLIST
  launchctl bootout "gui/$(id -u)" "${PLIST}" 2>/dev/null || true
  launchctl bootstrap "gui/$(id -u)" "${PLIST}"
  echo "ตั้งเวลาแล้ว: ทุกวันจันทร์ 09:00 (launchd: com.hermes.violation-audit)"
else
  echo "เครื่องนี้ไม่ใช่ macOS — ข้ามการตั้งเวลา launchd (บน VPS ให้ใช้ cron/systemd timer แทน)"
fi

echo "ติดตั้งเสร็จ · ลองรันเองได้เลย: violation-audit"
