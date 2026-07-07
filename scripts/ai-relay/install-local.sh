#!/usr/bin/env bash
# Install AI Relay helper commands for the current user.
# ไม่แตะรหัสลับ ไม่ login แทนพนักงาน และไม่เขียน token ลงไฟล์
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BIN_DIR="${HOME}/.local/bin"
PROJECT_RELAY_DIR="${ROOT}/.hermes/ai-relay"
HOME_RELAY_DIR="${HOME}/.hermes/ai-relay"

mkdir -p "${BIN_DIR}"
mkdir -p "${PROJECT_RELAY_DIR}"
mkdir -p "${HOME_RELAY_DIR}"

chmod +x "${ROOT}/scripts/ai-relay/relay-call.py"
chmod +x "${ROOT}/scripts/ai-relay/gate-run.py"
chmod +x "${ROOT}/scripts/ai-relay/relay-report.py"
chmod +x "${ROOT}/scripts/ai-relay/relay-status.py"
chmod +x "${ROOT}/scripts/ai-relay/relay-suggest.py"
chmod +x "${ROOT}/scripts/ai-relay/relay-relogin.py"
chmod +x "${ROOT}/scripts/ai-relay/relay-doctor.sh"
chmod +x "${ROOT}/scripts/ai-relay/relay-status.sh"
chmod +x "${ROOT}/scripts/ai-relay/relay-now.sh"
chmod +x "${ROOT}/scripts/ai-relay/relay-add-grok.py"
chmod +x "${ROOT}/scripts/ai-relay/relay-setup.sh"
chmod +x "${ROOT}/scripts/ai-relay/relay-grok.sh"
chmod +x "${ROOT}/scripts/ai-relay/install-gemini-vps.sh"

ln -sf "${ROOT}/scripts/ai-relay/relay-call.py" "${BIN_DIR}/relay-call"
ln -sf "${ROOT}/scripts/ai-relay/gate-run.py" "${BIN_DIR}/gate-run"
ln -sf "${ROOT}/scripts/ai-relay/relay-report.py" "${BIN_DIR}/relay-report"
ln -sf "${ROOT}/scripts/ai-relay/relay-suggest.py" "${BIN_DIR}/relay-suggest"
ln -sf "${ROOT}/scripts/ai-relay/relay-relogin.py" "${BIN_DIR}/relay-relogin"
ln -sf "${ROOT}/scripts/ai-relay/relay-doctor.sh" "${BIN_DIR}/relay-doctor"
ln -sf "${ROOT}/scripts/ai-relay/relay-status.sh" "${BIN_DIR}/relay-status"
ln -sf "${ROOT}/scripts/ai-relay/relay-now.sh" "${BIN_DIR}/relay-now"
ln -sf "${ROOT}/scripts/ai-relay/relay-add-grok.py" "${BIN_DIR}/relay-add-grok"
ln -sf "${ROOT}/scripts/ai-relay/relay-setup.sh" "${BIN_DIR}/relay-setup"
ln -sf "${ROOT}/scripts/ai-relay/relay-grok.sh" "${BIN_DIR}/relay-grok"
ln -sf "${ROOT}/scripts/ai-relay/install-gemini-vps.sh" "${BIN_DIR}/install-gemini-vps"

if [ ! -f "${PROJECT_RELAY_DIR}/adapters.yaml" ]; then
  cp "${ROOT}/scripts/ai-relay/sample-config/adapters.yaml" "${PROJECT_RELAY_DIR}/adapters.yaml"
fi

if [ ! -f "${PROJECT_RELAY_DIR}/accounts.yaml" ]; then
  cp "${ROOT}/scripts/ai-relay/sample-config/accounts.yaml" "${PROJECT_RELAY_DIR}/accounts.yaml"
fi

if [ ! -f "${HOME_RELAY_DIR}/adapters.yaml" ]; then
  cp "${ROOT}/scripts/ai-relay/sample-config/adapters.yaml" "${HOME_RELAY_DIR}/adapters.yaml"
fi

if [ ! -f "${HOME_RELAY_DIR}/accounts.yaml" ]; then
  cp "${ROOT}/scripts/ai-relay/sample-config/accounts.yaml" "${HOME_RELAY_DIR}/accounts.yaml"
fi

"${ROOT}/scripts/ai-relay/relay-add-grok.py" --cwd "${ROOT}" >/dev/null
"${ROOT}/scripts/ai-relay/relay-add-grok.py" --cwd "${HOME}" >/dev/null

cat <<MSG
AI Relay ติดตั้งคำสั่งให้เครื่องนี้แล้ว

คำสั่งที่เพิ่ม:
  relay-call
  gate-run
  relay-report
  relay-status
  relay-suggest
  relay-relogin
  relay-doctor
  relay-now
  relay-add-grok
  relay-grok
  install-gemini-vps

ไฟล์ local-only ที่สร้างถ้ายังไม่มี:
  ${PROJECT_RELAY_DIR}/adapters.yaml
  ${PROJECT_RELAY_DIR}/accounts.yaml
  ${HOME_RELAY_DIR}/adapters.yaml
  ${HOME_RELAY_DIR}/accounts.yaml

ขั้นต่อไป:
  1. ให้พนักงานรัน: grok login --oauth
  2. เลือก Continue with Google ในเบราว์เซอร์
  3. กลับมารัน: relay-doctor
  4. เรียกใช้จากที่ไหนก็ได้: relay-grok "ข้อความที่ต้องการส่งให้ Grok"
  5. ถ้าจะเปิด Gemini บน VPS ให้รัน: install-gemini-vps

ถ้า shell หา relay-doctor ไม่เจอ ให้เพิ่ม ~/.local/bin เข้า PATH ก่อน
MSG
