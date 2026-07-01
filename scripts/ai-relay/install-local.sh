#!/usr/bin/env bash
# Install AI Relay helper commands for the current user.
# ไม่แตะรหัสลับ ไม่ login แทนพนักงาน และไม่เขียน token ลงไฟล์
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BIN_DIR="${HOME}/.local/bin"
PROJECT_RELAY_DIR="${ROOT}/.hermes/ai-relay"

mkdir -p "${BIN_DIR}"
mkdir -p "${PROJECT_RELAY_DIR}"

chmod +x "${ROOT}/scripts/ai-relay/relay-call.py"
chmod +x "${ROOT}/scripts/ai-relay/gate-run.py"
chmod +x "${ROOT}/scripts/ai-relay/relay-doctor.sh"
chmod +x "${ROOT}/scripts/ai-relay/relay-status.sh"
chmod +x "${ROOT}/scripts/ai-relay/relay-now.sh"
chmod +x "${ROOT}/scripts/ai-relay/relay-add-grok.py"

ln -sf "${ROOT}/scripts/ai-relay/relay-call.py" "${BIN_DIR}/relay-call"
ln -sf "${ROOT}/scripts/ai-relay/gate-run.py" "${BIN_DIR}/gate-run"
ln -sf "${ROOT}/scripts/ai-relay/relay-doctor.sh" "${BIN_DIR}/relay-doctor"
ln -sf "${ROOT}/scripts/ai-relay/relay-status.sh" "${BIN_DIR}/relay-status"
ln -sf "${ROOT}/scripts/ai-relay/relay-now.sh" "${BIN_DIR}/relay-now"
ln -sf "${ROOT}/scripts/ai-relay/relay-add-grok.py" "${BIN_DIR}/relay-add-grok"

if [ ! -f "${PROJECT_RELAY_DIR}/adapters.yaml" ]; then
  cp "${ROOT}/scripts/ai-relay/sample-config/adapters.yaml" "${PROJECT_RELAY_DIR}/adapters.yaml"
fi

if [ ! -f "${PROJECT_RELAY_DIR}/accounts.yaml" ]; then
  cp "${ROOT}/scripts/ai-relay/sample-config/accounts.yaml" "${PROJECT_RELAY_DIR}/accounts.yaml"
fi

"${ROOT}/scripts/ai-relay/relay-add-grok.py" --cwd "${ROOT}" >/dev/null

cat <<MSG
AI Relay ติดตั้งคำสั่งให้เครื่องนี้แล้ว

คำสั่งที่เพิ่ม:
  relay-call
  gate-run
  relay-doctor
  relay-status
  relay-now
  relay-add-grok

ไฟล์ local-only ที่สร้างถ้ายังไม่มี:
  ${PROJECT_RELAY_DIR}/adapters.yaml
  ${PROJECT_RELAY_DIR}/accounts.yaml

ขั้นต่อไป:
  1. ให้พนักงานรัน: grok login --oauth
  2. เลือก Continue with Google ในเบราว์เซอร์
  3. กลับมารัน: relay-doctor

ถ้า shell หา relay-doctor ไม่เจอ ให้เพิ่ม ~/.local/bin เข้า PATH ก่อน
MSG
