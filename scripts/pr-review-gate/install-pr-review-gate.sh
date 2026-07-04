#!/usr/bin/env bash
# ติดตั้งด่านตรวจโค้ดก่อน merge (PR-Agent) ให้เครื่องนี้:
# 1. สร้าง venv แยกที่ ~/.hermes/pr-review-gate/venv แล้วลง pr-agent
# 2. วางไฟล์ตั้งค่า (ถ้ายังไม่มี) — key จริงอยู่ ~/.hermes/.env ไม่อยู่ในไฟล์ตั้งค่า
# 3. ทำคำสั่ง pr-review เรียกได้จากทุกที่
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC="${ROOT}/scripts/pr-review-gate"
GATE_DIR="${HOME}/.hermes/pr-review-gate"
VENV="${GATE_DIR}/venv"
BIN_DIR="${HOME}/.local/bin"

# เลือก python ที่เหมาะ (pr-agent ยังไม่รับประกันกับ python ใหม่มาก)
PYTHON_BIN=""
for c in python3.12 python3.11 python3.10 python3; do
  if command -v "$c" >/dev/null 2>&1; then PYTHON_BIN="$(command -v "$c")"; break; fi
done
[ -n "${PYTHON_BIN}" ] || { echo "ERROR: ไม่พบ python3 บนเครื่องนี้" >&2; exit 1; }

mkdir -p "${GATE_DIR}" "${BIN_DIR}"

if [ ! -x "${VENV}/bin/pip" ]; then
  "${PYTHON_BIN}" -m venv "${VENV}"
fi
"${VENV}/bin/pip" install --quiet --upgrade pip
# ล็อกเวอร์ชันกันพฤติกรรมเปลี่ยนเองตอนติดตั้งเครื่องใหม่ (จาก Codex review)
PR_AGENT_VERSION="${PR_AGENT_VERSION:-0.2.4}"
"${VENV}/bin/pip" install --quiet "pr-agent==${PR_AGENT_VERSION}" || {
  echo "ลงจาก PyPI ไม่ผ่าน — ลองลงจาก GitHub ตรง (tag v${PR_AGENT_VERSION})"
  "${VENV}/bin/pip" install --quiet "git+https://github.com/qodo-ai/pr-agent.git@v${PR_AGENT_VERSION}"
}

if [ ! -f "${GATE_DIR}/pr_agent.toml" ]; then
  cp "${SRC}/sample-config/pr_agent.toml" "${GATE_DIR}/pr_agent.toml"
fi

chmod +x "${SRC}/pr-review"
ln -sf "${SRC}/pr-review" "${BIN_DIR}/pr-review"

echo "ติดตั้งเสร็จ · ตรวจ: ${VENV}/bin/pr-agent --help"
echo "ใช้: pr-review <ลิงก์ PR> — ต้องมี GEMINI_API_KEY ใน ~/.hermes/.env และ gh login แล้ว"
