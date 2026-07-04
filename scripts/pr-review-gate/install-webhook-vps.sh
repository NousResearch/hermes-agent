#!/usr/bin/env bash
# ตั้งบริการ webhook ให้ทุก MR ถูกรีวิวอัตโนมัติ (เฟส 2 ของ F2 · รันบน VPS เท่านั้น)
# - เปิดบริการ pr-agent gitlab_webhook ใต้ systemd ฟังที่พอร์ต PORT
# - สร้างรหัสลับ webhook + ไฟล์ env เฉพาะ (โหมด 600)
# - จำกัด iptables ให้เฉพาะ GitLab (และ localhost) ยิงเข้าพอร์ตนี้ได้
set -euo pipefail

PORT="${PR_WEBHOOK_PORT:-3010}"
GITLAB_IP="${GITLAB_IP:-103.142.150.85}"
HOME_ENV="${HOME}/.hermes/.env"
GATE_DIR="${HOME}/.hermes/pr-review-gate"
WEBHOOK_ENV="${GATE_DIR}/webhook.env"
VENV="${GATE_DIR}/venv"
REPO="${HOME}/SynerryTools/hermes-agent/main"
UNIT="/etc/systemd/system/pr-review-webhook.service"

read_env() { grep "^$1=" "${HOME_ENV}" 2>/dev/null | head -1 | cut -d= -f2-; }

TOKEN="$(read_env GITLAB__PERSONAL_ACCESS_TOKEN)"
GEMINI="$(read_env GEMINI_API_KEY)"
GLURL="$(read_env GITLAB_URL)"
[ -n "${TOKEN}" ] || { echo "ERROR: ไม่พบ GITLAB__PERSONAL_ACCESS_TOKEN ใน ${HOME_ENV}"; exit 1; }
[ -n "${GLURL}" ] || GLURL="https://gitlab.dev.jigsawgroups.work"

# รหัสลับ webhook: ใช้ของเดิมถ้ามี ไม่งั้นสุ่มใหม่
if [ -f "${WEBHOOK_ENV}" ] && grep -q "^GITLAB__SHARED_SECRET=" "${WEBHOOK_ENV}"; then
  SECRET="$(grep '^GITLAB__SHARED_SECRET=' "${WEBHOOK_ENV}" | cut -d= -f2-)"
else
  SECRET="$(head -c 24 /dev/urandom | base64 | tr -d '/+=' | head -c 32)"
fi

mkdir -p "${GATE_DIR}"
umask 077
cat > "${WEBHOOK_ENV}" <<ENV
GITLAB__PERSONAL_ACCESS_TOKEN=${TOKEN}
GITLAB__URL=${GLURL}
GITLAB__SHARED_SECRET=${SECRET}
CONFIG__GIT_PROVIDER=gitlab
CONFIG__MODEL=gemini/gemini-2.5-flash
CONFIG__FALLBACK_MODELS=["openrouter/anthropic/claude-haiku-4.5"]
CONFIG__CUSTOM_MODEL_MAX_TOKENS=128000
GOOGLE_AI_STUDIO__GEMINI_API_KEY=${GEMINI}
PORT=${PORT}
ENV
chmod 600 "${WEBHOOK_ENV}"

# systemd service (WorkingDirectory = repo ที่มี .pr_agent.toml → รับกติกาไทย)
sudo tee "${UNIT}" >/dev/null <<UNIT
[Unit]
Description=PR Review Gate — GitLab webhook auto-reviewer (F2)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${USER}
WorkingDirectory=${REPO}
EnvironmentFile=${WEBHOOK_ENV}
ExecStart=${VENV}/bin/python -m pr_agent.servers.gitlab_webhook
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT

sudo systemctl daemon-reload
sudo systemctl enable --now pr-review-webhook.service

# จำกัดพอร์ตเฉพาะ GitLab + localhost (default policy ACCEPT จึงต้อง DROP เอง)
ensure_rule() { sudo iptables -C "$@" 2>/dev/null || sudo iptables "$@"; }
# ลบ DROP เก่าออกก่อน (กันซ้อน) แล้วใส่ ACCEPT (localhost, gitlab) ไว้บน DROP
sudo iptables -D INPUT -p tcp --dport "${PORT}" -j DROP 2>/dev/null || true
ensure_rule -I INPUT 1 -p tcp --dport "${PORT}" -s 127.0.0.1 -j ACCEPT
ensure_rule -I INPUT 2 -p tcp --dport "${PORT}" -s "${GITLAB_IP}" -j ACCEPT
sudo iptables -A INPUT -p tcp --dport "${PORT}" -j DROP

echo "เสร็จ · บริการฟังที่พอร์ต ${PORT} · secret ยาว ${#SECRET} ตัว"
echo "SECRET_FOR_HOOK=${SECRET}"
