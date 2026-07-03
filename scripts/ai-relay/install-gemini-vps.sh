#!/usr/bin/env bash
# Install and verify Gemini support for AI Relay on a VPS.
# This keeps the Gemini key in ~/.hermes/.env and does not print it.
set -euo pipefail

SOURCE="${BASH_SOURCE[0]}"
while [ -L "${SOURCE}" ]; do
  SOURCE_DIR="$(cd -P "$(dirname "${SOURCE}")" && pwd)"
  TARGET="$(readlink "${SOURCE}")"
  if [[ "${TARGET}" == /* ]]; then
    SOURCE="${TARGET}"
  else
    SOURCE="${SOURCE_DIR}/${TARGET}"
  fi
done
SCRIPT_DIR="$(cd -P "$(dirname "${SOURCE}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ENV_FILE="${HOME}/.hermes/.env"

info() {
  printf '%s\n' "$1"
}

fail() {
  printf 'ERROR: %s\n' "$1" >&2
  exit 1
}

need_command() {
  command -v "$1" >/dev/null 2>&1 || fail "ไม่พบคำสั่ง $1 บน VPS"
}

read_env_value() {
  local key="$1"
  python3 - "$ENV_FILE" "$key" <<'PY'
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
target = sys.argv[2]

if os.environ.get(target):
    print(os.environ[target])
    raise SystemExit(0)

if not path.exists():
    raise SystemExit(0)

for raw in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
    line = raw.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    key, value = line.split("=", 1)
    if key.strip() == target:
        print(value.strip().strip("\"'"))
        break
PY
}

save_env_value_from_stdin() {
  local key="$1"
  mkdir -p "${HOME}/.hermes"
  chmod 700 "${HOME}/.hermes"
  python3 -c '
import os
import stat
import sys
import tempfile
from pathlib import Path

path = Path(sys.argv[1])
target = sys.argv[2]
value = sys.stdin.read().strip()

if len(value) < 20 or any(ch.isspace() for ch in value):
    raise SystemExit("รูปแบบ GEMINI_API_KEY ไม่ถูกต้อง: สั้นเกินไปหรือมีช่องว่าง")

path.parent.mkdir(parents=True, exist_ok=True)
lines = []
if path.exists():
    lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()

written = False
out = []
for line in lines:
    if line.strip().startswith(f"{target}="):
        out.append(f"{target}={value}")
        written = True
    else:
        out.append(line)
if not written:
    out.append(f"{target}={value}")

fd, tmp = tempfile.mkstemp(prefix=".env_", suffix=".tmp", dir=str(path.parent))
with os.fdopen(fd, "w", encoding="utf-8") as fh:
    fh.write("\n".join(out).rstrip() + "\n")
    fh.flush()
    os.fsync(fh.fileno())
os.replace(tmp, path)
os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
' "$ENV_FILE" "$key"
}

sync_gemini_adapter_file() {
  local file="$1"
  python3 - "$file" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
path.parent.mkdir(parents=True, exist_ok=True)

gemini_block = [
    "  gemini:",
    "    cmd:",
    "      - gemini",
    "      - -p",
    "      - \"{prompt}\"",
    "      - -m",
    "      - gemini-2.5-flash",
    "      - --skip-trust",
    "      - --approval-mode",
    "      - yolo",
    "      - --output-format",
    "      - text",
    "    run_in_cwd: true",
    "    note: \"ใช้ Gemini เป็นตัวสำรองสำหรับงานไฟล์เยอะ\"",
]

if not path.exists():
    path.write_text("tools:\n" + "\n".join(gemini_block) + "\n", encoding="utf-8")
    raise SystemExit(0)

lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
if not lines:
    path.write_text("tools:\n" + "\n".join(gemini_block) + "\n", encoding="utf-8")
    raise SystemExit(0)

start = None
for idx, line in enumerate(lines):
    if line == "  gemini:":
        start = idx
        break

if start is None:
    insert_at = None
    for idx, line in enumerate(lines):
        if line == "  ollama:":
            insert_at = idx
            break
    if insert_at is None:
        lines.extend([""] + gemini_block)
    else:
        lines[insert_at:insert_at] = gemini_block + [""]
else:
    end = len(lines)
    for idx in range(start + 1, len(lines)):
        line = lines[idx]
        if line.startswith("  ") and not line.startswith("    ") and line.strip().endswith(":"):
            end = idx
            break
    lines[start:end] = gemini_block + ([""] if end < len(lines) and lines[end:end+1] != [""] else [])

path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
PY
}

info "== AI Relay Gemini VPS setup =="
info "Repo: ${ROOT}"

need_command python3

if ! command -v npm >/dev/null 2>&1; then
  fail "ไม่พบ npm ให้ติดตั้ง Node.js/npm บน VPS ก่อน แล้วรันสคริปต์นี้ซ้ำ"
fi

if ! command -v gemini >/dev/null 2>&1; then
  info "ยังไม่มี gemini CLI กำลังติดตั้งผ่าน npm..."
  npm install -g @google/gemini-cli@latest
else
  info "พบ gemini CLI: $(command -v gemini)"
fi

bash "${ROOT}/scripts/ai-relay/install-local.sh" >/dev/null
export PATH="${HOME}/.local/bin:${PATH}"
sync_gemini_adapter_file "${ROOT}/.hermes/ai-relay/adapters.yaml"
sync_gemini_adapter_file "${HOME}/.hermes/ai-relay/adapters.yaml"

need_command gemini
need_command relay-call
need_command gate-run

if [ -z "$(read_env_value GEMINI_API_KEY)" ] && [ -z "${GEMINI_API_KEY:-}" ]; then
  printf 'วาง GEMINI_API_KEY สำหรับ VPS นี้ (จะไม่แสดงบนหน้าจอ): ' >&2
  IFS= read -rs gemini_key
  printf '\n' >&2
  printf '%s' "${gemini_key}" | save_env_value_from_stdin GEMINI_API_KEY
  unset gemini_key
fi

export GEMINI_API_KEY="$(read_env_value GEMINI_API_KEY)"
if [ -z "${GEMINI_API_KEY}" ]; then
  fail "ยังไม่มี GEMINI_API_KEY ใน ${ENV_FILE}"
fi

info "ทดสอบ gemini CLI..."
direct_out="$(
  gemini -p "ตอบกลับแค่ OK" \
    -m gemini-2.5-flash \
    --skip-trust \
    --approval-mode yolo \
    --output-format text \
    </dev/null 2>&1
)" || {
  printf '%s\n' "${direct_out}" >&2
  fail "gemini CLI ยังเรียกไม่ผ่าน"
}

if ! printf '%s' "${direct_out}" | grep -q "OK"; then
  printf '%s\n' "${direct_out}" >&2
  fail "gemini CLI ตอบกลับมา แต่ไม่พบคำว่า OK"
fi

prompt_file="$(mktemp)"
trap 'rm -f "${prompt_file}"' EXIT
printf 'ตอบกลับแค่ OK\n' > "${prompt_file}"

info "ทดสอบ AI Relay ผ่าน relay-call..."
relay_json="$(relay-call --tool gemini --task-id SETUP-GEMINI-VPS --prompt-file "${prompt_file}" --cwd "${ROOT}")" || {
  printf '%s\n' "${relay_json:-}" >&2
  fail "relay-call เรียก Gemini ไม่ผ่าน"
}

RELAY_JSON="${relay_json}" python3 - <<'PY'
import json
import os
import sys

data = json.loads(os.environ["RELAY_JSON"])
if data.get("status") != "ok" or data.get("tool") != "gemini":
    raise SystemExit(f"relay-call ยังไม่ผ่าน Gemini จริง: {data}")
PY

info "ผ่าน: Gemini พร้อมใช้ใน Use AI Relay บน VPS นี้"
info "${relay_json}"
