#!/usr/bin/env bash
# relay-doctor — ตรวจว่าเครื่องนี้พร้อมใช้ AI Relay กับ Grok ไหม
# ค่าเริ่มต้นไม่เรียก Grok CLI ที่อาจเปิด OAuth/device URL เอง
# ใช้: relay-doctor          ตรวจติดตั้งแบบไม่เปิดเว็บ
#      relay-doctor --probe  ตรวจ login จริงหลังตั้งใจ probe แล้ว
set -u

probe_login=0
for arg in "$@"; do
  case "$arg" in
    --probe) probe_login=1 ;;
    -h|--help)
      echo "ใช้: relay-doctor [--probe]"
      echo "  relay-doctor          ตรวจติดตั้ง/config แบบไม่เปิด OAuth URL"
      echo "  relay-doctor --probe  เรียก grok models เพื่อตรวจ login จริง"
      exit 0
      ;;
  esac
done

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
RELAY_DIR="${ROOT}/.hermes/ai-relay"

load_env_file() {
  local env_file="$1"
  [ -f "$env_file" ] || return 0
  local line key value
  while IFS= read -r line || [ -n "$line" ]; do
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    [ -z "$line" ] && continue
    case "$line" in \#*) continue ;; esac
    case "$line" in *=*) ;; *) continue ;; esac
    key="${line%%=*}"
    value="${line#*=}"
    key="${key%"${key##*[![:space:]]}"}"
    case "$key" in
      ''|*[!A-Za-z0-9_]*|[0-9]*) continue ;;
    esac
    [ -n "${!key:-}" ] && continue
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    value="${value%\"}"; value="${value#\"}"
    value="${value%\'}"; value="${value#\'}"
    export "$key=$value"
  done < "$env_file"
}

load_env_file "$HOME/.hermes/.env"
load_env_file "$ROOT/.hermes/.env"

if [ ! -f "${RELAY_DIR}/adapters.yaml" ] || [ ! -f "${RELAY_DIR}/accounts.yaml" ]; then
  relay_add_grok_bin="$(command -v relay-add-grok 2>/dev/null || true)"
  if [ -n "${relay_add_grok_bin}" ]; then
    "${relay_add_grok_bin}" --cwd "${ROOT}" >/dev/null 2>&1 || true
  fi
fi

echo "═══ AI Relay · ตรวจความพร้อมเครื่องนี้ ═══"
echo "เครื่อง: $(hostname -s 2>/dev/null || hostname)  ·  ระบบ: $(uname -s) $(uname -m)"
echo "โปรเจกต์: ${ROOT}"
echo

ok=0
warn=0
fail=0

mark_ok() {
  printf "  ✅ %s\n" "$1"
  ok=$((ok+1))
}

mark_warn() {
  printf "  ⚠️  %s\n" "$1"
  warn=$((warn+1))
}

mark_fail() {
  printf "  ❌ %s\n" "$1"
  fail=$((fail+1))
}

find_codex() {
  for b in "${RELAY_CODEX_BIN:-}" "${XC_CODEX_BIN:-}"; do
    [ -n "$b" ] && [ -x "$b" ] && { echo "$b"; return; }
  done
  local e
  e=$(ls -1 "$HOME"/.cursor/extensions/openai.chatgpt-*/bin/*/codex 2>/dev/null | sort | tail -1)
  [ -n "$e" ] && [ -x "$e" ] && { echo "$e"; return; }
  command -v codex 2>/dev/null && return
  [ -x "$HOME/.codex/bin/codex" ] && echo "$HOME/.codex/bin/codex"
}

find_relay_call() {
  command -v relay-call 2>/dev/null && return
  [ -x "${ROOT}/scripts/ai-relay/relay-call.py" ] && echo "${ROOT}/scripts/ai-relay/relay-call.py"
}

find_relay_portal() {
  command -v relay-portal 2>/dev/null && return
  [ -x "${ROOT}/scripts/ai-relay/relay-portal.py" ] && echo "${ROOT}/scripts/ai-relay/relay-portal.py"
}

find_gate_run() {
  command -v gate-run 2>/dev/null && return
  [ -x "${ROOT}/scripts/ai-relay/gate-run.py" ] && echo "${ROOT}/scripts/ai-relay/gate-run.py"
}

echo "[1/4 โปรแกรมที่ต้องมี]"
grok_bin="$(command -v grok 2>/dev/null || true)"
codex_bin="$(find_codex || true)"
gemini_bin="$(command -v gemini 2>/dev/null || true)"
ollama_bin="$(command -v ollama 2>/dev/null || true)"
relay_call_bin="$(find_relay_call || true)"
relay_portal_bin="$(find_relay_portal || true)"
gate_run_bin="$(find_gate_run || true)"

claude_bin="$(command -v claude 2>/dev/null || true)"
[ -n "$claude_bin" ] && mark_warn "พบ claude local ที่ ${claude_bin} แต่ AI Relay จะใช้ AI Portal แทน ไม่ต้อง login Claude local" || mark_ok "ไม่ต้องมี claude local เพราะสมองหลักวิ่งผ่าน AI Portal"
[ -n "$grok_bin" ] && mark_warn "พบ grok local ที่ ${grok_bin} แต่ AI Relay จะใช้ AI Portal ก่อน" || mark_ok "ไม่ต้องมี grok local เพราะ Grok วิ่งผ่าน AI Portal"
[ -n "$codex_bin" ] && mark_warn "พบ codex local ที่ ${codex_bin} แต่ AI Relay จะใช้ AI Portal ก่อน" || mark_ok "ไม่ต้องมี codex local เพราะ Codex วิ่งผ่าน AI Portal"
[ -n "$gemini_bin" ] && mark_ok "gemini พบที่ ${gemini_bin}" || mark_warn "gemini ไม่พบบนเครื่องนี้ ใช้เป็นตัวสำรองไม่ได้"
[ -n "$ollama_bin" ] && mark_ok "ollama พบที่ ${ollama_bin}" || mark_warn "ollama ไม่พบบนเครื่องนี้ ใช้เป็นตัวสำรองในเครื่องไม่ได้"
[ -n "$relay_call_bin" ] && mark_ok "relay-call พบที่ ${relay_call_bin}" || mark_fail "relay-call ไม่พบ ให้รัน bash scripts/ai-relay/install-local.sh"
[ -n "$relay_portal_bin" ] && mark_ok "relay-portal พบที่ ${relay_portal_bin}" || mark_fail "relay-portal ไม่พบ ให้รัน relay-setup ใหม่"
[ -n "$gate_run_bin" ] && mark_ok "gate-run พบที่ ${gate_run_bin}" || mark_fail "gate-run ไม่พบ ให้รัน bash scripts/ai-relay/install-local.sh"
expected_relay_call="${HOME}/.local/bin/relay-call"
expected_relay_portal="${HOME}/.local/bin/relay-portal"
if [ -n "$relay_call_bin" ] && [ "$relay_call_bin" != "$expected_relay_call" ]; then
  mark_warn "PATH ตอนนี้เจอ relay-call ที่ ${relay_call_bin} ก่อน ${expected_relay_call} อาจเป็นตัวเก่าที่เรียก Claude/Codex/Grok local ให้รัน: export PATH=\"\$HOME/.local/bin:\$PATH\" && hash -r"
fi
if [ -n "$relay_portal_bin" ] && [ "$relay_portal_bin" != "$expected_relay_portal" ]; then
  mark_warn "PATH ตอนนี้เจอ relay-portal ที่ ${relay_portal_bin} ก่อน ${expected_relay_portal} ให้เปิด Terminal/Cursor ใหม่หลังรัน relay-setup"
fi
echo

echo "[2/4 สถานะ AI Portal token]"
if [ -n "${AI_PORTAL_URL:-}" ] || [ -n "${AI_PORTAL_BASE_URL:-}" ]; then
  mark_ok "พบ AI_PORTAL_URL/AI_PORTAL_BASE_URL"
else
  mark_fail "ยังไม่มี AI_PORTAL_URL ใน ~/.hermes/.env"
fi
if [ -n "${AI_PORTAL_CLAUDE_TOKEN:-}" ] || [ -n "${AI_RELAY_CLAUDE_TOKEN:-}" ] || [ -n "${AI_PORTAL_TOKEN:-}" ]; then
  mark_ok "พบ Claude Portal token"
else
  mark_fail "ยังไม่มี AI_PORTAL_CLAUDE_TOKEN ใน ~/.hermes/.env"
fi
if [ -n "${AI_PORTAL_CODEX_TOKEN_01:-}" ] && [ -n "${AI_PORTAL_CODEX_TOKEN_02:-}" ]; then
  mark_ok "พบ Codex Portal token แบบ cross-route 2 ID"
elif [ -n "${AI_PORTAL_CODEX_TOKEN:-}" ] || [ -n "${AI_RELAY_CODEX_TOKEN:-}" ] || [ -n "${OPENAI_API_KEY:-}" ]; then
  mark_ok "พบ Codex Portal token"
else
  mark_fail "ยังไม่มี AI_PORTAL_CODEX_TOKEN_01/02 หรือ AI_PORTAL_CODEX_TOKEN ใน ~/.hermes/.env"
fi
if [ -n "${AI_PORTAL_GROK_TOKEN:-}" ] || [ -n "${AI_RELAY_GROK_TOKEN:-}" ] || [ -n "${GROK_API_KEY:-}" ]; then
  mark_ok "พบ Grok Portal token"
else
  mark_fail "ยังไม่มี AI_PORTAL_GROK_TOKEN ใน ~/.hermes/.env"
fi

echo
echo "[2b/4 สถานะ Login local เฉพาะตัวสำรอง]"
grok_logged_in=0
if [ -z "$grok_bin" ]; then
  mark_ok "ข้าม Grok local login เพราะใช้ AI Portal token"
elif [ "$probe_login" -ne 1 ]; then
  if [ -f "$HOME/.grok/models_cache.json" ] || [ -f "$HOME/.grok/config.toml" ]; then
    mark_warn "ข้ามการ probe Grok เพื่อไม่ให้เปิด OAuth ซ้ำ พบไฟล์ Grok local แล้ว ถ้าต้องยืนยัน login จริงให้รัน relay-doctor --probe"
  else
    mark_warn "ข้ามการ probe Grok เพื่อไม่ให้เปิด OAuth ซ้ำ ถ้ายังไม่เคย login ให้รัน grok login --oauth แค่ครั้งเดียว"
  fi
else
  grok_models_output="$(grok models 2>&1 || true)"
  if printf "%s" "$grok_models_output" | grep -qiE "you are not authenticated|not authenticated|not logged|logged out"; then
    mark_fail "Grok ยังไม่ได้ login ให้รัน grok login --oauth แล้วเลือก Continue with Google"
  elif printf "%s" "$grok_models_output" | grep -qiE "available models|grok-"; then
    mark_ok "Grok login แล้ว และอ่านรายชื่อ model ได้"
    grok_logged_in=1
  else
    mark_warn "Grok ตอบกลับมา แต่รูปแบบไม่ชัด ให้รัน grok models ดูข้อความเต็มเอง"
  fi
fi

if command -v hermes >/dev/null 2>&1; then
  hermes_xai_output="$(hermes auth status xai-oauth 2>&1 || true)"
  if printf "%s" "$hermes_xai_output" | grep -qiE "logged out|no xai oauth|not logged"; then
    mark_warn "Hermes xAI ยังไม่ได้ login ถ้าต้องให้ Hermes ใช้ Grok ให้รัน hermes auth add xai-oauth"
  else
    mark_ok "Hermes xAI ไม่ได้รายงานว่า logged out"
  fi
else
  mark_warn "ไม่พบคำสั่ง hermes ข้ามการตรวจ Hermes xAI"
fi
echo

echo "[3/4 ไฟล์ตั้งค่า local-only]"
if [ -f "${RELAY_DIR}/adapters.yaml" ]; then
  mark_ok "มี ${RELAY_DIR}/adapters.yaml"
else
    mark_fail "ยังไม่มี ${RELAY_DIR}/adapters.yaml ให้รัน relay-add-grok --cwd ${ROOT}"
fi

if [ -f "${RELAY_DIR}/accounts.yaml" ]; then
  if grep -vE '^[[:space:]]*#' "${RELAY_DIR}/accounts.yaml" | grep -qiE "xai-|sk-|api[_-]?key|token|password|secret"; then
    mark_fail "accounts.yaml อาจมีรหัสลับ ให้ลบค่าลับออกก่อนใช้งาน"
  else
    mark_ok "มี ${RELAY_DIR}/accounts.yaml และไม่พบรูปแบบรหัสลับพื้นฐาน"
  fi
else
    mark_fail "ยังไม่มี ${RELAY_DIR}/accounts.yaml ให้รัน relay-add-grok --cwd ${ROOT}"
fi
echo

echo "[4/4 สรุป]"
echo "ผ่าน: ${ok} · เตือน: ${warn} · ไม่ผ่าน: ${fail}"

if [ "$fail" -eq 0 ] && [ "$probe_login" -ne 1 ]; then
  echo "พร้อมระดับติดตั้งแล้ว (ยังไม่ได้ probe login จริงเพื่อเลี่ยง OAuth popup ซ้ำ)"
  echo "Claude/Codex/Grok ใช้ AI Portal token ไม่ต้อง login local"
  exit 0
fi

if [ "$fail" -eq 0 ] && [ "$grok_logged_in" -eq 1 ]; then
  echo "พร้อมใช้ AI Relay 100% บนเครื่องนี้"
  exit 0
fi

echo "ยังไม่พร้อมใช้ AI Relay 100% ให้แก้รายการ ❌ ก่อน"
exit 1
