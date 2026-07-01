#!/usr/bin/env bash
# relay-doctor — ตรวจว่าเครื่องนี้พร้อมใช้ AI Relay กับ Grok ไหม
# อ่านอย่างเดียว ไม่ส่ง prompt ให้ AI และไม่พิมพ์รหัสลับ
# ใช้: relay-doctor
set -u

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
RELAY_DIR="${ROOT}/.hermes/ai-relay"

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
gate_run_bin="$(find_gate_run || true)"

[ -n "$grok_bin" ] && mark_ok "grok พบที่ ${grok_bin}" || mark_fail "grok ไม่พบบนเครื่องนี้"
[ -n "$codex_bin" ] && mark_ok "codex พบที่ ${codex_bin}" || mark_warn "codex ไม่พบบนเครื่องนี้ ใช้เป็นตัวสำรองไม่ได้"
[ -n "$gemini_bin" ] && mark_ok "gemini พบที่ ${gemini_bin}" || mark_warn "gemini ไม่พบบนเครื่องนี้ ใช้เป็นตัวสำรองไม่ได้"
[ -n "$ollama_bin" ] && mark_ok "ollama พบที่ ${ollama_bin}" || mark_warn "ollama ไม่พบบนเครื่องนี้ ใช้เป็นตัวสำรองในเครื่องไม่ได้"
[ -n "$relay_call_bin" ] && mark_ok "relay-call พบที่ ${relay_call_bin}" || mark_fail "relay-call ไม่พบ ให้รัน bash scripts/ai-relay/install-local.sh"
[ -n "$gate_run_bin" ] && mark_ok "gate-run พบที่ ${gate_run_bin}" || mark_fail "gate-run ไม่พบ ให้รัน bash scripts/ai-relay/install-local.sh"
echo

echo "[2/4 สถานะ Login]"
grok_logged_in=0
if [ -z "$grok_bin" ]; then
  mark_fail "ตรวจ Grok login ไม่ได้ เพราะยังไม่มีคำสั่ง grok"
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
  mark_fail "ยังไม่มี ${RELAY_DIR}/adapters.yaml ให้รัน bash scripts/ai-relay/install-local.sh"
fi

if [ -f "${RELAY_DIR}/accounts.yaml" ]; then
  if grep -vE '^[[:space:]]*#' "${RELAY_DIR}/accounts.yaml" | grep -qiE "xai-|sk-|api[_-]?key|token|password|secret"; then
    mark_fail "accounts.yaml อาจมีรหัสลับ ให้ลบค่าลับออกก่อนใช้งาน"
  else
    mark_ok "มี ${RELAY_DIR}/accounts.yaml และไม่พบรูปแบบรหัสลับพื้นฐาน"
  fi
else
  mark_fail "ยังไม่มี ${RELAY_DIR}/accounts.yaml ให้รัน bash scripts/ai-relay/install-local.sh"
fi
echo

echo "[4/4 สรุป]"
echo "ผ่าน: ${ok} · เตือน: ${warn} · ไม่ผ่าน: ${fail}"

if [ "$fail" -eq 0 ] && [ "$grok_logged_in" -eq 1 ]; then
  echo "พร้อมใช้ AI Relay กับ Grok 100% บนเครื่องนี้"
  exit 0
fi

echo "ยังไม่พร้อมใช้ Grok ผ่าน AI Relay 100% ให้แก้รายการ ❌ ก่อน"
exit 1
