#!/usr/bin/env bash
# relay-doctor — ตรวจว่าเครื่องนี้พร้อมใช้ AI Relay ไหม (มีโปรแกรมเขียนโค้ดตัวไหนเรียกได้บ้าง)
# ใช้ได้ทุกเครื่อง: Mac เจ้าของ · VPS · โน้ตบุ๊กพนักงาน · อ่านอย่างเดียว ไม่เปลืองเครดิต AI
# ใช้:  bash scripts/ai-relay/relay-doctor.sh
set -u

echo "═══ AI Relay · ตรวจความพร้อมเครื่องนี้ ═══"
echo "เครื่อง: $(hostname -s 2>/dev/null || hostname)  ·  ระบบ: $(uname -s) $(uname -m)"
echo

# หา codex ข้ามที่เก็บ (env → Cursor extension → PATH → ~/.codex/bin) — ตรงกับตัวค้นใน relay-call.py
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

ok=0; miss=0
report() {  # ชื่อ, path-ที่หาเจอ
  local name="$1" bin="$2"
  if [ -n "$bin" ] && [ -x "$bin" ]; then
    printf "  ✅ %-7s → %s\n" "$name" "$bin"; ok=$((ok+1))
  else
    printf "  ❌ %-7s → ไม่พบบนเครื่องนี้\n" "$name"; miss=$((miss+1))
  fi
}

echo "[โปรแกรมเขียนโค้ด (coder) ที่ Relay เรียกได้]"
report codex  "$(find_codex)"
report grok   "$(command -v grok   2>/dev/null)"
report gemini "$(command -v gemini 2>/dev/null)"
report ollama "$(command -v ollama 2>/dev/null)"
echo

echo "สรุป: เรียกได้ $ok ตัว · ไม่พบ $miss ตัว"
if [ "$ok" -ge 1 ]; then
  echo "พร้อมใช้ Relay ✅ (มีโปรแกรมเขียนโค้ดอย่างน้อย 1 ตัว — สลับสำรองได้)"
  exit 0
else
  echo "ยังไม่พร้อม ❌ (ไม่มีโปรแกรมเขียนโค้ดเลยบนเครื่องนี้)"
  exit 1
fi
