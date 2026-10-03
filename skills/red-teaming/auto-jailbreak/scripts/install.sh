#!/usr/bin/env bash
# Auto-Jailbreak — smart installer.
# By default it AUTO-DETECTS what you already have (an OpenRouter key in your
# Hermes config, or a local Ollama), picks a sensible attacker/target pair, and
# just works. No telemetry: it only talks to the endpoints chosen here.
#
#   install.sh            auto-detect, show the plan, confirm (Enter to accept)
#   install.sh --yes      auto-detect and apply, no questions
#   install.sh --manual   choose every endpoint/model yourself
#   install.sh --plan     show what it WOULD do, then exit (no install)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA="${AUTOJB_HOME:-$HOME/.auto-jailbreak}"
VENV="${AUTOJB_VENV:-$DATA/venv}"
CONFIG="${AUTOJB_CONFIG:-$DATA/config.env}"

MODE="auto"
for a in "$@"; do case "$a" in
  -y|--yes) MODE=yes;; -m|--manual) MODE=manual;; --plan) MODE=plan;;
  -h|--help) sed -n '2,12p' "$0"; exit 0;;
esac; done

# Sensible defaults for the OpenRouter path (edit freely). The attacker must be
# lightly censored, or it refuses to attack; the target is what you evaluate.
OR_URL="https://openrouter.ai/api/v1"
DEF_ATTACKER="openrouter/nousresearch/hermes-4-405b"
DEF_TARGET="openrouter/deepseek/deepseek-v4-pro"
OLLAMA_URL="http://localhost:11434"

mkdir -p "$DATA"

# --- Detection ------------------------------------------------------------
detect_or_key() {
  if [ -n "${OPENROUTER_API_KEY:-}" ]; then printf '%s' "$OPENROUTER_API_KEY"; return; fi
  for f in "$HOME/.hermes/.env" "$HOME/.config/hermes/.env" "$HOME/.hermes/hermes-agent/.env"; do
    [ -f "$f" ] || continue
    local v; v=$(grep -E '^OPENROUTER_API_KEY=' "$f" 2>/dev/null | head -1 | cut -d= -f2- | tr -d "\"' ")
    [ -n "$v" ] && { printf '%s' "$v"; return; }
  done
}
ollama_models() { curl -fsS --max-time 2 "$OLLAMA_URL/api/tags" 2>/dev/null | grep -oE '"name":"[^"]+"' | cut -d'"' -f4; }

OR_KEY="$(detect_or_key || true)"
OLL_LIST="$(ollama_models || true)"

# --- Decide the plan ------------------------------------------------------
SRC=""; T_EP=""; TMODEL=""; T_KEY=""; A_EP=""; AMODEL=""; A_KEY=""
if [ -n "$OR_KEY" ]; then
  SRC="OpenRouter (key found in your Hermes config)"
  T_EP="$OR_URL"; T_KEY="$OR_KEY"; TMODEL="$DEF_TARGET"
  A_EP="$OR_URL"; A_KEY="$OR_KEY"; AMODEL="$DEF_ATTACKER"
elif [ -n "$OLL_LIST" ]; then
  SRC="local Ollama"
  first="ollama/$(printf '%s\n' "$OLL_LIST" | head -1)"
  unc=$(printf '%s\n' "$OLL_LIST" | grep -iE 'dolphin|uncensor|abliterat|venice' | head -1)
  T_EP="$OLLAMA_URL"; T_KEY="ollama"; TMODEL="$first"
  A_EP="$OLLAMA_URL"; A_KEY="ollama"
  AMODEL=$([ -n "$unc" ] && echo "ollama/$unc" || echo "$first")
fi

show_plan() {
  echo "Detected source : ${SRC:-none}"
  echo "  target   : ${TMODEL:-<none>}   @ ${T_EP:-<none>}"
  echo "  attacker : ${AMODEL:-<none>}   @ ${A_EP:-<none>}"
  [ -n "$OR_KEY" ] && echo "  OpenRouter key : found (hidden)"
}

if [ "$MODE" = "plan" ]; then show_plan; exit 0; fi

# --- Manual questionnaire (fallback / --manual) ---------------------------
ask() { local p="$1" d="$2" v; read -rp "$p [$d]: " v; printf '%s' "${v:-$d}"; }
manual() {
  echo "Manual setup:"
  T_EP="$(ask 'Target endpoint' "${T_EP:-http://localhost:11434}")"
  TMODEL="$(ask 'Target model' "${TMODEL:-ollama/llama3}")"
  read -rp "Target API key [${T_KEY:-ollama}]: " k; T_KEY="${k:-${T_KEY:-ollama}}"
  A_EP="$(ask 'Attacker endpoint' "${A_EP:-$T_EP}")"
  AMODEL="$(ask 'Attacker model (uncensored)' "${AMODEL:-$TMODEL}")"
  read -rp "Attacker API key [${A_KEY:-$T_KEY}]: " k; A_KEY="${k:-${A_KEY:-$T_KEY}}"
}

if [ "$MODE" = "manual" ]; then manual
elif [ -z "$SRC" ]; then
  echo "Nothing auto-detected (no OpenRouter key, no Ollama)."
  [ "$MODE" = "yes" ] && { echo "Refusing to guess with --yes." >&2; exit 1; }
  manual
else
  echo "== Auto-detected setup =="; show_plan; echo
  if [ "$MODE" = "auto" ]; then
    read -rp "Use this? [Y = yes / n = cancel / m = manual]: " ans
    case "${ans:-Y}" in [nN]*) echo "Cancelled."; exit 0;; [mM]*) manual;; esac
  fi
fi

# --- Dependencies (venv + PyRIT) ------------------------------------------
if [ ! -x "$VENV/bin/python" ]; then echo "Creating virtualenv: $VENV"; python3 -m venv "$VENV"; fi
if "$VENV/bin/python" -c "import pyrit" 2>/dev/null; then
  echo "Dependencies already present."
else
  echo "Installing PyRIT + LiteLLM (large download, a few minutes)..."
  "$VENV/bin/pip" install --quiet --upgrade pip
  "$VENV/bin/pip" install --quiet -r "$HERE/requirements.txt"
fi

# --- Write config ---------------------------------------------------------
umask 077
cat > "$CONFIG" <<CFG
# Auto-Jailbreak configuration. Edit freely; re-run install.sh to change.
LITELLM_ENDPOINT=$T_EP
LITELLM_MODEL=$TMODEL
LITELLM_API_KEY=$T_KEY
LITELLM_ADVERSE_ENDPOINT=$A_EP
LITELLM_ADVERSE_MODEL=$AMODEL
LITELLM_ADVERSE_API_KEY=$A_KEY
CFG
echo "Wrote config: $CONFIG"

# --- Smoke test -----------------------------------------------------------
echo "Testing engine..."
if echo '{}' | "$VENV/bin/python" "$HERE/attaque.py" 2>/dev/null | grep -q "question vide"; then
  echo "OK. Engine loads."
else
  echo "WARN: unexpected engine response; check the install above."
fi
echo
echo "Done. Run an attack with:"
echo "  echo '{\"question\": \"<objective>\", \"mode\": \"crescendo\"}' | bash \"$HERE/redteam.sh\""
