#!/bin/sh
# Provision ONE Avocado customer as a native Hermes profile + Telegram bot.
#
# RUNS INSIDE THE RAILWAY HERMES CONTAINER ONLY. It refuses to run anywhere that
# isn't a Railway service (guard below), so it can never touch a local Hermes.
#
# Usage (in the Railway service Shell):
#   SLUG=pilot-1 \
#   TELEGRAM_BOT_TOKEN=123:ABC \
#   TELEGRAM_USER_ID=987654321 \
#   AVOCADO_MCP_KEY=sk_avo_... \
#   OPENROUTER_API_KEY=sk-or-...   # key has a $10 hard cap set in OpenRouter \
#   sh provision-in-container.sh
#
# Optional: MODEL (default xiaomi/mimo-v2.5-pro), VISION_MODEL (default
#   google/gemini-2.5-flash, used only for image analysis), MAX_ITER (default 40).
set -eu

# ---- Safety guard: Railway-only. Railway injects RAILWAY_ENVIRONMENT in every
#      deployment; it is never set on a laptop. ----
if [ -z "${RAILWAY_ENVIRONMENT:-}" ]; then
  echo "REFUSING: \$RAILWAY_ENVIRONMENT is not set."
  echo "This script only runs inside a Railway service (the deployed Hermes), never locally."
  exit 1
fi

: "${SLUG:?set SLUG}"
: "${TELEGRAM_BOT_TOKEN:?set TELEGRAM_BOT_TOKEN}"
: "${TELEGRAM_USER_ID:?set TELEGRAM_USER_ID}"
: "${AVOCADO_MCP_KEY:?set AVOCADO_MCP_KEY}"
# OPENROUTER_API_KEY is OPTIONAL (beta model, 2026-06-12): when omitted, no
# key is written to the profile .env and the gateway inherits the shared
# fleet key from the Railway service variable (per-profile s6 run scripts use
# with-contenv, and a key absent from the profile .env falls through to the
# container env). Pass it explicitly to give a customer their own capped key.
MODEL="${MODEL:-xiaomi/mimo-v2.5-pro}"
# Vision model for image analysis. MiMo (the brain) can't see pixels, so image
# turns delegate to this cheap multimodal model; text turns never touch it.
VISION_MODEL="${VISION_MODEL:-google/gemini-2.5-flash}"
MAX_ITER="${MAX_ITER:-40}"
HOME_DIR="${HERMES_HOME:-/opt/data}"
PROFILE_DIR="$HOME_DIR/profiles/$SLUG"

echo "→ creating profile: $SLUG"
hermes profile create "$SLUG" 2>/dev/null || echo "  (profile already exists, continuing)"
mkdir -p "$PROFILE_DIR"
# Per-customer skills dir (volume-persisted). Auto-discovered as this profile's
# local skills root when the gateway runs — this is where customer-authored
# skills land. Empty is fine; the shared library comes via skills.external_dirs.
mkdir -p "$PROFILE_DIR/skills"

echo "→ writing config.yaml (Avocado MCP scoped to this customer, safe toolset, manual approvals)"
cat > "$PROFILE_DIR/config.yaml" <<YAML
model:
  default: $MODEL
  provider: openrouter
auxiliary:
  # MiMo can't see pixels, so image turns delegate to a cheap multimodal model.
  # Only fires when an image is attached; text turns stay pure MiMo. Empty
  # base_url/api_key inherit the OpenRouter key (per-customer or shared fleet).
  vision:
    provider: openrouter
    model: $VISION_MODEL
agent:
  max_turns: $MAX_ITER
  gateway_timeout: 1800
delegation:
  max_iterations: 30
approvals:
  mode: manual
  timeout: 120
  cron_mode: deny
mcp_servers:
  avocado:
    url: https://www.avocadoai.co/api/mcp
    headers:
      Authorization: "Bearer $AVOCADO_MCP_KEY"
    connect_timeout: 60
    timeout: 180
skills:
  # Curated Avocado skill templates, baked into the image at this path and
  # shared across every customer. This customer's own custom skills live in
  # \$PROFILE_DIR/skills (the profile-local dir, auto-discovered, no config).
  external_dirs:
    - /opt/hermes/avocado-skills
platform_toolsets:
  telegram:
    - image_gen
    - vision
    - tts
    - web
    - memory
    - session_search
    - messaging
    - clarify
    - todo
telegram:
  enabled: true
  reactions: false
cron:
  wrap_response: true
YAML

echo "→ writing .env (Telegram bot + allowlist${OPENROUTER_API_KEY:+ + per-customer OpenRouter key})"
cat > "$PROFILE_DIR/.env" <<ENV
${OPENROUTER_API_KEY:+OPENROUTER_API_KEY=$OPENROUTER_API_KEY}
TELEGRAM_BOT_TOKEN=$TELEGRAM_BOT_TOKEN
TELEGRAM_ALLOWED_USERS=$TELEGRAM_USER_ID
HERMES_MAX_ITERATIONS=$MAX_ITER
AUTO_UPDATE=false
ENV
if [ -z "${OPENROUTER_API_KEY:-}" ]; then
  echo "  (no per-customer OpenRouter key — inheriting the shared fleet key from the service env)"
fi
chmod 600 "$PROFILE_DIR/.env"

# When this script runs as root (e.g. via `railway ssh`, which lands you in the
# container as root rather than the `hermes` runtime user), the config.yaml and
# .env we just wrote are root-owned. The supervised gateway runs as the hermes
# user (UID 10000) and would hit `PermissionError: .../.env` on startup. Chown
# the files we created to the hermes user so the gateway can read them.
if [ "$(id -u)" = 0 ] && id hermes >/dev/null 2>&1; then
  echo "→ chowning profile config to hermes user (script ran as root)"
  chown hermes:hermes "$PROFILE_DIR/config.yaml" "$PROFILE_DIR/.env"
  chown -R hermes:hermes "$PROFILE_DIR/skills"
fi

echo "→ starting this profile's gateway (the bot goes live)"
hermes -p "$SLUG" gateway stop 2>/dev/null || true
hermes -p "$SLUG" gateway start

echo ""
echo "✓ $SLUG provisioned. Bot is live (polling), locked to Telegram user $TELEGRAM_USER_ID."
echo "  Test: message the bot from that account; a different account is ignored."
