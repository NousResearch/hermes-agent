#!/bin/bash
# ──────────────────────────────────────────────────────────────
# Hermes Agent – Docker entrypoint (gateway only, no web UI)
# Bootstraps ~/.hermes/ from Railway environment variables,
# then runs the gateway process directly.
# ──────────────────────────────────────────────────────────────
set -euo pipefail

export PYTHONUNBUFFERED=1

HERMES_HOME="${HERMES_HOME:-/root/.hermes}"

# ── Directory structure ───────────────────────────────────────
mkdir -p "$HERMES_HOME/sessions" "$HERMES_HOME/logs" \
         "$HERMES_HOME/skills"   "$HERMES_HOME/cache"

# ── Bootstrap .env ────────────────────────────────────────────
cat > "$HERMES_HOME/.env" <<EOF
# LLM provider
OPENROUTER_API_KEY=${OPENROUTER_API_KEY:-}
NOUS_API_KEY=${NOUS_API_KEY:-}
LLM_MODEL=${LLM_MODEL:-anthropic/claude-opus-4-6}

# Tools (all optional)
FIRECRAWL_API_KEY=${FIRECRAWL_API_KEY:-}
FAL_KEY=${FAL_KEY:-}
GITHUB_TOKEN=${GITHUB_TOKEN:-}
BROWSERBASE_API_KEY=${BROWSERBASE_API_KEY:-}
BROWSERBASE_PROJECT_ID=${BROWSERBASE_PROJECT_ID:-}

# Terminal backend
TERMINAL_ENV=${TERMINAL_ENV:-local}
TERMINAL_TIMEOUT=${TERMINAL_TIMEOUT:-60}

# Messaging – Telegram
TELEGRAM_BOT_TOKEN=${TELEGRAM_BOT_TOKEN:-}
TELEGRAM_HOME_CHANNEL=${TELEGRAM_HOME_CHANNEL:-}
TELEGRAM_ALLOWED_USERS=${TELEGRAM_ALLOWED_USERS:-}

# Messaging – Discord
DISCORD_BOT_TOKEN=${DISCORD_BOT_TOKEN:-}
DISCORD_HOME_CHANNEL=${DISCORD_HOME_CHANNEL:-}
DISCORD_ALLOWED_USERS=${DISCORD_ALLOWED_USERS:-}

# Messaging – Slack
SLACK_BOT_TOKEN=${SLACK_BOT_TOKEN:-}
SLACK_APP_TOKEN=${SLACK_APP_TOKEN:-}
SLACK_HOME_CHANNEL=${SLACK_HOME_CHANNEL:-}

# Gateway
GATEWAY_ALLOW_ALL_USERS=${GATEWAY_ALLOW_ALL_USERS:-false}
SESSION_IDLE_MINUTES=${SESSION_IDLE_MINUTES:-}
EOF

# ── Bootstrap config.yaml ─────────────────────────────────────
PROVIDER="${HERMES_INFERENCE_PROVIDER:-auto}"
# Auto-detect provider from available keys if not set
if [ "$PROVIDER" = "auto" ]; then
  if [ -n "${NOUS_API_KEY:-}" ]; then
    PROVIDER="nous-api"
  elif [ -n "${OPENROUTER_API_KEY:-}" ]; then
    PROVIDER="openrouter"
  fi
fi

cat > "$HERMES_HOME/config.yaml" <<EOF
model:
  default: "${LLM_MODEL:-anthropic/claude-opus-4-6}"
  provider: "${PROVIDER}"
terminal:
  backend: "${TERMINAL_ENV:-local}"
compression:
  enabled: true
  threshold: 0.85
EOF

# ── Bootstrap auth.json from env var (Nous Portal OAuth token) ─
if [ -n "${HERMES_AUTH_JSON:-}" ]; then
  if [ ! -f "$HERMES_HOME/auth.json" ] || [ "${FORCE_AUTH_RESET:-false}" = "true" ]; then
    echo "$HERMES_AUTH_JSON" | base64 -d > "$HERMES_HOME/auth.json"
    echo "[entrypoint] Wrote fresh auth.json from HERMES_AUTH_JSON"
  else
    echo "[entrypoint] Using existing auth.json from volume (set FORCE_AUTH_RESET=true to override)"
  fi
fi

# ── SOUL.md placeholder ───────────────────────────────────────
if [ ! -f "$HERMES_HOME/SOUL.md" ]; then
  echo "You are Hermes, a helpful AI assistant." > "$HERMES_HOME/SOUL.md"
fi

echo "[entrypoint] Hermes home: $HERMES_HOME"
echo "[entrypoint] Provider: $PROVIDER"
echo "[entrypoint] Model: ${LLM_MODEL:-anthropic/claude-opus-4-6}"
echo "[entrypoint] Starting gateway..."

# ── Start gateway directly ────────────────────────────────────
exec python -u -m gateway.run
