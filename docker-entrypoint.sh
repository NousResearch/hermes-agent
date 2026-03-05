#!/bin/bash
# ──────────────────────────────────────────────────────────────
# Hermes Agent – Docker entrypoint
# Bootstraps ~/.hermes config from environment variables before
# starting the gateway process.
# ──────────────────────────────────────────────────────────────
set -euo pipefail

HERMES_HOME="${HERMES_HOME:-/root/.hermes}"

# ── Create directory structure ────────────────────────────────
mkdir -p \
    "$HERMES_HOME/sessions" \
    "$HERMES_HOME/logs" \
    "$HERMES_HOME/skills" \
    "$HERMES_HOME/cache"

# ── Write .env (hermes reads this on startup) ─────────────────
cat > "$HERMES_HOME/.env" <<EOF
# LLM
OPENROUTER_API_KEY=${OPENROUTER_API_KEY:-}
LLM_MODEL=${LLM_MODEL:-anthropic/claude-opus-4-6}

# Tools
FIRECRAWL_API_KEY=${FIRECRAWL_API_KEY:-}
NOUS_API_KEY=${NOUS_API_KEY:-}
FAL_KEY=${FAL_KEY:-}
HONCHO_API_KEY=${HONCHO_API_KEY:-}
GITHUB_TOKEN=${GITHUB_TOKEN:-}

# Terminal
TERMINAL_ENV=${TERMINAL_ENV:-local}
TERMINAL_TIMEOUT=${TERMINAL_TIMEOUT:-60}
TERMINAL_LIFETIME_SECONDS=${TERMINAL_LIFETIME_SECONDS:-300}

# Browser
BROWSERBASE_API_KEY=${BROWSERBASE_API_KEY:-}
BROWSERBASE_PROJECT_ID=${BROWSERBASE_PROJECT_ID:-}
BROWSERBASE_PROXIES=${BROWSERBASE_PROXIES:-true}

# Voice
VOICE_TOOLS_OPENAI_KEY=${VOICE_TOOLS_OPENAI_KEY:-}

# Messaging – Telegram
TELEGRAM_BOT_TOKEN=${TELEGRAM_BOT_TOKEN:-}
TELEGRAM_HOME_CHANNEL=${TELEGRAM_HOME_CHANNEL:-}
TELEGRAM_HOME_CHANNEL_NAME=${TELEGRAM_HOME_CHANNEL_NAME:-}

# Messaging – Discord
DISCORD_BOT_TOKEN=${DISCORD_BOT_TOKEN:-}
DISCORD_HOME_CHANNEL=${DISCORD_HOME_CHANNEL:-}
DISCORD_HOME_CHANNEL_NAME=${DISCORD_HOME_CHANNEL_NAME:-}

# Messaging – Slack
SLACK_BOT_TOKEN=${SLACK_BOT_TOKEN:-}
SLACK_APP_TOKEN=${SLACK_APP_TOKEN:-}
SLACK_ALLOWED_USERS=${SLACK_ALLOWED_USERS:-}
SLACK_HOME_CHANNEL=${SLACK_HOME_CHANNEL:-}
SLACK_HOME_CHANNEL_NAME=${SLACK_HOME_CHANNEL_NAME:-}

# Messaging – WhatsApp
WHATSAPP_ENABLED=${WHATSAPP_ENABLED:-false}
WHATSAPP_ALLOWED_USERS=${WHATSAPP_ALLOWED_USERS:-}

# Gateway
GATEWAY_ALLOW_ALL_USERS=${GATEWAY_ALLOW_ALL_USERS:-false}
SESSION_IDLE_MINUTES=${SESSION_IDLE_MINUTES:-}

# Context compression
CONTEXT_COMPRESSION_ENABLED=${CONTEXT_COMPRESSION_ENABLED:-true}
CONTEXT_COMPRESSION_THRESHOLD=${CONTEXT_COMPRESSION_THRESHOLD:-0.85}

# Human delay pacing
HERMES_HUMAN_DELAY_MODE=${HERMES_HUMAN_DELAY_MODE:-off}

# RL / experiment tracking
TINKER_API_KEY=${TINKER_API_KEY:-}
WANDB_API_KEY=${WANDB_API_KEY:-}
RL_API_URL=${RL_API_URL:-http://localhost:8080}
EOF

# ── Write config.yaml (minimal; hermes setup can extend it) ───
if [ ! -f "$HERMES_HOME/config.yaml" ]; then
    cat > "$HERMES_HOME/config.yaml" <<EOF
model: "${LLM_MODEL:-anthropic/claude-opus-4-6}"
EOF
fi

# ── Write SOUL.md placeholder ─────────────────────────────────
if [ ! -f "$HERMES_HOME/SOUL.md" ]; then
    cat > "$HERMES_HOME/SOUL.md" <<EOF
You are Hermes, a helpful AI assistant.
EOF
fi

echo "[entrypoint] Hermes home: $HERMES_HOME"
echo "[entrypoint] Starting: $*"
exec "$@"
