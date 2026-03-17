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
# LLM providers
OPENROUTER_API_KEY=${OPENROUTER_API_KEY:-}
NOUS_API_KEY=${NOUS_API_KEY:-}
LLM_MODEL=${LLM_MODEL:-anthropic/claude-opus-4-6}
ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY:-}
OPENAI_API_KEY=${OPENAI_API_KEY:-}
KIMI_API_KEY=${KIMI_API_KEY:-}
MINIMAX_API_KEY=${MINIMAX_API_KEY:-}
MINIMAX_GROUP_ID=${MINIMAX_GROUP_ID:-}
ZHIPU_API_KEY=${ZHIPU_API_KEY:-}
CUSTOM_API_KEY=${CUSTOM_API_KEY:-}
CUSTOM_ENDPOINT_URL=${CUSTOM_ENDPOINT_URL:-}

# Tools (all optional)
FIRECRAWL_API_KEY=${FIRECRAWL_API_KEY:-}
FAL_KEY=${FAL_KEY:-}
GITHUB_TOKEN=${GITHUB_TOKEN:-}
BROWSERBASE_API_KEY=${BROWSERBASE_API_KEY:-}
BROWSERBASE_PROJECT_ID=${BROWSERBASE_PROJECT_ID:-}
PARALLEL_API_KEY=${PARALLEL_API_KEY:-}
HONCHO_API_KEY=${HONCHO_API_KEY:-}
SKILLS_HUB_REPO=${SKILLS_HUB_REPO:-}

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

# Messaging – Signal (requires signal-cli bridge)
SIGNAL_HTTP_URL=${SIGNAL_HTTP_URL:-}
SIGNAL_PHONE_NUMBER=${SIGNAL_PHONE_NUMBER:-}
SIGNAL_ALLOWED_USERS=${SIGNAL_ALLOWED_USERS:-}

# Messaging – Email (IMAP + SMTP)
EMAIL_ADDRESS=${EMAIL_ADDRESS:-}
EMAIL_PASSWORD=${EMAIL_PASSWORD:-}
EMAIL_IMAP_HOST=${EMAIL_IMAP_HOST:-}
EMAIL_SMTP_HOST=${EMAIL_SMTP_HOST:-}
EMAIL_ALLOWED_SENDERS=${EMAIL_ALLOWED_SENDERS:-}

# Gateway
GATEWAY_ALLOW_ALL_USERS=${GATEWAY_ALLOW_ALL_USERS:-false}
SESSION_IDLE_MINUTES=${SESSION_IDLE_MINUTES:-}
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

# ── Auto-detect provider ──────────────────────────────────────
PROVIDER="${HERMES_INFERENCE_PROVIDER:-auto}"
if [ "$PROVIDER" = "auto" ]; then
  if [ -n "${NOUS_API_KEY:-}" ]; then
    PROVIDER="nous-api"
  elif [ -n "${OPENROUTER_API_KEY:-}" ]; then
    PROVIDER="openrouter"
  elif [ -n "${ANTHROPIC_API_KEY:-}" ]; then
    PROVIDER="anthropic"
  elif [ -n "${OPENAI_API_KEY:-}" ]; then
    PROVIDER="openai"
  fi
fi

# ── Write config.yaml on first boot only ─────────────────────
# Subsequent boots use the patch step below so TUI-set values
# (from `hermes model` or `railway ssh`) are preserved.
if [ ! -f "$HERMES_HOME/config.yaml" ]; then
  echo "[entrypoint] Writing fresh config.yaml …"
  cat > "$HERMES_HOME/config.yaml" <<EOF
model:
  default: "${LLM_MODEL:-anthropic/claude-opus-4-6}"
  provider: "${PROVIDER}"
  base_url: "${HERMES_BASE_URL:-https://openrouter.ai/api/v1}"
terminal:
  backend: "${TERMINAL_ENV:-local}"
agent:
  max_turns: ${HERMES_MAX_TURNS:-60}
  reasoning_effort: "${HERMES_REASONING_EFFORT:-medium}"
compression:
  enabled: true
  threshold: 0.85
memory:
  memory_enabled: true
  user_profile_enabled: true
EOF
fi

# ── Patch specific keys from env vars (runs every boot) ───────
# Applies Railway env var changes without overwriting the full
# config, so manual TUI edits on the volume are preserved.
python3 - <<'PYEOF'
import os, sys
try:
    import yaml
except ImportError:
    sys.exit(0)

cfg_path = os.path.join(os.environ.get("HERMES_HOME", "/root/.hermes"), "config.yaml")
if not os.path.exists(cfg_path):
    sys.exit(0)

with open(cfg_path) as f:
    cfg = yaml.safe_load(f) or {}

changed = False

max_turns = os.environ.get("HERMES_MAX_TURNS")
if max_turns:
    cfg.setdefault("agent", {})["max_turns"] = int(max_turns)
    changed = True

reasoning = os.environ.get("HERMES_REASONING_EFFORT")
if reasoning:
    cfg.setdefault("agent", {})["reasoning_effort"] = reasoning
    changed = True

model = os.environ.get("LLM_MODEL") or os.environ.get("HERMES_MODEL")
if model:
    cfg.setdefault("model", {})["default"] = model
    changed = True

base_url = os.environ.get("HERMES_BASE_URL") or os.environ.get("CUSTOM_ENDPOINT_URL")
if base_url:
    cfg.setdefault("model", {})["base_url"] = base_url
    changed = True

provider = os.environ.get("HERMES_INFERENCE_PROVIDER")
if provider and provider != "auto":
    cfg.setdefault("model", {})["provider"] = provider
    changed = True

if changed:
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
    print("[entrypoint] config.yaml patched from env vars")
PYEOF

echo "[entrypoint] Hermes home: $HERMES_HOME"
echo "[entrypoint] Provider: $PROVIDER"
echo "[entrypoint] Model: ${LLM_MODEL:-anthropic/claude-opus-4-6}"
echo "[entrypoint] Starting gateway..."

# ── Start gateway directly ────────────────────────────────────
exec python -u -m gateway.run
