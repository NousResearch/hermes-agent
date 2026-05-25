#!/bin/bash
set -e

echo "[start.sh] Initializing Hermes environment..."

# --- Set paths ---
export HOME=/data
export HERMES_HOME=/data/.hermes

# --- Create required directories ---
echo "[start.sh] Ensuring directory structure..."
mkdir -p "$HERMES_HOME"/{cron,sessions,logs,memories,skills,pairing,hooks,image_cache,audio_cache,workspace}

# --- Build .env file ---
echo "[start.sh] Writing environment variables to .env..."

ENV_FILE="$HERMES_HOME/.env"
> "$ENV_FILE"

# ✅ ONLY Google (Gemini)
[ -n "$GOOGLE_API_KEY" ] && echo "GOOGLE_API_KEY=$GOOGLE_API_KEY" >> "$ENV_FILE"

# Messaging
[ -n "$TELEGRAM_BOT_TOKEN" ] && echo "TELEGRAM_BOT_TOKEN=$TELEGRAM_BOT_TOKEN" >> "$ENV_FILE"
[ -n "$TELEGRAM_ALLOWED_USERS" ] && echo "TELEGRAM_ALLOWED_USERS=$TELEGRAM_ALLOWED_USERS" >> "$ENV_FILE"

# Optional tools
[ -n "$GH_TOKEN" ] && echo "GH_TOKEN=$GH_TOKEN" >> "$ENV_FILE"
[ -n "$GITHUB_TOKEN" ] && echo "GITHUB_TOKEN=$GITHUB_TOKEN" >> "$ENV_FILE"

# --- Seed config.yaml if missing ---
if [ ! -f "$HERMES_HOME/config.yaml" ] && [ -f /opt/hermes-agent/cli-config.yaml.example ]; then
  echo "[start.sh] Seeding config.yaml"
  cp /opt/hermes-agent/cli-config.yaml.example "$HERMES_HOME/config.yaml"
fi

# --- Force Gemini provider + model ---
echo "[start.sh] Configuring Gemini model..."
cat <<EOF > "$HERMES_HOME/config.yaml"
model:
  provider: gemini
  default: gemini-1.5-flash
EOF

# --- Start application ---
echo "[start.sh] Starting admin server..."
exec python /app/server.py
exec python /app/server.py
