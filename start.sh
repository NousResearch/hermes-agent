#!/bin/bash
set -e

echo "[start.sh] Initializing Hermes environment..."

# --- Set paths ---
export HOME=/data
export HERMES_HOME=/data/.hermes

# --- Validate required variables ---
if [ -z "$GOOGLE_API_KEY" ]; then
  echo "[ERROR] GOOGLE_API_KEY is missing"
  exit 1
fi

# --- Create directories ---
echo "[start.sh] Ensuring directory structure..."
mkdir -p "$HERMES_HOME"/{cron,sessions,logs,memories,skills,pairing,hooks,image_cache,audio_cache,workspace}

# --- Build .env ---
echo "[start.sh] Writing environment variables..."

ENV_FILE="$HERMES_HOME/.env"
> "$ENV_FILE"

echo "GOOGLE_API_KEY=$GOOGLE_API_KEY" >> "$ENV_FILE"
[ -n "$TELEGRAM_BOT_TOKEN" ] && echo "TELEGRAM_BOT_TOKEN=$TELEGRAM_BOT_TOKEN" >> "$ENV_FILE"
[ -n "$TELEGRAM_ALLOWED_USERS" ] && echo "TELEGRAM_ALLOWED_USERS=$TELEGRAM_ALLOWED_USERS" >> "$ENV_FILE"
[ -n "$GH_TOKEN" ] && echo "GH_TOKEN=$GH_TOKEN" >> "$ENV_FILE"
[ -n "$GITHUB_TOKEN" ] && echo "GITHUB_TOKEN=$GITHUB_TOKEN" >> "$ENV_FILE"

# --- Create config.yaml ---
echo "[start.sh] Writing config.yaml..."

cat <<EOF > "$HERMES_HOME/config.yaml"
model:
  provider: gemini
  default: gemini-1.5-flash
EOF

# --- Debug section ---
echo "[debug] ===== ENV ====="
cat "$HERMES_HOME/.env"

echo "[debug] ===== CONFIG ====="
cat "$HERMES_HOME/config.yaml"

echo "[debug] ===== HERMES CHECK ====="
hermes config check || echo "[debug] config check failed"

echo "[debug] ===== LLM TEST (background) ====="
(hermes chat -z "Say hello briefly" || echo "[debug] LLM test failed") &

# --- Start server ---
echo "[start.sh] Starting admin server..."
exec python /app/server.py
