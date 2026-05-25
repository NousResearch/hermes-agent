#!/bin/bash
set -e

export HOME=/data
export HERMES_HOME=/data/.hermes

echo "[start] preparing..."

mkdir -p "$HERMES_HOME"

if [ -z "$GOOGLE_API_KEY" ]; then
  echo "[ERROR] GOOGLE_API_KEY missing"
  exit 1
fi

ENV_FILE="$HERMES_HOME/.env"
: > "$ENV_FILE"

echo "GOOGLE_API_KEY=$GOOGLE_API_KEY" >> "$ENV_FILE"

cat <<EOF > "$HERMES_HOME/config.yaml"
model:
  provider: gemini
  default: gemini-1.5-flash
EOF

echo "[start] launching server..."
exec python /app/server.py
