#!/bin/bash
# Docker entrypoint: bootstrap config files into the mounted volume, then run hermes.
set -e

HERMES_HOME="/opt/data"
INSTALL_DIR="/opt/hermes"

mkdir -p "$HERMES_HOME"/{cron,sessions,logs,hooks,memories,skills}

# .env - always write from environment variables if they are set
if [ -n "$OPENAI_API_KEY" ] || [ -n "$TELEGRAM_BOT_TOKEN" ]; then
    : > "$HERMES_HOME/.env"
    [ -n "$OPENAI_API_KEY" ] && echo "OPENAI_API_KEY=$OPENAI_API_KEY" >> "$HERMES_HOME/.env"
    [ -n "$OPENAI_BASE_URL" ] && echo "OPENAI_BASE_URL=$OPENAI_BASE_URL" >> "$HERMES_HOME/.env"
    [ -n "$TELEGRAM_BOT_TOKEN" ] && echo "TELEGRAM_BOT_TOKEN=$TELEGRAM_BOT_TOKEN" >> "$HERMES_HOME/.env"
    [ -n "$TELEGRAM_ALLOWED_USERS" ] && echo "TELEGRAM_ALLOWED_USERS=$TELEGRAM_ALLOWED_USERS" >> "$HERMES_HOME/.env"
elif [ ! -f "$HERMES_HOME/.env" ]; then
    cp "$INSTALL_DIR/.env.example" "$HERMES_HOME/.env"
fi

# config.yaml - write model config from env if set
if [ -n "$OPENAI_BASE_URL" ]; then
    cat > "$HERMES_HOME/config.yaml" << EOF
model: "${HERMES_MODEL:-auto}"
base_url: "$OPENAI_BASE_URL"
api_key: "$OPENAI_API_KEY"
EOF
elif [ ! -f "$HERMES_HOME/config.yaml" ]; then
    cp "$INSTALL_DIR/cli-config.yaml.example" "$HERMES_HOME/config.yaml"
fi

# SOUL.md
if [ ! -f "$HERMES_HOME/SOUL.md" ]; then
    cp "$INSTALL_DIR/docker/SOUL.md" "$HERMES_HOME/SOUL.md"
fi

# Sync bundled skills (manifest-based so user edits are preserved)
if [ -d "$INSTALL_DIR/skills" ]; then
    python3 "$INSTALL_DIR/tools/skills_sync.py"
fi

exec hermes "$@"
