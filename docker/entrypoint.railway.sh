#!/bin/bash
# Railway entrypoint for Hermes Agent
# Restores config from git, then starts gateway

set -e

HERMES_HOME="${HERMES_HOME:-/opt/data}"
INSTALL_DIR="/opt/hermes"
HARNESS_REPO="git@github.com:swapkats/harness.git"
HARNESS_BRANCH="${HARNESS_BRANCH:-main}"

echo "[railway-entrypoint] Starting Hermes Agent on Railway"
echo "[railway-entrypoint] HERMES_HOME=$HERMES_HOME"

# --- Privilege dropping ---
if [ "$(id -u)" = "0" ]; then
    echo "[railway-entrypoint] Dropping to hermes user"
    exec gosu hermes "$0" "$@"
fi

# --- Create directory structure ---
mkdir -p "$HERMES_HOME"/{cron,sessions,logs,hooks,memories,skills,skins,plans,workspace,home}

# --- Sync from git (restore config/state from harness) ---
if [ -d "$HERMES_HOME/.git" ]; then
    echo "[railway-entrypoint] Pulling latest config from git..."
    (cd "$HERMES_HOME" && git pull origin main 2>&1 | tail -5) || echo "[railway-entrypoint] Git pull failed, continuing with existing state"
else
    echo "[railway-entrypoint] Cloning harness repo to restore config..."
    git clone -b "$HARNESS_BRANCH" "$HARNESS_REPO" "$HERMES_HOME" 2>&1 | tail -5
fi

# --- Bootstrap .env if missing ---
if [ ! -f "$HERMES_HOME/.env" ]; then
    echo "[railway-entrypoint] Creating .env from example..."
    cp "$INSTALL_DIR/.env.example" "$HERMES_HOME/.env"
fi

# --- Bootstrap config.yaml if missing ---
if [ ! -f "$HERMES_HOME/config.yaml" ]; then
    echo "[railway-entrypoint] Creating config.yaml from example..."
    cp "$INSTALL_DIR/cli-config.yaml.example" "$HERMES_HOME/config.yaml"
fi

# --- Copy SOUL.md ---
if [ ! -f "$HERMES_HOME/SOUL.md" ]; then
    cp "$INSTALL_DIR/docker/SOUL.md" "$HERMES_HOME/SOUL.md" 2>/dev/null || true
fi

# --- Sync bundled skills ---
if [ -d "$INSTALL_DIR/skills" ]; then
    python3 "$INSTALL_DIR/tools/skills_sync.py" 2>/dev/null || true
fi

echo "[railway-entrypoint] Config restored. Starting hermes gateway..."

# --- Final exec ---
source "${INSTALL_DIR}/.venv/bin/activate"
exec hermes gateway run