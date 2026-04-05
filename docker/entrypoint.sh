#!/bin/bash
# Docker entrypoint: bootstrap config files into the mounted volume, then run hermes.
set -e

HERMES_HOME="/opt/data"
INSTALL_DIR="/opt/hermes"
INSTALL_CAMOUFOX_BROWSER="${INSTALL_CAMOUFOX_BROWSER:-0}"

# Create essential directory structure.  Cache and platform directories
# (cache/images, cache/audio, platforms/whatsapp, etc.) are created on
# demand by the application — don't pre-create them here so new installs
# get the consolidated layout from get_hermes_dir().
mkdir -p "$HERMES_HOME"/{cron,sessions,logs,hooks,memories,skills}

# .env
if [ ! -f "$HERMES_HOME/.env" ]; then
    cp "$INSTALL_DIR/.env.example" "$HERMES_HOME/.env"
fi

# config.yaml
if [ ! -f "$HERMES_HOME/config.yaml" ]; then
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

# Optional: install the Camoufox browser package at container start.
if [ "$INSTALL_CAMOUFOX_BROWSER" = "1" ] || [ "$INSTALL_CAMOUFOX_BROWSER" = "true" ] || [ "$INSTALL_CAMOUFOX_BROWSER" = "TRUE" ]; then
    if [ -d "$INSTALL_DIR/node_modules/@askjo/camoufox-browser" ]; then
        echo "Camoufox browser package already installed."
    else
        echo "Installing Camoufox browser package..."
        npm install --prefix "$INSTALL_DIR" --no-audit @askjo/camoufox-browser
    fi
fi

exec hermes gateway run
