#!/bin/bash
set -e

# Make sure Hermes home folders exist
mkdir -p /data/.hermes/cron \
         /data/.hermes/sessions \
         /data/.hermes/logs \
         /data/.hermes/memories \
         /data/.hermes/skills \
         /data/.hermes/pairing \
         /data/.hermes/hooks \
         /data/.hermes/image_cache \
         /data/.hermes/audio_cache \
         /data/.hermes/workspace

# Create empty env file if missing
if [ ! -f /data/.hermes/.env ]; then
  touch /data/.hermes/.env
fi

# Seed config.yaml only if it does not exist yet
# server.py will rewrite this later using the selected provider/model
if [ ! -f /data/.hermes/config.yaml ] && [ -f /opt/hermes-agent/cli-config.yaml.example ]; then
  cp /opt/hermes-agent/cli-config.yaml.example /data/.hermes/config.yaml
fi

# Ensure expected environment values exist
export HOME=/data
export HERMES_HOME=/data/.hermes

# Start the admin wrapper
exec python /app/server.py
