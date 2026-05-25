#!/bin/bash
set -e

echo "[start.sh] Preparing Hermes directories..."

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

if [ ! -f /data/.hermes/.env ]; then
  echo "[start.sh] Creating empty .env"
  touch /data/.hermes/.env
fi

if [ ! -f /data/.hermes/config.yaml ] && [ -f /opt/hermes-agent/cli-config.yaml.example ]; then
  echo "[start.sh] Seeding config.yaml from example"
  cp /opt/hermes-agent/cli-config.yaml.example /data/.hermes/config.yaml
fi

export HOME=/data
export HERMES_HOME=/data/.hermes

echo "[start.sh] Starting admin server..."
exec python /app/server.py
