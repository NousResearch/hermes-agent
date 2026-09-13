#!/usr/bin/env bash
# setup_service.sh — Configure permissions and install runit service for Termux Chromium.
# Credits: @pjy010218
set -u

PREFIX="${PREFIX:-/data/data/com.termux/files/usr}"
SERVICE_DIR="$PREFIX/var/service/chromium-headless"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_RUN_SRC="$(cd "$SCRIPT_DIR/../service" && pwd)/run"

echo "[1/4] Checking prerequisites (chromium, runit, python)..."
if ! command -v chromium-browser >/dev/null 2>&1 && ! command -v chromium >/dev/null 2>&1 && [ ! -f "$PREFIX/lib/chromium/chrome" ]; then
    echo "Chromium not found. Please install: pkg install -y x11-repo tur-repo && pkg install -y chromium runit python"
fi

echo "[2/4] Setting proper file permissions on Chromium binaries..."
if [ -d "$PREFIX/lib/chromium" ]; then
    chmod -R 755 "$PREFIX/lib/chromium" 2>/dev/null || true
    find "$PREFIX/lib/chromium" -type f -exec chmod 644 {} + 2>/dev/null || true
    chmod 755 "$PREFIX/lib/chromium/chrome" "$PREFIX/lib/chromium/chromedriver" \
              "$PREFIX/lib/chromium/chromium-launcher.sh" 2>/dev/null || true
fi

echo "[3/4] Provisioning runit service..."
mkdir -p "$SERVICE_DIR"
if [ -f "$SERVICE_RUN_SRC" ]; then
    cp "$SERVICE_RUN_SRC" "$SERVICE_DIR/run"
    chmod +x "$SERVICE_DIR/run"
    echo "Installed service script to $SERVICE_DIR/run"
fi

echo "[4/4] Starting runsvdir if not running..."
if command -v runsvdir >/dev/null 2>&1; then
    pgrep -x runsvdir >/dev/null 2>&1 || nohup runsvdir "$PREFIX/var/service" >/dev/null 2>&1 &
    echo "runsvdir active."
fi

echo "Chromium service configuration complete."
