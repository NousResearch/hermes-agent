#!/usr/bin/env bash
# install_all.sh — automated installer for the full Hermes Gateway Keep-Alive & Phone Survival stack.
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOME_DIR="$HOME"
RESURRECT_DIR="$HOME_DIR/.hermes/resurrect"
BOOT_DIR="$HOME_DIR/.termux/boot"

echo "[1/6] Making scripts executable..."
chmod +x "$SCRIPT_DIR"/*.sh "$SCRIPT_DIR"/*.py 2>/dev/null || true

echo "[2/6] Setting up WorkManager resurrection script in $RESURRECT_DIR..."
mkdir -p "$RESURRECT_DIR"
cp "$SCRIPT_DIR/resurrect.sh" "$RESURRECT_DIR/"
cp "$SCRIPT_DIR/termux_priority.sh" "$RESURRECT_DIR/"
chmod +x "$RESURRECT_DIR"/*.sh

echo "[3/6] Registering Android WorkManager resurrection job..."
if command -v termux-job-scheduler >/dev/null 2>&1; then
  termux-job-scheduler --job-id 1 \
    --script "$RESURRECT_DIR/resurrect.sh" \
    --period-ms 900000 \
    --network any --battery-not-low false 2>/dev/null || true
  echo "Registered WorkManager job (id 1, period 15m)."
else
  echo "Note: termux-job-scheduler not found. Install termux-api for WorkManager survival."
fi

echo "[4/6] Setting up Termux boot script..."
mkdir -p "$BOOT_DIR"
cat > "$BOOT_DIR/hermes_start.sh" << 'BOOTEOF'
#!/data/data/com.termux/files/usr/bin/bash
termux-wake-lock 2>/dev/null || true
sleep 10
bash "$HOME/.hermes/resurrect/resurrect.sh"
BOOTEOF
chmod +x "$BOOT_DIR/hermes_start.sh"

echo "[5/6] Starting RAM watchdog & applying OOM priority shaping..."
bash "$SCRIPT_DIR/start_ram_watchdog.sh" 2>/dev/null || true
bash "$SCRIPT_DIR/termux_priority.sh" 2>/dev/null || true

echo "[6/6] Configuring daily storage hygiene cron..."
if command -v crontab >/dev/null 2>&1; then
  (crontab -l 2>/dev/null | grep -v "ram_management.sh"; echo "0 4 * * * bash $SCRIPT_DIR/ram_management.sh") | crontab -
  echo "Added daily 04:00 AM hygiene cron job."
fi

echo "Hermes Gateway Keep-Alive stack installation complete."
