#!/usr/bin/env bash
set -euo pipefail

TUNNEL_NAME="${1:-hermes-line}"
HOSTNAME="${2:-}"
SERVICE_URL="${SERVICE_URL:-http://127.0.0.1:8646}"
CLOUDFLARED_DIR="${CLOUDFLARED_DIR:-$HOME/.cloudflared}"
CONFIG_PATH="${CONFIG_PATH:-$CLOUDFLARED_DIR/config.yml}"
PLIST_PATH="${PLIST_PATH:-$HOME/Library/LaunchAgents/com.soichiyo.cloudflared.hermes-line.plist}"

usage() {
  cat <<EOF
Usage:
  $(basename "$0") <tunnel-name> <hostname>

Example:
  $(basename "$0") hermes-line line-hermes.example.com

What this script does:
  1. Verifies Cloudflare login (cert.pem)
  2. Creates a named tunnel if needed
  3. Routes DNS for the hostname
  4. Writes ~/.cloudflared/config.yml
  5. Writes a user launchd plist for auto-start

Human prerequisites:
  - Run: cloudflared tunnel login
  - Own the DNS zone for <hostname> in Cloudflare
EOF
}

if [[ -z "$HOSTNAME" ]]; then
  usage
  exit 1
fi

CLOUDFLARED_BIN="${CLOUDFLARED_BIN:-}"
if [[ -z "$CLOUDFLARED_BIN" ]]; then
  if command -v cloudflared >/dev/null 2>&1; then
    CLOUDFLARED_BIN="$(command -v cloudflared)"
  elif [[ -x /opt/homebrew/bin/cloudflared ]]; then
    CLOUDFLARED_BIN="/opt/homebrew/bin/cloudflared"
  elif [[ -x /usr/local/bin/cloudflared ]]; then
    CLOUDFLARED_BIN="/usr/local/bin/cloudflared"
  else
    echo "cloudflared not found on PATH" >&2
    exit 1
  fi
fi

mkdir -p "$CLOUDFLARED_DIR"

if [[ ! -f "$CLOUDFLARED_DIR/cert.pem" ]]; then
  cat >&2 <<EOF
Missing $CLOUDFLARED_DIR/cert.pem

Run this first in a browser-enabled shell:
  "$CLOUDFLARED_BIN" tunnel login
EOF
  exit 1
fi

tmp_list="$(mktemp)"
"$CLOUDFLARED_BIN" tunnel list >"$tmp_list"

if grep -Eq "[[:space:]]$TUNNEL_NAME\$" "$tmp_list"; then
  TUNNEL_ID="$(awk -v name="$TUNNEL_NAME" '$NF == name { print $1; exit }' "$tmp_list")"
  echo "Using existing tunnel: $TUNNEL_NAME ($TUNNEL_ID)"
else
  echo "Creating tunnel: $TUNNEL_NAME"
  create_out="$("$CLOUDFLARED_BIN" tunnel create "$TUNNEL_NAME")"
  echo "$create_out"
  TUNNEL_ID="$(printf '%s\n' "$create_out" | sed -nE 's/.*Created tunnel[[:space:]]+([0-9a-f-]+).*/\1/p' | head -n1)"
fi

rm -f "$tmp_list"

if [[ -z "${TUNNEL_ID:-}" ]]; then
  echo "Could not determine tunnel ID for $TUNNEL_NAME" >&2
  exit 1
fi

CREDENTIALS_FILE="$CLOUDFLARED_DIR/$TUNNEL_ID.json"
if [[ ! -f "$CREDENTIALS_FILE" ]]; then
  echo "Missing credentials file: $CREDENTIALS_FILE" >&2
  exit 1
fi

echo "Routing DNS: $HOSTNAME -> $TUNNEL_NAME"
"$CLOUDFLARED_BIN" tunnel route dns "$TUNNEL_NAME" "$HOSTNAME"

cat >"$CONFIG_PATH" <<EOF
tunnel: $TUNNEL_ID
credentials-file: $CREDENTIALS_FILE

ingress:
  - hostname: $HOSTNAME
    service: $SERVICE_URL
  - service: http_status:404
EOF

mkdir -p "$(dirname "$PLIST_PATH")"
cat >"$PLIST_PATH" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.soichiyo.cloudflared.hermes-line</string>
  <key>ProgramArguments</key>
  <array>
    <string>$CLOUDFLARED_BIN</string>
    <string>tunnel</string>
    <string>--config</string>
    <string>$CONFIG_PATH</string>
    <string>run</string>
    <string>$TUNNEL_NAME</string>
  </array>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>StandardOutPath</key>
  <string>$HOME/.hermes/logs/cloudflared-line.log</string>
  <key>StandardErrorPath</key>
  <string>$HOME/.hermes/logs/cloudflared-line.log</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key>
    <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
  </dict>
</dict>
</plist>
EOF

cat <<EOF

Done.

Tunnel name:      $TUNNEL_NAME
Tunnel ID:        $TUNNEL_ID
Hostname:         $HOSTNAME
Service URL:      $SERVICE_URL
Config:           $CONFIG_PATH
LaunchAgent:      $PLIST_PATH
Webhook URL:      https://$HOSTNAME/line/webhook

Next:
  launchctl unload "$PLIST_PATH" 2>/dev/null || true
  launchctl load "$PLIST_PATH"

And for Hermes gateway persistence:
  cd /Users/soichiyo/.hermes/hermes-agent
  source venv/bin/activate
  hermes gateway install
  hermes gateway start
EOF
