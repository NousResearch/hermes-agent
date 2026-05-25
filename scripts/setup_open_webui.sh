#!/usr/bin/env bash
set -euo pipefail

# Bootstrap Open WebUI against Hermes Agent's OpenAI-compatible API server.
#
# Idempotent by design:
# - ensures ~/.hermes/.env has API server settings
# - installs Open WebUI into ~/.local/open-webui-venv
# - writes a reusable launcher at ~/.local/bin/start-open-webui-hermes.sh
# - optionally installs a user service (launchd on macOS, systemd --user on Linux)
#
# Usage:
#   bash scripts/setup_open_webui.sh
#
# Optional environment overrides:
#   OPEN_WEBUI_PORT=8080
#   OPEN_WEBUI_HOST=127.0.0.1
#   OPEN_WEBUI_NAME='Johnny Hermes'
#   OPEN_WEBUI_ENABLE_SIGNUP=true
#   OPEN_WEBUI_ENABLE_SERVICE=auto   # auto|true|false
#   OPEN_WEBUI_VENV=~/.local/open-webui-venv
#   OPEN_WEBUI_DATA_DIR=~/.local/share/open-webui/data
#   HERMES_API_PORT=8642
#   HERMES_API_HOST=127.0.0.1
#   HERMES_API_MODEL_NAME='Hermes Agent'

OPEN_WEBUI_PORT="${OPEN_WEBUI_PORT:-8080}"
OPEN_WEBUI_HOST="${OPEN_WEBUI_HOST:-127.0.0.1}"
OPEN_WEBUI_NAME="${OPEN_WEBUI_NAME:-Hermes Agent WebUI}"
OPEN_WEBUI_ENABLE_SIGNUP="${OPEN_WEBUI_ENABLE_SIGNUP:-true}"
OPEN_WEBUI_ENABLE_SERVICE="${OPEN_WEBUI_ENABLE_SERVICE:-auto}"
OPEN_WEBUI_VENV="${OPEN_WEBUI_VENV:-$HOME/.local/open-webui-venv}"
OPEN_WEBUI_DATA_DIR="${OPEN_WEBUI_DATA_DIR:-$HOME/.local/share/open-webui/data}"
HERMES_ENV_FILE="${HERMES_ENV_FILE:-$HOME/.hermes/.env}"
HERMES_API_PORT="${HERMES_API_PORT:-8642}"
HERMES_API_HOST="${HERMES_API_HOST:-127.0.0.1}"
HERMES_API_CONNECT_HOST="${HERMES_API_CONNECT_HOST:-127.0.0.1}"
HERMES_API_MODEL_NAME="${HERMES_API_MODEL_NAME:-Hermes Agent}"
HERMES_API_BASE_URL="http://${HERMES_API_CONNECT_HOST}:${HERMES_API_PORT}/v1"
LAUNCHER_PATH="$HOME/.local/bin/start-open-webui-hermes.sh"
LOG_DIR="$HOME/.hermes/logs"

log() {
  printf '[open-webui-bootstrap] %s\n' "$*"
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

choose_python() {
  if command -v python3.11 >/dev/null 2>&1; then
    echo python3.11
  elif command -v python3 >/dev/null 2>&1; then
    echo python3
  else
    echo "Python 3 is required." >&2
    exit 1
  fi
}

upsert_env() {
  local key="$1"
  local value="$2"
  local file="$3"

  mkdir -p "$(dirname "$file")"
  touch "$file"

  python3 - "$file" "$key" "$value" <<'PY'
from pathlib import Path
import sys
path = Path(sys.argv[1])
key = sys.argv[2]
value = sys.argv[3]
lines = path.read_text().splitlines() if path.exists() else []
out = []
seen = False
for raw in lines:
    stripped = raw.strip()
    if stripped.startswith(f"{key}="):
        if not seen:
            out.append(f"{key}={value}")
            seen = True
        continue
    out.append(raw)
if not seen:
    if out and out[-1] != "":
        out.append("")
    out.append(f"{key}={value}")
path.write_text("\n".join(out).rstrip() + "\n")
PY
}

get_env_value() {
  local key="$1"
  local file="$2"
  python3 - "$file" "$key" <<'PY'
from pathlib import Path
import sys
path = Path(sys.argv[1])
key = sys.argv[2]
if not path.exists():
    raise SystemExit(0)
for raw in path.read_text().splitlines():
    line = raw.strip()
    if line.startswith(f"{key}="):
        print(line.split("=", 1)[1])
        raise SystemExit(0)
PY
}

generate_secret() {
  python3 - <<'PY'
import secrets
print(secrets.token_urlsafe(32))
PY
}

shell_quote() {
  python3 - "$1" <<'PY'
import shlex
import sys
print(shlex.quote(sys.argv[1]))
PY
}

can_use_systemd_user() {
  [[ "$(uname -s)" == "Linux" ]] || return 1
  command -v systemctl >/dev/null 2>&1 || return 1

  local uid runtime_dir bus_path
  uid="$(id -u)"
  runtime_dir="${XDG_RUNTIME_DIR:-/run/user/$uid}"
  bus_path="$runtime_dir/bus"

  if [[ -z "${XDG_RUNTIME_DIR:-}" && -d "$runtime_dir" ]]; then
    export XDG_RUNTIME_DIR="$runtime_dir"
  fi
  if [[ -z "${DBUS_SESSION_BUS_ADDRESS:-}" && -S "$bus_path" ]]; then
    export DBUS_SESSION_BUS_ADDRESS="unix:path=$bus_path"
  fi

  systemctl --user show-environment >/dev/null 2>&1
}

install_macos_dependencies() {
  if [[ "$(uname -s)" == "Darwin" ]] && command -v brew >/dev/null 2>&1; then
    if ! command -v pandoc >/dev/null 2>&1; then
      log 'Installing pandoc with Homebrew (recommended by Open WebUI docs)...'
      brew install pandoc
    fi
  fi
}

install_open_webui() {
  local py
  py="$(choose_python)"
  log "Using Python interpreter: $py"
  "$py" -m venv "$OPEN_WEBUI_VENV"
  # shellcheck disable=SC1090
  source "$OPEN_WEBUI_VENV/bin/activate"
  python -m pip install --upgrade pip setuptools wheel
  python -m pip install open-webui
}

write_launcher() {
  mkdir -p "$(dirname "$LAUNCHER_PATH")" "$OPEN_WEBUI_DATA_DIR" "$LOG_DIR"

  local quoted_data_dir quoted_name quoted_base_url quoted_host quoted_port quoted_venv
  quoted_data_dir="$(shell_quote "$OPEN_WEBUI_DATA_DIR")"
  quoted_name="$(shell_quote "$OPEN_WEBUI_NAME")"
  quoted_base_url="$(shell_quote "$HERMES_API_BASE_URL")"
  quoted_host="$(shell_quote "$OPEN_WEBUI_HOST")"
  quoted_port="$(shell_quote "$OPEN_WEBUI_PORT")"
  quoted_venv="$(shell_quote "$OPEN_WEBUI_VENV")"

  cat > "$LAUNCHER_PATH" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
API_KEY=\$(python3 - <<'PY'
from pathlib import Path
p = Path.home()/'.hermes'/'.env'
for raw in p.read_text().splitlines():
    line = raw.strip()
    if line.startswith('API_SERVER_KEY='):
        print(line.split('=', 1)[1])
        break
PY
)
export DATA_DIR=${quoted_data_dir}
export WEBUI_NAME=${quoted_name}
export ENABLE_SIGNUP=${OPEN_WEBUI_ENABLE_SIGNUP}
export ENABLE_PUBLIC_ACTIVE_USERS_COUNT=False
export ENABLE_VERSION_UPDATE_CHECK=False
export OPENAI_API_BASE_URL=${quoted_base_url}
export OPENAI_API_KEY="\$API_KEY"
export ENABLE_OPENAI_API=True
export ENABLE_OLLAMA_API=False
export OFFLINE_MODE=True
export BYPASS_EMBEDDING_AND_RETRIEVAL=True
export RAG_EMBEDDING_MODEL_AUTO_UPDATE=False
export RAG_RERANKING_MODEL_AUTO_UPDATE=False
export SCARF_NO_ANALYTICS=true
export DO_NOT_TRACK=true
export ANONYMIZED_TELEMETRY=false
export HOST=${quoted_host}
export PORT=${quoted_port}
source ${quoted_venv}/bin/activate
exec open-webui serve
EOF

  chmod +x "$LAUNCHER_PATH"
}

ensure_env_permissions() {
  chmod 600 "$HERMES_ENV_FILE" 2>/dev/null || true
}

install_launchd_service() {
  local plist="$HOME/Library/LaunchAgents/ai.openwebui.hermes.plist"
  mkdir -p "$(dirname "$plist")"
  cat > "$plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>ai.openwebui.hermes</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>${LAUNCHER_PATH}</string>
  </array>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>WorkingDirectory</key>
  <string>${HOME}</string>
  <key>StandardOutPath</key>
  <string>${LOG_DIR}/openwebui.log</string>
  <key>StandardErrorPath</key>
  <string>${LOG_DIR}/openwebui.error.log</string>
</dict>
</plist>
EOF
  launchctl bootout "gui/$(id -u)" "$plist" >/dev/null 2>&1 || true
  launchctl bootstrap "gui/$(id -u)" "$plist"
  launchctl enable "gui/$(id -u)/ai.openwebui.hermes"
  launchctl kickstart -k "gui/$(id -u)/ai.openwebui.hermes"
}

install_systemd_user_service() {
  require_cmd systemctl
  local unit_dir="$HOME/.config/systemd/user"
  local unit="$unit_dir/openwebui-hermes.service"
  mkdir -p "$unit_dir"
  cat > "$unit" <<EOF
[Unit]
Description=Open WebUI connected to Hermes Agent
After=default.target

[Service]
Type=simple
ExecStart=/bin/bash %h/.local/bin/start-open-webui-hermes.sh
Restart=always
RestartSec=3
WorkingDirectory=%h
StandardOutput=append:%h/.hermes/logs/openwebui.log
StandardError=append:%h/.hermes/logs/openwebui.error.log

[Install]
WantedBy=default.target
EOF
  systemctl --user daemon-reload
  systemctl --user enable --now openwebui-hermes.service
}

start_foreground_hint() {
  log "Launcher created at: ${LAUNCHER_PATH}"
  log "Start Open WebUI manually with: ${LAUNCHER_PATH}"
}

main() {
  require_cmd hermes
  require_cmd curl
  require_cmd python3

  install_macos_dependencies

  local api_key
  api_key="$(get_env_value API_SERVER_KEY "$HERMES_ENV_FILE")"
  if [[ -z "$api_key" ]]; then
    api_key="$(generate_secret)"
  fi

  log 'Ensuring Hermes API server is configured...'
  upsert_env API_SERVER_ENABLED true "$HERMES_ENV_FILE"
  upsert_env API_SERVER_HOST "$HERMES_API_HOST" "$HERMES_ENV_FILE"
  upsert_env API_SERVER_PORT "$HERMES_API_PORT" "$HERMES_ENV_FILE"
  upsert_env API_SERVER_MODEL_NAME "$HERMES_API_MODEL_NAME" "$HERMES_ENV_FILE"
  upsert_env API_SERVER_KEY "$api_key" "$HERMES_ENV_FILE"
  ensure_env_permissions

  log 'Restarting Hermes gateway so API server settings take effect...'
  hermes gateway restart >/dev/null 2>&1 || true
  sleep 4
  if ! curl -fsS "http://${HERMES_API_CONNECT_HOST}:${HERMES_API_PORT}/health" >/dev/null; then
    log 'Hermes API server did not answer on the first check. Trying to start gateway in the background...'
    nohup hermes gateway run >/dev/null 2>&1 &
    sleep 6
  fi
  curl -fsS "http://${HERMES_API_CONNECT_HOST}:${HERMES_API_PORT}/health" >/dev/null

  log 'Installing Open WebUI into a dedicated virtualenv...'
  install_open_webui
  write_launcher

  case "$OPEN_WEBUI_ENABLE_SERVICE" in
    true|auto)
      if [[ "$(uname -s)" == "Darwin" ]]; then
        install_launchd_service
      elif can_use_systemd_user; then
        install_systemd_user_service
      else
        log 'No usable user service manager detected; falling back to the launcher script.'
        start_foreground_hint
      fi
      ;;
    false)
      start_foreground_hint
      ;;
    *)
      echo "OPEN_WEBUI_ENABLE_SERVICE must be one of: auto, true, false" >&2
      exit 1
      ;;
  esac

  log "Done. Open WebUI should be available at: http://${OPEN_WEBUI_HOST}:${OPEN_WEBUI_PORT}"
  log "Hermes API endpoint: ${HERMES_API_BASE_URL}"
  log 'Important: Open WebUI persists connection settings after first launch. If you later save a wrong API key in the Admin UI, update/delete that connection there or reset its database.'
}

main "$@"
