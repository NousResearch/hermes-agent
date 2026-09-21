#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-run}"
APP_NAME="Hermes"
BUNDLE_ID="com.nousresearch.hermes"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DESKTOP_DIR="$ROOT_DIR/apps/desktop"
APP_BUNDLE="$DESKTOP_DIR/release/mac-arm64/Hermes.app"
APP_BINARY="$APP_BUNDLE/Contents/MacOS/Hermes"
STAMP="$APP_BUNDLE/Contents/Resources/install-stamp.json"

drain_running_app() {
  if ! pgrep -x "$APP_NAME" >/dev/null 2>&1; then
    return
  fi

  /usr/bin/osascript -e "tell application id \"$BUNDLE_ID\" to quit" >/dev/null 2>&1 || true
  for _ in $(seq 1 60); do
    if ! pgrep -x "$APP_NAME" >/dev/null 2>&1; then
      return
    fi
    sleep 0.5
  done

  echo "Hermes did not finish its graceful background-work drain; refusing to build beside it." >&2
  exit 1
}

build_app() {
  local python_bin

  drain_running_app
  (
    cd "$DESKTOP_DIR"
    CSC_IDENTITY_AUTO_DISCOVERY=false npm run pack
  )

  test -x "$APP_BINARY"
  test -f "$STAMP"

  if [ -x "$ROOT_DIR/.venv/bin/python" ]; then
    python_bin="$ROOT_DIR/.venv/bin/python"
  elif [ -x "$ROOT_DIR/venv/bin/python" ]; then
    python_bin="$ROOT_DIR/venv/bin/python"
  else
    echo "No project Hermes Python runtime is available for macOS signing fixup." >&2
    exit 1
  fi

  (
    cd "$ROOT_DIR"
    "$python_bin" - "$DESKTOP_DIR" <<'PYEOF'
import sys
from pathlib import Path

from hermes_cli.main import _desktop_macos_relaunchable_fixup

ok = _desktop_macos_relaunchable_fixup(
    Path(sys.argv[1]),
    publisher_signing_configured=False,
)
raise SystemExit(0 if ok else 1)
PYEOF
  )
  codesign --verify --deep --strict "$APP_BUNDLE"
}

open_app() {
  /usr/bin/open -n "$APP_BUNDLE"
}

case "$MODE" in
  run)
    build_app
    open_app
    ;;
  --debug|debug)
    build_app
    lldb -- "$APP_BINARY"
    ;;
  --logs|logs)
    build_app
    open_app
    /usr/bin/log stream --info --style compact --predicate "process == \"$APP_NAME\""
    ;;
  --telemetry|telemetry)
    build_app
    open_app
    /usr/bin/log stream --info --style compact --predicate "subsystem == \"$BUNDLE_ID\""
    ;;
  --verify|verify)
    build_app
    open_app
    sleep 2
    pgrep -x "$APP_NAME" >/dev/null
    ;;
  *)
    echo "usage: $0 [run|--debug|--logs|--telemetry|--verify]" >&2
    exit 2
    ;;
esac
