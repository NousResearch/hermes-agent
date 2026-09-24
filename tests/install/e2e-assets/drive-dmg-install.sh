#!/usr/bin/env bash
# Drive the Hermes-Setup dmg bootstrap through its first-run GUI.
#
# The Setup app is Tauri (Rust + system webview), so Playwright/Electron
# attach never works. Launch the binary bare in the background (it inherits
# the redirect env), click "Install Hermes ->" with native input, then watch
# the install land on disk: checkout + venv console script.
#
# Usage:
#   drive-dmg-install.sh --app-bin <path> --install-dir <path> \
#     [--install-timeout-secs 2700] [--proof-dir <dir>]
#
# Exit 0: install landed. Non-zero: click never registered or the install
# never landed; desktop screenshots in --proof-dir say which.
set -euo pipefail

APP_BIN=""
INSTALL_DIR=""
INSTALL_TIMEOUT_SECS=2700
PROOF_DIR="."
while [ "$#" -gt 0 ]; do
  case "$1" in
    --app-bin) APP_BIN="$2"; shift 2 ;;
    --install-dir) INSTALL_DIR="$2"; shift 2 ;;
    --install-timeout-secs) INSTALL_TIMEOUT_SECS="$2"; shift 2 ;;
    --proof-dir) PROOF_DIR="$2"; shift 2 ;;
    *) echo "error: unknown argument: $1" >&2; exit 1 ;;
  esac
done
[ -n "$APP_BIN" ] && [ -n "$INSTALL_DIR" ] || { echo 'error: --app-bin and --install-dir are required' >&2; exit 1; }
mkdir -p "$PROOF_DIR"

log() { echo "[drive-dmg-install] $*"; }
shot() {
  # screencapture works regardless of app internals; -x mutes the shutter.
  screencapture -x "$PROOF_DIR/$1.png" 2>/dev/null || true
  log "screenshot: $PROOF_DIR/$1.png"
}

"$APP_BIN" &
SETUP_PID=$!
log "launched $APP_BIN (pid $SETUP_PID)"
cleanup() { kill "$SETUP_PID" 2>/dev/null || true; }
trap cleanup EXIT

# Wait for the installer window, then click the install button. The webview's
# button is not a native control and System Events is not permitted assistive
# access on the runners (error -25208), so post real CGEvents with cliclick.
# The AppleScript half only READS window geometry (position/size need no
# assistive grant). Re-click every probe until the install starts: a click
# that lands before the webview finishes painting is swallowed, and
# re-clicking a started install hits a progress screen (harmless).
command -v cliclick >/dev/null 2>&1 || brew install --quiet cliclick

window_geometry() {
  osascript <<'OSA' 2>/dev/null
tell application "System Events"
  set procs to (every process whose name contains "Hermes")
  if (count of procs) = 0 then return "no-process"
  set p to item 1 of procs
  if (count of windows of p) = 0 then return "no-window"
  set w to window 1 of p
  set {x, y} to position of w
  set {wd, ht} to size of w
  return (x as text) & " " & (y as text) & " " & (wd as text) & " " & (ht as text)
end tell
OSA
}

click_install() {
  local geo
  geo="$(window_geometry)"
  case "$geo" in
    no-process|no-window|'') echo "$geo"; return 0 ;;
  esac
  # shellcheck disable=SC2086
  set -- $geo
  local x=$1 y=$2 wd=$3 ht=$4
  if [ -n "$LAST_ERR" ]; then
    # Error screen: Retry install sits left of center at ~59% height
    # (measured: button x 359-492, y 402-441 in the 880x620 window).
    local cx=$((x + wd * 48 / 100))
    local cy=$((y + ht * 59 / 100))
    cliclick "c:${cx},${cy}" 2>&1 || true
    echo "clicked retry ${cx},${cy} (window ${x},${y} ${wd}x${ht})"
    return 0
  fi
  # First-run screen: button center sits at ~65% of window height.
  local cx=$((x + wd / 2))
  local cy=$((y + ht * 65 / 100))
  cliclick "c:${cx},${cy}" 2>&1 || true
  echo "clicked ${cx},${cy} (window ${x},${y} ${wd}x${ht})"
}

HERMES_BIN="$INSTALL_DIR/venv/bin/hermes"
# The bootstrap runs 11 stages; checkout + venv land in the first few and
# the desktop app build is near the end, so success requires all three or
# the EXIT trap kills the installer mid-build.
installed_app() {
  local cand
  for cand in \
    "$INSTALL_DIR/apps/desktop/release/mac-arm64/Hermes.app" \
    "$INSTALL_DIR/apps/desktop/release/mac/Hermes.app" \
    "/Applications/Hermes.app"; do
    [ -d "$cand" ] && return 0
  done
  return 1
}
install_complete() {
  [ -d "$INSTALL_DIR/.git" ] && [ -x "$HERMES_BIN" ] && installed_app
}
# The bootstrap parks on an error screen instead of exiting when a stage
# fails (e.g. a transient 429 downloading install.sh), with a Retry button
# in the same button zone the install click hits. Its log names the cause;
# watch for new failure lines, let the regular click drive the retry, and
# give up after a few so a persistent failure reports the real error
# instead of burning the whole install timeout.
BOOTSTRAP_LOG="$HOME/.hermes/logs/bootstrap-installer.log"
bootstrap_error() {
  [ -f "$BOOTSTRAP_LOG" ] || return 0
  # Match REAL failure shapes only: the structured stage log's state=Failed,
  # or the error screen's own message. Healthy INFO lines carry the literal
  # field error=None, so a bare error/failed substring match false-positives
  # on every stage transition.
  tail -5 "$BOOTSTRAP_LOG" 2>/dev/null \
    | grep -E 'state=Failed|install script failed|didn.t finish|ERROR ' | tail -1 || true
}
RETRIES=0
MAX_RETRIES=3
LAST_ERR=""
PENDING_ERR_COUNT=0
DEADLINE=$((SECONDS + INSTALL_TIMEOUT_SECS))
FIRST_SHOT=0
CLICKS=0
while :; do
  if install_complete; then
    log "install landed: checkout + venv console script + Hermes.app present"
    shot "02-install-landed"
    break
  fi
  if ! kill -0 "$SETUP_PID" 2>/dev/null; then
    # The bootstrap relaunches the desktop and exits on success; only fail
    # if it died without the install landing (give the FS one last look).
    sleep 5
    install_complete && continue
    shot "ERROR-setup-exited"
    log "Hermes-Setup exited (pid $SETUP_PID) before the install landed"
    exit 1
  fi
  err="$(bootstrap_error)"
  if [ -n "$err" ]; then
    PENDING_ERR_COUNT=$((PENDING_ERR_COUNT + 1))
  else
    PENDING_ERR_COUNT=0
  fi
  # Two consecutive error probes before switching click targets: a single
  # log-parse glitch must never redirect the clicker.
  if [ "$PENDING_ERR_COUNT" -ge 2 ] && [ "$err" != "$LAST_ERR" ]; then
    LAST_ERR="$err"
    RETRIES=$((RETRIES + 1))
    log "bootstrap error (retry $RETRIES/$MAX_RETRIES): $err"
    shot "ERROR-bootstrap-attempt-$RETRIES"
    if [ "$RETRIES" -gt "$MAX_RETRIES" ]; then
      log "bootstrap failing persistently; giving up"
      exit 1
    fi
  fi
  if [ -z "$err" ] && [ -n "$LAST_ERR" ]; then
    log "bootstrap error cleared; resuming install-button targeting"
    LAST_ERR=""
  fi
  if [ "$FIRST_SHOT" -eq 0 ] && [ "$SECONDS" -gt 10 ]; then
    shot "00-setup-window"
    FIRST_SHOT=1
  fi
  # Keep clicking until the install shows up on disk; count for the log.
  result="$(click_install || true)"
  CLICKS=$((CLICKS + 1))
  [ $((CLICKS % 6)) -eq 1 ] && log "click attempt $CLICKS: $result"
  if [ "$CLICKS" -eq 3 ]; then shot "01-after-first-clicks"; fi
  if [ "$SECONDS" -ge "$DEADLINE" ]; then
    shot "ERROR-install-timeout"
    log "install did not land within ${INSTALL_TIMEOUT_SECS}s ($CLICKS clicks attempted)"
    exit 1
  fi
  sleep 10
done

# Success: the bootstrap owns its own exit (it relaunches the desktop).
# Leave it a grace window, then stop it if it lingers; the caller's asserts
# take over from here.
for _ in 1 2 3 4 5 6; do
  kill -0 "$SETUP_PID" 2>/dev/null || break
  sleep 5
done
log "done"
exit 0
