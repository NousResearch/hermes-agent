#!/bin/bash
# posix.sh -- repo-owned macOS/Linux Desktop update hand-off.
#
# The whole job: wait for the Desktop to exit, run `hermes update`, tell the
# shim how it went, reopen the app. The Desktop spawns this detached and
# quits; because it lives in the checkout, every update refreshes the code
# that drives the next one. Replaces the in-app updater
# (applyUpdatesPosixInApp) -- with the app gone before the update starts,
# the HERMES_DESKTOP_CHILD_PID reaper-exclusion dance dies with it.
#
# CONTRACT (keep in sync with apps/desktop/electron/main.ts):
#   bash scripts/desktop-update/posix.sh
#     --install-root <path>    repo checkout (HERMES_HOME/hermes-agent)
#     --branch <ref>           branch to update against
#     --desktop-pid <pid>      the Electron main process to wait out
#     [--relaunch-target <p>]  mac: running .app to swap+reopen;
#                              linux: running binary (omit = no relaunch)
#     [--no-ui] [--no-marker-cleanup] [--self-test-ui]
#
# The shim (ui.html in a chromeless browser app window) is decoration: it
# polls /progress for `done` or `error` and reacts. It owns nothing --
# relaunch, result file, marker hygiene all happen here, identically, when
# no renderer exists. No chromium-family browser found = no UI, fine.

set -u

INSTALL_ROOT="" BRANCH="main" DESKTOP_PID=0 RELAUNCH_TARGET=""
NO_UI=0 NO_MARKER_CLEANUP=0 SELF_TEST_UI=0
while [ $# -gt 0 ]; do
  case "$1" in
    --install-root) INSTALL_ROOT="$2"; shift 2 ;;
    --branch) BRANCH="$2"; shift 2 ;;
    --desktop-pid) DESKTOP_PID="$2"; shift 2 ;;
    --relaunch-target) RELAUNCH_TARGET="$2"; shift 2 ;;
    --no-ui) NO_UI=1; shift ;;
    --no-marker-cleanup) NO_MARKER_CLEANUP=1; shift ;;
    --self-test-ui) SELF_TEST_UI=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 64 ;;
  esac
done
[ "$SELF_TEST_UI" -eq 1 ] || [ -n "$INSTALL_ROOT" ] || { echo "--install-root is required" >&2; exit 64; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HERMES_HOME="${INSTALL_ROOT:+$(dirname "$INSTALL_ROOT")}"
HERMES_HOME="${HERMES_HOME:-${TMPDIR:-/tmp}}"
MARKER="$HERMES_HOME/.hermes-update-in-progress"
LOG_DIR="$HERMES_HOME/logs"; mkdir -p "$LOG_DIR" 2>/dev/null || true
LOG="$LOG_DIR/desktop-update-handoff.log"
RESULT="$HERMES_HOME/.hermes-update-result.json"
STATUS="${TMPDIR:-/tmp}/hermes-update-status.$$"

UI_SERVER_PID="" UI_BROWSER_PID="" FINAL_CODE=1
FINAL_MSG="update did not complete"

log() { echo "$(date +%Y-%m-%dT%H:%M:%S%z) $1" | tee -a "$LOG" 2>/dev/null; }

# ── shim ────────────────────────────────────────────────────────────────────
publish() { # status message -- atomic replace; the server reads per poll
  printf '{"status":"%s","message":"%s"}' "$1" "$2" > "$STATUS.tmp" && mv -f "$STATUS.tmp" "$STATUS" 2>/dev/null || true
  [ -n "$UI_SERVER_PID" ] && sleep 1  # one poll beat to render the state
}

find_browser() {
  local c
  if [ "$(uname)" = "Darwin" ]; then
    for c in "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
             "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge" \
             "/Applications/Chromium.app/Contents/MacOS/Chromium" \
             "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser"; do
      [ -x "$c" ] && { echo "$c"; return; }
    done
  else
    for c in google-chrome google-chrome-stable chromium chromium-browser microsoft-edge brave-browser; do
      command -v "$c" 2>/dev/null && return
    done
  fi
}

start_ui() {
  [ "$NO_UI" -eq 1 ] && return
  local html="$SCRIPT_DIR/ui.html" py browser port="" i
  py="${INSTALL_ROOT:+$INSTALL_ROOT/venv/bin/python3}"
  [ -x "${py:-/nonexistent}" ] || py="$(command -v python3 2>/dev/null)"
  browser="$(find_browser)"
  { [ -f "$html" ] && [ -n "$py" ] && [ -n "$browser" ]; } || { log "shim: no renderer; skipping UI"; return; }

  publish "running" ""
  "$py" "$SCRIPT_DIR/serve-ui.py" "$html" "$STATUS" > "$LOG_DIR/desktop-update-ui-port" 2>>"$LOG" &
  UI_SERVER_PID=$!
  for i in $(seq 1 10); do
    port="$(tr -cd '0-9' < "$LOG_DIR/desktop-update-ui-port" 2>/dev/null)"
    [ -n "$port" ] && break
    sleep 0.2
  done
  [ -n "$port" ] || { kill "$UI_SERVER_PID" 2>/dev/null; UI_SERVER_PID=""; return; }

  # Throwaway profile: new window/process we own; user's browser untouched.
  "$browser" --app="http://127.0.0.1:$port/" --user-data-dir="${TMPDIR:-/tmp}/hermes-update-ui-$$" \
    --no-first-run --no-default-browser-check --window-size=280,320 >/dev/null 2>&1 &
  UI_BROWSER_PID=$!
  log "shim: app window on 127.0.0.1:$port"
}

stop_ui() { # error state leaves the window up for the user to read
  if [ -n "$UI_SERVER_PID" ]; then
    { kill "$UI_SERVER_PID" && wait "$UI_SERVER_PID"; } 2>/dev/null
  fi
  if [ "${1:-}" != "leave-window" ] && [ -n "$UI_BROWSER_PID" ]; then
    { kill "$UI_BROWSER_PID" && wait "$UI_BROWSER_PID"; } 2>/dev/null
  fi
  UI_SERVER_PID="" UI_BROWSER_PID=""
}

# ── relaunch ────────────────────────────────────────────────────────────────
relaunch() {
  [ -n "$RELAUNCH_TARGET" ] || return 0
  if [ "$(uname)" = "Darwin" ]; then
    # Swap the rebuilt bundle over the running one when both resolve, then
    # `open` (fully detached). POSIX doesn't lock running executables.
    local rebuilt="" c
    for c in "$INSTALL_ROOT/apps/desktop/release/mac-arm64/Hermes.app" \
             "$INSTALL_ROOT/apps/desktop/release/mac/Hermes.app"; do
      [ -d "$c" ] && { rebuilt="$c"; break; }
    done
    if [ -n "$rebuilt" ] && [ -d "$RELAUNCH_TARGET" ] && [ "$rebuilt" != "$RELAUNCH_TARGET" ]; then
      if /usr/bin/ditto "$rebuilt" "$RELAUNCH_TARGET.new"; then
        mv "$RELAUNCH_TARGET" "$RELAUNCH_TARGET.old" 2>/dev/null || rm -rf "$RELAUNCH_TARGET"
        mv "$RELAUNCH_TARGET.new" "$RELAUNCH_TARGET"
        rm -rf "$RELAUNCH_TARGET.old" 2>/dev/null || true
        log "swapped app bundle"
      else
        rm -rf "$RELAUNCH_TARGET.new" 2>/dev/null || true
        log "WARNING: bundle copy failed; relaunching existing app"
      fi
    fi
    /usr/bin/xattr -dr com.apple.quarantine "$RELAUNCH_TARGET" 2>/dev/null || true
    /usr/bin/open "$RELAUNCH_TARGET" || log "WARNING: relaunch failed"
  else
    # Linux: only relaunch a binary the rebuild actually replaced, with a
    # launchable sandbox helper -- otherwise say so instead of lying (#37541).
    case "$RELAUNCH_TARGET" in
      */release/*-unpacked/*)
        if [ -u "$(dirname "$RELAUNCH_TARGET")/chrome-sandbox" ] || [ -n "${HERMES_DESKTOP_NO_SANDBOX:-}" ]; then
          (setsid "$RELAUNCH_TARGET" >/dev/null 2>&1 &) || log "WARNING: relaunch failed"
        else
          FINAL_MSG="Update complete. Reopen Hermes to finish (the app could not restart itself)."
        fi ;;
      *)
        FINAL_MSG="Backend updated, but the desktop app package (AppImage/deb/rpm) was not changed. Update it to match." ;;
    esac
  fi
}

finish() {
  printf '{"ok":%s,"exit_code":%s,"message":"%s","branch":"%s","finished_at":%s}' \
    "$([ "$FINAL_CODE" -eq 0 ] && echo true || echo false)" "$FINAL_CODE" "$FINAL_MSG" "$BRANCH" "$(date +%s)" \
    > "$RESULT" 2>/dev/null || true
  if [ "$NO_MARKER_CLEANUP" -eq 0 ] && [ "$(head -1 "$MARKER" 2>/dev/null | tr -d '[:space:]')" = "$$" ]; then
    rm -f "$MARKER" 2>/dev/null || true
  fi
  if [ "$FINAL_CODE" -eq 0 ]; then publish "done" ""; stop_ui
  else publish "error" "$FINAL_MSG"; stop_ui leave-window; fi
  relaunch
  rm -f "$STATUS" "$STATUS.tmp" "$LOG_DIR/desktop-update-ui-port" 2>/dev/null || true
}
trap finish EXIT

# ── self-test: shim only, no update, touches nothing ───────────────────────
if [ "$SELF_TEST_UI" -eq 1 ]; then
  start_ui
  log "SELF-TEST: shim simulation (no update will run)"
  sleep "${HERMES_SELFTEST_HOLD_SECONDS:-6}"
  RELAUNCH_TARGET=""
  if [ -n "${HERMES_SELFTEST_FAIL:-}" ]; then FINAL_MSG="self-test error state"
  else FINAL_CODE=0 FINAL_MSG="self-test complete"; fi
  exit "$FINAL_CODE"
fi

# ── the actual job ──────────────────────────────────────────────────────────
log "hand-off start: root=$INSTALL_ROOT branch=$BRANCH desktopPid=$DESKTOP_PID pid=$$"
rm -f "$RESULT" 2>/dev/null || true
start_ui

# Marker claim: same cross-process lock contract as windows.ps1 /
# update_lock.py (the `hermes update` child adopts it via process ancestry).
printf '%s\n%s\n' "$$" "$(date +%s)" > "$MARKER" 2>/dev/null || log "WARNING: could not write update marker"

# Wait out the Desktop (FAIL CLOSED: updating under live backends bricks).
if [ "$DESKTOP_PID" -gt 0 ] 2>/dev/null; then
  for _ in $(seq 1 100); do kill -0 "$DESKTOP_PID" 2>/dev/null || break; sleep 0.3; done
  if kill -0 "$DESKTOP_PID" 2>/dev/null; then
    FINAL_CODE=4 FINAL_MSG="Update aborted: the Hermes window (pid $DESKTOP_PID) did not exit within 30s. Nothing was changed. Close Hermes fully and try again."
    log "$FINAL_MSG"; exit "$FINAL_CODE"
  fi
fi

HERMES_BIN="$INSTALL_ROOT/venv/bin/hermes"
[ -x "$HERMES_BIN" ] || { FINAL_CODE=3 FINAL_MSG="Update aborted: $HERMES_BIN is missing. The install needs repair (run the Hermes installer or hermes doctor)."; log "$FINAL_MSG"; exit 3; }

export PYTHONUNBUFFERED=1
log "running: hermes update --yes --gateway --branch $BRANCH"
OUT="$("$HERMES_BIN" update --yes --gateway --branch "$BRANCH" 2>&1)"; CODE=$?
printf '%s\n' "$OUT" >> "$LOG" 2>/dev/null
log "hermes update exit code: $CODE"

if [ "$CODE" -ne 0 ] && [ "$CODE" -ne 2 ]; then
  # Retry once: update-boundary class (fresh code on disk, stale in memory).
  # Exit 2 ("close all Hermes windows") is not retryable.
  log "retrying once (freshly pulled fix loads on the second run)"
  OUT="$("$HERMES_BIN" update --yes --gateway --branch "$BRANCH" 2>&1)"; CODE=$?
  printf '%s\n' "$OUT" >> "$LOG" 2>/dev/null
  log "retry exit code: $CODE"
fi

# Truthful completion: `hermes update` calls a GUI build failure non-fatal
# (exit 0). For a Desktop-driven update that would relaunch the OLD build
# and call it success -- retry the build once, propagate honestly.
if [ "$CODE" -eq 0 ] && printf '%s' "$OUT" | grep -q "Desktop build failed"; then
  log "desktop build failed inside hermes update; retrying build"
  "$HERMES_BIN" desktop --force-build --build-only >> "$LOG" 2>&1 || {
    FINAL_CODE=6 FINAL_MSG="Code and dependencies updated, but the Desktop app rebuild failed - you are running the previous build. Run hermes desktop --force-build from a terminal to retry."
    exit 6
  }
fi

if [ "$CODE" -eq 0 ]; then FINAL_CODE=0 FINAL_MSG="Update complete."
else FINAL_CODE="$CODE" FINAL_MSG="Update failed (exit $CODE). Run hermes debug share in a terminal to send a report."; fi
exit "$FINAL_CODE"
