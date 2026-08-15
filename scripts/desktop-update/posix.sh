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
#     [--relaunch-cwd <p>]     linux: working directory to restore on relaunch
#     [--sandbox-fallback]     linux: the caller vouches for a sandbox opt-out
#                              (ELECTRON_DISABLE_SANDBOX / --no-sandbox launch)
#     [--no-ui] [--no-marker-cleanup] [--self-test-ui] [--self-test-gate]
#     [-- <args...>]           linux: filtered launch args to replay
#
# The shim (ui.html in a chromeless browser app window) is decoration: it
# polls /progress for `done` or `error` and reacts. It owns nothing --
# relaunch, result file, marker hygiene all happen here, identically, when
# no renderer exists. No chromium-family browser found = no UI, fine.
#
# ORDERING (the durable-truth rule): swap and relaunch are DECIDED AND
# EXECUTED before the result file is written, the marker is removed, or a
# terminal event reaches the shim. Nothing user-visible may claim an outcome
# the filesystem hasn't already delivered.

set -u

ORIGINAL_ARGS=("$@")
INSTALL_ROOT="" BRANCH="main" DESKTOP_PID=0 RELAUNCH_TARGET=""
RELAUNCH_CWD="" SANDBOX_FALLBACK=0 RELAUNCH_ARGS=()
NO_UI=0 NO_MARKER_CLEANUP=0 SELF_TEST_UI=0 SELF_TEST_GATE=0 HANDOFF_DAEMONIZED=0
while [ $# -gt 0 ]; do
  case "$1" in
    --install-root) INSTALL_ROOT="$2"; shift 2 ;;
    --branch) BRANCH="$2"; shift 2 ;;
    --desktop-pid) DESKTOP_PID="$2"; shift 2 ;;
    --relaunch-target) RELAUNCH_TARGET="$2"; shift 2 ;;
    --relaunch-cwd) RELAUNCH_CWD="$2"; shift 2 ;;
    --sandbox-fallback) SANDBOX_FALLBACK=1; shift ;;
    --no-ui) NO_UI=1; shift ;;
    --no-marker-cleanup) NO_MARKER_CLEANUP=1; shift ;;
    --self-test-ui) SELF_TEST_UI=1; shift ;;
    --self-test-gate) SELF_TEST_GATE=1; shift ;;
    --daemonized) HANDOFF_DAEMONIZED=1; shift ;;
    --) shift; RELAUNCH_ARGS=("$@"); shift $# ;;
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
DONE_NOTE=""  # set when the update succeeded but the app will NOT reopen itself

log() { echo "$(date +%Y-%m-%dT%H:%M:%S%z) $1" | tee -a "$LOG" 2>/dev/null; }

# Keep a durable signal breadcrumb.  A detached hand-off used to leave only the
# generic FINAL_MSG when it was terminated while the updater child was running,
# which erased the one fact needed to diagnose the failure.
TERM_TEARDOWN_IGNORED=0
on_signal() {
  local sig="$1" pgid="unknown"
  pgid="$(ps -o pgid= -p $$ 2>/dev/null | tr -d '[:space:]')"
  # Electron sends one final TERM to the detached hand-off process group while
  # quitting, even after the orchestrator has been re-parented to PID 1.  That
  # TERM is teardown noise, not a user cancellation.  Ignore it once only when
  # the originating desktop PID is already gone; a later TERM still stops us.
  if [ "$sig" = "TERM" ] && [ "$HANDOFF_DAEMONIZED" -eq 1 ] \
      && [ "$TERM_TEARDOWN_IGNORED" -eq 0 ] && ! kill -0 "$DESKTOP_PID" 2>/dev/null; then
    TERM_TEARDOWN_IGNORED=1
    log "SIGNAL: TERM ignored after desktop teardown pid=$$ ppid=$PPID pgid=${pgid:-unknown} desktopPid=$DESKTOP_PID"
    return 0
  fi
  log "SIGNAL: $sig pid=$$ ppid=$PPID pgid=${pgid:-unknown}"
  FINAL_MSG="Update hand-off was interrupted by $sig (pid $$)."
  case "$sig" in
    HUP) FINAL_CODE=129 ;;
    INT) FINAL_CODE=130 ;;
    QUIT) FINAL_CODE=131 ;;
    TERM) FINAL_CODE=143 ;;
  esac
  exit "$FINAL_CODE"
}
trap 'on_signal HUP' HUP
trap 'on_signal INT' INT
trap 'on_signal QUIT' QUIT
trap 'on_signal TERM' TERM

# ── shim ────────────────────────────────────────────────────────────────────
json_escape() { # minimal JSON string escape: \ " and control whitespace
  local s=${1//\\/\\\\}
  s=${s//\"/\\\"}
  s=${s//$'\n'/\\n}
  s=${s//$'\r'/\\r}
  s=${s//$'\t'/\\t}
  printf '%s' "$s"
}

notify_fallback() { # status message — renderer-free recovery surface.
  # Fires only when there is no shim window. BEST-EFFORT immediate channel:
  # each rung requires EXECUTION acceptance, not existence — notify-send's
  # exit code is its acceptance (fire-and-forget), zenity/kdialog must
  # survive their first second (a dialog that dies instantly had no display
  # and must not eat the message). The GUARANTEED channel is the result
  # file: a manual/error outcome is durably marked and the next Desktop
  # boot surfaces it in a dialog (handoff-result.ts + main.ts).
  case "$1" in manual|error) ;; *) return 0 ;; esac
  if [ "$(uname)" = "Darwin" ]; then
    /usr/bin/osascript -e "display notification \"$(printf '%s' "$2" | sed 's/"/\\"/g')\" with title \"Hermes update\"" 2>/dev/null && return 0
  else
    if command -v notify-send >/dev/null 2>&1; then
      notify-send -u critical "Hermes update" "$2" 2>/dev/null && return 0
    fi
    local p
    if command -v zenity >/dev/null 2>&1; then
      zenity --warning --title="Hermes update" --text="$2" 2>/dev/null &
      p=$!; sleep 1
      kill -0 "$p" 2>/dev/null && return 0
      wait "$p" 2>/dev/null
    fi
    if command -v kdialog >/dev/null 2>&1; then
      kdialog --title "Hermes update" --sorry "$2" 2>/dev/null &
      p=$!; sleep 1
      kill -0 "$p" 2>/dev/null && return 0
      wait "$p" 2>/dev/null
    fi
  fi
  # No immediate surface landed. The durable channel takes over: the result
  # is marked manual/failed and the next boot shows it in a real dialog.
  log "NOTICE: no notification surface accepted; outcome reaches the user via the result dialog on next launch: $2"
}

publish() { # status message -- atomic replace; the server reads per poll
  printf '{"status":"%s","message":"%s"}' "$(json_escape "$1")" "$(json_escape "$2")" > "$STATUS.tmp" \
    && mv -f "$STATUS.tmp" "$STATUS" 2>/dev/null || true
  [ -n "$UI_SERVER_PID" ] && sleep 1  # one poll beat to render the state
  [ -z "$UI_SERVER_PID" ] && notify_fallback "$1" "$2"
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
  # The Desktop's final teardown targets the updater process group.  Put both
  # UI processes in their own sessions so neither the HTTP server nor a Chrome
  # renderer becomes collateral damage (Chrome surfaces that renderer death as
  # an "Aw, Snap!" page with error code 15 even while /progress still returns
  # HTTP 200).  The Python wrapper immediately execs the real process, so $!
  # remains the PID that stop_ui can terminate.
  # TERM/HUP stay IGNORED in the server (SIG_IGN survives execv): a stray
  # teardown TERM killed the shim ~1s into `hermes update` (2026-08-14 16:44,
  # window showed ERR_CONNECTION_REFUSED for the whole run; upstream #66753).
  # stop_ui ends the server with SIGKILL instead — it is stateless HTTP.
  "$py" -c 'import os, signal, sys; os.setsid(); signal.signal(signal.SIGTERM, signal.SIG_IGN); signal.signal(signal.SIGHUP, signal.SIG_IGN); os.execv(sys.argv[1], sys.argv[1:])' \
    "$py" "$SCRIPT_DIR/serve-ui.py" "$html" "$STATUS" > "$LOG_DIR/desktop-update-ui-port" 2>>"$LOG" &
  UI_SERVER_PID=$!
  for i in $(seq 1 10); do
    port="$(tr -cd '0-9' < "$LOG_DIR/desktop-update-ui-port" 2>/dev/null)"
    [ -n "$port" ] && break
    sleep 0.2
  done
  [ -n "$port" ] || { kill -9 "$UI_SERVER_PID" 2>/dev/null; UI_SERVER_PID=""; return; }

  # Throwaway profile: new window/process we own; user's browser untouched.
  "$py" -c 'import os, signal, sys; os.setsid(); signal.signal(signal.SIGTERM, signal.SIG_DFL); os.execv(sys.argv[1], sys.argv[1:])' \
    "$browser" --app="http://127.0.0.1:$port/" --user-data-dir="${TMPDIR:-/tmp}/hermes-update-ui-$$" \
    --no-first-run --no-default-browser-check --window-size=280,320 >/dev/null 2>&1 &
  UI_BROWSER_PID=$!
  log "shim: app window on 127.0.0.1:$port"
}

stop_ui() { # error state leaves the window up for the user to read
  if [ -n "$UI_SERVER_PID" ]; then
    # The server ignores TERM/HUP (see start_ui) — KILL is its off switch.
    { kill -9 "$UI_SERVER_PID" && wait "$UI_SERVER_PID"; } 2>/dev/null
  fi
  if [ "${1:-}" != "leave-window" ] && [ -n "$UI_BROWSER_PID" ]; then
    { kill "$UI_BROWSER_PID" && wait "$UI_BROWSER_PID"; } 2>/dev/null
  fi
  UI_SERVER_PID="" UI_BROWSER_PID=""
}

# ── relaunch ────────────────────────────────────────────────────────────────
# Linux relaunch gate -- an exact port of the deleted update-relaunch.ts
# decision (#45205/#37541), not a loosened rewrite:
#   * the running binary must live under THIS checkout's rebuilt
#     apps/desktop/release/linux-unpacked (anchored, path-segment-aware --
#     proof the update we just ran replaced the selected executable);
#   * chrome-sandbox ABSENT is fine (namespace-sandbox build; nothing to
#     block on), PRESENT means root-owned AND setuid or Electron refuses to
#     boot ("quit and never came back");
#   * a user sandbox opt-out (ELECTRON_DISABLE_SANDBOX=1/true in our
#     inherited env, --no-sandbox among the replayed launch args, or the
#     Desktop vouching via --sandbox-fallback) makes the relaunch safe
#     despite a failed preflight.
# Outcomes mirror decideRelaunchOutcome: relaunch | skew | manual.
GATE="" GATE_MSG=""
linux_gate() {
  local unpacked="$INSTALL_ROOT/apps/desktop/release/linux-unpacked" sb arg
  case "$RELAUNCH_TARGET" in
    "$unpacked"/*) ;;
    *) GATE=skew GATE_MSG="Backend updated, but the desktop app package (AppImage/deb/rpm) was not changed. Update or reinstall it to match."; return ;;
  esac

  sb="$unpacked/chrome-sandbox"
  if [ ! -e "$sb" ]; then GATE=relaunch; return; fi
  if [ -u "$sb" ] && [ "$(stat -c %u "$sb" 2>/dev/null)" = "0" ]; then GATE=relaunch; return; fi

  case "${ELECTRON_DISABLE_SANDBOX:-}" in 1|true|TRUE|True) GATE=relaunch; return ;; esac
  [ "$SANDBOX_FALLBACK" -eq 1 ] && { GATE=relaunch; return; }
  for arg in ${RELAUNCH_ARGS[@]+"${RELAUNCH_ARGS[@]}"}; do
    [ "$arg" = "--no-sandbox" ] && { GATE=relaunch; return; }
  done

  GATE=manual GATE_MSG="Update complete, but the rebuilt app can't relaunch itself (its sandbox helper needs root ownership). Reopen Hermes to finish."
}

mac_swap() {
  local rebuilt="" c
  for c in "$INSTALL_ROOT/apps/desktop/release/mac-arm64/Hermes.app" \
           "$INSTALL_ROOT/apps/desktop/release/mac/Hermes.app"; do
    [ -d "$c" ] && { rebuilt="$c"; break; }
  done

  # Transactional swap: stage a full copy, move the old bundle aside, move
  # the copy in. Every step checked; a failed final move ROLLS BACK so the
  # user always has a launchable app, and the result file tells the truth.
  if [ "$FINAL_CODE" -eq 0 ] && [ -n "$rebuilt" ] && [ -d "$RELAUNCH_TARGET" ] && [ "$rebuilt" != "$RELAUNCH_TARGET" ]; then
    rm -rf "$RELAUNCH_TARGET.new" "$RELAUNCH_TARGET.old" 2>/dev/null || true
    if ! /usr/bin/ditto "$rebuilt" "$RELAUNCH_TARGET.new"; then
      rm -rf "$RELAUNCH_TARGET.new" 2>/dev/null || true
      DONE_NOTE="Update complete, but the new app could not be staged; the previous version was kept. Run the update again."
      log "WARNING: bundle copy failed; keeping existing app"
    elif ! mv "$RELAUNCH_TARGET" "$RELAUNCH_TARGET.old"; then
      rm -rf "$RELAUNCH_TARGET.new" 2>/dev/null || true
      DONE_NOTE="Update complete, but the new app could not replace the old one; the previous version was kept. Run the update again."
      log "WARNING: could not move old bundle aside; keeping existing app"
    elif ! mv "$RELAUNCH_TARGET.new" "$RELAUNCH_TARGET"; then
      if mv "$RELAUNCH_TARGET.old" "$RELAUNCH_TARGET"; then
        rm -rf "$RELAUNCH_TARGET.new" 2>/dev/null || true
        DONE_NOTE="Update complete, but the new app could not be installed; the previous version was restored. Run the update again."
        log "WARNING: bundle install failed; rolled back to the previous app"
      else
        FINAL_CODE=7 FINAL_MSG="The update finished but installing the new app failed and the previous app could not be restored. Reinstall Hermes (the rebuilt app is at $rebuilt)."
        log "ERROR: bundle install failed AND rollback failed"
      fi
    else
      rm -rf "$RELAUNCH_TARGET.old" 2>/dev/null || true
      log "swapped app bundle"
    fi
  fi
}

deliver_outcome() { # the truth-determining half: swap bundles / gate the relaunch
  [ -n "$RELAUNCH_TARGET" ] || return 0
  if [ "$(uname)" = "Darwin" ]; then
    mac_swap
  else
    linux_gate
    if [ "$GATE" != "relaunch" ] && [ "$FINAL_CODE" -eq 0 ]; then
      DONE_NOTE="$GATE_MSG"
      log "no relaunch ($GATE): $GATE_MSG"
    fi
  fi
}

launch_app() { # attempted BEFORE the terminal event (launch acceptance is
  # part of the outcome — gille's review). Returns nonzero when a launch
  # was due but did not verifiably happen; caller downgrades to manual.
  [ -n "$RELAUNCH_TARGET" ] || return 0
  if [ "$(uname)" = "Darwin" ]; then
    # A supplied target that no longer exists is a REJECTED launch (the
    # swap failed badly or the bundle vanished) — not "no launch due".
    [ -d "$RELAUNCH_TARGET" ] || { log "WARNING: relaunch target missing: $RELAUNCH_TARGET"; return 1; }
    /usr/bin/xattr -dr com.apple.quarantine "$RELAUNCH_TARGET" 2>/dev/null || true
    # `open` talks to launchd and FAILS LOUDLY on a broken/unlaunchable
    # bundle — its exit code IS launch acceptance here.
    /usr/bin/open "$RELAUNCH_TARGET" || { log "WARNING: open rejected the app"; return 1; }
  elif [ "$GATE" = "relaunch" ]; then
    # setsid only proves the wrapper shell started, so verify acceptance:
    # spawn, then confirm the child is still alive shortly after — an
    # immediate exec failure (ENOENT, ELF mismatch, dead sandbox) dies
    # within the window and downgrades to manual instead of lying.
    (cd "${RELAUNCH_CWD:-/}" 2>/dev/null || cd /
     setsid "$RELAUNCH_TARGET" ${RELAUNCH_ARGS[@]+"${RELAUNCH_ARGS[@]}"} >/dev/null 2>&1 &
     echo $! > "$STATUS.launchpid") || { log "WARNING: relaunch spawn failed"; return 1; }
    local lp
    lp="$(cat "$STATUS.launchpid" 2>/dev/null)"; rm -f "$STATUS.launchpid" 2>/dev/null
    [ -n "$lp" ] || { log "WARNING: relaunch pid unknown"; return 1; }
    sleep 1.5
    kill -0 "$lp" 2>/dev/null || { log "WARNING: relaunched app exited immediately"; return 1; }
  fi
}

MANUAL=0  # 1 = update landed but the user must act (result protocol field)

write_result() {
  printf '{"ok":%s,"exit_code":%s,"manual":%s,"message":"%s","branch":"%s","finished_at":%s}' \
    "$([ "$FINAL_CODE" -eq 0 ] && echo true || echo false)" "$FINAL_CODE" \
    "$([ "$MANUAL" -eq 1 ] && echo true || echo false)" \
    "$(json_escape "$FINAL_MSG")" "$(json_escape "$BRANCH")" "$(date +%s)" \
    > "$RESULT.tmp" 2>/dev/null && mv -f "$RESULT.tmp" "$RESULT" 2>/dev/null || true
}

finish() {
  # Ordering (gille's reviews, both rounds):
  #   1. deliver the outcome (swap/gate) so the truth exists;
  #   2. durable result + marker removal (the relaunched app consumes the
  #      result on boot and must not park on our marker — this must be on
  #      disk BEFORE any launch attempt);
  #   3. attempt the launch and require ACCEPTANCE;
  #   4. only then the terminal shim event — done means "the app is coming
  #      back", manual means "it is not, here's what to do", error is error.
  # A rejected launch rewrites the result (nothing consumed it — the app
  # never started) so the next boot tells the truth too.
  deliver_outcome
  [ "$FINAL_CODE" -eq 0 ] && [ -n "$DONE_NOTE" ] && { FINAL_MSG="$DONE_NOTE"; MANUAL=1; }
  write_result

  if [ "$NO_MARKER_CLEANUP" -eq 0 ] && [ "$(head -1 "$MARKER" 2>/dev/null | tr -d '[:space:]')" = "$$" ]; then
    rm -f "$MARKER" 2>/dev/null || true
  fi

  if [ "$FINAL_CODE" -ne 0 ]; then
    publish "error" "$FINAL_MSG"; stop_ui leave-window
    launch_app || true   # error path still tries to bring the app back
    rm -f "$STATUS" "$STATUS.tmp" "$LOG_DIR/desktop-update-ui-port" 2>/dev/null || true
    return
  fi

  if [ -n "$DONE_NOTE" ]; then
    if [ "$(uname)" = "Darwin" ]; then
      # mac DONE_NOTE = swap failed but the PREVIOUS bundle was kept/rolled
      # back — bring it back up; the note still tells the user to re-run.
      # A gated linux outcome (skew/manual) skips the launch BY DESIGN.
      if ! launch_app; then
        # Even the kept bundle didn't come back: the durable message must
        # carry BOTH facts (update ok, previous app not reopened).
        FINAL_MSG="$DONE_NOTE Hermes also could not reopen itself - open it manually."
        write_result
      fi
    fi
    publish "manual" "$FINAL_MSG"; stop_ui leave-window
  elif launch_app; then
    publish "done" ""; stop_ui
  else
    # Launch was due and did not land. Downgrade: truthful result for the
    # next boot, manual state held on screen now.
    FINAL_MSG="Update complete. Reopen Hermes to finish (it could not restart itself)."
    MANUAL=1
    write_result
    publish "manual" "$FINAL_MSG"; stop_ui leave-window
  fi
  rm -f "$STATUS" "$STATUS.tmp" "$LOG_DIR/desktop-update-ui-port" 2>/dev/null || true
}
trap finish EXIT

# ── self-tests: no update, touch nothing ────────────────────────────────────
if [ "$SELF_TEST_GATE" -eq 1 ]; then
  # Prints the gate decision for the given --install-root/--relaunch-target
  # and exits; scripts/desktop-update/repro.sh gate asserts the matrix.
  trap - EXIT
  linux_gate
  echo "$GATE${GATE_MSG:+:$GATE_MSG}"
  exit 0
fi

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
# Electron's macOS quit teardown sends SIGTERM to its still-parented updater
# child on this machine. `detached + unref` gives the child a process group but
# does not re-parent it before `before-quit` runs, so the hand-off consistently
# died two seconds after starting `hermes update`. Re-exec through a one-shot
# setsid child and let this direct Electron child exit first. The real
# orchestrator is then owned by launchd (PPID 1) and is outside Electron's quit
# teardown, while retaining the same marker/result protocol.
if [ "$HANDOFF_DAEMONIZED" -ne 1 ]; then
  # This launcher is disposable. In particular it must not run finish() on
  # EXIT: that would publish a false failure and relaunch Hermes while the
  # re-parented orchestrator is only just starting.
  trap - EXIT HUP INT QUIT TERM
  /usr/bin/nohup /usr/bin/python3 -c '
import os, sys
env = os.environ.copy()
os.setsid()
os.execve("/bin/bash", ["/bin/bash", sys.argv[1], *sys.argv[2:], "--daemonized"], env)
' "$SCRIPT_DIR/posix.sh" "${ORIGINAL_ARGS[@]}" >/dev/null 2>&1 &
  exit 0
fi

# Electron terminates the entire detached updater process group during quit,
# including the loopback status server.  Arm TERM immunity before `start_ui`
# so the shim server and the later `hermes update` subprocess both inherit
# SIG_IGN.  The orchestrator restores its normal TERM handler after the update
# command has returned; the already-running server keeps the inherited setting
# until normal cleanup closes it.
trap '' TERM
log "hand-off start: root=$INSTALL_ROOT branch=$BRANCH desktopPid=$DESKTOP_PID pid=$$"
rm -f "$RESULT" 2>/dev/null || true

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

# Do not create Chrome until Electron has fully left.  During its 2.5s quit
# dwell/before-quit teardown macOS can terminate descendants of the hand-off;
# a Chrome renderer reports that SIGTERM as "Aw, Snap!" error code 15.  The
# update marker above prevents a second click during this short UI-less gap.
sleep 1
start_ui

HERMES_BIN="$INSTALL_ROOT/venv/bin/hermes"
[ -x "$HERMES_BIN" ] || { FINAL_CODE=3 FINAL_MSG="Update aborted: $HERMES_BIN is missing. The install needs repair (run the Hermes installer or hermes doctor)."; log "$FINAL_MSG"; exit 3; }

# Run FROM the install root: `hermes update` resolves the tree it mutates
# from the working directory, and we inherit the Desktop's cwd (which can be
# an unrelated repo — updating THAT instead of the install is the failure
# the sandbox repro caught). FAIL CLOSED: set -u without set -e means a
# failed cd would otherwise continue in the wrong tree — the exact class
# this correction exists to eliminate.
cd "$INSTALL_ROOT" || {
  FINAL_CODE=3 FINAL_MSG="Update aborted: cannot enter the install root ($INSTALL_ROOT). Nothing was changed."
  log "$FINAL_MSG"; exit 3
}
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
trap 'on_signal TERM' TERM

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
