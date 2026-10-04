#!/usr/bin/env bash
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
#     [--branch <ref> | --channel stable|canary|main]  default: branch main
#     --desktop-pid <pid>      the Electron main process to wait out
#     [--handoff-run <id>]     hand-off protocol 2: adopt ONLY the Desktop's
#                              bridge marker carrying `run:<id>`
#     [--marker-op reclaim|withdraw]  the Desktop's marker helper: one verdict
#                              line on stdout, no update (see marker.sh)
#     [--relaunch-target <p>]  mac: running .app to swap+reopen;
#                              linux: running binary (omit = no relaunch)
#     [--relaunch-cwd <p>]     linux: working directory to restore on relaunch
#     [--sandbox-fallback]     linux: the caller vouches for a sandbox opt-out
#                              (ELECTRON_DISABLE_SANDBOX / --no-sandbox launch)
#     [--no-ui] [--no-marker-cleanup] [--self-test-ui] [--self-test-gate]
#     [--self-test-marker]
#     [-- <args...>]           linux: filtered launch args to replay
#
# The shim (ui.html in a chromeless browser app window) is decoration: it
# polls /progress for the current stage or a terminal event and reacts. The
# stages come from the gates below, never from child output. It owns nothing --
# relaunch, result file, marker hygiene all happen here, identically, when
# no renderer exists. No chromium-family browser found = no UI, fine; macOS
# never opens one (see find_browser).
#
# The Desktop reads the next line to pick the hand-off protocol it speaks.
# hermes-handoff-protocol: 2
#
# ORDERING (the durable-truth rule): swap and relaunch are DECIDED AND
# EXECUTED before the result file is written, the marker is removed, or a
# terminal event reaches the shim. Nothing user-visible may claim an outcome
# the filesystem hasn't already delivered.

set -u

ORIGINAL_ARGS=("$@")
INSTALL_ROOT="" BRANCH="main" CHANNEL="" DESKTOP_PID=0 RELAUNCH_TARGET=""
BRANCH_EXPLICIT=0
RELAUNCH_CWD="" SANDBOX_FALLBACK=0 RELAUNCH_ARGS=()
NO_GATEWAY=0
NO_UI=0 NO_MARKER_CLEANUP=0 SELF_TEST_UI=0 SELF_TEST_GATE=0 SELF_TEST_MARKER=0
SELF_TEST_TCC_HEAL=0
HANDOFF_DAEMONIZED=0 HANDOFF_RUN="" MARKER_OP=""
while [ $# -gt 0 ]; do
  case "$1" in
    --install-root) INSTALL_ROOT="$2"; shift 2 ;;
    --branch) BRANCH="$2"; BRANCH_EXPLICIT=1; shift 2 ;;
    --channel)
      case "${2:-}" in
        stable|canary|main) CHANNEL="$2" ;;
        *) echo "--channel must be stable, canary, or main" >&2; exit 64 ;;
      esac
      shift 2 ;;
    --desktop-pid) DESKTOP_PID="$2"; shift 2 ;;
    --handoff-run) HANDOFF_RUN="$2"; shift 2 ;;
    --marker-op) MARKER_OP="$2"; NO_UI=1; shift 2 ;;
    --relaunch-target) RELAUNCH_TARGET="$2"; shift 2 ;;
    --relaunch-cwd) RELAUNCH_CWD="$2"; shift 2 ;;
    --sandbox-fallback) SANDBOX_FALLBACK=1; shift ;;
    --no-gateway) NO_GATEWAY=1; shift ;;
    --no-ui) NO_UI=1; shift ;;
    --no-marker-cleanup) NO_MARKER_CLEANUP=1; shift ;;
    --self-test-ui) SELF_TEST_UI=1; shift ;;
    --self-test-gate) SELF_TEST_GATE=1; shift ;;
    --self-test-tcc-heal) SELF_TEST_TCC_HEAL=1; shift ;;
    --daemonized) HANDOFF_DAEMONIZED=1; shift ;;
    --self-test-marker) SELF_TEST_MARKER=1; NO_UI=1; NO_MARKER_CLEANUP=1; shift ;;
    --) shift; RELAUNCH_ARGS=("$@"); shift $# ;;
    *) echo "unknown arg: $1" >&2; exit 64 ;;
  esac
done
[ "$SELF_TEST_UI" -eq 1 ] || [ -n "$INSTALL_ROOT" ] || { echo "--install-root is required" >&2; exit 64; }
case "$DESKTOP_PID" in ''|*[!0-9]*) echo "--desktop-pid must be a pid" >&2; exit 64 ;; esac
[ -z "$HANDOFF_RUN" ] || [[ "$HANDOFF_RUN" =~ ^[A-Za-z0-9._-]{1,128}$ ]] || { echo "--handoff-run must match [A-Za-z0-9._-]{1,128}" >&2; exit 64; }
[ "$BRANCH_EXPLICIT" -eq 0 ] || [ -z "$CHANNEL" ] || { echo "--branch and --channel are mutually exclusive" >&2; exit 64; }
TARGET_ARGS=(--branch "$BRANCH")
[ -z "$CHANNEL" ] || TARGET_ARGS=(--channel "$CHANNEL")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HERMES_HOME="${HERMES_HOME:-${INSTALL_ROOT:+$(dirname "$INSTALL_ROOT")}}"
HERMES_HOME="${HERMES_HOME:-${TMPDIR:-/tmp}}"
export HERMES_HOME
MARKER="$HERMES_HOME/.hermes-update-in-progress"
LOG_DIR="$HERMES_HOME/logs"; mkdir -p "$LOG_DIR" 2>/dev/null || true
LOG="$LOG_DIR/desktop-update-handoff.log"
RESULT="$HERMES_HOME/.hermes-update-result.json"
STATUS="${TMPDIR:-/tmp}/hermes-update-status.$$"
STARTED_AT="$(date +%s)"  # the shim's elapsed clock; see serve-ui.py

UI_SERVER_PID="" UI_BROWSER_PID="" UI_PANEL_PID="" UI_PROFILE_DIR="" FINAL_CODE=1
FINAL_MSG="update did not complete"
DONE_NOTE=""  # set when the update succeeded but the app will NOT reopen itself
# Follow-up work that failed AFTER `hermes update` committed the new code. The
# result stays ok:true (contract C3: ok:false only while still on the previous
# version); each entry reaches the Desktop as a warning.
WARNINGS=()
MARKER_BODY=""  # the claim we published (logging; release judges identity, not bytes)
MARKER_CLAIMED=0
APP_REBUILD_FAILED=0

log() {
  if [ -n "$MARKER_OP" ]; then echo "$(date +%Y-%m-%dT%H:%M:%S%z) $1" >> "$LOG" 2>/dev/null
  else echo "$(date +%Y-%m-%dT%H:%M:%S%z) $1" | tee -a "$LOG" 2>/dev/null; fi
}

# The marker (contract C1/C2 + the A7 lock): parsing, identity, claim, delegate,
# release and the Desktop's helper ops live in marker.sh.
MY_PID=$$ MY_CT=""
# shellcheck source=marker.sh
. "$SCRIPT_DIR/marker.sh"

marker_release() { # A7 rule 5, under the lock. Never while a survivor of the
  # update still holds the checkout lock (R6): the marker would read free while
  # that completion still mutates the checkout. Wait it out; past the bound,
  # leave the marker in place -- dead to every reader, and the Desktop's
  # reclaim helper refuses to delete it while the checkout lock is held.
  local waited=0
  [ "$MARKER_CLAIMED" -eq 1 ] && [ "$NO_MARKER_CLEANUP" -eq 0 ] || return 0
  while checkout_lock_held; do
    [ "$waited" -gt 0 ] || log "a process still holds the checkout update lock; keeping the update marker until it exits"
    if [ "$waited" -ge "$RELEASE_WAIT_S" ]; then
      log "WARNING: checkout update lock still held after ${waited}s; leaving the update marker"
      return 0
    fi
    sleep 1; waited=$((waited + 1))
  done
  marker_locked marker_release_locked
  MARKER_CLAIMED=0
}
RELEASE_WAIT_S=7200

run_bounded() { # seconds cmd... -> cmd's stdout; 124 when it had to be killed
  local secs="$1" out pid i rc
  shift
  out="$(mktemp "${TMPDIR:-/tmp}/hermes-update-probe.$$.XXXXXX")" || return 1
  # Its own process group (pgid = pid), so a timeout kills the probe's whole
  # tree -- a hung `hermes` launcher's python child included, not just the shell.
  if command -v perl >/dev/null 2>&1; then
    perl -e 'setpgrp(0, 0); exec { $ARGV[0] } @ARGV or exit 127' "$@" > "$out" 2>/dev/null < /dev/null &
  else
    "$@" > "$out" 2>/dev/null < /dev/null &
  fi
  pid=$!
  for ((i = 0; i < secs * 10; i++)); do kill -0 "$pid" 2>/dev/null || break; sleep 0.1; done
  if kill -0 "$pid" 2>/dev/null; then
    kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null
    wait "$pid" 2>/dev/null
    rm -f "$out" 2>/dev/null
    log "WARNING: probe timed out after ${secs}s: $*"
    return 124
  fi
  wait "$pid"; rc=$?
  cat "$out" 2>/dev/null; rm -f "$out" 2>/dev/null
  return $rc
}

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

write_status() { # status message -- atomic replace; the server reads per poll
  printf '{"status":"%s","message":"%s"}' "$(json_escape "$1")" "$(json_escape "$2")" > "$STATUS.tmp" \
    && mv -f "$STATUS.tmp" "$STATUS" 2>/dev/null || true
}

publish_stage() { # a long wait the orchestrator is already gating on. No poll
  # beat (that would add a second per stage to every update) and no
  # notification fallback (there is nothing here for the user to act on).
  write_status "running" "$1"
}

publish() { # terminal event -- the page must render it before teardown
  write_status "$1" "$2"
  [ -n "$UI_SERVER_PID" ] && sleep 1  # one poll beat to render the state
  [ -z "$UI_SERVER_PID" ] && notify_fallback "$1" "$2"
}

find_browser() {
  local c
  # No Microsoft Edge and no Brave, on purpose. Edge's OS-level
  # Microsoft-account integration signs a fresh throwaway profile into the
  # user's MSA and renders its own "syncing your data" notification — MSA
  # email included — inside this window that is titled "Hermes" (#88410).
  # Brave paints its own P3A privacy-notice bar over the progress page in
  # the same window — cramped to unreadability at the shim's small size
  # (#88682). The throwaway --user-data-dir below cannot block either; the
  # remaining Chromium-family browsers carry no first-run chrome of their
  # own into a fresh profile.
  #
  # No browser at all on macOS. A second --user-data-dir is a second instance
  # of the same bundle, and the Dock records every one as a new recent-app
  # tile it never merges with the pinned browser: one more duplicate Chrome
  # icon per update (#96374). A stable profile would not help (still a second
  # instance). notify_fallback + the next-boot result dialog carry the outcome.
  [ "$(uname)" = "Darwin" ] && return
  for c in google-chrome google-chrome-stable chromium chromium-browser; do
    command -v "$c" 2>/dev/null && return
  done
}

# The shim is decoration; launching a browser the user does NOT use is not.
# A Safari/Firefox/Helium user who merely has Chrome installed watched Chrome
# open on every update — a "why is Chrome opening?" surprise (community
# report, Aug 2026). Only render the shim when the system DEFAULT browser is
# itself Chromium-family; otherwise skip the window and let notify_fallback +
# the durable result file carry the outcome. Best-effort on purpose: any
# detection failure keeps today's behavior (0 = allowed).
default_browser_is_chromium() {
  local handler=""
  # Linux only (find_browser never picks one on macOS). xdg-settings is the
  # authority; missing tool = permissive.
  command -v xdg-settings >/dev/null 2>&1 || return 0
  handler="$(xdg-settings get default-web-browser 2>/dev/null)" || return 0
  [ -n "$handler" ] || return 0
  case "$handler" in
    *chrome*|*chromium*|*Chrome*|*Chromium*) return 0 ;;
    *) return 1 ;;
  esac
}

start_status_panel() { # the macOS no-browser shim: osascript (JXA) panel.
  # Best-effort by design: osascript missing or the panel exiting instantly
  # (syntax, headless session) just means no UI, exactly as before. The panel
  # polls $STATUS itself and self-exits after a terminal state, so this pid
  # only needs killing on OUR early teardown paths.
  local py="$1" panel="$SCRIPT_DIR/update-panel.js"
  [ -n "$py" ] || py="/usr/bin/python3"  # unused by osascript; keeps the wrapper shape
  "$py" -c 'import os, signal, sys; os.setsid(); signal.signal(signal.SIGTERM, signal.SIG_IGN); os.execv(sys.argv[1], sys.argv[1:])' \
    /usr/bin/osascript -l JavaScript "$panel" "$STATUS" >>"$LOG" 2>&1 &
  UI_PANEL_PID=$!
  sleep 1
  kill -0 "$UI_PANEL_PID" 2>/dev/null || { UI_PANEL_PID=""; return; }
  log "shim: status panel pid=$UI_PANEL_PID"
}

start_ui() {
  [ "$NO_UI" -eq 1 ] && return
  local html="$SCRIPT_DIR/ui.html" py browser port="" i
  py="${INSTALL_ROOT:+$INSTALL_ROOT/venv/bin/python3}"
  [ -x "${py:-/nonexistent}" ] || py="$(command -v python3 2>/dev/null)"
  browser="$(find_browser)"
  if [ -n "$browser" ] && ! default_browser_is_chromium; then
    log "shim: default browser is not Chromium-family; skipping UI window"
    browser=""
  fi
  if [ "$(uname)" = "Darwin" ] && [ -z "$browser" ] && [ -f "$SCRIPT_DIR/update-panel.js" ]; then
    # No Chromium renderer may host ui.html (Safari/Firefox default, or no
    # Chrome): draw the same progress as a native panel instead. Reads the
    # same $STATUS JSON — no server, no browser, no other-app scripting.
    start_status_panel "$py"
    [ -n "$UI_PANEL_PID" ] && return
    log "shim: status panel did not start; continuing without UI"
  fi
  { [ -f "$html" ] && [ -n "$py" ] && [ -n "$browser" ]; } || { log "shim: no renderer; skipping UI"; return; }

  UI_PROFILE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/hermes-update-ui-$$-XXXXXXXX")" || {
    UI_PROFILE_DIR=""
    log "shim: could not allocate a browser profile; skipping UI"
    return
  }
  publish_stage ""
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
    "$py" "$SCRIPT_DIR/serve-ui.py" "$html" "$STATUS" "$STARTED_AT" "$$" > "$STATUS.port" 2>>"$LOG" &
  UI_SERVER_PID=$!
  for i in $(seq 1 10); do
    port="$(tr -cd '0-9' < "$STATUS.port" 2>/dev/null)"
    [ -n "$port" ] && break
    sleep 0.2
  done
  [ -n "$port" ] || { kill -9 "$UI_SERVER_PID" 2>/dev/null; UI_SERVER_PID=""; return; }

  # Throwaway profile: new window/process we own; user's browser untouched.
  "$py" -c 'import os, signal, sys; os.setsid(); signal.signal(signal.SIGTERM, signal.SIG_DFL); os.execv(sys.argv[1], sys.argv[1:])' \
    "$browser" --app="http://127.0.0.1:$port/" --user-data-dir="$UI_PROFILE_DIR" \
    --no-first-run --no-default-browser-check --window-size=280,320 >/dev/null 2>&1 &
  UI_BROWSER_PID=$!
  log "shim: app window on 127.0.0.1:$port"
}

stop_ui() { # error/manual outcomes keep the window up briefly so a watching
  # user can read the message, then close it. The outcome is also durably
  # written to the result file and surfaced in a dialog on the next Desktop
  # boot (handoff-result.ts), so the shim window never lingers indefinitely —
  # before this, each aborted update left another orphan browser window on
  # screen until the user closed it by hand.
  if [ "${1:-}" = "leave-window" ]; then
    sleep "${HERMES_UPDATE_SHIM_GRACE_SECONDS:-15}"
  fi
  if [ -n "$UI_SERVER_PID" ]; then
    # The server ignores TERM/HUP (see start_ui) — KILL is its off switch.
    { kill -9 "$UI_SERVER_PID" && wait "$UI_SERVER_PID"; } 2>/dev/null
  fi
  if [ -n "$UI_BROWSER_PID" ]; then
    { kill "$UI_BROWSER_PID" && wait "$UI_BROWSER_PID"; } 2>/dev/null
  fi
  if [ -n "$UI_PANEL_PID" ]; then
    # The panel ignores TERM (SIG_IGN survives execv, same contract as the
    # HTTP server) — KILL is its off switch. leave-window grace is the panel's
    # own delay terminal-state sleep, so we just end it.
    { kill -9 "$UI_PANEL_PID" && wait "$UI_PANEL_PID"; } 2>/dev/null
  fi
  if [ -n "$UI_PROFILE_DIR" ]; then
    rm -rf "$UI_PROFILE_DIR" 2>/dev/null || true
    UI_PROFILE_DIR=""
  fi
  UI_SERVER_PID="" UI_BROWSER_PID="" UI_PANEL_PID=""
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
  local unpacked="" sb arg cand
  # Canonicalise both sides before the prefix compare. On some distros
  # (e.g. Fedora/ostree) /home is a symlink to /var/home; the relaunch
  # target is read from /proc/<pid>/exe, which the kernel canonicalises
  # through symlinks, while INSTALL_ROOT keeps the original spelling —
  # the raw prefix match then false-gates as "skew" and tells the user
  # to reinstall an app that is fine. readlink -m canonicalises existing
  # leading components without requiring the full path to exist (unlike
  # -f); a no-op when both sides are already spelled the same.
  # electron-builder names the unpacked dir `linux-unpacked` on x86_64 but
  # `linux-<arch>-unpacked` on every other arch (linux-arm64-unpacked is
  # what ARM ships). Hardcoding the x86_64 name made a healthy ARM install
  # false-gate as "skew" on every update (#94703). Resolve the dir the
  # running binary actually lives in; fall back to the first
  # linux*-unpacked dir present when the target doesn't match any (the
  # skew case below then rejects it).
  [ -n "$RELAUNCH_TARGET" ] && RELAUNCH_TARGET="$(readlink -m -- "$RELAUNCH_TARGET")"
  for cand in "$INSTALL_ROOT"/apps/desktop/release/linux*-unpacked; do
    [ -d "$cand" ] || continue
    cand="$(readlink -m -- "$cand")"
    case "$RELAUNCH_TARGET" in
      "$cand"/*) unpacked="$cand"; break ;;
    esac
    [ -n "$unpacked" ] || unpacked="$cand"
  done
  [ -n "$unpacked" ] || unpacked="$(readlink -m -- "$INSTALL_ROOT/apps/desktop/release/linux-unpacked")"
  case "$RELAUNCH_TARGET" in
    "$unpacked"/*) ;;
    *) GATE=skew GATE_MSG="Backend updated, but the desktop app package (AppImage/deb/rpm) was not changed. Update or reinstall it to match."; return ;;
  esac

  sb="$unpacked/chrome-sandbox"
  if [ ! -e "$sb" ]; then GATE=relaunch; return; fi
  if [ -u "$sb" ] && [ "$(stat -c %u "$sb" 2>/dev/null)" = "0" ]; then GATE=relaunch; return; fi
  # Namespace sandbox usable => Electron never consults the setuid helper,
  # so a non-root chrome-sandbox does not block relaunch (mirrors the
  # _desktop_linux_userns_sandbox_available() probe in hermes_cli/main.py).
  if unshare --user --map-root-user true 2>/dev/null; then GATE=relaunch; return; fi

  case "${ELECTRON_DISABLE_SANDBOX:-}" in 1|true|TRUE|True) GATE=relaunch; return ;; esac
  [ "$SANDBOX_FALLBACK" -eq 1 ] && { GATE=relaunch; return; }
  for arg in ${RELAUNCH_ARGS[@]+"${RELAUNCH_ARGS[@]}"}; do
    [ "$arg" = "--no-sandbox" ] && { GATE=relaunch; return; }
  done

  GATE=manual GATE_MSG="Update complete, but the rebuilt app can't relaunch itself (its sandbox helper needs root ownership). Reopen Hermes to finish."
}

mac_bundle_recover() { # bundle path -- finish or roll back an interrupted swap
  # Start-of-run recovery for BOTH bundle swappers (this script's .old/.new and
  # `hermes desktop`'s .hermes-update-old/-new). The update marker serialises
  # swappers, so anything left here belongs to a dead run. A missing bundle
  # with its aside copy present means the swap died between its two renames:
  # put the previous app back (the staged copy may be partial). A present
  # bundle makes every aside/staged copy a leftover.
  local target="$1" aside staged
  [ -n "$target" ] || return 0
  for aside in "$target.old" "$target.hermes-update-old"; do
    if [ ! -e "$target" ] && [ -d "$aside" ]; then
      if mv "$aside" "$target" 2>/dev/null; then
        log "recovered an interrupted app swap: restored $target from $aside"
      else
        log "WARNING: could not restore $target from $aside"
      fi
    fi
  done
  [ -d "$target" ] || return 0
  for staged in "$target.old" "$target.new" "$target.hermes-update-old" "$target.hermes-update-new"; do
    if [ -e "$staged" ]; then
      rm -rf "$staged" 2>/dev/null && log "removed a leftover $staged"
    fi
  done
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
  # The code update already committed, so every failure here is a follow-up
  # warning on an ok:true result, never "still on the previous version".
  if [ "$FINAL_CODE" -eq 0 ] && [ "$APP_REBUILD_FAILED" -eq 0 ] && [ -n "$rebuilt" ] \
      && [ -d "$RELAUNCH_TARGET" ] && [ "$rebuilt" != "$RELAUNCH_TARGET" ]; then
    publish_stage "Installing the new app"
    mac_bundle_recover "$RELAUNCH_TARGET"
    if ! /usr/bin/ditto "$rebuilt" "$RELAUNCH_TARGET.new"; then
      rm -rf "$RELAUNCH_TARGET.new" 2>/dev/null || true
      DONE_NOTE="Hermes was updated, but the new app could not be staged; the previous app was kept. Run the update again."
      add_warning "app-swap" "bundle copy failed; previous app kept"
      return
    fi
    # The two renames are one critical section: a signal between them would
    # leave no app at the user's path. Defer HUP/INT/QUIT/TERM across it;
    # SIGKILL there is finished by mac_bundle_recover on the next run.
    trap '' HUP INT QUIT TERM
    if ! mv "$RELAUNCH_TARGET" "$RELAUNCH_TARGET.old"; then
      rm -rf "$RELAUNCH_TARGET.new" 2>/dev/null || true
      DONE_NOTE="Hermes was updated, but the new app could not replace the old one; the previous app was kept. Run the update again."
      add_warning "app-swap" "could not move the old bundle aside; previous app kept"
    elif ! mv "$RELAUNCH_TARGET.new" "$RELAUNCH_TARGET"; then
      if mv "$RELAUNCH_TARGET.old" "$RELAUNCH_TARGET"; then
        rm -rf "$RELAUNCH_TARGET.new" 2>/dev/null || true
        DONE_NOTE="Hermes was updated, but the new app could not be installed; the previous app was restored. Run the update again."
        add_warning "app-swap" "bundle install failed; rolled back to the previous app"
      else
        DONE_NOTE="Hermes was updated, but installing the new app failed and the previous app could not be restored. Reinstall Hermes (the rebuilt app is at $rebuilt)."
        add_warning "app-swap" "bundle install failed AND rollback failed"
      fi
    else
      rm -rf "$RELAUNCH_TARGET.old" 2>/dev/null || true
      log "swapped app bundle"
    fi
    trap 'on_signal HUP' HUP; trap 'on_signal INT' INT; trap 'on_signal QUIT' QUIT; trap 'on_signal TERM' TERM
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
      add_warning "relaunch" "$GATE: $GATE_MSG"
    fi
  fi
}

launch_app() { # attempted BEFORE the terminal event (launch acceptance is
  # part of the outcome — gille's review). Returns nonzero when a launch
  # was due but did not verifiably happen; caller downgrades to manual.
  [ -n "$RELAUNCH_TARGET" ] || return 0
  if [ "$DESKTOP_PID" -gt 0 ] 2>/dev/null && ident_alive "$DESKTOP_PID" "$DESKTOP_CT"; then
    # The Desktop never quit (exit 4) or refused us: it is still open, so a
    # second instance would only fight its single-instance lock.
    log "relaunch skipped: the Desktop (pid $DESKTOP_PID) is still running"
    return 0
  fi
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
     # The relaunched app must not inherit the hand-off's update env.
     env -u HERMES_UPDATE_STATUS_FILE -u HERMES_UPDATE_STARTED_AT -u HERMES_UPDATE_HANDOFF_PID -u PYTHONUNBUFFERED \
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

write_result() { # atomic (tmp + rename); started_at ties it to marker line 2
  local w="" item
  for item in ${WARNINGS[@]+"${WARNINGS[@]}"}; do
    w="$w${w:+,}\"$(json_escape "$item")\""
  done
  printf '{"ok":%s,"exit_code":%s,"manual":%s,"message":"%s","branch":"%s","channel":"%s","started_at":%s,"finished_at":%s,"warnings":[%s]}' \
    "$([ "$FINAL_CODE" -eq 0 ] && echo true || echo false)" "$FINAL_CODE" \
    "$([ "$MANUAL" -eq 1 ] && echo true || echo false)" \
    "$(json_escape "$FINAL_MSG")" "$(json_escape "$BRANCH")" "$(json_escape "$CHANNEL")" \
    "${STARTED_AT:-0}" "$(date +%s)" "$w" \
    > "$RESULT.$$.tmp" 2>/dev/null && mv -f "$RESULT.$$.tmp" "$RESULT" 2>/dev/null
  rm -f "$RESULT.$$.tmp" 2>/dev/null || true
}

add_warning() { # step reason -- post-commit follow-up failed; the update still counts
  WARNINGS+=("$1: $2")
  log "WARNING ($1): $2"
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

  marker_release

  if [ "$FINAL_CODE" -ne 0 ]; then
    publish "error" "$FINAL_MSG"; stop_ui leave-window
    launch_app || true   # error path still tries to bring the app back
    rm -f "$STATUS" "$STATUS.tmp" "$STATUS.port" 2>/dev/null || true
    return
  fi

  if [ -n "$DONE_NOTE" ]; then
    if [ "$GATE" != "skew" ] && [ "$GATE" != "manual" ]; then
      # A kept/rolled-back mac bundle or a failed post-commit follow-up: bring
      # the app back up; the note still tells the user what to re-run. A gated
      # linux outcome (skew/manual) skips the launch BY DESIGN.
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
  rm -f "$STATUS" "$STATUS.tmp" "$STATUS.port" 2>/dev/null || true
}
trap finish EXIT

# ── legacy macOS TCC anchor self-heal (#95759) ──────────────────────────────
# The reverted TCC interpreter anchor (#95425/#95541) left some installs with
# a real-file `venv/bin/python` copy plus a `.tcc-anchor-source` marker, and
# `python3`/`python3.N` aliases that die at interpreter init ("No module
# named 'encodings'"). On those installs EVERY normal CLI entrypoint is dead
# (`venv/bin/hermes` has a `python3` shebang), so no Python-side heal —
# doctor OR in-update — can ever run. This shell is the last surface that
# still executes, so the heal lives here (recovery design after @aeonsong's
# #96231; heal-point observation by @ahrazzle / @tokenfires on #95759).
#
# Ping-pong coherence with the re-landed forward anchor
# (hermes_cli/macos_tcc_anchor.ensure_tcc_anchor, which re-anchors whenever
# `venv/bin/python` is a uv-managed symlink): this heal is gated on the
# interpreter FAILING its boot probe, so a healthy anchored install is never
# touched; and when it does restore symlinks, the very `hermes update` run it
# unblocks re-installs a boot-gated healthy anchor — a one-shot convergence,
# not a loop.

tcc_probe_python() { # interpreter path → 0 iff it boots a real stdlib.
  # PYTHONHOME/PYTHONPATH are scrubbed: an inherited PYTHONHOME papers over
  # exactly the prefix-resolution failure this probe exists to detect.
  [ -x "$1" ] || return 1
  env -u PYTHONHOME -u PYTHONPATH -u PYTHONSTARTUP -u __PYVENV_LAUNCHER__ \
    "$1" -c 'import encodings' >/dev/null 2>&1
}

TCC_HEAL_STATE="not-run"

tcc_heal_rollback() { # restore every .tcc-heal-old.$$ backup in a bin dir
  local b
  for b in "$1"/*.tcc-heal-old.$$; do
    [ -e "$b" ] || [ -L "$b" ] || continue
    mv -f "$b" "${b%.tcc-heal-old.$$}" 2>/dev/null || true
  done
}

tcc_heal_cleanup() { # heal landed: drop the backups
  local b
  for b in "$1"/*.tcc-heal-old.$$; do
    [ -e "$b" ] || [ -L "$b" ] || continue
    rm -f "$b" 2>/dev/null || true
  done
}

tcc_alias_names() { # existing python3 / python3.N entries in a bin dir
  local a
  for a in "$1"/python3 "$1"/python3.*; do
    [ -e "$a" ] || [ -L "$a" ] || continue
    case "${a##*/}" in
      python3|python3.[0-9]|python3.[0-9][0-9]) printf '%s\n' "$a" ;;
    esac
  done
}

tcc_anchor_heal() { # $1 = venv bin dir. 0 iff python3 boots on exit.
  local bin="$1" marker src alias staged
  local py="$bin/python" py3="$bin/python3"
  if tcc_probe_python "$py3"; then TCC_HEAL_STATE="healthy"; return 0; fi
  marker="$bin/.tcc-anchor-source"
  [ -f "$marker" ] || { TCC_HEAL_STATE="no-marker"; return 1; }
  src="$(head -1 "$marker" 2>/dev/null)"
  case "$src" in
    "$bin"/*) TCC_HEAL_STATE="unsafe-source"; return 1 ;;
    /*) ;;
    *) TCC_HEAL_STATE="invalid-marker"; return 1 ;;
  esac
  if tcc_probe_python "$py"; then
    # #95541 alias-brick: the anchored copy itself boots; only the aliases
    # are dead. Aliases over a real-file anchor must be REAL FILES (hard
    # link, else copy) — an alias *symlink* onto the copy is the exact
    # crash shape being healed here.
    TCC_HEAL_STATE="healed-aliases"
    while IFS= read -r alias; do
      [ -n "$alias" ] || continue
      mv "$alias" "$alias.tcc-heal-old.$$" 2>/dev/null \
        || { tcc_heal_rollback "$bin"; TCC_HEAL_STATE="failed"; return 1; }
      if ! { ln "$py" "$alias" 2>/dev/null || cp -p "$py" "$alias" 2>/dev/null; }; then
        tcc_heal_rollback "$bin"; TCC_HEAL_STATE="failed"; return 1
      fi
    done <<EOF_ALIASES
$(tcc_alias_names "$bin")
EOF_ALIASES
  elif [ -x "$src" ] && tcc_probe_python "$src"; then
    # Full restore to the pre-anchor layout: python → symlink to the
    # marker-recorded store interpreter, aliases → symlinks to python.
    TCC_HEAL_STATE="healed-symlinks"
    mv "$py" "$py.tcc-heal-old.$$" 2>/dev/null \
      || { TCC_HEAL_STATE="failed"; return 1; }
    if ! ln -s "$src" "$py" 2>/dev/null; then
      tcc_heal_rollback "$bin"; TCC_HEAL_STATE="failed"; return 1
    fi
    while IFS= read -r alias; do
      [ -n "$alias" ] || continue
      staged="$alias.tcc-heal-new.$$"
      mv "$alias" "$alias.tcc-heal-old.$$" 2>/dev/null \
        || { tcc_heal_rollback "$bin"; TCC_HEAL_STATE="failed"; return 1; }
      if ! { ln -s python "$staged" 2>/dev/null && mv "$staged" "$alias" 2>/dev/null; }; then
        rm -f "$staged" 2>/dev/null
        tcc_heal_rollback "$bin"; TCC_HEAL_STATE="failed"; return 1
      fi
    done <<EOF_ALIASES
$(tcc_alias_names "$bin")
EOF_ALIASES
  else
    # Recorded source gone or itself unbootable (the vanished-uv-store
    # class from #95759): fail closed, touch nothing.
    TCC_HEAL_STATE="source-missing"; return 1
  fi
  if tcc_probe_python "$py3"; then
    if [ "$TCC_HEAL_STATE" = "healed-symlinks" ]; then
      # Marker asserts the anchored layout; the symlink layout is done
      # with it. (The alias-heal branch KEEPS it: real-file python +
      # real-file aliases is exactly the layout ensure_tcc_anchor marks.)
      rm -f "$marker" 2>/dev/null || true
    fi
    tcc_heal_cleanup "$bin"
    return 0
  fi
  tcc_heal_rollback "$bin"
  TCC_HEAL_STATE="failed"
  return 1
}

tcc_pick_update_invoke() { # sets UPDATE_INVOKE; safety net past a failed heal
  # Last-resort class: aliases still dead but the anchored copy boots. The
  # launchd gateway proves `venv/bin/python -m hermes_cli.main` works when
  # every alias entrypoint is bricked — drive the update the same way.
  local bin="$1"
  UPDATE_INVOKE=("$bin/hermes")
  if ! tcc_probe_python "$bin/python3" && tcc_probe_python "$bin/python"; then
    UPDATE_INVOKE=("$bin/python" -m hermes_cli.main)
  fi
}

# ── self-tests: no update, touch nothing ────────────────────────────────────
if [ "$SELF_TEST_TCC_HEAL" -eq 1 ]; then
  # Runs the REAL heal + invoke selection against --install-root and reports;
  # tests/scripts/desktop_update/test_desktop_update_tcc_heal.py drives the state matrix through it.
  trap - EXIT
  tcc_anchor_heal "$INSTALL_ROOT/venv/bin" || true
  tcc_pick_update_invoke "$INSTALL_ROOT/venv/bin"
  echo "state=$TCC_HEAL_STATE invoke=${UPDATE_INVOKE[*]}"
  exit 0
fi

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

# ── the Desktop's marker helper: verdict on stdout, nothing else ────────────
if [ -n "$MARKER_OP" ]; then
  trap - EXIT HUP INT QUIT TERM
  marker_op "$MARKER_OP"
  exit $?
fi

# ── the actual job ──────────────────────────────────────────────────────────
# Everything below is ONE brace group: bash parses a compound command whole
# before running it, so `hermes update` rewriting this very file (the script
# lives in the checkout it updates) can never feed the rest of the run from a
# shifted offset of the new text.
{
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
  # --daemonized must precede ORIGINAL_ARGS, not follow it: ORIGINAL_ARGS may
  # contain a `--` separator (Linux relaunch args), and anything appended
  # after that point is swallowed into RELAUNCH_ARGS instead of being parsed
  # as a flag. Appending here previously left HANDOFF_DAEMONIZED unset on
  # every re-exec, causing this block to re-fire forever (self-exec loop,
  # unbounded argv growth) whenever relaunch args were present.
  # nohup leaves SIGHUP ignored, and an ignored-at-entry signal can never be
  # re-trapped by bash, so every child (hermes update, gateways it starts, the
  # relaunched app) inherited it. The daemon has no terminal after setsid:
  # restore the default before exec.
  /usr/bin/nohup /usr/bin/python3 -c '
import os, signal, sys
env = os.environ.copy()
os.setsid()
signal.signal(signal.SIGHUP, signal.SIG_DFL)
os.execve("/bin/bash", ["/bin/bash", sys.argv[1], *sys.argv[2:]], env)
' "$SCRIPT_DIR/posix.sh" --daemonized "${ORIGINAL_ARGS[@]}" >/dev/null 2>&1 &
  DAEMON_PID=$!
  # Contract C2: the hand-off has started only once the daemon has claimed
  # the marker. Report a daemon that never got that far (no /usr/bin/python3,
  # a refused claim) as a failed launch so the Desktop stays up.
  for _ in $(seq 1 100); do
    [ "$(head -1 "$MARKER" 2>/dev/null | tr -d '[:space:]')" = "$DAEMON_PID" ] && exit 0
    if ! kill -0 "$DAEMON_PID" 2>/dev/null; then
      wait "$DAEMON_PID"; code=$?
      [ "$code" -ne 0 ] || code=70
      exit "$code"
    fi
    sleep 0.1
  done
  exit 70
fi

# Electron terminates the entire detached updater process group during quit,
# including the loopback status server.  Arm TERM immunity before `start_ui`
# so the shim server inherits SIG_IGN.  The orchestrator restores its normal
# TERM handler after the update command has returned.
trap '' TERM

# Marker claim FIRST (contract C1/C2): before the Desktop wait, the UI or any
# probe. The Desktop supplies one acquisition time for the whole ownership
# chain; a bridge marker owned by --desktop-pid is adopted.
NOW="$(date +%s)"
STARTED_AT="${HERMES_UPDATE_STARTED_AT:-$NOW}"
case "$STARTED_AT" in ''|*[!0-9]*) STARTED_AT="$NOW" ;; esac
MIN_STARTED_AT=$((NOW - 1200))
# Compare the validated decimal strings before doing arithmetic. Shell integer
# expansion can wrap on an attacker-controlled value wider than signed 64-bit.
if [ "${#STARTED_AT}" -ne "${#NOW}" ] \
    || [[ "$STARTED_AT" > "$NOW" || "$STARTED_AT" < "$MIN_STARTED_AT" ]]; then
  STARTED_AT="$NOW"
fi
DESKTOP_CT="$(proc_ct "$DESKTOP_PID")"
MARKER_REFUSED_PID=""
if ! marker_claim; then
  # A4: this run changed nothing and owns no result -- the other update (or
  # the Desktop that gave up on this hand-off) reports its own. An older
  # Desktop quits right after spawning us, so bring one back if it is gone.
  log "Another Hermes update is already running (process $MARKER_REFUSED_PID), or the Desktop gave up on this hand-off. Nothing was changed."
  trap - EXIT
  [ "$(uname)" = "Darwin" ] || linux_gate
  launch_app || true
  exit 2
fi
log "hand-off start: root=$INSTALL_ROOT branch=$BRANCH channel=$CHANNEL desktopPid=$DESKTOP_PID pid=$$"

if [ "$SELF_TEST_MARKER" -eq 1 ]; then
  trap - EXIT
  exit 0
fi

# Start-of-run recovery of litter only a dead hand-off can leave (we hold the
# marker, so no other hand-off is running): an interrupted app swap, a TCC heal
# backup of a killed run, UI/probe temp files whose owning pid is gone.
case "$RELAUNCH_TARGET" in *.app) mac_bundle_recover "$RELAUNCH_TARGET" ;; esac
for _stale in "$INSTALL_ROOT"/venv/bin/*.tcc-heal-old.* "$INSTALL_ROOT"/venv/bin/*.tcc-heal-new.* \
    "${TMPDIR:-/tmp}"/hermes-update-status.[0-9]* "${TMPDIR:-/tmp}"/hermes-update-probe.[0-9]* \
    "${TMPDIR:-/tmp}"/hermes-update-go.[0-9]* \
    "${TMPDIR:-/tmp}"/hermes-update-ui-[0-9]*-*; do
  [ -e "$_stale" ] || [ -L "$_stale" ] || continue
  case "$_stale" in
    *.tcc-heal-*) _owner="${_stale##*.}" ;;
    *) _owner="$(printf '%s' "${_stale##*/}" | sed -n 's/^[^0-9]*\([0-9][0-9]*\).*/\1/p')" ;;
  esac
  case "$_owner" in ''|*[!0-9]*) continue ;; esac
  { [ "$_owner" != "$$" ] && ! pid_alive "$_owner"; } || continue
  case "$_stale" in
    *.tcc-heal-old.*)
      _orig="${_stale%.tcc-heal-old.*}"
      if [ ! -e "$_orig" ] && [ ! -L "$_orig" ]; then
        mv -f "$_stale" "$_orig" 2>/dev/null && log "restored $_orig from a killed TCC heal"
      else
        rm -f "$_stale" 2>/dev/null
      fi ;;
    *) rm -rf "$_stale" 2>/dev/null ;;
  esac
done

# Wait out the Desktop (FAIL CLOSED: updating under live backends bricks).
# Zombie-aware and identity-checked: a reused pid is not the Desktop.
if [ "$DESKTOP_PID" -gt 0 ] 2>/dev/null; then
  for _ in $(seq 1 200); do ident_alive "$DESKTOP_PID" "$DESKTOP_CT" || break; sleep 0.3; done
  if ident_alive "$DESKTOP_PID" "$DESKTOP_CT"; then
    FINAL_CODE=4 FINAL_MSG="Update aborted: the Hermes window (pid $DESKTOP_PID) did not exit within 60s. Nothing was changed. Close Hermes fully and try again."
    log "$FINAL_MSG"; exit "$FINAL_CODE"
  fi
fi

# Do not create Chrome until Electron has fully left.  During its 2.5s quit
# dwell/before-quit teardown macOS can terminate descendants of the hand-off;
# a Chrome renderer reports that SIGTERM as "Aw, Snap!" error code 15.  The
# update marker above prevents a second click during this short UI-less gap.
sleep 1
start_ui

# Current installs publish an installation-bound launcher. Only pre-PM
# checkouts use the old shim/TCC rescue; a damaged PM install must not retarget.
LEGACY_INSTALL=0
[ -d "$INSTALL_ROOT/pm" ] || LEGACY_INSTALL=1
select_update_invoke() {
  HERMES_BIN="$INSTALL_ROOT/.hermes/bin/hermes"
  if [ -x "$HERMES_BIN" ]; then
    UPDATE_INVOKE=("$HERMES_BIN")
    return 0
  fi
  if [ -f "$INSTALL_ROOT/hermes_cli/_launchers.py" ]; then
    local candidate version reported expected
    expected="$(cd "$INSTALL_ROOT" && pwd -P)" || return 1
    for candidate in "$HOME/.local/bin/hermes" "$HERMES_HOME/bin/hermes"; do
      [ -x "$candidate" ] || continue
      version="$(run_bounded 60 "$candidate" --version)" || continue
      reported="$(printf '%s\n' "$version" | sed -n 's/^Install directory: //p')"
      [ -d "$reported" ] || continue
      [ "$(cd "$reported" && pwd -P)" = "$expected" ] || continue
      HERMES_BIN="$candidate"
      UPDATE_INVOKE=("$candidate")
      return 0
    done
  fi
  if [ "$LEGACY_INSTALL" -eq 1 ] && [ ! -d "$INSTALL_ROOT/pm" ]; then
    HERMES_BIN="$INSTALL_ROOT/venv/bin/hermes"
    [ -x "$HERMES_BIN" ] || return 1
    if [ "$(uname)" = Darwin ]; then
      tcc_anchor_heal "$INSTALL_ROOT/venv/bin" || log "TCC anchor rescue failed ($TCC_HEAL_STATE)"
    fi
    tcc_pick_update_invoke "$INSTALL_ROOT/venv/bin"
    return 0
  fi
  return 1
}
select_update_invoke || { FINAL_CODE=3 FINAL_MSG="Update aborted: the installation launcher at $HERMES_BIN is missing. Repair this installation."; log "$FINAL_MSG"; exit 3; }

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
# The takeover children (hermes update -> _update_takeover/update_finish and
# the PM sync / build stages they drive) publish their stages back into the
# shim's UI through this file; without a watching UI the variable is simply
# absent and the helper no-ops.
export HERMES_UPDATE_STATUS_FILE="$STATUS"
# `hermes update` runs under OUR marker claim (contract C1 rule 4/6).
export HERMES_UPDATE_HANDOFF_PID="$$"
# --keep-stash: never re-apply local source edits after the update (they stay
# parked in git stash). Probe --help first: older installed backends don't
# know the flag and argparse would abort with exit 2, which collides with the
# "close all Hermes windows" sentinel. The probe is bounded: a hung launcher
# must not hold the marker forever.
KEEP_STASH=""
if run_bounded 60 "${UPDATE_INVOKE[@]}" update --help | grep -q -- '--keep-stash'; then
  KEEP_STASH="--keep-stash"
else
  log "installed hermes predates --keep-stash (or the probe failed); running without it"
fi
# --gateway restarts the local messaging gateway after the update. The
# Desktop omits it (--no-gateway) when it is served by a remote gateway
# (#117529): restarting a local one there is never wanted, and with the same
# channel credentials as the remote host it becomes a competing long-poll
# consumer (e.g. Telegram rejects one of the two getUpdates callers).
GATEWAY_FLAG="--gateway"
if [ "$NO_GATEWAY" -eq 1 ]; then
  GATEWAY_FLAG=""
  log "update requested without --gateway (remote-served Desktop)"
fi

run_update() { # streams straight into the log (a killed run keeps its output);
  # OUT = this run's slice of the log. TERM goes back to default for the child
  # so neither it nor anything it starts (gateways) inherits our SIG_IGN.
  # The child runs asynchronously only so its pid can go into the marker
  # (marker_add_delegate) before it does anything: it waits on a go-file that
  # is created only AFTER the delegate line is published, and gives up when
  # this script dies first -- so no instant exists in which the update runs
  # but the marker names neither a live owner nor a live delegate. A
  # HUP/INT/QUIT that lands meanwhile is handled once it exits, exactly like a
  # foreground child.
  local offset pid sig go="${TMPDIR:-/tmp}/hermes-update-go.$$" refresher
  offset="$(wc -c < "$LOG" 2>/dev/null | tr -d '[:space:]')"
  PENDING_SIGNAL=""
  for sig in HUP INT QUIT; do trap "PENDING_SIGNAL=\${PENDING_SIGNAL:-$sig}" "$sig"; done
  rm -f "$go" 2>/dev/null
  ( trap - TERM
    while [ ! -e "$go" ]; do kill -0 "$MY_PID" 2>/dev/null || exit 70; sleep 0.05; done
    exec "${UPDATE_INVOKE[@]}" update --yes $GATEWAY_FLAG $KEEP_STASH "${TARGET_ARGS[@]}" ) >> "$LOG" 2>&1 <&0 &
  pid=$!
  marker_add_delegate "$pid"
  : > "$go"
  # An older packaged Desktop judges a marker by line 2 alone and deletes it
  # 20 minutes in, live owner or not: keep line 2 young while we own it.
  ( for _ in $(seq 1 100000); do
      for _ in $(seq 1 60); do sleep 5; kill -0 "$MY_PID" 2>/dev/null || exit 0; done
      marker_locked marker_refresh_locked
    done ) >/dev/null 2>&1 &
  refresher=$!
  while :; do
    wait "$pid"; CODE=$?
    [ "$CODE" -ne 127 ] || break          # not our child any more
    kill -0 "$pid" 2>/dev/null || break   # reaped: CODE is its exit status
  done
  kill -KILL "$refresher" 2>/dev/null; wait "$refresher" 2>/dev/null
  rm -f "$go" 2>/dev/null
  for sig in HUP INT QUIT; do trap "on_signal $sig" "$sig"; done
  OUT="$(tail -c +"$(( ${offset:-0} + 1 ))" "$LOG" 2>/dev/null)"
  [ -z "$PENDING_SIGNAL" ] || on_signal "$PENDING_SIGNAL"
}

log "running: ${UPDATE_INVOKE[*]} update --yes $GATEWAY_FLAG $KEEP_STASH ${TARGET_ARGS[*]}"
publish_stage "Updating code and dependencies"
run_update
log "hermes update exit code: $CODE"

if [ "$LEGACY_INSTALL" -eq 1 ] && [ "$CODE" -ne 0 ] && [ "$CODE" -ne 2 ]; then
  # Retry once: update-boundary class (fresh code on disk, stale in memory).
  # Exit 2 ("close all Hermes windows") is not retryable.
  #
  # A parked-branch SKIP (checkout on a feature branch with unmerged
  # commits) is also deterministic — the retry would hit the exact same
  # branch state and skip again, so it only wastes time. Detect the skip
  # by its banner, skip the retry, and surface an honest message with a
  # dedicated exit code (8) so callers can distinguish "skipped" from a
  # real failure.
  if printf '%s' "$OUT" | grep -q "CODE UPDATE SKIPPED"; then
    log "hermes update skipped (checkout parked on a non-target branch); not retrying"
    FINAL_CODE=8
    FINAL_MSG="Update skipped: the git checkout is on a branch that isn't fully merged into $BRANCH. Switch to the target branch and update again (see the terminal output for the exact commands)."
    exit 8
  fi
  log "retrying once (freshly pulled fix loads on the second run)"
  publish_stage "Retrying update"
  select_update_invoke || { FINAL_CODE=3 FINAL_MSG="Updated installation launcher is missing; repair this installation."; exit 3; }
  run_update
  log "retry exit code: $CODE"
fi
trap 'on_signal TERM' TERM

if [ "$CODE" -eq 0 ]; then FINAL_CODE=0 FINAL_MSG="Update complete."
else
  FINAL_CODE="$CODE" FINAL_MSG="Update failed (exit $CODE). Run hermes debug share in a terminal to send a report."
  # The bricked-venv class is fixable and must not read as a generic exit 1:
  # a dead interpreter with a failed/impossible heal means retrying can never
  # succeed — tell the user what is actually wrong (#95759).
  if [ "$LEGACY_INSTALL" -eq 1 ] && ! tcc_probe_python "$INSTALL_ROOT/venv/bin/python3" \
      && ! tcc_probe_python "$INSTALL_ROOT/venv/bin/python"; then
    FINAL_MSG="Update failed: the Python interpreter inside $INSTALL_ROOT/venv cannot start (heal state: $TCC_HEAL_STATE). Reinstall the runtime with the Hermes installer, or run hermes doctor --fix from a terminal if any hermes command still works."
  fi
fi

# Pre-PM update code could report a failed desktop build with exit zero.
# Current composition propagates failure and never enters this legacy repair.
# The code already committed: a failed rebuild is a follow-up warning on an
# ok:true result (contract C3), never "you're still on the previous version".
if [ "$LEGACY_INSTALL" -eq 1 ] && [ "$CODE" -eq 0 ] && printf '%s' "$OUT" | grep -q "Desktop build failed"; then
  log "desktop build failed inside hermes update; retrying build"
  publish_stage "Rebuilding Desktop"
  if ! ( trap - TERM; exec "${UPDATE_INVOKE[@]}" desktop --force-build --build-only ) >> "$LOG" 2>&1; then
    APP_REBUILD_FAILED=1
    DONE_NOTE="Hermes was updated, but the Desktop app rebuild failed - you are running the previous app build. Run hermes desktop --force-build from a terminal to retry."
    add_warning "desktop-rebuild" "hermes desktop --force-build --build-only failed"
  fi
fi
exit "$FINAL_CODE"
}
