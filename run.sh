#!/usr/bin/env bash
# ============================================================================
# run.sh — Hermes Agent unified launcher
# ============================================================================
# Boots the Hermes Agent system with real-time logs streamed to the terminal.
#
#   Usage:
#     ./run.sh up                 # auto-detect: desktop GUI > interactive CLI > gateway
#     ./run.sh up --desktop       # force Electron desktop shell (needs a display)
#     ./run.sh up --cli           # force the interactive terminal UI (needs a TTY)
#     ./run.sh up --gateway       # force the headless messaging gateway service
#     ./run.sh up --setup-only    # only provision the environment, then exit
#     ./run.sh doctor             # report what the launcher detected, change nothing
#     ./run.sh help
#
# Design notes (why this script looks the way it does):
#   * The Python environment is managed by `uv` against ./pyproject.toml +
#     ./uv.lock and lives in ./venv (same convention as setup-hermes.sh). There
#     is no requirements.txt / main.py in this project.
#   * The desktop GUI is an Electron shell (apps/desktop). It bootstraps its OWN
#     Python backend, so we never start a separate backend for the GUI path.
#   * On a headless VPS (no $DISPLAY) Electron cannot render, so we fall back to
#     the interactive CLI (if attached to a TTY) or the gateway service.
#   * Provisioning is idempotent: an existing venv / node_modules is reused and
#     never rebuilt unless it is missing. Re-running `./run.sh up` is safe.
#   * The chosen service runs in the FOREGROUND with logs tee'd live to the
#     terminal and to ./logs — nothing is daemonized or silenced. A signal trap
#     tears the whole process group down so no child is orphaned.
# ============================================================================

set -Eeuo pipefail

# --- Resolve project root so the script works from any CWD (idempotency) -----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# uv must not pick up config from a different user's HOME under sudo (see #21269).
export UV_NO_CONFIG="${UV_NO_CONFIG:-1}"

VENV_DIR="$SCRIPT_DIR/venv"
LOG_DIR="$SCRIPT_DIR/logs"
PYTHON_VERSION="3.11"

# --- Pretty, greppable logging ----------------------------------------------
if [ -t 1 ]; then
  C_INFO=$'\033[0;36m'; C_OK=$'\033[0;32m'; C_WARN=$'\033[0;33m'
  C_ERR=$'\033[0;31m';  C_DIM=$'\033[0;90m'; C_OFF=$'\033[0m'
else
  C_INFO=""; C_OK=""; C_WARN=""; C_ERR=""; C_DIM=""; C_OFF=""
fi
log()  { printf '%s[hermes-run]%s %s\n'  "$C_INFO" "$C_OFF" "$*"; }
ok()   { printf '%s[hermes-run]%s %s\n'  "$C_OK"   "$C_OFF" "$*"; }
warn() { printf '%s[hermes-run]%s %s\n'  "$C_WARN" "$C_OFF" "$*" >&2; }
die()  { printf '%s[hermes-run] ERROR:%s %s\n' "$C_ERR" "$C_OFF" "$*" >&2; exit 1; }

# Report the failing line on any unexpected error (set -E makes this fire in fns).
trap 'die "failed at line $LINENO (command: $BASH_COMMAND)"' ERR

have() { command -v "$1" >/dev/null 2>&1; }

# ============================================================================
# Environment provisioning (Python via uv, reused if already present)
# ============================================================================

resolve_uv() {
  if have uv; then echo "uv"; return 0; fi
  for c in "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv"; do
    [ -x "$c" ] && { echo "$c"; return 0; }
  done
  return 1
}

ensure_python_env() {
  # Fast path: a provisioned venv with the `hermes` console script already exists.
  if [ -x "$VENV_DIR/bin/hermes" ]; then
    ok "Python env present (venv/bin/hermes) — reusing, not reinstalling."
    return 0
  fi

  log "No provisioned venv found — creating one (first run only)."

  # Prefer the project's own setup script when available: it handles Termux,
  # extras selection, locale data files and CLI symlinking correctly.
  if [ -x "$SCRIPT_DIR/setup-hermes.sh" ]; then
    log "Delegating provisioning to ./setup-hermes.sh ..."
    # HERMES_SKIP_WIZARD keeps provisioning non-interactive; if the script does
    # not honor it the wizard is still safe to skip with Ctrl-C and re-run up.
    HERMES_SKIP_WIZARD=1 bash "$SCRIPT_DIR/setup-hermes.sh" </dev/null || \
      warn "setup-hermes.sh exited non-zero; verifying the venv below."
    [ -x "$VENV_DIR/bin/hermes" ] && { ok "Environment provisioned."; return 0; }
    warn "setup-hermes.sh did not yield venv/bin/hermes — falling back to uv."
  fi

  # Fallback: provision directly with uv against the lockfile.
  local UV; UV="$(resolve_uv)" || die \
    "Neither a venv nor 'uv' is available. Install uv (https://docs.astral.sh/uv/) or run ./setup-hermes.sh first."

  [ -d "$VENV_DIR" ] || "$UV" venv "$VENV_DIR" --python "$PYTHON_VERSION"

  log "Installing dependencies (this can take 1-5 minutes on a fresh venv) ..."
  if ! UV_PROJECT_ENVIRONMENT="$VENV_DIR" "$UV" sync --extra all --locked; then
    warn "Locked sync failed (stale lockfile?) — retrying with editable install."
    "$UV" pip install --python "$VENV_DIR/bin/python" -e ".[all]" \
      || "$UV" pip install --python "$VENV_DIR/bin/python" -e "."
  fi

  [ -x "$VENV_DIR/bin/hermes" ] || die "Provisioning finished but venv/bin/hermes is missing."
  ok "Environment provisioned."
}

ensure_node_modules() {
  # Only needed for the desktop GUI path. npm workspaces require a root install
  # before the desktop workspace can build (scripts/assert-root-install.cjs).
  have node || die "Node.js >= 20 is required for the desktop GUI but was not found."
  have npm  || die "npm is required for the desktop GUI but was not found."

  if [ -d "$SCRIPT_DIR/node_modules" ] && [ -d "$SCRIPT_DIR/apps/desktop/node_modules" ]; then
    ok "Node modules present — reusing, not reinstalling."
    return 0
  fi
  log "Installing Node workspace dependencies (first run only) ..."
  npm install
  ok "Node dependencies installed."
}

# ============================================================================
# Runtime helpers
# ============================================================================

# Warn (do not fail) if a port the desktop dev server wants is already taken.
warn_if_port_busy() {
  local port="$1" who=""
  if have lsof; then
    who="$(lsof -nP -iTCP:"$port" -sTCP:LISTEN -t 2>/dev/null || true)"
  elif have ss; then
    ss -ltn 2>/dev/null | grep -q ":$port[[:space:]]" && who="busy"
  fi
  [ -n "$who" ] && warn "Port $port is already in use (pid: ${who:-unknown}); the desktop dev server may fail to bind."
}

display_available() {
  # A GUI is renderable if we have an X/Wayland display, or we're on macOS.
  [ -n "${DISPLAY:-}" ] || [ -n "${WAYLAND_DISPLAY:-}" ] || [ "$(uname -s)" = "Darwin" ]
}

# Foreground-exec a command with its own process group, tee logs live, and make
# sure the whole group dies with us so there are no orphaned children.
run_foreground() {
  local name="$1"; shift
  mkdir -p "$LOG_DIR"
  local logfile="$LOG_DIR/${name}.log"
  log "Launching '${name}' — streaming logs live to terminal and ${C_DIM}${logfile}${C_OFF}"
  log "Press Ctrl-C to stop."

  set -m
  "$@" > >(tee -a "$logfile") 2>&1 &
  local child=$!
  # Forward termination to the whole child process group (negative pid).
  trap 'warn "Stopping ${name} ..."; kill -- -"$child" 2>/dev/null || kill "$child" 2>/dev/null || true' INT TERM
  local rc=0
  wait "$child" || rc=$?
  trap - INT TERM
  if [ "$rc" -ne 0 ]; then
    warn "'${name}' exited with code ${rc}. Last lines: ${C_DIM}${logfile}${C_OFF}"
    return "$rc"
  fi
  ok "'${name}' exited cleanly."
}

# ============================================================================
# Service launchers
# ============================================================================

launch_desktop() {
  ensure_node_modules
  warn_if_port_busy 5174
  if ! display_available; then
    warn "No \$DISPLAY/\$WAYLAND_DISPLAY detected — Electron will likely fail to open a window."
    warn "On a headless VPS, run under a virtual framebuffer, e.g.:  xvfb-run -a ./run.sh up --desktop"
    warn "Or use a headless interface instead:  ./run.sh up --cli   |   ./run.sh up --gateway"
  fi
  # `npm run dev` (apps/desktop) starts vite renderer + electron; electron boots
  # the Python backend itself, so this single foreground tree IS the full stack.
  run_foreground "desktop" npm run dev --workspace apps/desktop
}

launch_cli() {
  [ -t 0 ] && [ -t 1 ] || die \
    "The interactive CLI needs a TTY. Use './run.sh up --gateway' for a non-interactive/background context."
  log "Starting the Hermes interactive terminal UI."
  # Interactive TUI: hand over the terminal directly (no tee — it owns the TTY).
  exec "$VENV_DIR/bin/hermes"
}

launch_gateway() {
  warn "Headless gateway mode assumes you have already run 'hermes setup' / 'hermes gateway setup'."
  run_foreground "gateway" "$VENV_DIR/bin/hermes" gateway start
}

# ============================================================================
# Mode selection
# ============================================================================

choose_mode_auto() {
  if display_available && have node && have npm; then
    echo "desktop"
  elif [ -t 0 ] && [ -t 1 ]; then
    echo "cli"
  else
    echo "gateway"
  fi
}

doctor() {
  log "Hermes launcher diagnostics:"
  printf '  %-22s %s\n' "project root"     "$SCRIPT_DIR"
  printf '  %-22s %s\n' "os"               "$(uname -srm 2>/dev/null || echo unknown)"
  printf '  %-22s %s\n' "venv/bin/hermes"  "$([ -x "$VENV_DIR/bin/hermes" ] && echo present || echo MISSING)"
  printf '  %-22s %s\n' "uv"               "$(resolve_uv 2>/dev/null || echo 'not found')"
  printf '  %-22s %s\n' "node"             "$(have node && node --version || echo 'not found')"
  printf '  %-22s %s\n' "npm"              "$(have npm && npm --version || echo 'not found')"
  printf '  %-22s %s\n' "DISPLAY"          "${DISPLAY:-<unset>}"
  printf '  %-22s %s\n' "WAYLAND_DISPLAY"  "${WAYLAND_DISPLAY:-<unset>}"
  printf '  %-22s %s\n' "display usable"   "$(display_available && echo yes || echo 'no (headless)')"
  printf '  %-22s %s\n' "stdin/stdout TTY" "$([ -t 0 ] && [ -t 1 ] && echo yes || echo no)"
  printf '  %-22s %s\n' "auto mode -> "    "$(choose_mode_auto)"
}

usage() {
  sed -n '5,15p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

# ============================================================================
# Entry point
# ============================================================================

main() {
  local cmd="${1:-help}"; shift || true
  case "$cmd" in
    up)
      local mode="auto" setup_only=0
      while [ $# -gt 0 ]; do
        case "$1" in
          --desktop)    mode="desktop" ;;
          --cli)        mode="cli" ;;
          --gateway)    mode="gateway" ;;
          --setup-only) setup_only=1 ;;
          *) die "Unknown option for 'up': $1 (see ./run.sh help)" ;;
        esac
        shift
      done

      ensure_python_env
      if [ "$setup_only" -eq 1 ]; then ok "Setup complete (--setup-only); not launching."; return 0; fi

      [ "$mode" = "auto" ] && mode="$(choose_mode_auto)"
      log "Selected run mode: ${C_OK}${mode}${C_OFF}"
      case "$mode" in
        desktop) launch_desktop ;;
        cli)     launch_cli ;;
        gateway) launch_gateway ;;
      esac
      ;;
    doctor)        doctor ;;
    help|-h|--help) usage ;;
    *) die "Unknown command: '$cmd'. Try: ./run.sh up   |   ./run.sh doctor   |   ./run.sh help" ;;
  esac
}

main "$@"
