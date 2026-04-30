#!/usr/bin/env bash
# ComfyUI Setup — Install, launch, and verify using the official comfy-cli.
#
# Improvements over v1:
#   - Prefers `pipx` / `uvx` over global `pip install` (avoids polluting system Python)
#   - Idempotent: detects already-running server and skips re-launch
#   - Configurable port via --port=N (default 8188)
#   - Configurable workspace via --workspace=PATH
#   - Persistent log file in /tmp/comfyui_setup.<pid>.log for debugging
#   - SIGINT trap cleans up partial state
#   - Refuses local install when hardware_check.py verdict is "cloud"
#   - Forwards extra flags to comfy-cli (e.g. --cuda-version=12.4)
#
# Usage:
#   bash scripts/comfyui_setup.sh
#       (auto-detects GPU; uses recommendation from hardware_check.py)
#   bash scripts/comfyui_setup.sh --nvidia
#   bash scripts/comfyui_setup.sh --m-series --port=8190
#   bash scripts/comfyui_setup.sh --amd --workspace=/data/comfy
#
# Flags:
#   --nvidia | --amd | --m-series | --cpu     GPU selection (skips hw check)
#   --port=N                                  HTTP port (default 8188)
#   --workspace=PATH                          ComfyUI install location
#   --skip-launch                             Install only, don't start server
#   --force-cloud-override                    Install locally even if hw says cloud
#   --                                        Pass remaining args to `comfy install`

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARDWARE_CHECK="$SCRIPT_DIR/hardware_check.py"
LOG_FILE="/tmp/comfyui_setup.$$.log"
PORT=8188
WORKSPACE=""
GPU_FLAG=""
SKIP_LAUNCH=0
FORCE_CLOUD_OVERRIDE=0
EXTRA_INSTALL_ARGS=()

cleanup() {
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo "==> Setup exited with status $exit_code. Log: $LOG_FILE" >&2
    fi
    exit $exit_code
}
trap cleanup EXIT INT TERM

log() { echo "==> $*" | tee -a "$LOG_FILE" >&2; }
err() { echo "ERROR: $*" | tee -a "$LOG_FILE" >&2; }

# --- Argument parsing ---
PASSTHROUGH=0
for arg in "$@"; do
    if [ "$PASSTHROUGH" -eq 1 ]; then
        EXTRA_INSTALL_ARGS+=("$arg")
        continue
    fi
    case "$arg" in
        --nvidia|--amd|--m-series|--cpu)
            GPU_FLAG="$arg"
            ;;
        --port=*)
            PORT="${arg#*=}"
            ;;
        --workspace=*)
            WORKSPACE="${arg#*=}"
            ;;
        --skip-launch)
            SKIP_LAUNCH=1
            ;;
        --force-cloud-override)
            FORCE_CLOUD_OVERRIDE=1
            ;;
        --)
            PASSTHROUGH=1
            ;;
        --help|-h)
            # Print the leading comment block, stripping the `# ` prefix.
            # Stops at the first blank line which separates docs from code.
            awk '
                NR == 1 { next }                      # skip shebang
                /^[^#]/ { exit }                      # stop at first non-comment line
                /^$/    { exit }                      # ...or first blank line
                { sub(/^# ?/, ""); print }
            ' "$0"
            exit 0
            ;;
        *)
            err "Unknown argument: $arg"
            exit 64
            ;;
    esac
done

log "Logging to $LOG_FILE"

# --- Step 0: Hardware check (skipped if user gave an explicit GPU flag) ---
if [ -z "$GPU_FLAG" ]; then
    if [ ! -f "$HARDWARE_CHECK" ]; then
        log "hardware_check.py not found — defaulting to --nvidia"
        GPU_FLAG="--nvidia"
    else
        log "Running hardware check…"
        set +e
        HW_JSON="$(python3 "$HARDWARE_CHECK" --json 2>>"$LOG_FILE")"
        HW_EXIT=$?
        set -e

        if [ -z "$HW_JSON" ]; then
            err "hardware_check.py produced no output (exit $HW_EXIT). Pass an explicit flag."
            exit 1
        fi
        echo "$HW_JSON" | tee -a "$LOG_FILE" >&2

        VERDICT="$(echo "$HW_JSON" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("verdict",""))')"
        FLAG="$(echo "$HW_JSON"   | python3 -c 'import sys,json; print(json.load(sys.stdin).get("comfy_cli_flag") or "")')"

        if [ "$VERDICT" = "cloud" ] && [ "$FORCE_CLOUD_OVERRIDE" -ne 1 ]; then
            log ""
            log "Hardware check: this machine is not suitable for local ComfyUI."
            log "Recommended: Comfy Cloud — https://platform.comfy.org"
            log ""
            log "To override and force a local install, re-run with --force-cloud-override"
            log "or pass an explicit GPU flag (--nvidia|--amd|--m-series|--cpu)."
            exit 2
        fi

        if [ "$VERDICT" = "marginal" ]; then
            log "Hardware check: verdict is MARGINAL."
            log "  SD1.5 should work; SDXL/Flux may be slow or OOM."
            log "  Consider Comfy Cloud for heavier workflows: https://platform.comfy.org"
        fi

        if [ -z "$FLAG" ]; then
            log "hardware_check could not pick a comfy-cli flag. Defaulting to --nvidia."
            log "(For Intel Arc or unsupported hardware, use the manual install path.)"
            GPU_FLAG="--nvidia"
        else
            GPU_FLAG="$FLAG"
        fi
    fi
fi

log "GPU flag: $GPU_FLAG"
log "Port: $PORT"
[ -n "$WORKSPACE" ] && log "Workspace: $WORKSPACE"
[ "${#EXTRA_INSTALL_ARGS[@]}" -gt 0 ] && log "Extra install args: ${EXTRA_INSTALL_ARGS[*]}"

# --- Step 1: Install comfy-cli (prefer pipx / uvx over global pip) ---
COMFY_BIN=""
if command -v comfy >/dev/null 2>&1; then
    COMFY_BIN="comfy"
    log "comfy-cli already on PATH: $(comfy -v 2>/dev/null || echo 'unknown version')"
elif command -v uvx >/dev/null 2>&1; then
    log "Using uvx (no install needed)"
    COMFY_BIN="uvx --from comfy-cli comfy"
elif command -v pipx >/dev/null 2>&1; then
    log "Installing comfy-cli via pipx…"
    pipx install comfy-cli >>"$LOG_FILE" 2>&1
    COMFY_BIN="comfy"
    # pipx adds shims to ~/.local/bin which may need to be on PATH
    if ! command -v comfy >/dev/null 2>&1; then
        if [ -x "$HOME/.local/bin/comfy" ]; then
            export PATH="$HOME/.local/bin:$PATH"
            COMFY_BIN="$HOME/.local/bin/comfy"
        fi
    fi
else
    log "Neither pipx nor uvx found. Falling back to pip install --user…"
    log "  (Recommend installing pipx: https://pipx.pypa.io)"
    if ! pip install --user comfy-cli >>"$LOG_FILE" 2>&1; then
        # macOS: PEP 668 externally-managed-environment may block --user
        log "pip install --user failed. Retrying with --break-system-packages…"
        pip install --user --break-system-packages comfy-cli >>"$LOG_FILE" 2>&1 || {
            err "Could not install comfy-cli. Install pipx or uv first."
            exit 1
        }
    fi
    # Resolve the actual `comfy` script — pip --user puts it in:
    #   Linux: ~/.local/bin/comfy
    #   macOS: ~/Library/Python/<ver>/bin/comfy  OR  ~/.local/bin/comfy
    COMFY_BIN=""
    for candidate in "$HOME/.local/bin/comfy" \
                     "$HOME/Library/Python/3.13/bin/comfy" \
                     "$HOME/Library/Python/3.12/bin/comfy" \
                     "$HOME/Library/Python/3.11/bin/comfy" \
                     "$HOME/Library/Python/3.10/bin/comfy"; do
        if [ -x "$candidate" ]; then
            COMFY_BIN="$candidate"
            export PATH="$(dirname "$candidate"):$PATH"
            break
        fi
    done
    if [ -z "$COMFY_BIN" ]; then
        if command -v comfy >/dev/null 2>&1; then
            COMFY_BIN="comfy"
        else
            err "Installed comfy-cli but couldn't find the 'comfy' script."
            err "Add the right Python user-bin directory to PATH and retry."
            exit 1
        fi
    fi
fi

# --- Step 2: Disable analytics tracking (avoid interactive prompt) ---
log "Disabling analytics tracking…"
$COMFY_BIN --skip-prompt tracking disable >>"$LOG_FILE" 2>&1 || true

# --- Step 3: Install ComfyUI ---
WORKSPACE_ARG=()
if [ -n "$WORKSPACE" ]; then
    WORKSPACE_ARG=(--workspace "$WORKSPACE")
fi

if $COMFY_BIN "${WORKSPACE_ARG[@]}" which 2>/dev/null | grep -q "ComfyUI"; then
    EXISTING_WS="$($COMFY_BIN "${WORKSPACE_ARG[@]}" which 2>/dev/null || true)"
    log "ComfyUI already installed at: $EXISTING_WS"
else
    log "Installing ComfyUI ($GPU_FLAG)…"
    if ! $COMFY_BIN "${WORKSPACE_ARG[@]}" --skip-prompt install "$GPU_FLAG" "${EXTRA_INSTALL_ARGS[@]}" >>"$LOG_FILE" 2>&1; then
        err "Install failed. Tail of log:"
        tail -20 "$LOG_FILE" >&2
        exit 1
    fi
fi

if [ "$SKIP_LAUNCH" -eq 1 ]; then
    log "Setup complete (--skip-launch). Run \`$COMFY_BIN launch --background -- --port $PORT\` when ready."
    exit 0
fi

# --- Step 4: Detect already-running server ---
if curl -fsS "http://127.0.0.1:$PORT/system_stats" >/dev/null 2>&1; then
    log "Server already running on port $PORT — skipping launch."
    log "Stop with \`$COMFY_BIN stop\` if you want a fresh start."
    curl -fsS "http://127.0.0.1:$PORT/system_stats" | python3 -m json.tool 2>/dev/null || true
    log "Done."
    exit 0
fi

# --- Step 5: Launch ---
log "Launching ComfyUI in background on port $PORT…"
LAUNCH_EXTRAS=("--" "--port" "$PORT")
if ! $COMFY_BIN "${WORKSPACE_ARG[@]}" launch --background "${LAUNCH_EXTRAS[@]}" >>"$LOG_FILE" 2>&1; then
    err "Background launch failed. Tail of log:"
    tail -20 "$LOG_FILE" >&2
    err "Try foreground launch to see real-time errors: $COMFY_BIN launch -- --port $PORT"
    exit 1
fi

# --- Step 6: Wait for server ---
log "Waiting for server…"
MAX_WAIT=60
ELAPSED=0
while [ $ELAPSED -lt $MAX_WAIT ]; do
    if curl -fsS "http://127.0.0.1:$PORT/system_stats" >/dev/null 2>&1; then
        log "Server is running!"
        curl -fsS "http://127.0.0.1:$PORT/system_stats" | python3 -m json.tool 2>/dev/null || true
        break
    fi
    sleep 2
    ELAPSED=$((ELAPSED + 2))
done

if [ $ELAPSED -ge $MAX_WAIT ]; then
    err "Server did not start within ${MAX_WAIT}s."
    err "Inspect log: $LOG_FILE"
    err "Or run foreground: $COMFY_BIN launch -- --port $PORT"
    exit 1
fi

log ""
log "Setup complete!"
log "  Server: http://127.0.0.1:$PORT"
log "  Web UI: http://127.0.0.1:$PORT (open in browser)"
log "  Stop:   $COMFY_BIN stop"
log "  Log:    $LOG_FILE (kept until shell closes)"
log ""
log "Next steps:"
log "  - Download a model: $COMFY_BIN model download --url <URL> --relative-path models/checkpoints"
log "  - Run a workflow:   python3 $SCRIPT_DIR/run_workflow.py --workflow <file.json> --args '{...}'"

# Disable trap on success path
trap - EXIT
