#!/usr/bin/env bash
# Ares bootstrap installer.
#
# Ares is a downstream, Hermes-compatible distribution maintained by
# RecursiveIntell. It intentionally creates no provider, MCP, hook, or plugin
# configuration. Those choices remain explicit operator actions after install.
set -euo pipefail

REPO_URL="https://github.com/RecursiveIntell/Ares.git"
BRANCH="main"
HERMES_HOME="${HERMES_HOME:-$HOME/.ares}"
INSTALL_DIR=""
ARES_BIN_DIR="${ARES_BIN_DIR:-$HOME/.local/bin}"
USE_VENV=true
RECURSIVE_AGENT_SOURCE=""

log() { printf '[ares] %s\n' "$*"; }
die() { printf '[ares] error: %s\n' "$*" >&2; exit 1; }

show_help() {
    cat <<'EOF'
Ares Installer

Bootstrap the Ares downstream distribution of Hermes Agent. Ares preserves the
existing Python package and `hermes` CLI for compatibility, and also installs
an `ares` launcher that uses an isolated Ares home by default.

Usage:
  bash install.sh [options]

Options:
  --branch NAME                     Git branch to install (default: main)
  --dir PATH                        Source checkout directory
  --hermes-home PATH                Ares data directory (default: ~/.ares)
  --ares-bin-dir PATH               Directory for the `ares` launcher (default: ~/.local/bin)
  --no-venv                         Use the active Python environment instead of a managed .venv
  --with-recursive-agent-source PATH
                                    Install the standalone Recursive Agent plugin from an existing
                                    RecursiveIntell/recursive-agent checkout.
                                    The Recursive Agent daemon is not installed or started by this option.
  -h, --help                        Show this help

Prerequisites: git, uv (unless --no-venv), and Python 3.11 through 3.13.

This installer does not create providers, credentials, MCP servers, hooks, or
plugins except the explicitly requested Recursive Agent plugin. Run `ares setup`
after installation to configure a model provider.
EOF
}

while (($#)); do
    case "$1" in
        --branch) BRANCH="${2:?--branch requires a value}"; shift 2 ;;
        --dir) INSTALL_DIR="${2:?--dir requires a value}"; shift 2 ;;
        --hermes-home) HERMES_HOME="${2:?--hermes-home requires a value}"; shift 2 ;;
        --ares-bin-dir) ARES_BIN_DIR="${2:?--ares-bin-dir requires a value}"; shift 2 ;;
        --no-venv) USE_VENV=false; shift ;;
        --with-recursive-agent-source) RECURSIVE_AGENT_SOURCE="${2:?--with-recursive-agent-source requires a value}"; shift 2 ;;
        -h|--help) show_help; exit 0 ;;
        *) die "unknown option: $1" ;;
    esac
done

require_command() {
    command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"
}

resolve_layout() {
    if [[ -z "$INSTALL_DIR" ]]; then
        INSTALL_DIR="$HERMES_HOME/ares-agent"
    fi
    INSTALL_DIR="$(python3 -c 'import os, sys; print(os.path.abspath(sys.argv[1]))' "$INSTALL_DIR")"
    HERMES_HOME="$(python3 -c 'import os, sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "$HERMES_HOME")"
}

checkout_source() {
    if [[ -e "$INSTALL_DIR" && ! -d "$INSTALL_DIR/.git" ]]; then
        die "install path exists but is not a Git checkout: $INSTALL_DIR"
    fi

    if [[ -d "$INSTALL_DIR/.git" ]]; then
        if [[ -n "$(git -C "$INSTALL_DIR" status --porcelain)" ]]; then
            die "refusing to update a dirty checkout: $INSTALL_DIR"
        fi
        log "updating Ares checkout at $INSTALL_DIR"
        git -C "$INSTALL_DIR" fetch origin "$BRANCH"
        git -C "$INSTALL_DIR" checkout "$BRANCH"
        git -C "$INSTALL_DIR" pull --ff-only origin "$BRANCH"
    else
        log "cloning Ares from $REPO_URL"
        mkdir -p "$(dirname "$INSTALL_DIR")"
        git clone --branch "$BRANCH" "$REPO_URL" "$INSTALL_DIR"
    fi
}

install_runtime() {
    if [[ "$USE_VENV" == true ]]; then
        require_command uv
        log "creating managed Python environment"
        (cd "$INSTALL_DIR" && uv sync --locked --extra all)
    else
        log "installing into the active Python environment"
        python3 -m pip install -e "$INSTALL_DIR[all]"
    fi
}

install_stable_runtime() {
    log "building the isolated Ares release runtime"
    if [[ "$USE_VENV" == true ]]; then
        ARES_HOME="$HERMES_HOME" ARES_BIN_DIR="$ARES_BIN_DIR" \
            "$INSTALL_DIR/.venv/bin/ares" setup --source "$INSTALL_DIR"
    else
        ARES_HOME="$HERMES_HOME" ARES_BIN_DIR="$ARES_BIN_DIR" \
            python3 -m ares_runtime.local_runtime setup --source "$INSTALL_DIR"
    fi
}

install_recursive_agent_plugin() {
    [[ -n "$RECURSIVE_AGENT_SOURCE" ]] || return 0
    RECURSIVE_AGENT_SOURCE="$(python3 -c 'import os, sys; print(os.path.abspath(sys.argv[1]))' "$RECURSIVE_AGENT_SOURCE")"
    local installer="$RECURSIVE_AGENT_SOURCE/scripts/install-hermes-plugin.sh"
    [[ -x "$installer" ]] || die "Recursive Agent plugin installer not found or not executable: $installer"
    log "installing the standalone Recursive Agent plugin"
    HERMES_HOME="$HERMES_HOME" "$installer"
    log "Recursive Agent daemon remains an explicit operator-managed prerequisite"
}

main() {
    require_command git
    require_command python3
    resolve_layout
    checkout_source
    install_runtime
    install_stable_runtime
    install_recursive_agent_plugin

    log "Ares installed"
    log "launcher: $ARES_BIN_DIR/ares"
    log "data home: $HERMES_HOME"
    log "next: ares setup"
}

main
