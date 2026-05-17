#!/usr/bin/env bash
#
# bootstrap_browser_tools.sh — install agent-browser + Playwright Chromium
# into ~/.hermes/node/ for use by Hermes Agent's browser tools.
#
# Targets the registry-install path: users who got Hermes via
# `uvx --from 'hermes-agent[acp]==X' hermes-acp` don't have a repo clone,
# so the install.sh `npm install`-in-repo flow doesn't apply. This script
# is a self-contained, idempotent slice of install.sh's browser block —
# safe to run from `hermes-acp --setup-browser`, from a fresh terminal,
# or from install.sh itself (it's a no-op when everything is already in place).
#
# Usage:
#   bootstrap_browser_tools.sh           # use defaults
#   bootstrap_browser_tools.sh --yes     # accept the ~400MB Chromium download
#   bootstrap_browser_tools.sh --skip-chromium    # only install Node + agent-browser
#   HERMES_HOME=/custom/path bootstrap_browser_tools.sh
#
# Idempotent: re-running this is safe and fast. Each step checks whether
# the work is already done.

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────

NODE_VERSION="22"
HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
NODE_PREFIX="$HERMES_HOME/node"

SKIP_CHROMIUM=false
ASSUME_YES=false

# ─────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────

if [ -t 1 ]; then
    C_GREEN='\033[0;32m'
    C_YELLOW='\033[0;33m'
    C_BLUE='\033[0;34m'
    C_RED='\033[0;31m'
    C_RESET='\033[0m'
else
    C_GREEN='' ; C_YELLOW='' ; C_BLUE='' ; C_RED='' ; C_RESET=''
fi

log_info()    { printf "${C_BLUE}[*]${C_RESET} %s\n"  "$*"; }
log_success() { printf "${C_GREEN}[✓]${C_RESET} %s\n" "$*"; }
log_warn()    { printf "${C_YELLOW}[!]${C_RESET} %s\n" "$*" >&2; }
log_error()   { printf "${C_RED}[✗]${C_RESET} %s\n"   "$*" >&2; }

# ─────────────────────────────────────────────────────────────────────────
# Arg parsing
# ─────────────────────────────────────────────────────────────────────────

while [ $# -gt 0 ]; do
    case "$1" in
        --skip-chromium) SKIP_CHROMIUM=true ;;
        --yes|-y)        ASSUME_YES=true ;;
        -h|--help)
            cat <<EOF
Bootstrap Hermes Agent browser tools.

Installs Node.js (into ~/.hermes/node/), the agent-browser npm package,
and the Playwright Chromium browser engine.

Options:
  --skip-chromium   Install Node + agent-browser but skip Chromium download
  --yes, -y         Accept the ~400 MB Chromium download without prompting
  -h, --help        Show this help

Environment:
  HERMES_HOME       Override Hermes data dir (default: \$HOME/.hermes)
EOF
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 2
            ;;
    esac
    shift
done

# ─────────────────────────────────────────────────────────────────────────
# OS / arch detection
# ─────────────────────────────────────────────────────────────────────────

OS="unknown"
case "$(uname -s)" in
    Linux*)  OS="linux"  ;;
    Darwin*) OS="macos"  ;;
    *)
        log_error "Unsupported OS: $(uname -s)"
        log_info "Windows users: run scripts/bootstrap_browser_tools.ps1 in PowerShell."
        exit 1
        ;;
esac

NODE_ARCH=""
case "$(uname -m)" in
    x86_64)         NODE_ARCH="x64"    ;;
    aarch64|arm64)  NODE_ARCH="arm64"  ;;
    armv7l)         NODE_ARCH="armv7l" ;;
    *)
        log_error "Unsupported architecture: $(uname -m)"
        exit 1
        ;;
esac

NODE_OS=""
case "$OS" in
    linux) NODE_OS="linux"  ;;
    macos) NODE_OS="darwin" ;;
esac

DISTRO=""
if [ -f /etc/os-release ]; then
    # shellcheck disable=SC1091
    . /etc/os-release
    DISTRO="${ID:-}"
fi

# ─────────────────────────────────────────────────────────────────────────
# Step 1: Node.js
# ─────────────────────────────────────────────────────────────────────────

ensure_node() {
    # Already on PATH and recent enough?
    if command -v node >/dev/null 2>&1; then
        local found_ver major
        found_ver=$(node --version 2>/dev/null)
        major=$(echo "$found_ver" | sed -E 's/^v([0-9]+).*/\1/')
        if [ -n "$major" ] && [ "$major" -ge 20 ]; then
            log_success "Node.js $found_ver found on PATH"
            return 0
        fi
        log_warn "Node.js $found_ver is older than v20 — installing managed Node."
    fi

    if [ -x "$NODE_PREFIX/bin/node" ]; then
        local found_ver
        found_ver=$("$NODE_PREFIX/bin/node" --version 2>/dev/null || echo "?")
        export PATH="$NODE_PREFIX/bin:$PATH"
        log_success "Node.js $found_ver found (Hermes-managed at $NODE_PREFIX)"
        return 0
    fi

    log_info "Installing Node.js $NODE_VERSION LTS into $NODE_PREFIX ..."

    local index_url="https://nodejs.org/dist/latest-v${NODE_VERSION}.x/"
    local tarball_name
    tarball_name=$(curl -fsSL "$index_url" \
        | grep -oE "node-v${NODE_VERSION}\.[0-9]+\.[0-9]+-${NODE_OS}-${NODE_ARCH}\.tar\.xz" \
        | head -1)

    if [ -z "$tarball_name" ]; then
        tarball_name=$(curl -fsSL "$index_url" \
            | grep -oE "node-v${NODE_VERSION}\.[0-9]+\.[0-9]+-${NODE_OS}-${NODE_ARCH}\.tar\.gz" \
            | head -1)
    fi

    if [ -z "$tarball_name" ]; then
        log_error "Could not locate Node.js $NODE_VERSION tarball for $NODE_OS-$NODE_ARCH"
        log_info "Install Node 20+ manually: https://nodejs.org/en/download/"
        return 1
    fi

    local tmp_dir
    tmp_dir=$(mktemp -d)
    trap 'rm -rf "$tmp_dir"' RETURN

    log_info "Downloading $tarball_name ..."
    if ! curl -fsSL "${index_url}${tarball_name}" -o "$tmp_dir/$tarball_name"; then
        log_error "Node.js download failed"
        return 1
    fi

    if [[ "$tarball_name" == *.tar.xz ]]; then
        tar xf "$tmp_dir/$tarball_name" -C "$tmp_dir"
    else
        tar xzf "$tmp_dir/$tarball_name" -C "$tmp_dir"
    fi

    local extracted_dir
    extracted_dir=$(ls -d "$tmp_dir"/node-v* 2>/dev/null | head -1)
    if [ ! -d "$extracted_dir" ]; then
        log_error "Node.js extraction failed"
        return 1
    fi

    mkdir -p "$HERMES_HOME"
    rm -rf "$NODE_PREFIX"
    mv "$extracted_dir" "$NODE_PREFIX"

    export PATH="$NODE_PREFIX/bin:$PATH"

    local installed_ver
    installed_ver=$("$NODE_PREFIX/bin/node" --version 2>/dev/null || echo "?")
    log_success "Node.js $installed_ver installed to $NODE_PREFIX"
}

# ─────────────────────────────────────────────────────────────────────────
# Step 2: agent-browser + @askjo/camofox-browser via global npm install
# ─────────────────────────────────────────────────────────────────────────

ensure_agent_browser() {
    if ! command -v npm >/dev/null 2>&1; then
        log_error "npm not on PATH after Node install — aborting"
        return 1
    fi

    # _find_agent_browser() in tools/browser_tool.py walks ~/.hermes/node/bin
    # plus a few standard prefixes, so installing globally into the managed
    # Node prefix is enough — no PATH manipulation needed from the agent side.
    if [ -x "$NODE_PREFIX/bin/agent-browser" ] || command -v agent-browser >/dev/null 2>&1; then
        log_success "agent-browser already installed"
        return 0
    fi

    # When the system's `npm` resolves to a root-owned prefix (e.g.
    # /usr/lib/node_modules), `npm install -g` fails with EACCES without
    # sudo. Force the prefix to the user-writable Hermes-managed Node
    # directory so we never need sudo and the agent can always find the
    # result. If we installed Node ourselves above, this is a no-op
    # (managed Node already uses $NODE_PREFIX). If the user has system
    # Node, we still drop agent-browser under $NODE_PREFIX/bin/ — which
    # is exactly where _browser_candidate_path_dirs() looks first.
    mkdir -p "$NODE_PREFIX"

    log_info "Installing agent-browser (npm, prefix=$NODE_PREFIX)..."
    if ! npm install -g --prefix "$NODE_PREFIX" --silent \
            agent-browser@^0.26.0 \
            "@askjo/camofox-browser@^1.5.2"; then
        log_error "npm install -g agent-browser failed"
        return 1
    fi

    # macOS/Linux global installs place the shim into $NODE_PREFIX/bin/.
    # Add it to PATH for any subsequent steps (npx playwright).
    export PATH="$NODE_PREFIX/bin:$PATH"

    log_success "agent-browser installed to $NODE_PREFIX/bin/"
}

# ─────────────────────────────────────────────────────────────────────────
# Step 3: Playwright Chromium
# ─────────────────────────────────────────────────────────────────────────

confirm_chromium_download() {
    if [ "$ASSUME_YES" = true ]; then return 0; fi
    if [ ! -t 0 ]; then
        log_warn "Non-interactive shell — skipping Chromium prompt."
        log_info "Re-run with --yes to install Chromium (~400 MB download)."
        return 1
    fi
    printf "Install Playwright Chromium (~400 MB download)? [y/N] "
    local reply=""
    read -r reply || reply=""
    case "$reply" in
        y|Y|yes|YES) return 0 ;;
        *) return 1 ;;
    esac
}

# Detect a usable system Chrome/Chromium. agent-browser's Chrome engine can
# use it instead of downloading Playwright's bundled Chromium, saving the
# download cost. Returns the path or empty string.
find_system_browser() {
    local candidate
    for candidate in google-chrome google-chrome-stable chromium chromium-browser chrome; do
        if command -v "$candidate" >/dev/null 2>&1; then
            command -v "$candidate"
            return 0
        fi
    done
    # macOS app-bundle locations
    if [ "$OS" = "macos" ]; then
        for candidate in \
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
            "/Applications/Chromium.app/Contents/MacOS/Chromium" ; do
            if [ -x "$candidate" ]; then
                echo "$candidate"
                return 0
            fi
        done
    fi
    return 1
}

write_browser_env() {
    local browser_path="$1"
    local env_file="$HERMES_HOME/.env"
    mkdir -p "$HERMES_HOME"
    if [ -f "$env_file" ] && grep -q "^AGENT_BROWSER_EXECUTABLE_PATH=" "$env_file"; then
        return 0
    fi
    {
        echo ""
        echo "# Hermes Agent browser tools — use the system Chrome/Chromium binary."
        echo "AGENT_BROWSER_EXECUTABLE_PATH=$browser_path"
    } >> "$env_file"
    log_success "Configured browser tools to use $browser_path"
}

ensure_chromium() {
    if [ "$SKIP_CHROMIUM" = true ]; then
        log_info "Skipping Chromium install (--skip-chromium)"
        return 0
    fi

    local system_browser
    system_browser="$(find_system_browser 2>/dev/null || true)"
    if [ -n "$system_browser" ]; then
        log_success "Found system browser: $system_browser"
        log_info "Skipping Playwright Chromium download; agent-browser will use it."
        write_browser_env "$system_browser"
        return 0
    fi

    if ! confirm_chromium_download; then
        log_info "Chromium install skipped. Browser tools will only work if you"
        log_info "set AGENT_BROWSER_EXECUTABLE_PATH or install Chromium later."
        return 0
    fi

    if ! command -v npx >/dev/null 2>&1; then
        log_error "npx not on PATH — cannot install Playwright Chromium"
        return 1
    fi

    log_info "Installing Playwright Chromium (~400 MB) ..."

    # On apt-based distros, --with-deps requires sudo. Try non-interactively
    # only — never prompt — and fall back to the bare browser-only install.
    local installed=false
    if [ "$OS" = "linux" ]; then
        case "$DISTRO" in
            ubuntu|debian|raspbian|pop|linuxmint|elementary|zorin|kali|parrot)
                if [ "$(id -u)" -eq 0 ] || (command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null); then
                    log_info "Installing system deps with --with-deps (sudo available)"
                    if npx --yes playwright install --with-deps chromium; then
                        installed=true
                    fi
                else
                    log_warn "sudo not available non-interactively — installing Chromium without system deps."
                    log_info "If browser tools fail to launch, an administrator should run:"
                    log_info "  sudo npx playwright install-deps chromium"
                fi
                ;;
            arch|manjaro|cachyos|endeavouros|garuda)
                log_info "Arch-family system dependencies are not auto-installed."
                log_info "If launch fails, run: sudo pacman -S nss atk at-spi2-core cups libdrm libxkbcommon mesa pango cairo alsa-lib"
                ;;
            fedora|rhel|centos|rocky|alma)
                log_info "Fedora/RHEL system dependencies are not auto-installed."
                log_info "If launch fails, run: sudo dnf install nss atk at-spi2-core cups-libs libdrm libxkbcommon mesa-libgbm pango cairo alsa-lib"
                ;;
            opensuse*|sles)
                log_info "openSUSE system dependencies are not auto-installed."
                ;;
        esac
    fi

    if [ "$installed" = false ]; then
        if npx --yes playwright install chromium; then
            installed=true
        fi
    fi

    if [ "$installed" = true ]; then
        log_success "Playwright Chromium installed"
    else
        log_error "Playwright Chromium install failed"
        log_info "Try again later: npx --yes playwright install chromium"
        return 1
    fi
}

# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────

main() {
    log_info "Hermes Agent: bootstrapping browser tools"
    log_info "  HERMES_HOME = $HERMES_HOME"
    log_info "  OS / arch   = $NODE_OS-$NODE_ARCH ${DISTRO:+($DISTRO)}"

    ensure_node
    ensure_agent_browser
    ensure_chromium

    log_success "Browser tools setup complete."
    log_info "Hermes Agent will pick up agent-browser from $NODE_PREFIX/bin/ on next launch."
}

main
