#!/bin/bash
# ============================================================================
# Hermes Agent Installer
# ============================================================================
# Installation script for Linux, macOS, and Android/Termux.
# Uses uv for desktop/server installs and Python's stdlib venv + pip on Termux.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
#
# Or with options:
#   curl -fsSL ... | bash -s -- --no-venv --skip-setup
#
# ============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Configuration
REPO_URL_SSH="git@github.com:NousResearch/hermes-agent.git"
REPO_URL_HTTPS="https://github.com/NousResearch/hermes-agent.git"
HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
INSTALL_DIR="${HERMES_INSTALL_DIR:-$HERMES_HOME/hermes-agent}"
PYTHON_VERSION="3.11"
NODE_VERSION="22"

# Options
USE_VENV=true
RUN_SETUP=true
BRANCH="main"

# Detect non-interactive mode (e.g. curl | bash)
# When stdin is not a terminal, read -p will fail with EOF,
# causing set -e to silently abort the entire script.
if [ -t 0 ]; then
    IS_INTERACTIVE=true
else
    IS_INTERACTIVE=false
fi

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-venv)
            USE_VENV=false
            shift
            ;;
        --skip-setup)
            RUN_SETUP=false
            shift
            ;;
        --branch)
            BRANCH="$2"
            shift 2
            ;;
        --dir)
            INSTALL_DIR="$2"
            shift 2
            ;;
        --hermes-home)
            HERMES_HOME="$2"
            shift 2
            ;;
        -h|--help)
            echo "Hermes Agent Installer"
            echo ""
            echo "Usage: install.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --no-venv      Don't create virtual environment"
            echo "  --skip-setup   Skip interactive setup wizard"
            echo "  --branch NAME  Git branch to install (default: main)"
            echo "  --dir PATH     Installation directory (default: ~/.hermes/hermes-agent)"
            echo "  --hermes-home PATH  Data directory (default: ~/.hermes, or \$HERMES_HOME)"
            echo "  -h, --help     Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ============================================================================
# Helper functions
# ============================================================================

print_banner() {
    echo ""
    echo -e "${MAGENTA}${BOLD}"
    echo "┌─────────────────────────────────────────────────────────┐"
    echo "│             ⚕ Hermes Agent Installer                    │"
    echo "├─────────────────────────────────────────────────────────┤"
    echo "│  An open source AI agent by Nous Research.              │"
    echo "└─────────────────────────────────────────────────────────┘"
    echo -e "${NC}"
}

log_info() {
    echo -e "${CYAN}→${NC} $1"
}

log_success() {
    echo -e "${GREEN}✓${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}⚠${NC} $1"
}

log_error() {
    echo -e "${RED}✗${NC} $1"
}

is_termux() {
    [ -n "${TERMUX_VERSION:-}" ] || [[ "${PREFIX:-}" == *"com.termux/files/usr"* ]]
}

get_command_link_dir() {
    if is_termux && [ -n "${PREFIX:-}" ]; then
        echo "$PREFIX/bin"
    else
        echo "$HOME/.local/bin"
    fi
}

get_command_link_display_dir() {
    if is_termux && [ -n "${PREFIX:-}" ]; then
        echo '$PREFIX/bin'
    else
        echo '~/.local/bin'
    fi
}

get_hermes_command_path() {
    local link_dir
    link_dir="$(get_command_link_dir)"
    if [ -x "$link_dir/hermes" ]; then
        echo "$link_dir/hermes"
    else
        echo "hermes"
    fi
}

# ============================================================================
# System detection
# ============================================================================

detect_os() {
    case "$(uname -s)" in
        Linux*)
            if is_termux; then
                OS="android"
                DISTRO="termux"
            else
                OS="linux"
                if [ -f /etc/os-release ]; then
                    . /etc/os-release
                    DISTRO="$ID"
                else
                    DISTRO="unknown"
                fi
            fi
            ;;
        Darwin*)
            OS="macos"
            DISTRO="macos"
            ;;
        CYGWIN*|MINGW*|MSYS*)
            OS="windows"
            DISTRO="windows"
            log_error "Windows detected. Please use the PowerShell installer:"
            log_info "  irm https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.ps1 | iex"
            exit 1
            ;;
        *)
            OS="unknown"
            DISTRO="unknown"
            log_warn "Unknown operating system"
            ;;
    esac

    log_success "Detected: $OS ($DISTRO)"
}

# ============================================================================
# Dependency checks
# ============================================================================

install_uv() {
    if [ "$DISTRO" = "termux" ]; then
        log_info "Termux detected — using Python's stdlib venv + pip instead of uv"
        UV_CMD=""
        return 0
    fi

    log_info "Checking for uv package manager..."

    # Check common locations for uv
    if command -v uv &> /dev/null; then
        UV_CMD="uv"
        UV_VERSION=$($UV_CMD --version 2>/dev/null)
        log_success "uv found ($UV_VERSION)"
        return 0
    fi

    # Check ~/.local/bin (default uv install location) even if not on PATH yet
    if [ -x "$HOME/.local/bin/uv" ]; then
        UV_CMD="$HOME/.local/bin/uv"
        UV_VERSION=$($UV_CMD --version 2>/dev/null)
        log_success "uv found at ~/.local/bin ($UV_VERSION)"
        return 0
    fi

    # Check ~/.cargo/bin (alternative uv install location)
    if [ -x "$HOME/.cargo/bin/uv" ]; then
        UV_CMD="$HOME/.cargo/bin/uv"
        UV_VERSION=$($UV_CMD --version 2>/dev/null)
        log_success "uv found at ~/.cargo/bin ($UV_VERSION)"
        return 0
    fi

    # Install uv
    log_info "Installing uv (fast Python package manager)..."
    if curl -LsSf https://astral.sh/uv/install.sh | sh 2>/dev/null; then
        # uv installs to ~/.local/bin by default
        if [ -x "$HOME/.local/bin/uv" ]; then
            UV_CMD="$HOME/.local/bin/uv"
        elif [ -x "$HOME/.cargo/bin/uv" ]; then
            UV_CMD="$HOME/.cargo/bin/uv"
        elif command -v uv &> /dev/null; then
            UV_CMD="uv"
        else
            log_error "uv installed but not found on PATH"
            log_info "Try adding ~/.local/bin to your PATH and re-running"
            exit 1
        fi
        UV_VERSION=$($UV_CMD --version 2>/dev/null)
        log_success "uv installed ($UV_VERSION)"
    else
        log_error "Failed to install uv"
        log_info "Install manually: https://docs.astral.sh/uv/getting-started/installation/"
        exit 1
    fi
}

check_python() {
    if [ "$DISTRO" = "termux" ]; then
        log_info "Checking Termux Python..."
        if command -v python >/dev/null 2>&1; then
            PYTHON_PATH="$(command -v python)"
            if "$PYTHON_PATH" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; then
                PYTHON_FOUND_VERSION=$($PYTHON_PATH --version 2>/dev/null)
                log_success "Python found: $PYTHON_FOUND_VERSION"
                return 0
            fi
        fi

        log_info "Installing Python via pkg..."
        pkg install -y python >/dev/null
        PYTHON_PATH="$(command -v python)"
        PYTHON_FOUND_VERSION=$($PYTHON_PATH --version 2>/dev/null)
        log_success "Python installed: $PYTHON_FOUND_VERSION"
        return 0
    fi

    log_info "Checking Python $PYTHON_VERSION..."

    # Let uv handle Python — it can download and manage Python versions
    # First check if a suitable Python is already available
    if $UV_CMD python find "$PYTHON_VERSION" &> /dev/null; then
        PYTHON_PATH=$($UV_CMD python find "$PYTHON_VERSION")
        PYTHON_FOUND_VERSION=$($PYTHON_PATH --version 2>/dev/null)
        log_success "Python found: $PYTHON_FOUND_VERSION"
        return 0
    fi

    # Python not found — use uv to install it (no sudo needed!)
    log_info "Python $PYTHON_VERSION not found, installing via uv..."
    if $UV_CMD python install "$PYTHON_VERSION"; then
        PYTHON_PATH=$($UV_CMD python find "$PYTHON_VERSION")
        PYTHON_FOUND_VERSION=$($PYTHON_PATH --version 2>/dev/null)
        log_success "Python installed: $PYTHON_FOUND_VERSION"
    else
        log_error "Failed to install Python $PYTHON_VERSION"
        log_info "Install Python $PYTHON_VERSION manually, then re-run this script"
        exit 1
    fi
}

check_git() {
    log_info "Checking Git..."

    if command -v git &> /dev/null; then
        GIT_VERSION=$(git --version | awk '{print $3}')
        log_success "Git $GIT_VERSION found"
        return 0
    fi

    log_error "Git not found"

    if [ "$DISTRO" = "termux" ]; then
        log_info "Installing Git via pkg..."
        pkg install -y git >/dev/null
        if command -v git >/dev/null 2>&1; then
            GIT_VERSION=$(git --version | awk '{print $3}')
            log_success "Git $GIT_VERSION installed"
            return 0
        fi
    fi

    log_info "Please install Git:"

    case "$OS" in
        linux)
            case "$DISTRO" in
                ubuntu|debian)
                    log_info "  sudo apt update && sudo apt install git"
                    ;;
                fedora)
                    log_info "  sudo dnf install git"
                    ;;
                arch)
                    log_info "  sudo pacman -S git"
                    ;;
                *)
                    log_info "  Use your package manager to install git"
                    ;;
            esac
            ;;
        android)
            log_info "  pkg install git"
            ;;
        macos)
            log_info "  xcode-select --install"
            log_info "  Or: brew install git"
            ;;
    esac

    exit 1
}

check_node() {
    log_info "Checking Node.js (for browser tools)..."

    if command -v node &> /dev/null; then
        local found_ver=$(node --version)
        log_success "Node.js $found_ver found"
        HAS_NODE=true
        return 0
    fi

    # Check our own managed install from a previous run
    if [ -x "$HERMES_HOME/node/bin/node" ]; then
        export PATH="$HERMES_HOME/node/bin:$PATH"
        local found_ver=$("$HERMES_HOME/node/bin/node" --version)
        log_success "Node.js $found_ver found (Hermes-managed)"
        HAS_NODE=true
        return 0
    fi

    if [ "$DISTRO" = "termux" ]; then
        log_info "Node.js not found — installing Node.js via pkg..."
    else
        log_info "Node.js not found — installing Node.js $NODE_VERSION LTS..."
    fi
    install_node
}

install_node() {
    if [ "$DISTRO" = "termux" ]; then
        log_info "Installing Node.js via pkg..."
        if pkg install -y nodejs >/dev/null; then
            local installed_ver
            installed_ver=$(node --version 2>/dev/null)
            log_success "Node.js $installed_ver installed via pkg"
            HAS_NODE=true
        else
            log_warn "Failed to install Node.js via pkg"
            HAS_NODE=false
        fi
        return 0
    fi

    local arch=$(uname -m)
    local node_arch
    case "$arch" in
        x86_64)        node_arch="x64"    ;;
        aarch64|arm64) node_arch="arm64"  ;;
        armv7l)        node_arch="armv7l" ;;
        *)
            log_warn "Unsupported architecture ($arch) for Node.js auto-install"
            log_info "Install manually: https://nodejs.org/en/download/"
            HAS_NODE=false
            return 0
            ;;
    esac

    local node_os
    case "$OS" in
        linux) node_os="linux"  ;;
        macos) node_os="darwin" ;;
        *)
            log_warn "Unsupported OS for Node.js auto-install"
            HAS_NODE=false
            return 0
            ;;
    esac

    # Resolve the latest v22.x.x tarball name from the index page
    local index_url="https://nodejs.org/dist/latest-v${NODE_VERSION}.x/"
    local tarball_name
    tarball_name=$(curl -fsSL "$index_url" \
        | grep -oE "node-v${NODE_VERSION}\.[0-9]+\.[0-9]+-${node_os}-${node_arch}\.tar\.xz" \
        | head -1)

    # Fallback to .tar.gz if .tar.xz not available
    if [ -z "$tarball_name" ]; then
        tarball_name=$(curl -fsSL "$index_url" \
            | grep -oE "node-v${NODE_VERSION}\.[0-9]+\.[0-9]+-${node_os}-${node_arch}\.tar\.gz" \
            | head -1)
    fi

    if [ -z "$tarball_name" ]; then
        log_warn "Could not find Node.js $NODE_VERSION binary for $node_os-$node_arch"
        log_info "Install manually: https://nodejs.org/en/download/"
        HAS_NODE=false
        return 0
    fi

    local download_url="${index_url}${tarball_name}"
    local tmp_dir
    tmp_dir=$(mktemp -d)

    log_info "Downloading $tarball_name..."
    if ! curl -fsSL "$download_url" -o "$tmp_dir/$tarball_name"; then
        log_warn "Download failed"
        rm -rf "$tmp_dir"
        HAS_NODE=false
        return 0
    fi

    log_info "Extracting to ~/.hermes/node/..."
    if [[ "$tarball_name" == *.tar.xz ]]; then
        tar xf "$tmp_dir/$tarball_name" -C "$tmp_dir"
    else
        tar xzf "$tmp_dir/$tarball_name" -C "$tmp_dir"
    fi

    local extracted_dir
    extracted_dir=$(ls -d "$tmp_dir"/node-v* 2>/dev/null | head -1)

    if [ ! -d "$extracted_dir" ]; then
        log_warn "Extraction failed"
        rm -rf "$tmp_dir"
        HAS_NODE=false
        return 0
    fi

    # Place into ~/.hermes/node/ and symlink binaries to ~/.local/bin/
    rm -rf "$HERMES_HOME/node"
    mkdir -p "$HERMES_HOME"
    mv "$extracted_dir" "$HERMES_HOME/node"
    rm -rf "$tmp_dir"

    mkdir -p "$HOME/.local/bin"
    ln -sf "$HERMES_HOME/node/bin/node" "$HOME/.local/bin/node"
    ln -sf "$HERMES_HOME/node/bin/npm"  "$HOME/.local/bin/npm"
    ln -sf "$HERMES_HOME/node/bin/npx"  "$HOME/.local/bin/npx"

    export PATH="$HERMES_HOME/node/bin:$PATH"

    local installed_ver
    installed_ver=$("$HERMES_HOME/node/bin/node" --version 2>/dev/null)
    log_success "Node.js $installed_ver installed to ~/.hermes/node/"
    HAS_NODE=true
}

install_system_packages() {
    # Detect what's missing
    HAS_RIPGREP=false
    HAS_FFMPEG=false
    local need_ripgrep=false
    local need_ffmpeg=false

    log_info "Checking ripgrep (fast file search)..."
    if command -v rg &> /dev/null; then
        log_success "$(rg --version | head -1) found"
        HAS_RIPGREP=true
    else
        need_ripgrep=true
    fi

    log_info "Checking ffmpeg (TTS voice messages)..."
    if command -v ffmpeg &> /dev/null; then
        local ffmpeg_ver=$(ffmpeg -version 2>/dev/null | head -1 | awk '{print $3}')
        log_success "ffmpeg $ffmpeg_ver found"
        HAS_FFMPEG=true
    else
        need_ffmpeg=true
    fi

    # Termux always needs the Android build toolchain for the tested pip path,
    # even when ripgrep/ffmpeg are already present.
    if [ "$DISTRO" = "termux" ]; then
        local termux_pkgs=(clang rust make pkg-config libffi openssl)
        if [ "$need_ripgrep" = true ]; then
            termux_pkgs+=("ripgrep")
        fi
        if [ "$need_ffmpeg" = true ]; then
            termux_pkgs+=("ffmpeg")
        fi

        log_info "Installing Termux packages: ${termux_pkgs[*]}"
        if pkg install -y "${termux_pkgs[@]}" >/dev/null; then
            [ "$need_ripgrep" = true ] && HAS_RIPGREP=true && log_success "ripgrep installed"
            [ "$need_ffmpeg" = true ]  && HAS_FFMPEG=true  && log_success "ffmpeg installed"
            log_success "Termux build dependencies installed"
            return 0
        fi

        log_warn "Could not auto-install all Termux packages"
        log_info "Install manually: pkg install ${termux_pkgs[*]}"
        return 0
    fi

    # Nothing to install — done
    if [ "$need_ripgrep" = false ] && [ "$need_ffmpeg" = false ]; then
        return 0
    fi

    # Build a human-readable description + package list
    local desc_parts=()
    local pkgs=()
    if [ "$need_ripgrep" = true ]; then
        desc_parts+=("ripgrep for faster file search")
        pkgs+=("ripgrep")
    fi
    if [ "$need_ffmpeg" = true ]; then
        desc_parts+=("ffmpeg for TTS voice messages")
        pkgs+=("ffmpeg")
    fi
    local description
    description=$(IFS=" and "; echo "${desc_parts[*]}")

    # ── macOS: brew ──
    if [ "$OS" = "macos" ]; then
        if command -v brew &> /dev/null; then
            log_info "Installing ${pkgs[*]} via Homebrew..."
            if brew install "${pkgs[@]}"; then
                [ "$need_ripgrep" = true ] && HAS_RIPGREP=true && log_success "ripgrep installed"
                [ "$need_ffmpeg" = true ]  && HAS_FFMPEG=true  && log_success "ffmpeg installed"
                return 0
            fi
        fi
        log_warn "Could not auto-install (brew not found or install failed)"
        log_info "Install manually: brew install ${pkgs[*]}"
        return 0
    fi

    # ── Linux: resolve package manager command ──
    local pkg_install=""
    case "$DISTRO" in
        ubuntu|debian) pkg_install="apt install -y"   ;;
        fedora)        pkg_install="dnf install -y"   ;;
        arch)          pkg_install="pacman -S --noconfirm" ;;
    esac

    if [ -n "$pkg_install" ]; then
        local install_cmd="$pkg_install ${pkgs[*]}"

        # Prevent needrestart/whiptail dialogs from blocking non-interactive installs
        case "$DISTRO" in
            ubuntu|debian) export DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=a ;;
        esac

        # Already root — just install
        if [ "$(id -u)" -eq 0 ]; then
            log_info "Installing ${pkgs[*]}..."
            if $install_cmd; then
                [ "$need_ripgrep" = true ] && HAS_RIPGREP=true && log_success "ripgrep installed"
                [ "$need_ffmpeg" = true ]  && HAS_FFMPEG=true  && log_success "ffmpeg installed"
                return 0
            fi
        # Passwordless sudo — just install
        elif command -v sudo &> /dev/null && sudo -n true 2>/dev/null; then
            log_info "Installing ${pkgs[*]}..."
            if sudo DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=a $install_cmd; then
                [ "$need_ripgrep" = true ] && HAS_RIPGREP=true && log_success "ripgrep installed"
                [ "$need_ffmpeg" = true ]  && HAS_FFMPEG=true  && log_success "ffmpeg installed"
                return 0
            fi
        # sudo needs password — ask once for everything
        elif command -v sudo &> /dev/null; then
            if [ "$IS_INTERACTIVE" = true ]; then
                echo ""
                log_info "sudo is needed ONLY to install optional system packages (${pkgs[*]}) via your package manager."
                log_info "Hermes Agent itself does not require or retain root access."
                read -p "Install ${description}? (requires sudo) [y/N] " -n 1 -r
                echo
                if [[ $REPLY =~ ^[Yy]$ ]]; then
                    if sudo DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=a $install_cmd; then
                        [ "$need_ripgrep" = true ] && HAS_RIPGREP=true && log_success "ripgrep installed"
                        [ "$need_ffmpeg" = true ]  && HAS_FFMPEG=true  && log_success "ffmpeg installed"
                        return 0
                    fi
                fi
            elif [ -e /dev/tty ]; then
                # Non-interactive (e.g. curl | bash) but a terminal is available.
                # Read the prompt from /dev/tty (same approach the setup wizard uses).
                echo ""
                log_info "sudo is needed ONLY to install optional system packages (${pkgs[*]}) via your package manager."
                log_info "Hermes Agent itself does not require or retain root access."
                read -p "Install ${description}? [Y/n] " -n 1 -r < /dev/tty
                echo
                if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
                    if sudo DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=a $install_cmd < /dev/tty; then
                        [ "$need_ripgrep" = true ] && HAS_RIPGREP=true && log_success "ripgrep installed"
                        [ "$need_ffmpeg" = true ]  && HAS_FFMPEG=true  && log_success "ffmpeg installed"
                        return 0
                    fi
                fi
            else
                log_warn "Non-interactive mode and no terminal available — cannot install system packages"
                log_info "Install manually after setup completes: sudo $install_cmd"
            fi
        fi
    fi

    # ── Fallback for ripgrep: cargo ──
    if [ "$need_ripgrep" = true ] && [ "$HAS_RIPGREP" = false ]; then
        if command -v cargo &> /dev/null; then
            log_info "Trying cargo install ripgrep (no sudo needed)..."
            if cargo install ripgrep; then
                log_success "ripgrep installed via cargo"
                HAS_RIPGREP=true
            fi
        fi
    fi

    # ── Show manual instructions for anything still missing ──
    if [ "$HAS_RIPGREP" = false ] && [ "$need_ripgrep" = true ]; then
        log_warn "ripgrep not installed (file search will use grep fallback)"
        show_manual_install_hint "ripgrep"
    fi
    if [ "$HAS_FFMPEG" = false ] && [ "$need_ffmpeg" = true ]; then
        log_warn "ffmpeg not installed (TTS voice messages will be limited)"
        show_manual_install_hint "ffmpeg"
    fi
}

show_manual_install_hint() {
    local pkg="$1"
    log_info "To install $pkg manually:"
    case "$OS" in
        linux)
            case "$DISTRO" in
                ubuntu|debian) log_info "  sudo apt install $pkg" ;;
                fedora)        log_info "  sudo dnf install $pkg" ;;
                arch)          log_info "  sudo pacman -S $pkg"   ;;
                *)             log_info "  Use your package manager or visit the project homepage" ;;
            esac
            ;;
        android)
            log_info "  pkg install $pkg"
            ;;
        macos) log_info "  brew install $pkg" ;;
    esac
}

# ============================================================================
# Installation
# ============================================================================

clone_repo() {
    log_info "Installing to $INSTALL_DIR..."

    if [ -d "$INSTALL_DIR" ]; then
        if [ -d "$INSTALL_DIR/.git" ]; then
            log_info "Existing installation found, updating..."
            cd "$INSTALL_DIR"

            local autostash_ref=""
            if [ -n "$(git status --porcelain)" ]; then
                local stash_name
                stash_name="hermes-install-autostash-$(date -u +%Y%m%d-%H%M%S)"
                log_info "Local changes detected, stashing before update..."
                git stash push --include-untracked -m "$stash_name"
                autostash_ref="$(git rev-parse --verify refs/stash)"
            fi

            git fetch origin
            git checkout "$BRANCH"
            git pull --ff-only origin "$BRANCH"

            if [ -n "$autostash_ref" ]; then
                local restore_now="yes"
                if [ -t 0 ] && [ -t 1 ]; then
                    echo
                    log_warn "Local changes were stashed before updating."
                    log_warn "Restoring them may reapply local customizations onto the updated codebase."
                    printf "Restore local changes now? [Y/n] "
                    read -r restore_answer
                    case "$restore_answer" in
                        ""|y|Y|yes|YES|Yes) restore_now="yes" ;;
                        *) restore_now="no" ;;
                    esac
                fi

                if [ "$restore_now" = "yes" ]; then
                    log_info "Restoring local changes..."
                    if git stash apply "$autostash_ref"; then
                        git stash drop "$autostash_ref" >/dev/null
                        log_warn "Local changes were restored on top of the updated codebase."
                        log_warn "Review git diff / git status if Hermes behaves unexpectedly."
                    else
                        log_error "Update succeeded, but restoring local changes failed. Your changes are still preserved in git stash."
                        log_info "Resolve manually with: git stash apply $autostash_ref"
                        exit 1
                    fi
                else
                    log_info "Skipped restoring local changes."
                    log_info "Your changes are still preserved in git stash."
                    log_info "Restore manually with: git stash apply $autostash_ref"
                fi
            fi
        else
            log_error "Directory exists but is not a git repository: $INSTALL_DIR"
            log_info "Remove it or choose a different directory with --dir"
            exit 1
        fi
    else
        # Try SSH first (for private repo access), fall back to HTTPS
        # GIT_SSH_COMMAND disables interactive prompts and sets a short timeout
        # so SSH fails fast instead of hanging when no key is configured.
        log_info "Trying SSH clone..."
        if GIT_SSH_COMMAND="ssh -o BatchMode=yes -o ConnectTimeout=5" \
           git clone --branch "$BRANCH" "$REPO_URL_SSH" "$INSTALL_DIR" 2>/dev/null; then
            log_success "Cloned via SSH"
        else
            rm -rf "$INSTALL_DIR" 2>/dev/null  # Clean up partial SSH clone
            log_info "SSH failed, trying HTTPS..."
            if git clone --branch "$BRANCH" "$REPO_URL_HTTPS" "$INSTALL_DIR"; then
                log_success "Cloned via HTTPS"
            else
                log_error "Failed to clone repository"
                exit 1
            fi
        fi
    fi

    cd "$INSTALL_DIR"

    log_success "Repository ready"
}

setup_venv() {
    if [ "$USE_VENV" = false ]; then
        log_info "Skipping virtual environment (--no-venv)"
        return 0
    fi

    if [ "$DISTRO" = "termux" ]; then
        log_info "Creating virtual environment with Termux Python..."

        if [ -d "venv" ]; then
            log_info "Virtual environment already exists, recreating..."
            rm -rf venv
        fi

        "$PYTHON_PATH" -m venv venv
        log_success "Virtual environment ready ($(./venv/bin/python --version 2>/dev/null))"
        return 0
    fi

    log_info "Creating virtual environment with Python $PYTHON_VERSION..."

    if [ -d "venv" ]; then
        log_info "Virtual environment already exists, recreating..."
        rm -rf venv
    fi

    # uv creates the venv and pins the Python version in one step
    $UV_CMD venv venv --python "$PYTHON_VERSION"

    log_success "Virtual environment ready (Python $PYTHON_VERSION)"
}

install_deps() {
    log_info "Installing dependencies..."

    if [ "$DISTRO" = "termux" ]; then
        if [ "$USE_VENV" = true ]; then
            export VIRTUAL_ENV="$INSTALL_DIR/venv"
            PIP_PYTHON="$INSTALL_DIR/venv/bin/python"
        else
            PIP_PYTHON="$PYTHON_PATH"
        fi

        if [ -z "${ANDROID_API_LEVEL:-}" ]; then
            ANDROID_API_LEVEL="$(getprop ro.build.version.sdk 2>/dev/null || true)"
            if [ -z "$ANDROID_API_LEVEL" ]; then
                ANDROID_API_LEVEL=24
            fi
            export ANDROID_API_LEVEL
            log_info "Using ANDROID_API_LEVEL=$ANDROID_API_LEVEL for Android wheel builds"
        fi

        "$PIP_PYTHON" -m pip install --upgrade pip setuptools wheel >/dev/null
        if ! "$PIP_PYTHON" -m pip install -e '.[termux]' -c constraints-termux.txt; then
            log_warn "Termux feature install (.[termux]) failed, trying base install..."
            if ! "$PIP_PYTHON" -m pip install -e '.' -c constraints-termux.txt; then
                log_error "Package installation failed on Termux."
                log_info "Ensure these packages are installed: pkg install clang rust make pkg-config libffi openssl"
                log_info "Then re-run: cd $INSTALL_DIR && python -m pip install -e '.[termux]' -c constraints-termux.txt"
                exit 1
            fi
        fi

        log_success "Main package installed"
        log_info "Termux note: browser/WhatsApp tooling is not installed by default; see the Termux guide for optional follow-up steps."

        if [ -d "tinker-atropos" ] && [ -f "tinker-atropos/pyproject.toml" ]; then
            log_info "tinker-atropos submodule found — skipping install (optional, for RL training)"
            log_info "  To install later: $PIP_PYTHON -m pip install -e \"./tinker-atropos\""
        fi

        log_success "All dependencies installed"
        return 0
    fi

    if [ "$USE_VENV" = true ]; then
        # Tell uv to install into our venv (no need to activate)
        export VIRTUAL_ENV="$INSTALL_DIR/venv"
    fi

    # On Debian/Ubuntu (including WSL), some Python packages need build tools.
    # Check and offer to install them if missing.
    if [ "$DISTRO" = "ubuntu" ] || [ "$DISTRO" = "debian" ]; then
        local need_build_tools=false
        for pkg in gcc python3-dev libffi-dev; do
            if ! dpkg -s "$pkg" &>/dev/null; then
                need_build_tools=true
                break
            fi
        done
        if [ "$need_build_tools" = true ]; then
            log_info "Some build tools may be needed for Python packages..."
            if command -v sudo &> /dev/null; then
                if sudo -n true 2>/dev/null; then
                    sudo DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=a apt-get update -qq && sudo DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=a apt-get install -y -qq build-essential python3-dev libffi-dev >/dev/null 2>&1 || true
                    log_success "Build tools installed"
                else
                    log_info "sudo is needed ONLY to install build tools (build-essential, python3-dev, libffi-dev) via apt."
                    log_info "Hermes Agent itself does not require or retain root access."
                    read -p "Install build tools? [Y/n] " -n 1 -r < /dev/tty
                    echo
                    if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
                        sudo DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=a apt-get update -qq && sudo DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=a apt-get install -y -qq build-essential python3-dev libffi-dev >/dev/null 2>&1 || true
                        log_success "Build tools installed"
                    fi
                fi
            fi
        fi
    fi

    # Install the main package in editable mode with all extras.
    # Try [all] first, fall back to base install if extras have issues.
    ALL_INSTALL_LOG=$(mktemp)
    if ! $UV_CMD pip install -e ".[all]" 2>"$ALL_INSTALL_LOG"; then
        log_warn "Full install (.[all]) failed, trying base install..."
        log_info "Reason: $(tail -5 "$ALL_INSTALL_LOG" | head -3)"
        rm -f "$ALL_INSTALL_LOG"
        if ! $UV_CMD pip install -e "."; then
            log_error "Package installation failed."
            log_info "Check that build tools are installed: sudo apt install build-essential python3-dev"
            log_info "Then re-run: cd $INSTALL_DIR && uv pip install -e '.[all]'"
            exit 1
        fi
    else
        rm -f "$ALL_INSTALL_LOG"
    fi

    log_success "Main package installed"

    # tinker-atropos (RL training) is optional — skip by default.
    # To enable RL tools: git submodule update --init tinker-atropos && uv pip install -e "./tinker-atropos"
    if [ -d "tinker-atropos" ] && [ -f "tinker-atropos/pyproject.toml" ]; then
        log_info "tinker-atropos submodule found — skipping install (optional, for RL training)"
        log_info "  To install: $UV_CMD pip install -e \"./tinker-atropos\""
    fi

    log_success "All dependencies installed"
}

setup_path() {
    log_info "Setting up hermes command..."

    if [ "$USE_VENV" = true ]; then
        HERMES_BIN="$INSTALL_DIR/venv/bin/hermes"
    else
        HERMES_BIN="$(which hermes 2>/dev/null || echo "")"
        if [ -z "$HERMES_BIN" ]; then
            log_warn "hermes not found on PATH after install"
            return 0
        fi
    fi

    # Verify the entry point script was actually generated
    if [ ! -x "$HERMES_BIN" ]; then
        log_warn "hermes entry point not found at $HERMES_BIN"
        log_info "This usually means the pip install didn't complete successfully."
        if [ "$DISTRO" = "termux" ]; then
            log_info "Try: cd $INSTALL_DIR && python -m pip install -e '.[termux]' -c constraints-termux.txt"
        else
            log_info "Try: cd $INSTALL_DIR && uv pip install -e '.[all]'"
        fi
        return 0
    fi

    local command_link_dir
    local command_link_display_dir
    command_link_dir="$(get_command_link_dir)"
    command_link_display_dir="$(get_command_link_display_dir)"

    # Create a user-facing shim for the hermes command.
    mkdir -p "$command_link_dir"
    ln -sf "$HERMES_BIN" "$command_link_dir/hermes"
    log_success "Symlinked hermes → $command_link_display_dir/hermes"

    if [ "$DISTRO" = "termux" ]; then
        export PATH="$command_link_dir:$PATH"
        log_info "$command_link_display_dir is the native Termux command path"
        log_success "hermes command ready"
        return 0
    fi

    # Check if ~/.local/bin is on PATH; if not, add it to shell config.
    # Detect the user's actual login shell (not the shell running this script,
    # which is always bash when piped from curl).
    if ! echo "$PATH" | tr ':' '\n' | grep -q "^$command_link_dir$"; then
        SHELL_CONFIGS=()
        IS_FISH=false
        LOGIN_SHELL="$(basename "${SHELL:-/bin/bash}")"
        case "$LOGIN_SHELL" in
            zsh)
                [ -f "$HOME/.zshrc" ] && SHELL_CONFIGS+=("$HOME/.zshrc")
                [ -f "$HOME/.zprofile" ] && SHELL_CONFIGS+=("$HOME/.zprofile")
                # If neither exists, create ~/.zshrc (common on fresh macOS installs)
                if [ ${#SHELL_CONFIGS[@]} -eq 0 ]; then
                    touch "$HOME/.zshrc"
                    SHELL_CONFIGS+=("$HOME/.zshrc")
                fi
                ;;
            bash)
                [ -f "$HOME/.bashrc" ] && SHELL_CONFIGS+=("$HOME/.bashrc")
                [ -f "$HOME/.bash_profile" ] && SHELL_CONFIGS+=("$HOME/.bash_profile")
                ;;
            fish)
                # fish uses ~/.config/fish/config.fish and fish_add_path — not export PATH=
                IS_FISH=true
                FISH_CONFIG="$HOME/.config/fish/config.fish"
                mkdir -p "$(dirname "$FISH_CONFIG")"
                touch "$FISH_CONFIG"
                ;;
            *)
                [ -f "$HOME/.bashrc" ] && SHELL_CONFIGS+=("$HOME/.bashrc")
                [ -f "$HOME/.zshrc" ] && SHELL_CONFIGS+=("$HOME/.zshrc")
                ;;
        esac
        # Also ensure ~/.profile has it (sourced by login shells on
        # Ubuntu/Debian/WSL even when ~/.bashrc is skipped)
        [ "$IS_FISH" = "false" ] && [ -f "$HOME/.profile" ] && SHELL_CONFIGS+=("$HOME/.profile")

        PATH_LINE='export PATH="$HOME/.local/bin:$PATH"'

        for SHELL_CONFIG in "${SHELL_CONFIGS[@]}"; do
            if ! grep -v '^[[:space:]]*#' "$SHELL_CONFIG" 2>/dev/null | grep -qE 'PATH=.*\.local/bin'; then
                echo "" >> "$SHELL_CONFIG"
                echo "# Hermes Agent — ensure ~/.local/bin is on PATH" >> "$SHELL_CONFIG"
                echo "$PATH_LINE" >> "$SHELL_CONFIG"
                log_success "Added ~/.local/bin to PATH in $SHELL_CONFIG"
            fi
        done

        # fish uses fish_add_path instead of export PATH=...
        if [ "$IS_FISH" = "true" ]; then
            if ! grep -q 'fish_add_path.*\.local/bin' "$FISH_CONFIG" 2>/dev/null; then
                echo "" >> "$FISH_CONFIG"
                echo "# Hermes Agent — ensure ~/.local/bin is on PATH" >> "$FISH_CONFIG"
                echo 'fish_add_path "$HOME/.local/bin"' >> "$FISH_CONFIG"
                log_success "Added ~/.local/bin to PATH in $FISH_CONFIG"
            fi
        fi

        if [ "$IS_FISH" = "false" ] && [ ${#SHELL_CONFIGS[@]} -eq 0 ]; then
            log_warn "Could not detect shell config file to add ~/.local/bin to PATH"
            log_info "Add manually: $PATH_LINE"
        fi
    else
        log_info "~/.local/bin already on PATH"
    fi

    # Export for current session so hermes works immediately
    export PATH="$command_link_dir:$PATH"

    log_success "hermes command ready"
}

copy_config_templates() {
    log_info "Setting up configuration files..."

    # Create ~/.hermes directory structure (config at top level, code in subdir)
    mkdir -p "$HERMES_HOME"/{cron,sessions,logs,pairing,hooks,image_cache,audio_cache,memories,skills,whatsapp/session}

    # Create .env at ~/.hermes/.env (top level, easy to find)
    if [ ! -f "$HERMES_HOME/.env" ]; then
        if [ -f "$INSTALL_DIR/.env.example" ]; then
            cp "$INSTALL_DIR/.env.example" "$HERMES_HOME/.env"
            log_success "Created ~/.hermes/.env from template"
        else
            touch "$HERMES_HOME/.env"
            log_success "Created ~/.hermes/.env"
        fi
    else
        log_info "~/.hermes/.env already exists, keeping it"
    fi

    # Create config.yaml at ~/.hermes/config.yaml (top level, easy to find)
    if [ ! -f "$HERMES_HOME/config.yaml" ]; then
        if [ -f "$INSTALL_DIR/cli-config.yaml.example" ]; then
            cp "$INSTALL_DIR/cli-config.yaml.example" "$HERMES_HOME/config.yaml"
            log_success "Created ~/.hermes/config.yaml from template"
        fi
    else
        log_info "~/.hermes/config.yaml already exists, keeping it"
    fi

    # Create SOUL.md if it doesn't exist (global persona file)
    if [ ! -f "$HERMES_HOME/SOUL.md" ]; then
        cat > "$HERMES_HOME/SOUL.md" << 'SOUL_EOF'
# Hermes Agent Persona

<!--
This file defines the agent's personality and tone.
The agent will embody whatever you write here.
Edit this to customize how Hermes communicates with you.

Examples:
  - "You are a warm, playful assistant who uses kaomoji occasionally."
  - "You are a concise technical expert. No fluff, just facts."
  - "You speak like a friendly coworker who happens to know everything."

This file is loaded fresh each message -- no restart needed.
Delete the contents (or this file) to use the default personality.
-->
SOUL_EOF
        log_success "Created ~/.hermes/SOUL.md (edit to customize personality)"
    fi

    log_success "Configuration directory ready: ~/.hermes/"

    # Seed bundled skills into ~/.hermes/skills/ (manifest-based, one-time per skill)
    log_info "Syncing bundled skills to ~/.hermes/skills/ ..."
    if "$INSTALL_DIR/venv/bin/python" "$INSTALL_DIR/tools/skills_sync.py" 2>/dev/null; then
        log_success "Skills synced to ~/.hermes/skills/"
    else
        # Fallback: simple directory copy if Python sync fails
        if [ -d "$INSTALL_DIR/skills" ] && [ ! "$(ls -A "$HERMES_HOME/skills/" 2>/dev/null | grep -v '.bundled_manifest')" ]; then
            cp -r "$INSTALL_DIR/skills/"* "$HERMES_HOME/skills/" 2>/dev/null || true
            log_success "Skills copied to ~/.hermes/skills/"
        fi
    fi
}

install_node_deps() {
    if [ "$HAS_NODE" = false ]; then
        log_info "Skipping Node.js dependencies (Node not installed)"
        return 0
    fi

    if [ "$DISTRO" = "termux" ]; then
        log_info "Skipping automatic Node/browser dependency setup on Termux"
        log_info "Browser automation and WhatsApp bridge are not part of the tested Termux install path yet."
        log_info "If you want to experiment manually later, run: cd $INSTALL_DIR && npm install"
        return 0
    fi

    if [ -f "$INSTALL_DIR/package.json" ]; then
        log_info "Installing Node.js dependencies (browser tools)..."
        cd "$INSTALL_DIR"
        npm install --silent 2>/dev/null || {
            log_warn "npm install failed (browser tools may not work)"
        }
        log_success "Node.js dependencies installed"

        # Install Playwright browser + system dependencies.
        # Playwright's --with-deps only supports apt-based systems natively.
        # For Arch/Manjaro we install the system libs via pacman first.
        # Other systems must install Chromium dependencies manually.
        log_info "Installing browser engine (Playwright Chromium)..."
        case "$DISTRO" in
            ubuntu|debian|raspbian|pop|linuxmint|elementary|zorin|kali|parrot)
                log_info "Playwright may request sudo to install browser system dependencies (shared libraries)."
                log_info "This is standard Playwright setup — Hermes itself does not require root access."
                cd "$INSTALL_DIR" && npx playwright install --with-deps chromium 2>/dev/null || {
                    log_warn "Playwright browser installation failed — browser tools will not work."
                    log_warn "Try running manually: cd $INSTALL_DIR && npx playwright install --with-deps chromium"
                }
                ;;
            arch|manjaro)
                if command -v pacman &> /dev/null; then
                    log_info "Arch/Manjaro detected — installing Chromium system dependencies via pacman..."
                    if command -v sudo &> /dev/null && sudo -n true 2>/dev/null; then
                        sudo NEEDRESTART_MODE=a pacman -S --noconfirm --needed \
                            nss atk at-spi2-core cups libdrm libxkbcommon mesa pango cairo alsa-lib >/dev/null 2>&1 || true
                    elif [ "$(id -u)" -eq 0 ]; then
                        pacman -S --noconfirm --needed \
                            nss atk at-spi2-core cups libdrm libxkbcommon mesa pango cairo alsa-lib >/dev/null 2>&1 || true
                    else
                        log_warn "Cannot install browser deps without sudo. Run manually:"
                        log_warn "  sudo pacman -S nss atk at-spi2-core cups libdrm libxkbcommon mesa pango cairo alsa-lib"
                    fi
                fi
                cd "$INSTALL_DIR" && npx playwright install chromium 2>/dev/null || {
                    log_warn "Playwright browser installation failed — browser tools will not work."
                }
                ;;
            fedora|rhel|centos|rocky|alma)
                log_warn "Playwright does not support automatic dependency installation on RPM-based systems."
                log_info "Install Chromium system dependencies manually before using browser tools:"
                log_info "  sudo dnf install nss atk at-spi2-core cups-libs libdrm libxkbcommon mesa-libgbm pango cairo alsa-lib"
                cd "$INSTALL_DIR" && npx playwright install chromium 2>/dev/null || {
                    log_warn "Playwright browser installation failed — install dependencies above and retry."
                }
                ;;
            opensuse*|sles)
                log_warn "Playwright does not support automatic dependency installation on zypper-based systems."
                log_info "Install Chromium system dependencies manually before using browser tools:"
                log_info "  sudo zypper install mozilla-nss libatk-1_0-0 at-spi2-core cups-libs libdrm2 libxkbcommon0 Mesa-libgbm1 pango cairo libasound2"
                cd "$INSTALL_DIR" && npx playwright install chromium 2>/dev/null || {
                    log_warn "Playwright browser installation failed — install dependencies above and retry."
                }
                ;;
            *)
                log_warn "Playwright does not support automatic dependency installation on $DISTRO."
                log_info "Install Chromium/browser system dependencies for your distribution, then run:"
                log_info "  cd $INSTALL_DIR && npx playwright install chromium"
                log_info "Browser tools will not work until dependencies are installed."
                cd "$INSTALL_DIR" && npx playwright install chromium 2>/dev/null || true
                ;;
        esac
        log_success "Browser engine setup complete"
    fi

    # Install WhatsApp bridge dependencies
    if [ -f "$INSTALL_DIR/scripts/whatsapp-bridge/package.json" ]; then
        log_info "Installing WhatsApp bridge dependencies..."
        cd "$INSTALL_DIR/scripts/whatsapp-bridge"
        npm install --silent 2>/dev/null || {
            log_warn "WhatsApp bridge npm install failed (WhatsApp may not work)"
        }
        log_success "WhatsApp bridge dependencies installed"
    fi
}

run_setup_wizard() {
    if [ "$RUN_SETUP" = false ]; then
        log_info "Skipping setup wizard (--skip-setup)"
        return 0
    fi

    # The setup wizard reads from /dev/tty, so it works even when the
    # install script itself is piped (curl | bash). Only skip if no
    # terminal is available at all (e.g. Docker build, CI).
    if ! [ -e /dev/tty ]; then
        log_info "Setup wizard skipped (no terminal available). Run 'hermes setup' after install."
        return 0
    fi

    echo ""
    log_info "Starting setup wizard..."
    echo ""

    cd "$INSTALL_DIR"

    # Run hermes setup using the venv Python directly (no activation needed).
    # Redirect stdin from /dev/tty so interactive prompts work when piped from curl.
    if [ "$USE_VENV" = true ]; then
        "$INSTALL_DIR/venv/bin/python" -m hermes_cli.main setup < /dev/tty
    else
        python -m hermes_cli.main setup < /dev/tty
    fi
}

maybe_start_gateway() {
    # Check if any messaging platform tokens were configured
    ENV_FILE="$HERMES_HOME/.env"
    if [ ! -f "$ENV_FILE" ]; then
        return 0
    fi

    HAS_MESSAGING=false
    for VAR in TELEGRAM_BOT_TOKEN DISCORD_BOT_TOKEN SLACK_BOT_TOKEN SLACK_APP_TOKEN WHATSAPP_ENABLED; do
        VAL=$(grep "^${VAR}=" "$ENV_FILE" 2>/dev/null | cut -d'=' -f2-)
        if [ -n "$VAL" ] && [ "$VAL" != "your-token-here" ]; then
            HAS_MESSAGING=true
            break
        fi
    done

    if [ "$HAS_MESSAGING" = false ]; then
        return 0
    fi

    echo ""
    log_info "Messaging platform token detected!"
    log_info "The gateway needs to be running for Hermes to send/receive messages."

    # If WhatsApp is enabled and no session exists yet, run foreground first for QR scan
    WHATSAPP_VAL=$(grep "^WHATSAPP_ENABLED=" "$ENV_FILE" 2>/dev/null | cut -d'=' -f2-)
    WHATSAPP_SESSION="$HERMES_HOME/whatsapp/session/creds.json"
    if [ "$WHATSAPP_VAL" = "true" ] && [ ! -f "$WHATSAPP_SESSION" ]; then
        if [ "$IS_INTERACTIVE" = true ]; then
            echo ""
            log_info "WhatsApp is enabled but not yet paired."
            log_info "Running 'hermes whatsapp' to pair via QR code..."
            echo ""
            read -p "Pair WhatsApp now? [Y/n] " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
                HERMES_CMD="$(get_hermes_command_path)"
                $HERMES_CMD whatsapp || true
            fi
        else
            log_info "WhatsApp pairing skipped (non-interactive). Run 'hermes whatsapp' to pair."
        fi
    fi

    if ! [ -e /dev/tty ]; then
        log_info "Gateway setup skipped (no terminal available). Run 'hermes gateway install' later."
        return 0
    fi

    echo ""
    if [ "$DISTRO" = "termux" ]; then
        read -p "Would you like to start the gateway in the background? [Y/n] " -n 1 -r < /dev/tty
    else
        read -p "Would you like to install the gateway as a background service? [Y/n] " -n 1 -r < /dev/tty
    fi
    echo

    if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
        HERMES_CMD="$(get_hermes_command_path)"

        if [ "$DISTRO" != "termux" ] && command -v systemctl &> /dev/null; then
            log_info "Installing systemd service..."
            if $HERMES_CMD gateway install 2>/dev/null; then
                log_success "Gateway service installed"
                if $HERMES_CMD gateway start 2>/dev/null; then
                    log_success "Gateway started! Your bot is now online."
                else
                    log_warn "Service installed but failed to start. Try: hermes gateway start"
                fi
            else
                log_warn "Systemd install failed. You can start manually: hermes gateway"
            fi
        else
            if [ "$DISTRO" = "termux" ]; then
                log_info "Termux detected — starting gateway in best-effort background mode..."
            else
                log_info "systemd not available — starting gateway in background..."
            fi
            nohup $HERMES_CMD gateway > "$HERMES_HOME/logs/gateway.log" 2>&1 &
            GATEWAY_PID=$!
            log_success "Gateway started (PID $GATEWAY_PID). Logs: ~/.hermes/logs/gateway.log"
            log_info "To stop: kill $GATEWAY_PID"
            log_info "To restart later: hermes gateway"
            if [ "$DISTRO" = "termux" ]; then
                log_warn "Android may stop background processes when Termux is suspended or the system reclaims resources."
            fi
        fi
    else
        log_info "Skipped. Start the gateway later with: hermes gateway"
    fi
}

print_success() {
    echo ""
    echo -e "${GREEN}${BOLD}"
    echo "┌─────────────────────────────────────────────────────────┐"
    echo "│              ✓ Installation Complete!                   │"
    echo "└─────────────────────────────────────────────────────────┘"
    echo -e "${NC}"
    echo ""

    # Show file locations
    echo -e "${CYAN}${BOLD}📁 Your files (all in ~/.hermes/):${NC}"
    echo ""
    echo -e "   ${YELLOW}Config:${NC}    ~/.hermes/config.yaml"
    echo -e "   ${YELLOW}API Keys:${NC}  ~/.hermes/.env"
    echo -e "   ${YELLOW}Data:${NC}      ~/.hermes/cron/, sessions/, logs/"
    echo -e "   ${YELLOW}Code:${NC}      ~/.hermes/hermes-agent/"
    echo ""

    echo -e "${CYAN}─────────────────────────────────────────────────────────${NC}"
    echo ""
    echo -e "${CYAN}${BOLD}🚀 Commands:${NC}"
    echo ""
    echo -e "   ${GREEN}hermes${NC}              Start chatting"
    echo -e "   ${GREEN}hermes setup${NC}        Configure API keys & settings"
    echo -e "   ${GREEN}hermes config${NC}       View/edit configuration"
    echo -e "   ${GREEN}hermes config edit${NC}  Open config in editor"
    echo -e "   ${GREEN}hermes gateway install${NC} Install gateway service (messaging + cron)"
    echo -e "   ${GREEN}hermes update${NC}       Update to latest version"
    echo ""

    echo -e "${CYAN}─────────────────────────────────────────────────────────${NC}"
    echo ""
    if [ "$DISTRO" = "termux" ]; then
        echo -e "${YELLOW}⚡ 'hermes' was linked into $(get_command_link_display_dir), which is already on PATH in Termux.${NC}"
        echo ""
    else
        echo -e "${YELLOW}⚡ Reload your shell to use 'hermes' command:${NC}"
        echo ""
        LOGIN_SHELL="$(basename "${SHELL:-/bin/bash}")"
        if [ "$LOGIN_SHELL" = "zsh" ]; then
            echo "   source ~/.zshrc"
        elif [ "$LOGIN_SHELL" = "bash" ]; then
            echo "   source ~/.bashrc"
        elif [ "$LOGIN_SHELL" = "fish" ]; then
            echo "   source ~/.config/fish/config.fish"
        else
            echo "   source ~/.bashrc   # or ~/.zshrc"
        fi
        echo ""
    fi

    # Show Node.js warning if auto-install failed
    if [ "$HAS_NODE" = false ]; then
        echo -e "${YELLOW}"
        echo "Note: Node.js could not be installed automatically."
        echo "Browser tools need Node.js. Install manually:"
        if [ "$DISTRO" = "termux" ]; then
            echo "  pkg install nodejs"
        else
            echo "  https://nodejs.org/en/download/"
        fi
        echo -e "${NC}"
    fi

    # Show ripgrep note if not installed
    if [ "$HAS_RIPGREP" = false ]; then
        echo -e "${YELLOW}"
        echo "Note: ripgrep (rg) was not found. File search will use"
        echo "grep as a fallback. For faster search in large codebases,"
        if [ "$DISTRO" = "termux" ]; then
            echo "install ripgrep: pkg install ripgrep"
        else
            echo "install ripgrep: sudo apt install ripgrep (or brew install ripgrep)"
        fi
        echo -e "${NC}"
    fi
}

# ============================================================================
# Main
# ============================================================================

main() {
    print_banner

    detect_os
    install_uv
    check_python
    check_git
    check_node
    install_system_packages

    clone_repo
    setup_venv
    install_deps
    install_node_deps
    setup_path
    copy_config_templates
    run_setup_wizard
    maybe_start_gateway

    print_success
}

main
