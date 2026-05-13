#!/bin/bash
# =============================================================================
# Hermes Agent Tarball Installer (Alternative)
# =============================================================================
# Alternative installation method using pre-built tarballs.
# This is lighter and faster than the full git clone method.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install-tarball.sh | bash
#
# Or with options:
#   curl -fsSL ... | bash -s -- --version 0.12.0
#
# =============================================================================

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
REPO_URL="https://github.com/NousResearch/hermes-agent"
HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
PYTHON_VERSION="3.11"

# Options
VERSION=""
INSTALL_DIR=""
SKIP_SETUP=false
USE_VENV=true

# Detect non-interactive mode
if [ -t 0 ]; then
    IS_INTERACTIVE=true
else
    IS_INTERACTIVE=false
fi

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --version) VERSION="$2"; shift 2 ;;
        --dir) INSTALL_DIR="$2"; shift 2 ;;
        --hermes-home) HERMES_HOME="$2"; shift 2 ;;
        --skip-setup) SKIP_SETUP=true; shift ;;
        --no-venv) USE_VENV=false; shift ;;
        -h|--help)
            echo "Hermes Agent Tarball Installer"
            echo ""
            echo "Usage: install-tarball.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --version VERSION  Version to install (default: latest)"
            echo "  --dir PATH         Installation directory (default: ~/.hermes/hermes-agent)"
            echo "  --hermes-home PATH Data directory (default: ~/.hermes)"
            echo "  --skip-setup       Skip interactive setup wizard"
            echo "  --no-venv          Don't create virtual environment"
            echo "  -h, --help         Show this help"
            echo ""
            echo "Notes:"
            echo "  This installer uses pre-built tarballs for faster installation."
            echo "  For full git-based installation, use install.sh instead."
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ============================================================================
# Helper functions
# ============================================================================

print_banner() {
    echo ""
    echo -e "${MAGENTA}${BOLD}"
    echo "┌─────────────────────────────────────────────────────────┐"
    echo "│        ⚕ Hermes Agent Tarball Installer                │"
    echo "├─────────────────────────────────────────────────────────┤"
    echo "│  A lightweight installer using pre-built packages.     │"
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

# ============================================================================
# System detection
# ============================================================================

detect_platform() {
    local arch=$(uname -m)
    local os=$(uname -s)
    
    case "$arch" in
        x86_64)  ARCH_SUFFIX="x64" ;;
        arm64|aarch64) ARCH_SUFFIX="arm64" ;;
        *)
            log_error "Unsupported architecture: $arch"
            log_info "Use install.sh for manual installation"
            exit 1
            ;;
    esac
    
    case "$os" in
        Linux)  PLATFORM="linux-${ARCH_SUFFIX}" ;;
        Darwin) PLATFORM="macos-${ARCH_SUFFIX}" ;;
        *)
            log_error "Unsupported OS: $os"
            log_info "Use install.sh for manual installation"
            exit 1
            ;;
    esac
    
    log_success "Detected platform: $PLATFORM"
}

# ============================================================================
# Version resolution
# ============================================================================

resolve_version() {
    if [ -n "$VERSION" ]; then
        log_info "Using specified version: $VERSION"
        return 0
    fi
    
    log_info "Resolving latest version..."
    
    # Get latest release from GitHub
    LATEST_URL="${REPO_URL}/releases/latest"
    REDIRECT_URL=$(curl -fsSI "$LATEST_URL" 2>/dev/null | grep -i "^location:" | awk '{print $2}' | tr -d '\r')
    
    if [ -n "$REDIRECT_URL" ]; then
        VERSION=$(echo "$REDIRECT_URL" | grep -oE 'v[0-9]+\.[0-9]+\.[0-9]+' | head -1 | sed 's/^v//')
    fi
    
    if [ -z "$VERSION" ]; then
        log_error "Could not determine latest version"
        log_info "Specify version with --version"
        exit 1
    fi
    
    log_success "Latest version: $VERSION"
}

# ============================================================================
# Download and extraction
# ============================================================================

download_tarball() {
    TARBALL_NAME="hermes-agent-${VERSION}-${PLATFORM}.tar.gz"
    DOWNLOAD_URL="${REPO_URL}/releases/download/v${VERSION}/${TARBALL_NAME}"
    
    log_info "Downloading ${TARBALL_NAME}..."
    
    TEMP_DIR=$(mktemp -d)
    TARBALL_PATH="$TEMP_DIR/$TARBALL_NAME"
    
    if ! curl -fsSL "$DOWNLOAD_URL" -o "$TARBALL_PATH"; then
        log_error "Failed to download tarball"
        log_info "Check if version $VERSION exists for platform $PLATFORM"
        rm -rf "$TEMP_DIR"
        exit 1
    fi
    
    log_success "Downloaded $(du -h "$TARBALL_PATH" | cut -f1)"
}

extract_tarball() {
    log_info "Extracting tarball..."
    
    INSTALL_DIR="${INSTALL_DIR:-$HERMES_HOME/hermes-agent}"
    
    # Create parent directory
    mkdir -p "$(dirname "$INSTALL_DIR")"
    
    # Extract
    tar -xzf "$TARBALL_PATH" -C "$(dirname "$INSTALL_DIR")"
    
    # Rename if needed
    EXTRACTED_DIR="$(dirname "$INSTALL_DIR")/hermes-agent-${VERSION}"
    if [ "$EXTRACTED_DIR" != "$INSTALL_DIR" ] && [ -d "$EXTRACTED_DIR" ]; then
        if [ -d "$INSTALL_DIR" ]; then
            log_warn "Existing installation at $INSTALL_DIR"
            if [ "$IS_INTERACTIVE" = true ]; then
                read -p "Replace existing installation? [y/N] " -n 1 -r
                echo
                if [[ $REPLY =~ ^[Yy]$ ]]; then
                    rm -rf "$INSTALL_DIR"
                    mv "$EXTRACTED_DIR" "$INSTALL_DIR"
                else
                    log_info "Keeping existing installation"
                    INSTALL_DIR="$EXTRACTED_DIR"
                fi
            else
                log_info "Non-interactive mode, keeping existing installation"
                INSTALL_DIR="$EXTRACTED_DIR"
            fi
        else
            mv "$EXTRACTED_DIR" "$INSTALL_DIR"
        fi
    fi
    
    log_success "Extracted to $INSTALL_DIR"
}

# ============================================================================
# Installation
# ============================================================================

setup_venv() {
    if [ "$USE_VENV" = false ]; then
        log_info "Skipping virtual environment (--no-venv)"
        return 0
    fi
    
    cd "$INSTALL_DIR"
    
    if [ -d "venv" ]; then
        log_info "Virtual environment already exists, using existing"
        return 0
    fi
    
    log_info "Creating virtual environment..."
    
    # Check for Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 not found"
        log_info "Install Python 3.11+ or use --no-venv"
        exit 1
    fi
    
    # Check Python version
    PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    if ! python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; then
        log_error "Python 3.11+ required (found $PYTHON_VERSION)"
        exit 1
    fi
    
    python3 -m venv venv
    source venv/bin/activate
    
    # Install package
    log_info "Installing Python dependencies..."
    pip install --upgrade pip setuptools wheel > /dev/null 2>&1
    pip install -e ".[all]" 2>&1 | tail -1
    
    log_success "Virtual environment ready"
}

setup_path() {
    log_info "Setting up hermes command..."
    
    if [ "$USE_VENV" = true ]; then
        HERMES_BIN="$INSTALL_DIR/venv/bin/hermes"
    else
        log_warn "Without venv, hermes command may not be available"
        return 0
    fi
    
    # Create symlink
    LINK_DIR="$HOME/.local/bin"
    mkdir -p "$LINK_DIR"
    ln -sf "$HERMES_BIN" "$LINK_DIR/hermes"
    
    # Add to PATH if needed
    if ! echo "$PATH" | tr ':' '\n' | grep -q "^$LINK_DIR$"; then
        SHELL_CONFIG=""
        case "$(basename "${SHELL:-/bin/bash}")" in
            zsh) SHELL_CONFIG="$HOME/.zshrc" ;;
            bash) SHELL_CONFIG="$HOME/.bashrc" ;;
        esac
        
        if [ -n "$SHELL_CONFIG" ] && [ -f "$SHELL_CONFIG" ]; then
            if ! grep -q 'hermes' "$SHELL_CONFIG" 2>/dev/null; then
                echo "" >> "$SHELL_CONFIG"
                echo "# Hermes Agent" >> "$SHELL_CONFIG"
                echo "export PATH=\"\$HOME/.local/bin:\$PATH\"" >> "$SHELL_CONFIG"
                log_success "Added ~/.local/bin to PATH in $SHELL_CONFIG"
            fi
        fi
    fi
    
    export PATH="$LINK_DIR:$PATH"
    log_success "hermes command ready"
}

setup_config() {
    log_info "Setting up configuration..."
    
    mkdir -p "$HERMES_HOME"/{cron,sessions,logs,pairing,hooks,image_cache,audio_cache,memories,skills}
    
    # Create .env
    if [ ! -f "$HERMES_HOME/.env" ]; then
        if [ -f "$INSTALL_DIR/.env.example" ]; then
            cp "$INSTALL_DIR/.env.example" "$HERMES_HOME/.env"
            log_success "Created ~/.hermes/.env"
        else
            touch "$HERMES_HOME/.env"
        fi
    fi
    
    # Create config.yaml
    if [ ! -f "$HERMES_HOME/config.yaml" ]; then
        if [ -f "$INSTALL_DIR/cli-config.yaml.example" ]; then
            cp "$INSTALL_DIR/cli-config.yaml.example" "$HERMES_HOME/config.yaml"
            log_success "Created ~/.hermes/config.yaml"
        fi
    fi
    
    # Create SOUL.md
    if [ ! -f "$HERMES_HOME/SOUL.md" ]; then
        cat > "$HERMES_HOME/SOUL.md" << 'EOF'
# Hermes Agent Persona

<!--
This file defines the agent's personality and tone.
Edit this to customize how Hermes communicates with you.
-->
EOF
        log_success "Created ~/.hermes/SOUL.md"
    fi
    
    # Sync skills
    if [ -f "$INSTALL_DIR/tools/skills_sync.py" ]; then
        "$INSTALL_DIR/venv/bin/python" "$INSTALL_DIR/tools/skills_sync.py" 2>/dev/null || true
    fi
    
    log_success "Configuration ready"
}

run_setup_wizard() {
    if [ "$SKIP_SETUP" = true ]; then
        log_info "Skipping setup wizard (--skip-setup)"
        return 0
    fi
    
    if ! (: </dev/tty) 2>/dev/null; then
        log_info "Setup wizard skipped (no terminal available). Run 'hermes setup' after install."
        return 0
    fi
    
    echo ""
    log_info "Starting setup wizard..."
    echo ""
    
    cd "$INSTALL_DIR"
    "$INSTALL_DIR/venv/bin/python" -m hermes_cli.main setup < /dev/tty
}

# ============================================================================
# Cleanup
# ============================================================================

cleanup() {
    if [ -n "$TEMP_DIR" ] && [ -d "$TEMP_DIR" ]; then
        rm -rf "$TEMP_DIR"
    fi
}

trap cleanup EXIT

# ============================================================================
# Success message
# ============================================================================

print_success() {
    echo ""
    echo -e "${GREEN}${BOLD}"
    echo "┌─────────────────────────────────────────────────────────┐"
    echo "│              ✓ Installation Complete!                   │"
    echo "└─────────────────────────────────────────────────────────┘"
    echo -e "${NC}"
    echo ""
    echo -e "${CYAN}${BOLD}📁 Your files:${NC}"
    echo ""
    echo -e "   ${YELLOW}Config:${NC}    $HERMES_HOME/config.yaml"
    echo -e "   ${YELLOW}API Keys:${NC}  $HERMES_HOME/.env"
    echo -e "   ${YELLOW}Data:${NC}      $HERMES_HOME/cron/, sessions/, logs/"
    echo -e "   ${YELLOW}Code:${NC}      $INSTALL_DIR"
    echo ""
    echo -e "${CYAN}─────────────────────────────────────────────────────────${NC}"
    echo ""
    echo -e "${CYAN}${BOLD}🚀 Commands:${NC}"
    echo ""
    echo -e "   ${GREEN}hermes${NC}              Start chatting"
    echo -e "   ${GREEN}hermes setup${NC}        Configure API keys & settings"
    echo -e "   ${GREEN}hermes update${NC}       Update to latest version"
    echo ""
    echo -e "${CYAN}─────────────────────────────────────────────────────────${NC}"
    echo ""
    echo -e "${YELLOW}⚡ Reload your shell to use 'hermes' command:${NC}"
    echo ""
    LOGIN_SHELL="$(basename "${SHELL:-/bin/bash}")"
    case "$LOGIN_SHELL" in
        zsh)    echo "   source ~/.zshrc" ;;
        bash)   echo "   source ~/.bashrc" ;;
        *)      echo "   source ~/.bashrc   # or ~/.zshrc" ;;
    esac
    echo ""
}

# ============================================================================
# Main
# ============================================================================

main() {
    print_banner
    
    detect_platform
    resolve_version
    download_tarball
    extract_tarball
    
    cd "$INSTALL_DIR"
    
    setup_venv
    setup_path
    setup_config
    run_setup_wizard
    
    print_success
}

main
