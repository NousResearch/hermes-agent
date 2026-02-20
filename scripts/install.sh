#!/bin/bash
# ============================================================================
# Hermes Agent Installer
# ============================================================================
# Installation script for Linux and macOS.
# Uses uv for fast Python provisioning and package management.
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
HERMES_HOME="$HOME/.hermes"
INSTALL_DIR="${HERMES_INSTALL_DIR:-$HERMES_HOME/hermes-agent}"
PYTHON_VERSION="3.11"

# Options
USE_VENV=true
RUN_SETUP=true
BRANCH="main"

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
    echo "│             🦋 Hermes Agent Installer                   │"
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

# ============================================================================
# System detection
# ============================================================================

detect_os() {
    case "$(uname -s)" in
        Linux*)
            OS="linux"
            if [ -f /etc/os-release ]; then
                . /etc/os-release
                DISTRO="$ID"
            else
                DISTRO="unknown"
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
        macos)
            log_info "  xcode-select --install"
            log_info "  Or: brew install git"
            ;;
    esac
    
    exit 1
}

check_node() {
    log_info "Checking Node.js (optional, for browser tools)..."
    
    if command -v node &> /dev/null; then
        NODE_VERSION=$(node --version)
        log_success "Node.js $NODE_VERSION found"
        HAS_NODE=true
        return 0
    fi
    
    log_warn "Node.js not found (browser tools will be limited)"
    log_info "To install Node.js (optional):"
    
    case "$OS" in
        linux)
            case "$DISTRO" in
                ubuntu|debian)
                    log_info "  curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -"
                    log_info "  sudo apt install -y nodejs"
                    ;;
                fedora)
                    log_info "  sudo dnf install nodejs"
                    ;;
                arch)
                    log_info "  sudo pacman -S nodejs npm"
                    ;;
                *)
                    log_info "  https://nodejs.org/en/download/"
                    ;;
            esac
            ;;
        macos)
            log_info "  brew install node"
            log_info "  Or: https://nodejs.org/en/download/"
            ;;
    esac
    
    HAS_NODE=false
    # Don't exit - Node is optional
}

check_ripgrep() {
    log_info "Checking ripgrep (optional, for faster file search)..."
    
    if command -v rg &> /dev/null; then
        RG_VERSION=$(rg --version | head -1)
        log_success "$RG_VERSION found"
        HAS_RIPGREP=true
        return 0
    fi
    
    log_warn "ripgrep not found (file search will use grep fallback)"
    
    # Offer to install
    echo ""
    read -p "Would you like to install ripgrep? (faster search, recommended) [Y/n] " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
        log_info "Installing ripgrep..."
        
        # Check if we can use sudo
        CAN_SUDO=false
        if command -v sudo &> /dev/null; then
            if sudo -n true 2>/dev/null || sudo -v 2>/dev/null; then
                CAN_SUDO=true
            fi
        fi
        
        case "$OS" in
            linux)
                if [ "$CAN_SUDO" = true ]; then
                    case "$DISTRO" in
                        ubuntu|debian)
                            if sudo apt install -y ripgrep 2>/dev/null; then
                                log_success "ripgrep installed"
                                HAS_RIPGREP=true
                                return 0
                            fi
                            ;;
                        fedora)
                            if sudo dnf install -y ripgrep 2>/dev/null; then
                                log_success "ripgrep installed"
                                HAS_RIPGREP=true
                                return 0
                            fi
                            ;;
                        arch)
                            if sudo pacman -S --noconfirm ripgrep 2>/dev/null; then
                                log_success "ripgrep installed"
                                HAS_RIPGREP=true
                                return 0
                            fi
                            ;;
                    esac
                else
                    log_warn "sudo not available - cannot auto-install system packages"
                    if command -v cargo &> /dev/null; then
                        log_info "Trying cargo install (no sudo required)..."
                        if cargo install ripgrep 2>/dev/null; then
                            log_success "ripgrep installed via cargo"
                            HAS_RIPGREP=true
                            return 0
                        fi
                    fi
                fi
                ;;
            macos)
                if command -v brew &> /dev/null; then
                    if brew install ripgrep 2>/dev/null; then
                        log_success "ripgrep installed"
                        HAS_RIPGREP=true
                        return 0
                    fi
                fi
                ;;
        esac
        log_warn "Auto-install failed. You can install manually later:"
    else
        log_info "Skipping ripgrep installation. To install manually:"
    fi
    
    # Show manual install instructions
    case "$OS" in
        linux)
            case "$DISTRO" in
                ubuntu|debian)
                    log_info "  sudo apt install ripgrep"
                    ;;
                fedora)
                    log_info "  sudo dnf install ripgrep"
                    ;;
                arch)
                    log_info "  sudo pacman -S ripgrep"
                    ;;
                *)
                    log_info "  https://github.com/BurntSushi/ripgrep#installation"
                    ;;
            esac
            if command -v cargo &> /dev/null; then
                log_info "  Or without sudo: cargo install ripgrep"
            fi
            ;;
        macos)
            log_info "  brew install ripgrep"
            ;;
    esac
    
    HAS_RIPGREP=false
    # Don't exit - ripgrep is optional (grep fallback exists)
}

check_ffmpeg() {
    log_info "Checking ffmpeg (optional, for TTS voice messages)..."
    
    if command -v ffmpeg &> /dev/null; then
        local ffmpeg_version=$(ffmpeg -version 2>/dev/null | head -1 | awk '{print $3}')
        log_success "ffmpeg found: $ffmpeg_version"
        HAS_FFMPEG=true
        return
    fi
    
    log_warn "ffmpeg not found"
    log_info "ffmpeg is needed for Telegram voice bubbles when using the default Edge TTS provider."
    log_info "Without it, Edge TTS audio is sent as a file instead of a voice bubble."
    log_info "(OpenAI and ElevenLabs TTS produce Opus natively and don't need ffmpeg.)"
    log_info ""
    log_info "To install ffmpeg:"
    
    case "$OS" in
        linux)
            case "$DISTRO" in
                ubuntu|debian)
                    log_info "  sudo apt install ffmpeg"
                    ;;
                fedora)
                    log_info "  sudo dnf install ffmpeg"
                    ;;
                arch)
                    log_info "  sudo pacman -S ffmpeg"
                    ;;
                *)
                    log_info "  https://ffmpeg.org/download.html"
                    ;;
            esac
            ;;
        macos)
            log_info "  brew install ffmpeg"
            ;;
    esac
    
    HAS_FFMPEG=false
    # Don't exit - ffmpeg is optional
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
            git fetch origin
            git checkout "$BRANCH"
            git pull origin "$BRANCH"
        else
            log_error "Directory exists but is not a git repository: $INSTALL_DIR"
            log_info "Remove it or choose a different directory with --dir"
            exit 1
        fi
    else
        # Try SSH first (for private repo access), fall back to HTTPS
        # Use --recurse-submodules to also clone mini-swe-agent and tinker-atropos
        log_info "Trying SSH clone..."
        if git clone --branch "$BRANCH" --recurse-submodules "$REPO_URL_SSH" "$INSTALL_DIR" 2>/dev/null; then
            log_success "Cloned via SSH"
        else
            log_info "SSH failed, trying HTTPS..."
            if git clone --branch "$BRANCH" --recurse-submodules "$REPO_URL_HTTPS" "$INSTALL_DIR"; then
                log_success "Cloned via HTTPS"
            else
                log_error "Failed to clone repository"
                log_info "For private repo access, ensure your SSH key is added to GitHub:"
                log_info "  ssh-add ~/.ssh/id_rsa"
                log_info "  ssh -T git@github.com  # Test connection"
                exit 1
            fi
        fi
    fi
    
    cd "$INSTALL_DIR"
    
    # Ensure submodules are initialized and updated (for existing installs or if --recurse failed)
    log_info "Initializing submodules (mini-swe-agent, tinker-atropos)..."
    git submodule update --init --recursive
    log_success "Submodules ready"
    
    log_success "Repository ready"
}

setup_venv() {
    if [ "$USE_VENV" = false ]; then
        log_info "Skipping virtual environment (--no-venv)"
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
    
    if [ "$USE_VENV" = true ]; then
        # Tell uv to install into our venv (no need to activate)
        export VIRTUAL_ENV="$INSTALL_DIR/venv"
    fi
    
    # Install the main package in editable mode with all extras
    $UV_CMD pip install -e ".[all]" || $UV_CMD pip install -e "."
    
    log_success "Main package installed"
    
    # Install submodules
    log_info "Installing mini-swe-agent (terminal tool backend)..."
    if [ -d "mini-swe-agent" ] && [ -f "mini-swe-agent/pyproject.toml" ]; then
        $UV_CMD pip install -e "./mini-swe-agent" || log_warn "mini-swe-agent install failed (terminal tools may not work)"
        log_success "mini-swe-agent installed"
    else
        log_warn "mini-swe-agent not found (run: git submodule update --init)"
    fi
    
    log_info "Installing tinker-atropos (RL training backend)..."
    if [ -d "tinker-atropos" ] && [ -f "tinker-atropos/pyproject.toml" ]; then
        $UV_CMD pip install -e "./tinker-atropos" || log_warn "tinker-atropos install failed (RL tools may not work)"
        log_success "tinker-atropos installed"
    else
        log_warn "tinker-atropos not found (run: git submodule update --init)"
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
    
    # Create symlink in ~/.local/bin (standard user binary location, usually on PATH)
    mkdir -p "$HOME/.local/bin"
    ln -sf "$HERMES_BIN" "$HOME/.local/bin/hermes"
    log_success "Symlinked hermes → ~/.local/bin/hermes"
    
    # Check if ~/.local/bin is on PATH; if not, add it to shell config
    if ! echo "$PATH" | tr ':' '\n' | grep -q "^$HOME/.local/bin$"; then
        SHELL_CONFIG=""
        if [ -n "$BASH_VERSION" ]; then
            if [ -f "$HOME/.bashrc" ]; then
                SHELL_CONFIG="$HOME/.bashrc"
            elif [ -f "$HOME/.bash_profile" ]; then
                SHELL_CONFIG="$HOME/.bash_profile"
            fi
        elif [ -n "$ZSH_VERSION" ] || [ -f "$HOME/.zshrc" ]; then
            SHELL_CONFIG="$HOME/.zshrc"
        fi
        
        PATH_LINE='export PATH="$HOME/.local/bin:$PATH"'
        
        if [ -n "$SHELL_CONFIG" ]; then
            if ! grep -q '\.local/bin' "$SHELL_CONFIG" 2>/dev/null; then
                echo "" >> "$SHELL_CONFIG"
                echo "# Hermes Agent — ensure ~/.local/bin is on PATH" >> "$SHELL_CONFIG"
                echo "$PATH_LINE" >> "$SHELL_CONFIG"
                log_success "Added ~/.local/bin to PATH in $SHELL_CONFIG"
            else
                log_info "~/.local/bin already referenced in $SHELL_CONFIG"
            fi
        fi
    else
        log_info "~/.local/bin already on PATH"
    fi
    
    # Export for current session so hermes works immediately
    export PATH="$HOME/.local/bin:$PATH"
    
    log_success "hermes command ready"
}

copy_config_templates() {
    log_info "Setting up configuration files..."
    
    # Create ~/.hermes directory structure (config at top level, code in subdir)
    mkdir -p "$HERMES_HOME"/{cron,sessions,logs,pairing,hooks,image_cache,audio_cache,memories,skills}
    
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
    
    if [ -f "$INSTALL_DIR/package.json" ]; then
        log_info "Installing Node.js dependencies..."
        cd "$INSTALL_DIR"
        npm install --silent 2>/dev/null || {
            log_warn "npm install failed (browser tools may not work)"
            return 0
        }
        log_success "Node.js dependencies installed"
    fi
}

run_setup_wizard() {
    if [ "$RUN_SETUP" = false ]; then
        log_info "Skipping setup wizard (--skip-setup)"
        return 0
    fi
    
    echo ""
    log_info "Starting setup wizard..."
    echo ""
    
    cd "$INSTALL_DIR"
    
    # Run hermes setup using the venv Python directly (no activation needed)
    if [ "$USE_VENV" = true ]; then
        "$INSTALL_DIR/venv/bin/python" -m hermes_cli.main setup
    else
        python -m hermes_cli.main setup
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
    echo ""
    read -p "Would you like to install the gateway as a background service? [Y/n] " -n 1 -r
    echo

    if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
        HERMES_CMD="$HOME/.local/bin/hermes"
        if [ ! -x "$HERMES_CMD" ]; then
            HERMES_CMD="hermes"
        fi

        if command -v systemctl &> /dev/null; then
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
            log_info "systemd not available — starting gateway in background..."
            nohup $HERMES_CMD gateway > "$HERMES_HOME/logs/gateway.log" 2>&1 &
            GATEWAY_PID=$!
            log_success "Gateway started (PID $GATEWAY_PID). Logs: ~/.hermes/logs/gateway.log"
            log_info "To stop: kill $GATEWAY_PID"
            log_info "To restart later: hermes gateway"
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
    echo -e "   ${GREEN}hermes gateway${NC}      Run messaging gateway"
    echo -e "   ${GREEN}hermes update${NC}       Update to latest version"
    echo ""
    
    echo -e "${CYAN}─────────────────────────────────────────────────────────${NC}"
    echo ""
    echo -e "${YELLOW}⚡ Reload your shell to use 'hermes' command:${NC}"
    echo ""
    echo "   source ~/.bashrc   # or ~/.zshrc"
    echo ""
    
    # Show Node.js warning if not installed
    if [ "$HAS_NODE" = false ]; then
        echo -e "${YELLOW}"
        echo "Note: Node.js was not found. Browser automation tools"
        echo "will have limited functionality. Install Node.js later"
        echo "if you need full browser support."
        echo -e "${NC}"
    fi
    
    # Show ripgrep note if not installed
    if [ "$HAS_RIPGREP" = false ]; then
        echo -e "${YELLOW}"
        echo "Note: ripgrep (rg) was not found. File search will use"
        echo "grep as a fallback. For faster search in large codebases,"
        echo "install ripgrep: sudo apt install ripgrep (or brew install ripgrep)"
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
    check_ripgrep
    check_ffmpeg
    
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
