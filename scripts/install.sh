#!/bin/bash
# ============================================================================
# Hermes Agent Installer
# ============================================================================
# Installation script for Linux and macOS.
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
PYTHON_MIN_VERSION="3.10"

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
            echo "  --dir PATH     Installation directory (default: ~/.hermes-agent)"
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
    echo "│  I'm just a butterfly with a lot of tools.             │"
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

check_python() {
    log_info "Checking Python..."
    
    # Try different python commands
    for cmd in python3.12 python3.11 python3.10 python3 python; do
        if command -v $cmd &> /dev/null; then
            PYTHON_CMD=$cmd
            PYTHON_VERSION=$($cmd -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
            
            # Check version
            if python3 -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)" 2>/dev/null; then
                log_success "Python $PYTHON_VERSION found"
                return 0
            fi
        fi
    done
    
    log_error "Python 3.10+ not found"
    log_info "Please install Python 3.10 or newer:"
    
    case "$OS" in
        linux)
            case "$DISTRO" in
                ubuntu|debian)
                    log_info "  sudo apt update && sudo apt install python3.11 python3.11-venv"
                    ;;
                fedora)
                    log_info "  sudo dnf install python3.11"
                    ;;
                arch)
                    log_info "  sudo pacman -S python"
                    ;;
                *)
                    log_info "  Use your package manager to install Python 3.10+"
                    ;;
            esac
            ;;
        macos)
            log_info "  brew install python@3.11"
            log_info "  Or download from https://www.python.org/downloads/"
            ;;
    esac
    
    exit 1
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
        log_info "Trying SSH clone..."
        if git clone --branch "$BRANCH" "$REPO_URL_SSH" "$INSTALL_DIR" 2>/dev/null; then
            log_success "Cloned via SSH"
        else
            log_info "SSH failed, trying HTTPS..."
            if git clone --branch "$BRANCH" "$REPO_URL_HTTPS" "$INSTALL_DIR"; then
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
    log_success "Repository ready"
}

setup_venv() {
    if [ "$USE_VENV" = false ]; then
        log_info "Skipping virtual environment (--no-venv)"
        return 0
    fi
    
    log_info "Creating virtual environment..."
    
    if [ -d "venv" ]; then
        log_info "Virtual environment already exists"
    else
        $PYTHON_CMD -m venv venv
    fi
    
    # Activate
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip wheel setuptools > /dev/null
    
    log_success "Virtual environment ready"
}

install_deps() {
    log_info "Installing dependencies..."
    
    if [ "$USE_VENV" = true ]; then
        source venv/bin/activate
    fi
    
    # Install the package in editable mode with all extras
    pip install -e ".[all]" > /dev/null 2>&1 || pip install -e "." > /dev/null
    
    log_success "Dependencies installed"
}

setup_path() {
    log_info "Setting up PATH..."
    
    # Determine the bin directory
    if [ "$USE_VENV" = true ]; then
        BIN_DIR="$INSTALL_DIR/venv/bin"
    else
        BIN_DIR="$HOME/.local/bin"
        mkdir -p "$BIN_DIR"
        
        # Create a wrapper script
        cat > "$BIN_DIR/hermes" << EOF
#!/bin/bash
cd "$INSTALL_DIR"
exec python -m hermes_cli.main "\$@"
EOF
        chmod +x "$BIN_DIR/hermes"
    fi
    
    # Add to PATH in shell config
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
    
    PATH_LINE="export PATH=\"$BIN_DIR:\$PATH\""
    
    if [ -n "$SHELL_CONFIG" ]; then
        if ! grep -q "hermes-agent" "$SHELL_CONFIG" 2>/dev/null; then
            echo "" >> "$SHELL_CONFIG"
            echo "# Hermes Agent" >> "$SHELL_CONFIG"
            echo "$PATH_LINE" >> "$SHELL_CONFIG"
            log_success "Added to $SHELL_CONFIG"
        else
            log_info "PATH already configured in $SHELL_CONFIG"
        fi
    fi
    
    # Also export for current session
    export PATH="$BIN_DIR:$PATH"
    
    log_success "PATH configured"
}

copy_config_templates() {
    log_info "Setting up configuration files..."
    
    # Create ~/.hermes directory structure (config at top level, code in subdir)
    mkdir -p "$HERMES_HOME/cron"
    mkdir -p "$HERMES_HOME/sessions"
    mkdir -p "$HERMES_HOME/logs"
    
    # Create .env at ~/.hermes/.env (top level, easy to find)
    if [ ! -f "$HERMES_HOME/.env" ]; then
        if [ -f "$INSTALL_DIR/.env.example" ]; then
            cp "$INSTALL_DIR/.env.example" "$HERMES_HOME/.env"
            log_success "Created ~/.hermes/.env from template"
        else
            # Create empty .env if no example exists
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
    
    log_success "Configuration directory ready: ~/.hermes/"
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
    
    if [ "$USE_VENV" = true ]; then
        source "$INSTALL_DIR/venv/bin/activate"
    fi
    
    cd "$INSTALL_DIR"
    python -m hermes_cli.main setup
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
}

# ============================================================================
# Main
# ============================================================================

main() {
    print_banner
    
    detect_os
    check_python
    check_git
    check_node
    
    clone_repo
    setup_venv
    install_deps
    install_node_deps
    setup_path
    copy_config_templates
    run_setup_wizard
    
    print_success
}

main
