#!/bin/bash
# ============================================================================
# Hermes Agent Setup Script
# ============================================================================
# Quick setup for developers who cloned the repo manually.
# Uses uv for fast Python provisioning and package management.
#
# Usage:
#   ./setup-hermes.sh
#
# This script:
# 1. Installs uv if not present
# 2. Creates a virtual environment with Python 3.11 via uv
# 3. Installs all dependencies (main package + submodules)
# 4. Creates .env from template (if not exists)
# 5. Symlinks the 'hermes' CLI command into ~/.local/bin
# 6. Runs the setup wizard (optional)
# ============================================================================

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_VERSION="3.11"

echo ""
echo -e "${CYAN}🦋 Hermes Agent Setup${NC}"
echo ""

# ============================================================================
# Install / locate uv
# ============================================================================

echo -e "${CYAN}→${NC} Checking for uv..."

UV_CMD=""
if command -v uv &> /dev/null; then
    UV_CMD="uv"
elif [ -x "$HOME/.local/bin/uv" ]; then
    UV_CMD="$HOME/.local/bin/uv"
elif [ -x "$HOME/.cargo/bin/uv" ]; then
    UV_CMD="$HOME/.cargo/bin/uv"
fi

if [ -n "$UV_CMD" ]; then
    UV_VERSION=$($UV_CMD --version 2>/dev/null)
    echo -e "${GREEN}✓${NC} uv found ($UV_VERSION)"
else
    echo -e "${CYAN}→${NC} Installing uv..."
    if curl -LsSf https://astral.sh/uv/install.sh | sh 2>/dev/null; then
        if [ -x "$HOME/.local/bin/uv" ]; then
            UV_CMD="$HOME/.local/bin/uv"
        elif [ -x "$HOME/.cargo/bin/uv" ]; then
            UV_CMD="$HOME/.cargo/bin/uv"
        fi
        
        if [ -n "$UV_CMD" ]; then
            UV_VERSION=$($UV_CMD --version 2>/dev/null)
            echo -e "${GREEN}✓${NC} uv installed ($UV_VERSION)"
        else
            echo -e "${RED}✗${NC} uv installed but not found. Add ~/.local/bin to PATH and retry."
            exit 1
        fi
    else
        echo -e "${RED}✗${NC} Failed to install uv. Visit https://docs.astral.sh/uv/"
        exit 1
    fi
fi

# ============================================================================
# Python check (uv can provision it automatically)
# ============================================================================

echo -e "${CYAN}→${NC} Checking Python $PYTHON_VERSION..."

if $UV_CMD python find "$PYTHON_VERSION" &> /dev/null; then
    PYTHON_PATH=$($UV_CMD python find "$PYTHON_VERSION")
    PYTHON_FOUND_VERSION=$($PYTHON_PATH --version 2>/dev/null)
    echo -e "${GREEN}✓${NC} $PYTHON_FOUND_VERSION found"
else
    echo -e "${CYAN}→${NC} Python $PYTHON_VERSION not found, installing via uv..."
    $UV_CMD python install "$PYTHON_VERSION"
    PYTHON_PATH=$($UV_CMD python find "$PYTHON_VERSION")
    PYTHON_FOUND_VERSION=$($PYTHON_PATH --version 2>/dev/null)
    echo -e "${GREEN}✓${NC} $PYTHON_FOUND_VERSION installed"
fi

# ============================================================================
# Virtual environment
# ============================================================================

echo -e "${CYAN}→${NC} Setting up virtual environment..."

if [ -d "venv" ]; then
    echo -e "${CYAN}→${NC} Removing old venv..."
    rm -rf venv
fi

$UV_CMD venv venv --python "$PYTHON_VERSION"
echo -e "${GREEN}✓${NC} venv created (Python $PYTHON_VERSION)"

# Tell uv to install into this venv (no activation needed for uv)
export VIRTUAL_ENV="$SCRIPT_DIR/venv"

# ============================================================================
# Dependencies
# ============================================================================

echo -e "${CYAN}→${NC} Installing dependencies..."

$UV_CMD pip install -e ".[all]" || $UV_CMD pip install -e "."

echo -e "${GREEN}✓${NC} Dependencies installed"

# ============================================================================
# Submodules (terminal backend + RL training)
# ============================================================================

echo -e "${CYAN}→${NC} Installing submodules..."

# mini-swe-agent (terminal tool backend)
if [ -d "mini-swe-agent" ] && [ -f "mini-swe-agent/pyproject.toml" ]; then
    $UV_CMD pip install -e "./mini-swe-agent" && \
        echo -e "${GREEN}✓${NC} mini-swe-agent installed" || \
        echo -e "${YELLOW}⚠${NC} mini-swe-agent install failed (terminal tools may not work)"
else
    echo -e "${YELLOW}⚠${NC} mini-swe-agent not found (run: git submodule update --init --recursive)"
fi

# tinker-atropos (RL training backend)
if [ -d "tinker-atropos" ] && [ -f "tinker-atropos/pyproject.toml" ]; then
    $UV_CMD pip install -e "./tinker-atropos" && \
        echo -e "${GREEN}✓${NC} tinker-atropos installed" || \
        echo -e "${YELLOW}⚠${NC} tinker-atropos install failed (RL tools may not work)"
else
    echo -e "${YELLOW}⚠${NC} tinker-atropos not found (run: git submodule update --init --recursive)"
fi

# ============================================================================
# Optional: ripgrep (for faster file search)
# ============================================================================

echo -e "${CYAN}→${NC} Checking ripgrep (optional, for faster search)..."

if command -v rg &> /dev/null; then
    echo -e "${GREEN}✓${NC} ripgrep found"
else
    echo -e "${YELLOW}⚠${NC} ripgrep not found (file search will use grep fallback)"
    read -p "Install ripgrep for faster search? [Y/n] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
        INSTALLED=false
        
        # Check if sudo is available
        if command -v sudo &> /dev/null && sudo -n true 2>/dev/null; then
            if command -v apt &> /dev/null; then
                sudo apt install -y ripgrep && INSTALLED=true
            elif command -v dnf &> /dev/null; then
                sudo dnf install -y ripgrep && INSTALLED=true
            fi
        fi
        
        # Try brew (no sudo needed)
        if [ "$INSTALLED" = false ] && command -v brew &> /dev/null; then
            brew install ripgrep && INSTALLED=true
        fi
        
        # Try cargo (no sudo needed)
        if [ "$INSTALLED" = false ] && command -v cargo &> /dev/null; then
            echo -e "${CYAN}→${NC} Trying cargo install (no sudo required)..."
            cargo install ripgrep && INSTALLED=true
        fi
        
        if [ "$INSTALLED" = true ]; then
            echo -e "${GREEN}✓${NC} ripgrep installed"
        else
            echo -e "${YELLOW}⚠${NC} Auto-install failed. Install options:"
            echo "    sudo apt install ripgrep     # Debian/Ubuntu"
            echo "    brew install ripgrep         # macOS"
            echo "    cargo install ripgrep        # With Rust (no sudo)"
            echo "    https://github.com/BurntSushi/ripgrep#installation"
        fi
    fi
fi

# ============================================================================
# Environment file
# ============================================================================

if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo -e "${GREEN}✓${NC} Created .env from template"
    fi
else
    echo -e "${GREEN}✓${NC} .env exists"
fi

# ============================================================================
# PATH setup — symlink hermes into ~/.local/bin
# ============================================================================

echo -e "${CYAN}→${NC} Setting up hermes command..."

HERMES_BIN="$SCRIPT_DIR/venv/bin/hermes"
mkdir -p "$HOME/.local/bin"
ln -sf "$HERMES_BIN" "$HOME/.local/bin/hermes"
echo -e "${GREEN}✓${NC} Symlinked hermes → ~/.local/bin/hermes"

# Ensure ~/.local/bin is on PATH in shell config
SHELL_CONFIG=""
if [ -f "$HOME/.zshrc" ]; then
    SHELL_CONFIG="$HOME/.zshrc"
elif [ -f "$HOME/.bashrc" ]; then
    SHELL_CONFIG="$HOME/.bashrc"
elif [ -f "$HOME/.bash_profile" ]; then
    SHELL_CONFIG="$HOME/.bash_profile"
fi

if [ -n "$SHELL_CONFIG" ]; then
    if ! echo "$PATH" | tr ':' '\n' | grep -q "^$HOME/.local/bin$"; then
        if ! grep -q '\.local/bin' "$SHELL_CONFIG" 2>/dev/null; then
            echo "" >> "$SHELL_CONFIG"
            echo "# Hermes Agent — ensure ~/.local/bin is on PATH" >> "$SHELL_CONFIG"
            echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$SHELL_CONFIG"
            echo -e "${GREEN}✓${NC} Added ~/.local/bin to PATH in $SHELL_CONFIG"
        else
            echo -e "${GREEN}✓${NC} ~/.local/bin already in $SHELL_CONFIG"
        fi
    else
        echo -e "${GREEN}✓${NC} ~/.local/bin already on PATH"
    fi
fi

# ============================================================================
# Seed bundled skills into ~/.hermes/skills/
# ============================================================================

HERMES_SKILLS_DIR="${HERMES_HOME:-$HOME/.hermes}/skills"
mkdir -p "$HERMES_SKILLS_DIR"

echo ""
echo "Syncing bundled skills to ~/.hermes/skills/ ..."
if "$SCRIPT_DIR/venv/bin/python" "$SCRIPT_DIR/tools/skills_sync.py" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} Skills synced"
else
    # Fallback: copy if sync script fails (missing deps, etc.)
    if [ -d "$SCRIPT_DIR/skills" ]; then
        cp -rn "$SCRIPT_DIR/skills/"* "$HERMES_SKILLS_DIR/" 2>/dev/null || true
        echo -e "${GREEN}✓${NC} Skills copied"
    fi
fi

# ============================================================================
# Done
# ============================================================================

echo ""
echo -e "${GREEN}✓ Setup complete!${NC}"
echo ""
echo "Next steps:"
echo ""
echo "  1. Reload your shell:"
echo "     source $SHELL_CONFIG"
echo ""
echo "  2. Run the setup wizard to configure API keys:"
echo "     hermes setup"
echo ""
echo "  3. Start chatting:"
echo "     hermes"
echo ""
echo "Other commands:"
echo "  hermes status        # Check configuration"
echo "  hermes gateway       # Start messaging gateway"
echo "  hermes cron daemon   # Run cron daemon"
echo "  hermes doctor        # Diagnose issues"
echo ""

# Ask if they want to run setup wizard now
read -p "Would you like to run the setup wizard now? [Y/n] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
    echo ""
    # Run directly with venv Python (no activation needed)
    "$SCRIPT_DIR/venv/bin/python" -m hermes_cli.main setup
fi
