#!/bin/sh
# ============================================================================
# Autolycus Agent Installer
# ============================================================================
# POSIX-compliant installer for FreeBSD, Linux, macOS, and BSD.
# Works with /bin/sh (dash, ash, ksh, bash, zsh).
# Assumes the repository is already cloned and the script is run from
# inside the autolycus-agent directory, or uses HERMES_HOME fallback.
#
# Prerequisites:
#   - Rust/Cargo installed
#   - make installed (required for cargo build on FreeBSD)
#
# Usage:
#   ./scripts/install-autolycus.sh
#   curl -fsSL https://raw.githubusercontent.com/waym0reom3ga/autolycus-agent/main/scripts/install-autolycus.sh | sh
#
# This script:
#   1. Detects the operating system
#   2. Installs uv via cargo
#   3. Creates a virtual environment with Python 3.11
#   4. Installs dependencies (OS-specific extras)
#   5. Sets up the hermes CLI command
#   6. Creates config files from templates
#   7. Syncs bundled skills
# ============================================================================

set -e

# Ensure HOME is set (may be unset on some BSD systems in non-login shells)
if [ -z "${HOME:-}" ]; then
    HOME="$(getent passwd "$(whoami)" 2>/dev/null | cut -d: -f6)" || \
    HOME="$(eval echo ~$(whoami))" || \
    HOME="/root"
fi

# FreeBSD installs packages to /usr/local/bin, which sudo may strip from PATH.
# Ensure it's always present so cargo/rustc/make are found under sudo.
case "$(uname -s)" in
    FreeBSD*)
        case ":${PATH}:" in
            *:/usr/local/bin:*) ;; # already there
            *) PATH="/usr/local/bin:${PATH}" ;;
        esac
        ;;
esac

# Colors
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m'

# Portable script directory detection (works when piped via curl | sh)
# When piped, $0 is "sh" and dirname gives ".", so we check if we're in the repo
SCRIPT_DIR="$(cd "$(dirname "$0" 2>/dev/null || echo .)" 2>/dev/null && pwd)" || SCRIPT_DIR="."
# If we can't find pyproject.toml relative to script location, use HERMES_HOME default
if [ ! -f "$SCRIPT_DIR/../pyproject.toml" ] && [ ! -f "$SCRIPT_DIR/pyproject.toml" ]; then
    SCRIPT_DIR="${HERMES_HOME:-$HOME/.hermes}/hermes-agent"
fi
REPO_DIR="$(cd "$SCRIPT_DIR/.." 2>/dev/null && pwd)" || REPO_DIR="$SCRIPT_DIR"
if [ ! -f "$REPO_DIR/pyproject.toml" ]; then
    REPO_DIR="${HERMES_HOME:-$HOME/.hermes}/hermes-agent"
fi
cd "$REPO_DIR"

PYTHON_VERSION="3.11"

echo ""
echo -e "${CYAN}${BOLD}⚕ Autolycus Agent Installer${NC}"
echo ""

# ============================================================================
# OS detection
# ============================================================================

detect_os() {
    case "$(uname -s)" in
        FreeBSD*)
            OS="freebsd"
            ;;
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
            ;;
        *)
            OS="unknown"
            echo -e "${RED}✗${NC} Unsupported operating system: $(uname -s)"
            exit 1
            ;;
    esac

    echo -e "${GREEN}✓${NC} Detected: $OS"
}

detect_os

# ============================================================================
# Prerequisite checks
# ============================================================================

check_prerequisites() {
    echo -e "${CYAN}→${NC} Checking prerequisites..."

    # Check for cargo/rust
    # On FreeBSD, cargo may be at /usr/local/bin even if not in PATH (e.g. under sudo)
    CARGO_CMD=""
    if command -v cargo &> /dev/null; then
        CARGO_CMD="cargo"
    elif [ -x "/usr/local/bin/cargo" ]; then
        CARGO_CMD="/usr/local/bin/cargo"
    fi

    if [ -z "$CARGO_CMD" ]; then
        echo -e "${RED}✗${NC} Rust/Cargo not found."
        echo ""
        echo "Install Rust:"
        echo "  FreeBSD:  pkg install rust"
        echo "  Arch:     sudo pacman -S rust"
        echo "  Ubuntu:   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
        echo "  macOS:    brew install rustup && rustup init"
        echo ""
        exit 1
    fi

    CARGO_VERSION=$($CARGO_CMD --version | awk '{print $2}')
    echo -e "${GREEN}✓${NC} Cargo $CARGO_VERSION found"

    # Check for make
    # On FreeBSD, Rust builds (jemalloc) need GNU make (gmake), not BSD make
    if [ "$OS" = "freebsd" ]; then
        if ! command -v gmake &> /dev/null; then
            echo -e "${RED}✗${NC} GNU make (gmake) not found (required for building Rust crates on FreeBSD)."
            echo ""
            echo "Install GNU make:"
            echo "  FreeBSD:  pkg install gmake"
            echo ""
            exit 1
        fi
        echo -e "${GREEN}✓${NC} gmake found"
    else
        if ! command -v make &> /dev/null; then
            echo -e "${RED}✗${NC} make not found (required for building uv from source)."
            echo ""
            echo "Install make:"
            echo "  FreeBSD:  pkg install make"
            echo "  Arch:     sudo pacman -S make"
            echo "  Ubuntu:   sudo apt install make"
            echo "  macOS:    xcode-select --install"
            echo ""
            exit 1
        fi
        echo -e "${GREEN}✓${NC} make found"
    fi
}

check_prerequisites

# On FreeBSD, set MAKE=gmake so Rust builds (jemalloc) use GNU make instead of BSD make
if [ "$OS" = "freebsd" ]; then
    export MAKE=gmake
fi

# ============================================================================
# Install uv via cargo
# ============================================================================

install_uv() {
    echo -e "${CYAN}→${NC} Installing uv..."

    # Check if uv is already available
    if command -v uv &> /dev/null; then
        UV_CMD="uv"
        UV_VERSION=$($UV_CMD --version 2>/dev/null)
        echo -e "${GREEN}✓${NC} uv already installed ($UV_VERSION)"
        return 0
    fi

    if [ -x "$HOME/.cargo/bin/uv" ]; then
        UV_CMD="$HOME/.cargo/bin/uv"
        UV_VERSION=$($UV_CMD --version 2>/dev/null)
        echo -e "${GREEN}✓${NC} uv found at ~/.cargo/bin ($UV_VERSION)"
        return 0
    fi

    if [ -x "$HOME/.local/bin/uv" ]; then
        UV_CMD="$HOME/.local/bin/uv"
        UV_VERSION=$($UV_CMD --version 2>/dev/null)
        echo -e "${GREEN}✓${NC} uv found at ~/.local/bin ($UV_VERSION)"
        return 0
    fi

    # Prefer pre-built binary installer (much faster, no Rust toolchain needed)
    echo -e "${CYAN}→${NC} Installing uv via official installer (pre-built binary)..."
    if curl -LsSf https://astral.sh/uv/install.sh | sh 2>/dev/null; then
        if [ -x "$HOME/.local/bin/uv" ]; then
            UV_CMD="$HOME/.local/bin/uv"
        elif command -v uv &> /dev/null; then
            UV_CMD="uv"
        else
            echo -e "${RED}✗${NC} uv installed but not found on PATH"
            exit 1
        fi
        UV_VERSION=$($UV_CMD --version 2>/dev/null)
        echo -e "${GREEN}✓${NC} uv installed ($UV_VERSION)"
        return 0
    fi

    # Fallback: install via cargo (requires Rust toolchain)
    echo -e "${YELLOW}⚠${NC} Pre-built installer failed, falling back to cargo install (slower)..."

    # Arch Linux enforces systemd user quotas on /tmp even when df shows free space.
    # Use home dir for cargo staging on Arch, /tmp everywhere else.
    if [ -f /etc/arch-release ]; then
        TMPDIR="$HOME/.cargo/cargo-tmp"
        mkdir -p "$TMPDIR"
    fi
    export TMPDIR
    if cargo install uv; then
        if [ -x "$HOME/.cargo/bin/uv" ]; then
            UV_CMD="$HOME/.cargo/bin/uv"
        elif [ -x "$HOME/.local/bin/uv" ]; then
            UV_CMD="$HOME/.local/bin/uv"
        elif command -v uv &> /dev/null; then
            UV_CMD="uv"
        else
            echo -e "${RED}✗${NC} uv installed but not found on PATH"
            echo "Try adding ~/.cargo/bin to PATH and re-running"
            exit 1
        fi
        UV_VERSION=$($UV_CMD --version 2>/dev/null)
        echo -e "${GREEN}✓${NC} uv installed ($UV_VERSION)"
        return 0
    else
        echo -e "${RED}✗${NC} Failed to install uv."
        echo ""
        echo "Possible causes:"
        echo "  - Rust/Cargo not installed"
        echo "  - Disk space / tmpfs quota exceeded"
        echo "  - Network issues"
        echo ""
        echo "Try installing uv manually: https://docs.astral.sh/uv/"
        exit 1
    fi
}

install_uv

# ============================================================================
# Python
# ============================================================================

setup_python() {
    echo -e "${CYAN}→${NC} Checking Python $PYTHON_VERSION..."

    if $UV_CMD python find "$PYTHON_VERSION" &> /dev/null; then
        PYTHON_PATH=$($UV_CMD python find "$PYTHON_VERSION")
        PYTHON_FOUND_VERSION=$($PYTHON_PATH --version 2>/dev/null)
        echo -e "${GREEN}✓${NC} Python found: $PYTHON_FOUND_VERSION"
    else
        echo -e "${CYAN}→${NC} Python $PYTHON_VERSION not found, installing via uv..."
        if $UV_CMD python install "$PYTHON_VERSION"; then
            PYTHON_PATH=$($UV_CMD python find "$PYTHON_VERSION")
            PYTHON_FOUND_VERSION=$($PYTHON_PATH --version 2>/dev/null)
            echo -e "${GREEN}✓${NC} Python installed: $PYTHON_FOUND_VERSION"
        else
            echo -e "${RED}✗${NC} Failed to install Python $PYTHON_VERSION"
            echo "Install Python manually, then re-run this script"
            exit 1
        fi
    fi
}

setup_python

# ============================================================================
# Virtual environment
# ============================================================================

setup_venv() {
    echo -e "${CYAN}→${NC} Setting up virtual environment..."

    if [ -d "venv" ]; then
        echo -e "${CYAN}→${NC} Removing old venv..."
        rm -rf venv
    fi

    $UV_CMD venv venv --python "$PYTHON_VERSION"
    echo -e "${GREEN}✓${NC} venv created (Python $PYTHON_VERSION)"

    export VIRTUAL_ENV="$REPO_DIR/venv"
}

setup_venv

# ============================================================================
# Dependencies (OS-specific extras)
# ============================================================================

install_deps() {
    echo -e "${CYAN}→${NC} Installing dependencies..."

    # Determine extras based on OS
    if [ "$OS" = "freebsd" ]; then
        EXTRAS="[modal,daytona,messaging,cron,cli,dev,tts-premium,slack,honcho,mcp]"
        echo -e "${CYAN}→${NC} FreeBSD detected — installing selective extras (voice excluded)"
    else
        EXTRAS="[all]"
        echo -e "${CYAN}→${NC} Installing full stack"
    fi

    # Prefer uv sync with lockfile
    if [ -f "uv.lock" ]; then
        echo -e "${CYAN}→${NC} Using uv.lock for hash-verified installation..."
        UV_PROJECT_ENVIRONMENT="$REPO_DIR/venv" $UV_CMD sync --all-extras --locked 2>/dev/null && \
            echo -e "${GREEN}✓${NC} Dependencies installed (lockfile verified)" || {
            echo -e "${YELLOW}⚠${NC} Lockfile install failed (may be outdated), falling back to pip..."
            $UV_CMD pip install -e ".$EXTRAS" || $UV_CMD pip install -e "."
            echo -e "${GREEN}✓${NC} Dependencies installed"
        }
    else
        $UV_CMD pip install -e ".$EXTRAS" || $UV_CMD pip install -e "."
        echo -e "${GREEN}✓${NC} Dependencies installed"
    fi
}

install_deps

# ============================================================================
# PATH setup
# ============================================================================

setup_path() {
    echo -e "${CYAN}→${NC} Setting up hermes and lycus commands..."

    HERMES_BIN="$REPO_DIR/venv/bin/hermes"
    LYCUS_BIN="$REPO_DIR/venv/bin/lycus"
    mkdir -p "$HOME/.local/bin"
    ln -sf "$HERMES_BIN" "$HOME/.local/bin/hermes"
    echo -e "${GREEN}✓${NC} Symlinked hermes → ~/.local/bin/hermes"
    ln -sf "$LYCUS_BIN" "$HOME/.local/bin/lycus"
    echo -e "${GREEN}✓${NC} Symlinked lycus → ~/.local/bin/lycus"

    # Add ~/.local/bin to shell config if needed
    if ! echo "$PATH" | tr ':' '\n' | grep -q "^$HOME/.local/bin$"; then
        SHELL_CONFIG=""
        # Detect actual login shell - on FreeBSD use pw command
        if [ "$OS" = "freebsd" ]; then
            LOGIN_SHELL="$(pw usershow "$(whoami)" -q 2>/dev/null | cut -d: -f7 | xargs basename)"
            [ -z "$LOGIN_SHELL" ] && LOGIN_SHELL="sh"
            if [ "$LOGIN_SHELL" = "csh" ] || [ "$LOGIN_SHELL" = "tcsh" ]; then
                SHELL_CONFIG="$HOME/.cshrc"
            elif [ "$LOGIN_SHELL" = "bash" ]; then
                SHELL_CONFIG="$HOME/.bashrc"
            elif [ "$LOGIN_SHELL" = "zsh" ]; then
                SHELL_CONFIG="$HOME/.zshrc"
            else
                SHELL_CONFIG="$HOME/.profile"
            fi
        elif echo "$SHELL" | grep -q "zsh"; then
            SHELL_CONFIG="$HOME/.zshrc"
        elif echo "$SHELL" | grep -q "bash"; then
            SHELL_CONFIG="$HOME/.bashrc"
            [ ! -f "$SHELL_CONFIG" ] && SHELL_CONFIG="$HOME/.bash_profile"
        else
            # Fallback
            if [ -f "$HOME/.zshrc" ]; then
                SHELL_CONFIG="$HOME/.zshrc"
            elif [ -f "$HOME/.bashrc" ]; then
                SHELL_CONFIG="$HOME/.bashrc"
            elif [ -f "$HOME/.bash_profile" ]; then
                SHELL_CONFIG="$HOME/.bash_profile"
            fi
        fi

        if [ -n "$SHELL_CONFIG" ]; then
            touch "$SHELL_CONFIG" 2>/dev/null || true
            if ! grep -q '\.local/bin' "$SHELL_CONFIG" 2>/dev/null; then
                echo "" >> "$SHELL_CONFIG"
                echo "# Autolycus Agent — ensure ~/.local/bin is on PATH" >> "$SHELL_CONFIG"
                # Use setenv for csh/tcsh, export for others
                if [ "$OS" = "freebsd" ] && { [ "$LOGIN_SHELL" = "csh" ] || [ "$LOGIN_SHELL" = "tcsh" ]; }; then
                    echo 'setenv PATH "$HOME/.local/bin:$PATH"' >> "$SHELL_CONFIG"
                else
                    echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$SHELL_CONFIG"
                fi
                echo -e "${GREEN}✓${NC} Added ~/.local/bin to PATH in $SHELL_CONFIG"
            else
                echo -e "${GREEN}✓${NC} ~/.local/bin already in $SHELL_CONFIG"
            fi
        fi
    else
        echo -e "${GREEN}✓${NC} ~/.local/bin already on PATH"
    fi
}

setup_path

# ============================================================================
# Config files
# ============================================================================

setup_config() {
    echo -e "${CYAN}→${NC} Setting up configuration files..."

    HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
    mkdir -p "$HERMES_HOME"/{cron,sessions,logs,memories,skills}

    # .env file
    if [ ! -f "$HERMES_HOME/.env" ]; then
        if [ -f ".env.example" ]; then
            cp .env.example "$HERMES_HOME/.env"
            echo -e "${GREEN}✓${NC} Created ~/.hermes/.env from template"
        else
            touch "$HERMES_HOME/.env"
            echo -e "${GREEN}✓${NC} Created ~/.hermes/.env"
        fi
    else
        echo -e "${GREEN}✓${NC} ~/.hermes/.env already exists"
    fi

    # config.yaml
    if [ ! -f "$HERMES_HOME/config.yaml" ]; then
        if [ -f "cli-config.yaml.example" ]; then
            cp cli-config.yaml.example "$HERMES_HOME/config.yaml"
            echo -e "${GREEN}✓${NC} Created ~/.hermes/config.yaml from template"
        fi
    else
        echo -e "${GREEN}✓${NC} ~/.hermes/config.yaml already exists"
    fi

    # Agent identity — assign a random name on first install
    if [ ! -f "$HERMES_HOME/.hermes_agent_name" ]; then
        AGENT_NAMES=("Atlas" "Bastion" "Cipher" "Drift" "Echo" "Flux" "Glint" "Haven" "Ion" "Jinx" "Kairo" "Lumen" "Nexus" "Orbit" "Prism" "Quill" "Rift" "Spark" "Talus" "Vex" "Warden" "Zephyr" "Axiom" "Blaze" "Coda" "Dusk" "Ember" "Frost" "Grail" "Haze" "Inferno" "Jade" "Kite" "Lance" "Mist" "Nova" "Onyx" "Pulse" "Quasar" "Rune" "Sage" "Titan" "Umber" "Volt" "Wrath" "Xenon" "Yucca" "Zenith" "Apex" "Bolt" "Crux" "Dune" "Eclipse" "Forge" "Gale" "Horizon" "Ignis" "Jolt" "Kinetic" "Lynx" "Magma" "Nimbus" "Opal" "Phoenix" "Quantum" "Radar" "Solar" "Terra" "Ultra" "Vector" "Wraith" "Yield" "Zen" "Arc")
        AGENT_NAME="${AGENT_NAMES[$((RANDOM % ${#AGENT_NAMES[@]}))]}"
        echo "$AGENT_NAME" > "$HERMES_HOME/.hermes_agent_name"
        echo -e "${GREEN}✓${NC} Agent identity assigned: $AGENT_NAME"
    else
        EXISTING_NAME=$(cat "$HERMES_HOME/.hermes_agent_name")
        echo -e "${GREEN}✓${NC} Agent identity: $EXISTING_NAME"
    fi

    # SOUL.md
    if [ ! -f "$HERMES_HOME/SOUL.md" ]; then
        cat > "$HERMES_HOME/SOUL.md" << 'SOUL_EOF'
# Autolycus Agent Persona

<!--
This file defines the agent's personality and tone.
The agent will embody whatever you write here.
Edit this to customize how Autolycus communicates with you.

This file is loaded fresh each message -- no restart needed.
Delete the contents (or this file) to use the default personality.
-->
SOUL_EOF
        echo -e "${GREEN}✓${NC} Created ~/.hermes/SOUL.md"
    fi
}

setup_config

# ============================================================================
# Skills sync
# ============================================================================

sync_skills() {
    echo -e "${CYAN}→${NC} Syncing bundled skills..."

    if "$REPO_DIR/venv/bin/python" "$REPO_DIR/tools/skills_sync.py" 2>/dev/null; then
        echo -e "${GREEN}✓${NC} Skills synced"
    else
        # Fallback: simple copy
        if [ -d "skills" ]; then
            cp -rn "skills/"* "$HOME/.hermes/skills/" 2>/dev/null || true
            echo -e "${GREEN}✓${NC} Skills copied"
        fi
    fi
}

sync_skills

# ============================================================================
# Done
# ============================================================================

echo ""
echo -e "${GREEN}${BOLD}✓ Installation complete!${NC}"
echo ""

# Detect actual shell for reload instructions
if [ "$OS" = "freebsd" ]; then
    LOGIN_SHELL="$(pw usershow "$(whoami)" -q 2>/dev/null | cut -d: -f7 | xargs basename)"
    [ -z "$LOGIN_SHELL" ] && LOGIN_SHELL="sh"
else
    LOGIN_SHELL="$(basename "${SHELL:-/bin/bash}")"
fi

echo "Next steps:"
echo ""
echo "  1. Reload your shell:"
if [ "$OS" = "freebsd" ]; then
    if [ "$LOGIN_SHELL" = "csh" ] || [ "$LOGIN_SHELL" = "tcsh" ]; then
        echo "     source ~/.cshrc"
    elif [ "$LOGIN_SHELL" = "bash" ]; then
        echo "     . ~/.bashrc"
    elif [ "$LOGIN_SHELL" = "zsh" ]; then
        echo "     . ~/.zshrc"
    else
        echo "     . ~/.profile   # sh"
    fi
else
    if [ "$LOGIN_SHELL" = "zsh" ]; then
        echo "     source ~/.zshrc"
    elif [ "$LOGIN_SHELL" = "bash" ]; then
        echo "     source ~/.bashrc"
    else
        echo "     source ~/.bashrc   # or ~/.zshrc"
    fi
fi
echo ""
echo "  2. Configure API keys:"
echo "     hermes setup"
echo ""
echo "  3. Start chatting:"
echo "     hermes"
echo ""

# FreeBSD: remind about python-sqlite
if [ "$OS" = "freebsd" ]; then
    echo -e "${YELLOW}⚠ Note:${NC} For long-term memory support, install python-sqlite:"
    echo "     pkg install py311-sqlite"
    echo ""
fi

echo "Other commands:"
echo "  hermes status        # Check configuration"
echo "  hermes doctor        # Diagnose issues"
echo "  hermes gateway install # Install gateway service"
echo ""
