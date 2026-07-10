#!/bin/bash
# ============================================================================
# HT AI Agent Setup Script
# ============================================================================
# Quick setup for developers who cloned the repo manually.
# Uses uv for desktop/server setup and Python's stdlib venv + pip on Termux.
#
# Usage:
#   ./setup-hermes.sh
#
# This script:
# 1. Detects desktop/server vs Android/Termux setup path
# 2. Creates a Python 3.11 virtual environment
# 3. Installs the appropriate dependency set for the platform
# 4. Creates .env from template (if not exists)
# 5. Symlinks the 'hermes' CLI command into a user-facing bin dir
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

# Prevent uv from discovering config files (uv.toml, pyproject.toml) from the
# wrong user's home directory when running under sudo -u <user>.  See #21269.
export UV_NO_CONFIG=1

PYTHON_VERSION="3.11"

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

echo ""
echo -e "${CYAN}HT AI Agent Setup${NC}"
echo ""

# ============================================================================
# Install / locate uv
# ============================================================================

echo -e "${CYAN}→${NC} Checking for uv..."

UV_CMD=""
if is_termux; then
    echo -e "${CYAN}→${NC} Termux detected — using Python's stdlib venv + pip instead of uv"
else
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
        # Capture installer output so a failure shows the user WHY
        # (network, glibc mismatch on old distros, missing curl, disk
        # full, etc.) instead of "✗ Failed to install uv" with zero
        # diagnostic.  Two-stage to avoid `curl | sh` masking curl
        # failures (sh exits 0 on empty stdin under no pipefail).
        _uv_log="$(mktemp 2>/dev/null || echo "/tmp/hermes-uv-install.$$.log")"
        _uv_installer="$(mktemp 2>/dev/null || echo "/tmp/hermes-uv-installer.$$.sh")"
        if ! curl -LsSf https://astral.sh/uv/install.sh -o "$_uv_installer" 2>"$_uv_log"; then
            echo -e "${RED}✗${NC} Failed to download uv installer."
            sed 's/^/    /' "$_uv_log" >&2
            echo -e "${CYAN}→${NC} Install manually: https://docs.astral.sh/uv/"
            rm -f "$_uv_log" "$_uv_installer"
            exit 1
        fi
        if sh "$_uv_installer" >>"$_uv_log" 2>&1; then
            rm -f "$_uv_installer"
            if [ -x "$HOME/.local/bin/uv" ]; then
                UV_CMD="$HOME/.local/bin/uv"
            elif [ -x "$HOME/.cargo/bin/uv" ]; then
                UV_CMD="$HOME/.cargo/bin/uv"
            fi

            if [ -n "$UV_CMD" ]; then
                rm -f "$_uv_log"
                UV_VERSION=$($UV_CMD --version 2>/dev/null)
                echo -e "${GREEN}✓${NC} uv installed ($UV_VERSION)"
            else
                echo -e "${RED}✗${NC} uv installer reported success but binary not found. Add ~/.local/bin to PATH and retry."
                echo -e "${CYAN}→${NC} Installer output:"
                sed 's/^/    /' "$_uv_log" >&2
                rm -f "$_uv_log"
                exit 1
            fi
        else
            echo -e "${RED}✗${NC} Failed to install uv."
            echo -e "${CYAN}→${NC} Installer output:"
            sed 's/^/    /' "$_uv_log" >&2
            echo -e "${CYAN}→${NC} Install manually: https://docs.astral.sh/uv/"
            rm -f "$_uv_log" "$_uv_installer"
            exit 1
        fi
    fi
fi

# ============================================================================
# Python check (uv can provision it automatically)
# ============================================================================

echo -e "${CYAN}→${NC} Checking Python $PYTHON_VERSION..."

if is_termux; then
    if command -v python >/dev/null 2>&1; then
        PYTHON_PATH="$(command -v python)"
        if "$PYTHON_PATH" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; then
            PYTHON_FOUND_VERSION=$($PYTHON_PATH --version 2>/dev/null)
            echo -e "${GREEN}✓${NC} $PYTHON_FOUND_VERSION found"
        else
            echo -e "${RED}✗${NC} Termux Python must be 3.11+"
            echo "    Run: pkg install python"
            exit 1
        fi
    else
        echo -e "${RED}✗${NC} Python not found in Termux"
        echo "    Run: pkg install python"
        exit 1
    fi
else
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
fi

# ============================================================================
# Virtual environment
# ============================================================================

echo -e "${CYAN}→${NC} Setting up virtual environment..."

if [ -d "venv" ]; then
    echo -e "${CYAN}→${NC} Removing old venv..."
    rm -rf venv
fi

if is_termux; then
    "$PYTHON_PATH" -m venv venv
    echo -e "${GREEN}✓${NC} venv created with stdlib venv"
else
    $UV_CMD venv venv --python "$PYTHON_VERSION"
    echo -e "${GREEN}✓${NC} venv created (Python $PYTHON_VERSION)"
fi

export VIRTUAL_ENV="$SCRIPT_DIR/venv"
SETUP_PYTHON="$SCRIPT_DIR/venv/bin/python"

# ============================================================================
# Dependencies
# ============================================================================

echo -e "${CYAN}→${NC} Installing dependencies..."

if is_termux; then
    export ANDROID_API_LEVEL="$(getprop ro.build.version.sdk 2>/dev/null || printf '%s' "${ANDROID_API_LEVEL:-}")"
    echo -e "${CYAN}→${NC} Termux detected — installing the tested Android bundle"
    "$SETUP_PYTHON" -m pip install --upgrade pip setuptools wheel
    if [ -f "constraints-termux.txt" ]; then
        "$SETUP_PYTHON" -m pip install -e ".[termux]" -c constraints-termux.txt || {
            echo -e "${YELLOW}⚠${NC} Termux bundle install failed, falling back to base install..."
            "$SETUP_PYTHON" -m pip install -e "." -c constraints-termux.txt
        }
    else
        "$SETUP_PYTHON" -m pip install -e ".[termux]" || "$SETUP_PYTHON" -m pip install -e "."
    fi
    echo -e "${GREEN}✓${NC} Dependencies installed"
else
    # Prefer uv sync with lockfile (hash-verified installs) when available,
    # fall back to pip install for compatibility or when lockfile is stale.
    #
    # Multi-tier pip fallback. Goal: ONE compromised PyPI package
    # (mistralai 2.4.6 in May 2026 → quarantined) shouldn't silently demote
    # a fresh setup to "core only". Edit _BROKEN_EXTRAS when a transitive
    # breaks; users keep voice / honcho / google / slack / matrix etc. even
    # if mistral can't resolve.
    _BROKEN_EXTRAS=()  # populate when an extra becomes unresolvable
    _ALL_EXTRAS=(
        modal daytona messaging matrix cron cli dev tts-premium slack
        pty honcho mcp homeassistant sms acp voice dingtalk feishu google
        bedrock web youtube
    )
    _SAFE_EXTRAS=()
    for _e in "${_ALL_EXTRAS[@]}"; do
        _skip=false
        for _b in "${_BROKEN_EXTRAS[@]}"; do
            [ "$_e" = "$_b" ] && _skip=true && break
        done
        [ "$_skip" = false ] && _SAFE_EXTRAS+=("$_e")
    done
    _SAFE_SPEC=".[$(IFS=,; echo "${_SAFE_EXTRAS[*]}")]"
    _try_install() {
        $UV_CMD pip install -e ".[all]" \
            || $UV_CMD pip install -e "$_SAFE_SPEC" \
            || $UV_CMD pip install -e "."
    }

    if [ -f "uv.lock" ]; then
        # Hash-verified install (preferred). The lockfile records SHA256
        # hashes for every transitive — a compromised transitive would have
        # a different hash and be REJECTED by uv. This is the only path
        # that protects against transitive-package supply-chain attacks
        # (the direct deps in pyproject.toml are exact-pinned, but
        # `uv pip install` re-resolves transitives fresh from PyPI).
        echo -e "${CYAN}→${NC} Using uv.lock for hash-verified installation..."
        echo -e "${CYAN}→${NC} (first run on a fresh venv can take 1-5 minutes; uv prints progress below)"
        # Critical flag choice: `--extra all`, NOT `--all-extras`. The
        # latter installs every [project.optional-dependencies] key,
        # bypassing the curated [all] extra and pulling backends like
        # [matrix] (python-olm needs make on Windows) and [rl] (git+https
        # deps that fail offline). See pyproject.toml's [all] for the
        # curated set, and tools/lazy_deps.py for backends that install
        # at first use.
        # Also: stream stderr through directly so the user sees uv's
        # progress UI instead of staring at a frozen prompt.
        if UV_PROJECT_ENVIRONMENT="$SCRIPT_DIR/venv" $UV_CMD sync --extra all --locked; then
            echo -e "${GREEN}✓${NC} Dependencies installed (hash-verified via uv.lock)"
        else
            echo -e "${YELLOW}⚠${NC} Lockfile sync failed (see uv output above)."
            echo -e "${YELLOW}⚠${NC} Falling back to PyPI resolve — transitives will NOT be hash-verified."
            _try_install
            echo -e "${GREEN}✓${NC} Dependencies installed (transitives re-resolved, not hash-verified)"
        fi
    else
        echo -e "${YELLOW}⚠${NC} uv.lock not found — installing without hash verification of transitives."
        _try_install
        echo -e "${GREEN}✓${NC} Dependencies installed (transitives re-resolved, not hash-verified)"
    fi
fi

# ============================================================================
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

        if is_termux; then
            pkg install -y ripgrep && INSTALLED=true
        else
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
        fi

        if [ "$INSTALLED" = true ]; then
            echo -e "${GREEN}✓${NC} ripgrep installed"
        else
            echo -e "${YELLOW}⚠${NC} Auto-install failed. Install options:"
            if is_termux; then
                echo "    pkg install ripgrep          # Termux / Android"
            else
                echo "    sudo apt install ripgrep     # Debian/Ubuntu"
                echo "    brew install ripgrep         # macOS"
                echo "    cargo install ripgrep        # With Rust (no sudo)"
            fi
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
        # .env holds API keys — restrict to owner-only access (matches
        # scripts/install.sh which already chmods 600 after creation).
        chmod 600 .env 2>/dev/null || true
        echo -e "${GREEN}✓${NC} Created .env from template"
    fi
else
    # Tighten an existing .env's perms in case it was created elsewhere
    # under a permissive umask.
    chmod 600 .env 2>/dev/null || true
    echo -e "${GREEN}✓${NC} .env exists"
fi

# ============================================================================
# PATH setup — symlink hermes into a user-facing bin dir
# ============================================================================

echo -e "${CYAN}→${NC} Setting up hermes command..."

HERMES_BIN="$SCRIPT_DIR/venv/bin/hermes"
COMMAND_LINK_DIR="$(get_command_link_dir)"
COMMAND_LINK_DISPLAY_DIR="$(get_command_link_display_dir)"
mkdir -p "$COMMAND_LINK_DIR"
ln -sf "$HERMES_BIN" "$COMMAND_LINK_DIR/hermes"
# `ht` is the primary command; `hermes` is kept as a legacy alias.
ln -sf "$SCRIPT_DIR/venv/bin/ht" "$COMMAND_LINK_DIR/ht"
echo -e "${GREEN}✓${NC} Symlinked ht → $COMMAND_LINK_DISPLAY_DIR/ht (legacy alias: hermes)"

if is_termux; then
    export PATH="$COMMAND_LINK_DIR:$PATH"
    echo -e "${GREEN}✓${NC} $COMMAND_LINK_DISPLAY_DIR is already on PATH in Termux"
else
    # Determine the appropriate shell config file
    SHELL_CONFIG=""
    if [[ "$SHELL" == *"zsh"* ]]; then
        SHELL_CONFIG="$HOME/.zshrc"
    elif [[ "$SHELL" == *"bash"* ]]; then
        SHELL_CONFIG="$HOME/.bashrc"
        [ ! -f "$SHELL_CONFIG" ] && SHELL_CONFIG="$HOME/.bash_profile"
    else
        # Fallback to checking existing files
        if [ -f "$HOME/.zshrc" ]; then
            SHELL_CONFIG="$HOME/.zshrc"
        elif [ -f "$HOME/.bashrc" ]; then
            SHELL_CONFIG="$HOME/.bashrc"
        elif [ -f "$HOME/.bash_profile" ]; then
            SHELL_CONFIG="$HOME/.bash_profile"
        fi
    fi

    if [ -n "$SHELL_CONFIG" ]; then
        # Touch the file just in case it doesn't exist yet but was selected
        touch "$SHELL_CONFIG" 2>/dev/null || true

        if ! echo "$PATH" | tr ':' '\n' | grep -q "^$HOME/.local/bin$"; then
            if ! grep -q '\.local/bin' "$SHELL_CONFIG" 2>/dev/null; then
                echo "" >> "$SHELL_CONFIG"
                echo "# HT AI Agent — ensure ~/.local/bin is on PATH" >> "$SHELL_CONFIG"
                echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$SHELL_CONFIG"
                echo -e "${GREEN}✓${NC} Added ~/.local/bin to PATH in $SHELL_CONFIG"
            else
                echo -e "${GREEN}✓${NC} ~/.local/bin already in $SHELL_CONFIG"
            fi
        else
            echo -e "${GREEN}✓${NC} ~/.local/bin already on PATH"
        fi
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
if is_termux; then
    echo "  1. Run the setup wizard to configure API keys:"
    echo "     ht setup"
    echo ""
    echo "  2. Start chatting:"
    echo "     ht"
    echo ""
else
    echo "  1. Reload your shell:"
    echo "     source $SHELL_CONFIG"
    echo ""
    echo "  2. Run the setup wizard to configure API keys:"
    echo "     ht setup"
    echo ""
    echo "  3. Start chatting:"
    echo "     ht"
    echo ""
fi
echo "Other commands:"
echo "  ht status            # Check configuration"
if is_termux; then
    echo "  ht gateway           # Run gateway in foreground"
else
    echo "  ht gateway install   # Install gateway service (messaging + cron)"
fi
echo "  ht cron list         # View scheduled jobs"
echo "  ht doctor            # Diagnose issues"
echo ""
echo "('hermes' still works as a legacy alias for 'ht')"
echo ""

# Ask if they want to run setup wizard now
read -p "Would you like to run the setup wizard now? [Y/n] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
    echo ""
    # Run directly with venv Python (no activation needed)
    "$SCRIPT_DIR/venv/bin/python" -m hermes_cli.main setup
fi
