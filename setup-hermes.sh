#!/bin/bash
# ============================================================================
# Hermes Agent Setup Script
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

PYTHON_VERSION="3.11"

# ============================================================================
# Git update — careful local-changes handling
# ============================================================================

_git_update() {
    # Not a git repo? Skip silently.
    if ! git -C "$SCRIPT_DIR" rev-parse --git-dir &>/dev/null; then
        return 0
    fi

    echo -e "${CYAN}→${NC} Checking for upstream updates..."

    # Fetch without merging so we can compare
    if ! git fetch origin main --quiet 2>/dev/null; then
        echo -e "${YELLOW}⚠${NC} Could not reach upstream — skipping update check."
        return 0
    fi

    LOCAL=$(git rev-parse HEAD)
    REMOTE=$(git rev-parse origin/main)

    if [ "$LOCAL" = "$REMOTE" ]; then
        echo -e "${GREEN}✓${NC} Already up to date."
        return 0
    fi

    COMMITS_BEHIND=$(git rev-list --count HEAD..origin/main)
    echo -e "${CYAN}→${NC} ${COMMITS_BEHIND} new commit(s) available upstream."

    # Check for local modifications (tracked files only)
    LOCAL_CHANGES=$(git status --porcelain --untracked-files=no)

    if [ -z "$LOCAL_CHANGES" ]; then
        # Clean working tree — just pull
        echo -e "${CYAN}→${NC} No local changes detected. Pulling..."
        git merge --ff-only origin/main
        echo -e "${GREEN}✓${NC} Updated to $(git rev-parse --short HEAD)."
        return 0
    fi

    # --- Local changes exist ---
    CHANGED_FILES=$(git diff --name-only HEAD)
    echo ""
    echo -e "${YELLOW}⚠  Local changes detected in:${NC}"
    echo "$CHANGED_FILES" | sed 's/^/    /'
    echo ""
    echo "  What do you want to do?"
    echo "  [1] Stash local changes, update, then restore them  (risky: merge conflicts possible)"
    echo "  [2] Skip the update, keep local changes as-is       (safe)"
    echo "  [3] Show a full diff first"
    echo ""
    read -p "  Choice [1/2/3]: " -n 1 -r GIT_CHOICE
    echo ""

    case "$GIT_CHOICE" in
        3)
            echo ""
            git diff HEAD
            echo ""
            read -p "  Choice [1/2]: " -n 1 -r GIT_CHOICE
            echo ""
            ;;
    esac

    case "$GIT_CHOICE" in
        1)
            STASH_MSG="hermes-setup-autostash-$(date +%Y%m%d-%H%M%S)"
            echo -e "${CYAN}→${NC} Stashing local changes as: ${STASH_MSG}"
            git stash push -m "$STASH_MSG" -- $CHANGED_FILES

            echo -e "${CYAN}→${NC} Pulling upstream..."
            if ! git merge --ff-only origin/main; then
                echo -e "${RED}✗${NC} Merge failed. Restoring your stash..."
                git stash pop
                echo -e "${GREEN}✓${NC} Your changes are back. No update was applied."
                return 1
            fi
            echo -e "${GREEN}✓${NC} Updated to $(git rev-parse --short HEAD)."

            echo -e "${CYAN}→${NC} Restoring your local changes..."
            if git stash pop; then
                echo -e "${GREEN}✓${NC} Local changes restored cleanly."
            else
                echo ""
                echo -e "${RED}⚠  MERGE CONFLICT during stash pop!${NC}"
                echo "    Your changes are partially applied. Conflicting files are marked in git status."
                echo "    Resolve conflicts manually:"
                echo "      git status"
                echo "      git diff"
                echo "      # edit conflicting files"
                echo "      git add <file>"
                echo "      git stash drop   # when done"
                echo ""
                echo "    Your original stash is: ${STASH_MSG}"
                echo "    To fully abort and go back to the upstream state:"
                echo "      git checkout -- ."
                echo "      git stash drop"
                return 1
            fi
            ;;
        2)
            echo -e "${YELLOW}⚠${NC} Skipping update. Your local changes are untouched."
            ;;
        *)
            echo -e "${YELLOW}⚠${NC} Invalid choice — skipping update to be safe."
            ;;
    esac
}

_git_update

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
echo -e "${CYAN}⚕ Hermes Agent Setup${NC}"
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
    if [ -f "uv.lock" ]; then
        echo -e "${CYAN}→${NC} Using uv.lock for hash-verified installation..."
        UV_PROJECT_ENVIRONMENT="$SCRIPT_DIR/venv" $UV_CMD sync --all-extras --locked 2>/dev/null && \
            echo -e "${GREEN}✓${NC} Dependencies installed (lockfile verified)" || {
            echo -e "${YELLOW}⚠${NC} Lockfile install failed (may be outdated), falling back to pip install..."
            $UV_CMD pip install -e ".[all]" || $UV_CMD pip install -e "."
            echo -e "${GREEN}✓${NC} Dependencies installed"
        }
    else
        $UV_CMD pip install -e ".[all]" || $UV_CMD pip install -e "."
        echo -e "${GREEN}✓${NC} Dependencies installed"
    fi
fi

# ============================================================================
# Submodules (terminal backend + RL training)
# ============================================================================

echo -e "${CYAN}→${NC} Installing optional submodules..."

# tinker-atropos (RL training backend)
if is_termux; then
    echo -e "${CYAN}→${NC} Skipping tinker-atropos on Termux (not part of the tested Android path)"
elif [ -d "tinker-atropos" ] && [ -f "tinker-atropos/pyproject.toml" ]; then
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
        echo -e "${GREEN}✓${NC} Created .env from template"
    fi
else
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
echo -e "${GREEN}✓${NC} Symlinked hermes → $COMMAND_LINK_DISPLAY_DIR/hermes"

if is_termux; then
    export PATH="$COMMAND_LINK_DIR:$PATH"
    echo -e "${GREEN}✓${NC} $COMMAND_LINK_DISPLAY_DIR is already on PATH in Termux"
else
    SHELL_CONFIG=""    # posix shells: bash/zsh
    FISH_CONFIG=""     # fish gets its own conf.d snippet

    # ── Fish shell ──────────────────────────────────────────────────────────
    if [[ "$SHELL" == *"fish"* ]]; then
        FISH_CONFIG="$HOME/.config/fish/conf.d/hermes.fish"
    # ── Zsh ─────────────────────────────────────────────────────────────────
    elif [[ "$SHELL" == *"zsh"* ]]; then
        SHELL_CONFIG="$HOME/.zshrc"
    # ── Bash ────────────────────────────────────────────────────────────────
    elif [[ "$SHELL" == *"bash"* ]]; then
        SHELL_CONFIG="$HOME/.bashrc"
        [ ! -f "$SHELL_CONFIG" ] && SHELL_CONFIG="$HOME/.bash_profile"
    # ── Fallback: probe existing files (fish first) ──────────────────────────
    else
        if [ -d "$HOME/.config/fish" ]; then
            FISH_CONFIG="$HOME/.config/fish/conf.d/hermes.fish"
        elif [ -f "$HOME/.zshrc" ]; then
            SHELL_CONFIG="$HOME/.zshrc"
        elif [ -f "$HOME/.bashrc" ]; then
            SHELL_CONFIG="$HOME/.bashrc"
        elif [ -f "$HOME/.bash_profile" ]; then
            SHELL_CONFIG="$HOME/.bash_profile"
        fi
    fi

    # ── Write fish PATH entry ────────────────────────────────────────────────
    if [ -n "$FISH_CONFIG" ]; then
        mkdir -p "$(dirname "$FISH_CONFIG")"
        if [ -f "$FISH_CONFIG" ] && grep -q '\.local/bin' "$FISH_CONFIG" 2>/dev/null; then
            echo -e "${GREEN}✓${NC} ~/.local/bin already in $FISH_CONFIG"
        else
            {
                echo "# Hermes Agent — ensure ~/.local/bin is on PATH"
                echo 'fish_add_path "$HOME/.local/bin"'
            } >> "$FISH_CONFIG"
            echo -e "${GREEN}✓${NC} Added ~/.local/bin to PATH in $FISH_CONFIG"
        fi
    fi

    # ── Write posix PATH entry ───────────────────────────────────────────────
    if [ -n "$SHELL_CONFIG" ]; then
        touch "$SHELL_CONFIG" 2>/dev/null || true
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
    echo "     hermes setup"
    echo ""
    echo "  2. Start chatting:"
    echo "     hermes"
    echo ""
else
    echo "  1. Reload your shell:"
    if [ -n "$FISH_CONFIG" ]; then
        echo "     source $FISH_CONFIG"
        echo "     # or just open a new terminal"
    elif [ -n "$SHELL_CONFIG" ]; then
        echo "     source $SHELL_CONFIG"
    else
        echo "     Open a new terminal"
    fi
    echo ""
    echo "  2. Run the setup wizard to configure API keys:"
    echo "     hermes setup"
    echo ""
    echo "  3. Start chatting:"
    echo "     hermes"
    echo ""
fi
echo "Other commands:"
echo "  hermes status        # Check configuration"
if is_termux; then
    echo "  hermes gateway       # Run gateway in foreground"
else
    echo "  hermes gateway install # Install gateway service (messaging + cron)"
fi
echo "  hermes cron list     # View scheduled jobs"
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
