#!/bin/bash
# =============================================================================
# Hermes Agent Tarball Builder
# =============================================================================
# Builds a pre-built tarball for distribution.
# Usage: bash scripts/build-tarball.sh [options]
#
# Options:
#   --version VERSION    Override version (default: from pyproject.toml)
#   --output DIR        Output directory (default: ./dist)
#   --no-node           Skip Node.js dependencies (smaller tarball)
#   --no-playwright     Skip Playwright browser installation
#   --target TARGET     Build target: linux-x64, linux-arm64, macos-x64, macos-arm64 (default: host)
# =============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="${PYTHON:-python3}"

# Defaults
VERSION=""
OUTPUT_DIR="$REPO_ROOT/dist"
INCLUDE_NODE=true
INCLUDE_PLAYWRIGHT=true
TARGET=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --version) VERSION="$2"; shift 2 ;;
        --output) OUTPUT_DIR="$2"; shift 2 ;;
        --no-node) INCLUDE_NODE=false; shift ;;
        --no-playwright) INCLUDE_PLAYWRIGHT=false; shift ;;
        --target) TARGET="$2"; shift 2 ;;
        -h|--help)
            head -15 "$0" | tail -13
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Resolve version from pyproject.toml if not provided
if [ -z "$VERSION" ]; then
    VERSION=$(grep -E '^version\s*=' "$REPO_ROOT/pyproject.toml" | head -1 | sed 's/.*"\(.*\)".*/\1/')
    if [ -z "$VERSION" ]; then
        echo "Could not determine version from pyproject.toml"
        exit 1
    fi
fi

# Detect host architecture if target not specified
if [ -z "$TARGET" ]; then
    ARCH=$(uname -m)
    OS=$(uname -s | tr '[:upper:]' '[:lower:]')
    
    case "$ARCH" in
        x86_64)  ARCH_SUFFIX="x64" ;;
        arm64|aarch64) ARCH_SUFFIX="arm64" ;;
        *) echo "Unsupported architecture: $ARCH"; exit 1 ;;
    esac
    
    case "$OS" in
        linux)  TARGET="linux-${ARCH_SUFFIX}" ;;
        darwin) TARGET="macos-${ARCH_SUFFIX}" ;;
        *) echo "Unsupported OS: $OS"; exit 1 ;;
    esac
fi

TARBALL_NAME="hermes-agent-${VERSION}-${TARGET}.tar.gz"
TARBALL_DIR="hermes-agent-${VERSION}"
BUILD_DIR=$(mktemp -d)

echo -e "${CYAN}${BOLD}══════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}${BOLD}  Hermes Agent Tarball Builder${NC}"
echo -e "${CYAN}${BOLD}══════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "  Version: ${YELLOW}${VERSION}${NC}"
echo -e "  Target:  ${YELLOW}${TARGET}${NC}"
echo -e "  Output:  ${YELLOW}${OUTPUT_DIR}/${TARBALL_NAME}${NC}"
echo ""

# Clean and prepare output directory
mkdir -p "$OUTPUT_DIR"
rm -f "$OUTPUT_DIR/$TARBALL_NAME"

# Create staging directory
echo -e "${BLUE}→${NC} Preparing build staging area..."
mkdir -p "$BUILD_DIR/$TARBALL_DIR"

# Copy source code (excluding unnecessary files)
echo -e "${BLUE}→${NC} Copying source code..."
rsync -a --exclude='.git' \
         --exclude='__pycache__' \
         --exclude='*.pyc' \
         --exclude='.pytest_cache' \
         --exclude='node_modules' \
         --exclude='dist' \
         --exclude='*.egg-info' \
         --exclude='.venv' \
         --exclude='venv' \
         "$REPO_ROOT/" "$BUILD_DIR/$TARBALL_DIR/"

cd "$BUILD_DIR/$TARBALL_DIR"

# Create virtual environment
echo -e "${BLUE}→${NC} Creating virtual environment..."
$PYTHON -m venv venv
source venv/bin/activate

# Install dependencies
echo -e "${BLUE}→${NC} Installing Python dependencies..."
pip install --upgrade pip setuptools wheel > /dev/null 2>&1

# Install package with all optional dependencies
pip install -e ".[all]" 2>&1 | tail -1

# Verify installation
echo -e "${BLUE}→${NC} Verifying installation..."
if ! python -c "import hermes_cli; print(f'  hermes_cli version: {hermes_cli.__version__}')"; then
    echo -e "${RED}✗${NC} Failed to verify hermes_cli installation"
    exit 1
fi

# Install Node.js dependencies if requested
if [ "$INCLUDE_NODE" = true ] && [ -f package.json ]; then
    echo -e "${BLUE}→${NC} Installing Node.js dependencies..."
    if command -v node &> /dev/null; then
        npm install --silent 2>/dev/null || {
            echo -e "${YELLOW}⚠${NC} npm install failed (browser tools may not work)"
        }
        
        # Install Playwright if requested
        if [ "$INCLUDE_PLAYWRIGHT" = true ] && command -v npx &> /dev/null; then
            echo -e "${BLUE}→${NC} Installing Playwright Chromium..."
            npx playwright install chromium 2>/dev/null || {
                echo -e "${YELLOW}⚠${NC} Playwright installation failed"
            }
        fi
    else
        echo -e "${YELLOW}⚠${NC} Node.js not found, skipping Node.js dependencies"
    fi
fi

# Install TUI dependencies
if [ -f ui-tui/package.json ]; then
    echo -e "${BLUE}→${NC} Installing TUI dependencies..."
    cd ui-tui
    npm install --silent 2>/dev/null || {
        echo -e "${YELLOW}⚠${NC} TUI npm install failed"
    }
    cd ..
fi

# Create version metadata
echo -e "${BLUE}→${NC} Creating version metadata..."
cat > .build-version.json << EOF
{
    "version": "${VERSION}",
    "target": "${TARGET}",
    "build_date": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "python_version": "$($PYTHON --version 2>&1 | awk '{print $2}')",
    "include_node": $([ "$INCLUDE_NODE" = true ] && echo "true" || echo "false"),
    "include_playwright": $([ "$INCLUDE_PLAYWRIGHT" = true ] && echo "true" || echo "false")
}
EOF

# Create installation manifest
echo -e "${BLUE}→${NC} Creating installation manifest..."
python - <<'PYEOF'
import json
import os
from pathlib import Path

# Find all Python packages
packages = []
for root, dirs, files in os.walk('.'):
    # Skip virtual environment and build directories
    dirs[:] = [d for d in dirs if d not in ['venv', '__pycache__', '.pytest_cache', 'node_modules']]
    
    if '__init__.py' in files:
        # Check if this is a top-level package
        parent = Path(root).parent
        if (parent / '__init__.py').exists():
            continue  # Skip subpackages
        packages.append(root[2:])  # Remove './'

# Create manifest
manifest = {
    "version": Path("pyproject.toml").read_text().split('version = "')[1].split('"')[0],
    "packages": packages,
    "entry_points": {
        "hermes": "hermes_cli.main:main",
        "hermes-agent": "run_agent:main",
        "hermes-acp": "acp_adapter.entry:main"
    },
    "config_files": [
        ".env.example",
        "cli-config.yaml.example",
        "SOUL.md"
    ],
    "skills_dir": "skills",
    "tools_dir": "tools"
}

with open('manifest.json', 'w') as f:
    json.dump(manifest, f, indent=2)
print("  ✓ Created manifest.json")
PYEOF

# Create a simple install script for the tarball
echo -e "${BLUE}→${NC} Creating embedded installer..."
cat > install-from-tarball.sh << 'INSTALL_EOF'
#!/bin/bash
# Hermes Agent Tarball Installer
# Usage: bash install-from-tarball.sh [OPTIONS]
#
# Options:
#   --dir PATH        Installation directory (default: ~/.hermes/hermes-agent)
#   --hermes-home PATH  Hermes home directory (default: ~/.hermes)
#   --skip-setup      Skip interactive setup wizard
#   --no-venv         Don't create virtual environment
#   -h, --help        Show this help

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# Defaults
HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
INSTALL_DIR=""
SKIP_SETUP=false
USE_VENV=true

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dir) INSTALL_DIR="$2"; shift 2 ;;
        --hermes-home) HERMES_HOME="$2"; shift 2 ;;
        --skip-setup) SKIP_SETUP=true; shift ;;
        --no-venv) USE_VENV=false; shift ;;
        -h|--help) head -10 "$0" | tail -8; exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Set default install dir
INSTALL_DIR="${INSTALL_DIR:-$HERMES_HOME/hermes-agent}"

echo -e "${CYAN}"
echo "┌─────────────────────────────────────────────────────────┐"
echo "│             ⚕ Hermes Agent Tarball Installer           │"
echo "└─────────────────────────────────────────────────────────┘"
echo -e "${NC}"

# Check Python version
echo -e "${CYAN}→${NC} Checking Python version..."
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}✗${NC} Python 3 not found"
    echo "  Please install Python 3.11 or later"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; then
    echo -e "${GREEN}✓${NC} Python $PYTHON_VERSION found"
else
    echo -e "${RED}✗${NC} Python 3.11+ required (found $PYTHON_VERSION)"
    exit 1
fi

# Create installation directory
mkdir -p "$INSTALL_DIR"

# Copy files
echo -e "${CYAN}→${NC} Installing to $INSTALL_DIR..."
cp -r ./* "$INSTALL_DIR/"
cp -r ./.build-version.json "$INSTALL_DIR/" 2>/dev/null || true
cp -r ./manifest.json "$INSTALL_DIR/" 2>/dev/null || true

# Create virtual environment if requested
if [ "$USE_VENV" = true ]; then
    echo -e "${CYAN}→${NC} Creating virtual environment..."
    cd "$INSTALL_DIR"
    if [ -d "venv" ]; then
        rm -rf venv
    fi
    python3 -m venv venv
    
    # Activate and install package
    source venv/bin/activate
    pip install --upgrade pip > /dev/null 2>&1
    pip install -e ".[all]" 2>&1 | tail -1
fi

# Create symlink
echo -e "${CYAN}→${NC} Setting up hermes command..."
if [ "$USE_VENV" = true ]; then
    HERMES_BIN="$INSTALL_DIR/venv/bin/hermes"
else
    HERMES_BIN="$INSTALL_DIR/venv/bin/hermes"  # Fallback
fi

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
            echo -e "${GREEN}✓${NC} Added ~/.local/bin to PATH in $SHELL_CONFIG"
        fi
    fi
fi

# Run setup wizard
if [ "$SKIP_SETUP" = false ]; then
    echo ""
    echo -e "${CYAN}→${NC} Running setup wizard..."
    cd "$INSTALL_DIR"
    "$INSTALL_DIR/venv/bin/python" -m hermes_cli.main setup
fi

echo ""
echo -e "${GREEN}${BOLD}✓ Installation complete!${NC}"
echo ""
echo "  Config:    $HERMES_HOME/config.yaml"
echo "  API Keys:  $HERMES_HOME/.env"
echo "  Code:      $INSTALL_DIR"
echo ""
echo "  Start chatting: hermes"
echo "  Run setup:     hermes setup"
echo ""
INSTALL_EOF
chmod +x install-from-tarball.sh

# Deactivate virtual environment
deactivate 2>/dev/null || true

# Create tarball
echo ""
echo -e "${BLUE}→${NC} Creating tarball..."
cd "$BUILD_DIR"
tar -czf "$OUTPUT_DIR/$TARBALL_NAME" "$TARBALL_DIR"

# Get tarball size
TARBALL_SIZE=$(ls -lh "$OUTPUT_DIR/$TARBALL_NAME" | awk '{print $5}')

echo ""
echo -e "${GREEN}${BOLD}══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}${BOLD}  ✓ Tarball created successfully!${NC}"
echo -e "${GREEN}${BOLD}══════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "  File:     ${YELLOW}$OUTPUT_DIR/$TARBALL_NAME${NC}"
echo -e "  Size:     ${YELLOW}$TARBALL_SIZE${NC}"
echo -e "  Version:  ${YELLOW}$VERSION${NC}"
echo -e "  Target:   ${YELLOW}$TARGET${NC}"
echo ""
echo -e "  Install with:"
echo -e "    ${CYAN}curl -fsSL https://github.com/NousResearch/hermes-agent/releases/download/v${VERSION}/${TARBALL_NAME} | tar -xzf - && cd hermes-agent-${VERSION} && bash install-from-tarball.sh${NC}"
echo ""

# Cleanup
rm -rf "$BUILD_DIR"

exit 0
