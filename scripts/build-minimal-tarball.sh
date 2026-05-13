#!/bin/bash
# =============================================================================
# Hermes Agent — Minimal Tarball Builder (for ADA Desktop App)
# =============================================================================
# Builds a lightweight tarball containing only the venv + source needed for
# the gateway runtime. Target size: ~80 MB instead of the full 500 MB+ build.
#
# Tarball structure (matches ADA Rust expectations):
#   hermes-agent/            ← top-level directory
#     bin/hermes             ← symlink → ../venv/bin/hermes
#     venv/                  ← Python virtualenv with gateway deps
#     src/                   ← hermes-agent source (agent, gateway, tools, etc.)
#     .build-version.json    ← version metadata
#
# The ADA Rust installer (hermes.rs) extracts to:
#   <runtime_dir>/hermes-agent/bin/hermes
# and verifies the binary exists + runs `hermes --version`.
#
# Usage:
#   bash scripts/build-minimal-tarball.sh [options]
#
# Options:
#   --version VERSION    Override version (default: from pyproject.toml)
#   --output DIR         Output directory (default: ./dist)
#   --target TARGET      Build target: linux-x86_64, linux-arm64, darwin-x86_64, darwin-arm64
#   --extras EXTRAS      Pip extras to install (default: "cron,pty,mcp,cli")
#   -h, --help           Show this help
# =============================================================================

set -euo pipefail

# ── Colors ──
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="${PYTHON:-python3}"

# ── Defaults ──
VERSION=""
OUTPUT_DIR="$REPO_ROOT/dist"
TARGET=""
EXTRAS="cron,pty,mcp,cli"

# ── Parse arguments ──
while [[ $# -gt 0 ]]; do
    case $1 in
        --version) VERSION="$2"; shift 2 ;;
        --output) OUTPUT_DIR="$2"; shift 2 ;;
        --target) TARGET="$2"; shift 2 ;;
        --extras) EXTRAS="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,20p' "$0" | sed 's/^# //; s/^#//'
            exit 0
            ;;
        *) echo -e "${RED}Unknown option: $1${NC}"; exit 1 ;;
    esac
done

# ── Ensure OUTPUT_DIR is absolute (before any cd) ──
if [[ "$OUTPUT_DIR" != /* ]]; then
    OUTPUT_DIR="${REPO_ROOT}/${OUTPUT_DIR}"
fi

# ── Resolve version ──
if [ -z "$VERSION" ]; then
    VERSION=$(grep -E '^version\s*=' "$REPO_ROOT/pyproject.toml" | head -1 | sed 's/.*"\(.*\)".*/\1/')
    if [ -z "$VERSION" ]; then
        echo -e "${RED}✗${NC} Cannot determine version from pyproject.toml"
        exit 1
    fi
fi

# ── Detect platform ──
if [ -z "$TARGET" ]; then
    ARCH=$(uname -m)
    OS=$(uname -s | tr '[:upper:]' '[:lower:]')

    case "$ARCH" in
        arm64|aarch64) ARCH_SUFFIX="arm64" ;;
        # Keep x86_64 as-is to match Rust's std::env::consts::ARCH normalization
        *) ARCH_SUFFIX="$ARCH" ;;
    esac

    # ADA Rust normalizes macos→darwin in tarball_filename()
    case "$OS" in
        linux)  TARGET="linux-${ARCH_SUFFIX}" ;;
        darwin) TARGET="darwin-${ARCH_SUFFIX}" ;;
        *) echo -e "${RED}✗${NC} Unsupported OS: $OS"; exit 1 ;;
    esac
fi

# ── Tarball naming ──
# Must match ADA's tarball_filename() pattern: hermes-venv-{os}-{arch}.tar.gz
# where os ∈ {darwin, linux}, arch ∈ {arm64, x64}
TARBALL_NAME="hermes-venv-${TARGET}.tar.gz"
TARBALL_DIR="hermes-agent"  # Fixed dirname — ADA expects <runtime>/hermes-agent/
BUILD_DIR=$(mktemp -d)

echo -e "${CYAN}${BOLD}══════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}${BOLD}  Hermes Agent — Minimal Tarball Builder (for ADA)${NC}"
echo -e "${CYAN}${BOLD}══════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "  Version:   ${YELLOW}${VERSION}${NC}"
echo -e "  Target:    ${YELLOW}${TARGET}${NC}"
echo -e "  Extras:    ${YELLOW}[${EXTRAS}]${NC}"
echo -e "  Output:    ${YELLOW}${OUTPUT_DIR}/${TARBALL_NAME}${NC}"
echo ""

# ── Check Python version ──
echo -e "${BLUE}→${NC} Checking Python version..."
if ! command -v "$PYTHON" &>/dev/null; then
    echo -e "${RED}✗${NC} Python not found: $PYTHON"
    exit 1
fi

PY_VER=$($PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if $PYTHON -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; then
    echo -e "${GREEN}✓${NC} Python $PY_VER found"
else
    echo -e "${RED}✗${NC} Python 3.11+ required (found $PY_VER)"
    exit 1
fi

# ── Prepare staging area ──
echo -e "${BLUE}→${NC} Preparing staging area: ${BUILD_DIR}/${TARBALL_DIR}"
STAGE="${BUILD_DIR}/${TARBALL_DIR}"
mkdir -p "$STAGE"

# ── Copy source (minimal — only what gateway needs) ──
echo -e "${BLUE}→${NC} Copying source files..."
mkdir -p "$STAGE/src"

# Top-level modules (required by pyproject.toml [tool.setuptools] py-modules)
for mod in run_agent.py model_tools.py toolsets.py batch_runner.py \
           trajectory_compressor.py toolset_distributions.py cli.py \
           hermes_constants.py hermes_state.py hermes_time.py \
           hermes_logging.py rl_cli.py utils.py; do
    if [ -f "$REPO_ROOT/$mod" ]; then
        cp "$REPO_ROOT/$mod" "$STAGE/src/"
    fi
done

# Packages (required by [tool.setuptools.packages.find] include list)
for pkg in agent tools hermes_cli gateway tui_gateway cron acp_adapter plugins; do
    if [ -d "$REPO_ROOT/$pkg" ]; then
        cp -r "$REPO_ROOT/$pkg" "$STAGE/src/"
    fi
done

# Copy pyproject.toml + setup files (needed for pip install -e)
cp "$REPO_ROOT/pyproject.toml" "$STAGE/src/"
if [ -f "$REPO_ROOT/setup.py" ]; then
    cp "$REPO_ROOT/setup.py" "$STAGE/src/"
fi
if [ -f "$REPO_ROOT/setup.cfg" ]; then
    cp "$REPO_ROOT/setup.cfg" "$STAGE/src/"
fi
if [ -f "$REPO_ROOT/MANIFEST.in" ]; then
    cp "$REPO_ROOT/MANIFEST.in" "$STAGE/src/"
fi
if [ -f "$REPO_ROOT/README.md" ]; then
    cp "$REPO_ROOT/README.md" "$STAGE/src/"
fi

# ── Create virtualenv ──
echo -e "${BLUE}→${NC} Creating virtualenv..."
$PYTHON -m venv "$STAGE/venv"
source "$STAGE/venv/bin/activate"

# ── Install minimal deps ──
echo -e "${BLUE}→${NC} Upgrading pip..."
pip install --upgrade pip setuptools wheel --quiet 2>&1 | tail -1

echo -e "${BLUE}→${NC} Installing hermes-agent[${EXTRAS}]..."
cd "$STAGE/src"
pip install -e ".[${EXTRAS}]" --quiet 2>&1 | tail -3

# ── Verify installation ──
echo -e "${BLUE}→${NC} Verifying installation..."
if ! python -c "import hermes_cli; print(f'  hermes_cli OK')" 2>/dev/null; then
    echo -e "${RED}✗${NC} hermes_cli import failed"
    deactivate
    rm -rf "$BUILD_DIR"
    exit 1
fi
if ! python -c "import gateway; print(f'  gateway OK')" 2>/dev/null; then
    echo -e "${YELLOW}⚠${NC}  gateway import failed (non-fatal for some extras)"
fi

# Verify hermes binary works
HERMES_BIN="$STAGE/venv/bin/hermes"
if [ ! -f "$HERMES_BIN" ]; then
    echo -e "${RED}✗${NC} hermes binary not found in venv"
    deactivate
    rm -rf "$BUILD_DIR"
    exit 1
fi

INSTALLED_VERSION=$("$HERMES_BIN" --version 2>/dev/null | head -1 || echo "unknown")
echo -e "${GREEN}✓${NC} hermes binary: ${INSTALLED_VERSION}"

# ── Create bin/ symlink (ADA expects <runtime>/hermes-agent/bin/hermes) ──
echo -e "${BLUE}→${NC} Creating bin/hermes symlink..."
mkdir -p "$STAGE/bin"
ln -sf ../venv/bin/hermes "$STAGE/bin/hermes"
ln -sf ../venv/bin/hermes-agent "$STAGE/bin/hermes-agent" 2>/dev/null || true

# ── Create version metadata ──
echo -e "${BLUE}→${NC} Creating .build-version.json..."
cat > "$STAGE/.build-version.json" <<EOF
{
    "version": "${VERSION}",
    "target": "${TARGET}",
    "build_date": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "python_version": "$($PYTHON --version 2>&1 | awk '{print $2}')",
    "extras": "${EXTRAS}",
    "build_type": "minimal",
    "for": "ada-desktop"
}
EOF

# ── Strip venv cruft ──
echo -e "${BLUE}→${NC} Stripping venv cruft..."
find "$STAGE/venv" -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
find "$STAGE/venv" -type d -name '.pytest_cache' -exec rm -rf {} + 2>/dev/null || true
find "$STAGE/venv" -name '*.pyc' -delete 2>/dev/null || true
find "$STAGE/venv" -name '*.pyo' -delete 2>/dev/null || true
# Remove pip/setuptools/wheel caches inside venv
rm -rf "$STAGE/venv/lib"/*/site-packages/pip/_vendor/distlib/*.exe 2>/dev/null || true
rm -rf "$STAGE/venv/lib"/*/site-packages/setuptools/*.dist-info/RECORD 2>/dev/null || true

# Deactivate before tarring
deactivate

# ── Create tarball ──
echo ""
echo -e "${BLUE}→${NC} Creating tarball..."
mkdir -p "$OUTPUT_DIR"
rm -f "$OUTPUT_DIR/$TARBALL_NAME"

cd "$BUILD_DIR"
tar -czf "$OUTPUT_DIR/$TARBALL_NAME" "$TARBALL_DIR"

TARBALL_SIZE=$(ls -lh "$OUTPUT_DIR/$TARBALL_NAME" | awk '{print $5}')

# ── Generate SHA256 ──
echo -e "${BLUE}→${NC} Generating SHA256 checksum..."
if command -v shasum &>/dev/null; then
    SHA256=$(shasum -a 256 "$OUTPUT_DIR/$TARBALL_NAME" | awk '{print $1}')
elif command -v sha256sum &>/dev/null; then
    SHA256=$(sha256sum "$OUTPUT_DIR/$TARBALL_NAME" | awk '{print $1}')
else
    echo -e "${YELLOW}⚠${NC}  Neither shasum nor sha256sum found; skipping SHA256"
    SHA256=""
fi

if [ -n "$SHA256" ]; then
    echo "${SHA256}  ${TARBALL_NAME}" > "$OUTPUT_DIR/${TARBALL_NAME}.sha256"
    echo -e "${GREEN}✓${NC} SHA256: ${SHA256:0:16}..."
fi

echo ""
echo -e "${GREEN}${BOLD}══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}${BOLD}  ✓ Minimal tarball created!${NC}"
echo -e "${GREEN}${BOLD}══════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "  File:     ${YELLOW}${OUTPUT_DIR}/${TARBALL_NAME}${NC}"
echo -e "  SHA256:   ${YELLOW}${OUTPUT_DIR}/${TARBALL_NAME}.sha256${NC}"
echo -e "  Size:     ${YELLOW}${TARBALL_SIZE}${NC}"
echo -e "  Version:  ${YELLOW}${VERSION}${NC}"
echo -e "  Target:   ${YELLOW}${TARGET}${NC}"
echo -e "  Extras:   ${YELLOW}[${EXTRAS}]${NC}"
echo ""
echo -e "  Upload to GitHub Release:"
echo -e "    ${CYAN}gh release upload v${VERSION} ${OUTPUT_DIR}/${TARBALL_NAME} ${OUTPUT_DIR}/${TARBALL_NAME}.sha256${NC}"
echo ""

# ── Cleanup ──
rm -rf "$BUILD_DIR"

exit 0
