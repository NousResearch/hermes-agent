#!/bin/bash
# ============================================================
# setup-pi2.sh
# Minimal install script for Raspberry Pi 2B (ARMv7, 1GB RAM)
# Usage: bash setup-pi2.sh
# ============================================================

set -e

# Auto-detect Python version (Pi2 might not have 3.10)
# Use python3 without version suffix to ensure compatibility
PYTHON="python3"

# Verify Python version
PYTHON_VERSION=$($PYTHON --version 2>&1 | cut -d' ' -f2)
echo "==> Using Python: $PYTHON_VERSION"

VENV_DIR="$HOME/hermes-venv"
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "==> [Pi2] Creating virtual environment ($VENV_DIR)"
$PYTHON -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo "==> [Pi2] Upgrading pip"
pip install --upgrade pip

echo "==> [Pi2] Installing core dependencies (heavy local ML/RAG packages excluded)"
pip install \
  openai==2.24.0 \
  certifi \
  python-dotenv \
  fire \
  "httpx[socks]" \
  rich \
  tenacity \
  pyyaml \
  ruamel.yaml \
  requests \
  jinja2 \
  pydantic \
  prompt_toolkit \
  croniter \
  packaging \
  Markdown \
  "PyJWT[crypto]" \
  "urllib3>=2.7.0,<3" \
  psutil \
  websockets \
  pathspec \
  ptyprocess \
  pypdf \
  beautifulsoup4

echo ""
echo "==> [Pi2] Core installation completed."
echo ""
echo "Optional packages (install only when needed):"
echo ""
echo "  # Anthropic Claude API"
echo "  pip install anthropic"
echo ""
echo "  # MCP tools"
echo "  pip install mcp starlette"
echo ""
echo "  # Voice input (requires libportaudio; recommended to skip on Pi2)"
echo "  sudo apt install portaudio19-dev -y"
echo "  pip install sounddevice numpy"
echo "  # faster-whisper requires source compilation on ARMv7; not recommended"
echo ""
echo "  # Telegram Bot"
echo "  pip install 'python-telegram-bot[webhooks]'"
echo ""
echo "  # Slack Bot"
echo "  pip install slack-bolt slack-sdk aiohttp"
echo ""
echo "  # Web API (FastAPI) - use native asyncio uvicorn; skip uvicorn[standard]/uvloop on Pi2"
echo "  pip install fastapi uvicorn"
echo ""
echo "  # Semantic RAG / embeddings (recommended: run remotely, not on Pi2)"
echo "  # Avoid local torch, sentence-transformers, and chromadb on ARMv7/1GB RAM."
echo "  # Use remote embeddings/cloud memory, or host Chroma/Qdrant/sentence-transformers on another machine."
echo ""
echo "  # Google API integrations"
echo "  pip install google-api-python-client google-auth-oauthlib"
echo ""

# ---- Configure .env ----
if [ ! -f "$REPO_DIR/.env" ]; then
  if [ -f "$REPO_DIR/.env.example" ]; then
    cp "$REPO_DIR/.env.example" "$REPO_DIR/.env"
    chmod 600 "$REPO_DIR/.env"
    echo "==> Created .env from .env.example; please fill in API keys"
  fi
fi

# ---- Create hermes command wrapper ----
echo "==> [Pi2] Checking old hermes command"
LINK_TARGET="$HOME/.local/bin/hermes"
mkdir -p "$HOME/.local/bin"

# Remove existing files and broken symlinks before writing the wrapper.
# A broken symlink makes -e/-f false but redirection still follows it and fails.
if [ -e "$LINK_TARGET" ] || [ -L "$LINK_TARGET" ]; then
  echo "Found old hermes command: $LINK_TARGET"
  if [ -L "$LINK_TARGET" ]; then
    echo "Removing symlink: $LINK_TARGET"
  else
    echo "Removing file: $LINK_TARGET"
  fi
  rm -f "$LINK_TARGET"
fi

if [ -f "$REPO_DIR/cli.py" ]; then
  cat > "$LINK_TARGET" <<EOF
#!/bin/bash
source "$VENV_DIR/bin/activate"
"$VENV_DIR/bin/python" "$REPO_DIR/cli.py" "\$@"
EOF
  chmod +x "$LINK_TARGET"
  echo "==> Installed 'hermes' command at $LINK_TARGET"
fi

echo ""
echo "==> Done! Run 'source ~/.bashrc && hermes' to start"
