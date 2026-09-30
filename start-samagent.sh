#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# SamAgent Local Platform — One-Step Local Project Launcher (macOS / Linux)
# ============================================================================
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

VENV_DIR="${SAMAGENT_VENV:-$REPO_DIR/.samagent-venv}"
PORT="${SAMAGENT_PORT:-8080}"
DEV_APP_PORT="${SAMAGENT_DEV_PORT:-3000}"

echo "============================================================"
echo " Starting SamAgent Local Platform"
echo "============================================================"

# 1. Create isolated local Python environment if needed
if [ ! -x "$VENV_DIR/bin/python" ]; then
  echo "[1/3] Creating local Python environment in $VENV_DIR ..."
  python3 -m venv "$VENV_DIR"
  "$VENV_DIR/bin/pip" install --upgrade pip -q
  "$VENV_DIR/bin/pip" install -q pyyaml fastapi uvicorn httpx pytest snowballstemmer pydantic "ruamel.yaml"
fi

# 2. Install VS Code extension & Desktop launchers + seed ~/SamAgentProjects
echo "[2/3] Syncing VS Code extension & local workspace (~/SamAgentProjects)..."
PYTHONPATH="$REPO_DIR" "$VENV_DIR/bin/python" - <<PY
from samagent.platform_installer import install_os_desktop_platform
install_os_desktop_platform(port=int("${PORT}"))
PY

# 3. Open browser automatically once port 8080 is ready
(
  sleep 1.5
  if command -v open >/dev/null 2>&1; then
    open "http://127.0.0.1:${PORT}"
  elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open "http://127.0.0.1:${PORT}" >/dev/null 2>&1 || true
  fi
) &

echo "[3/3] SamAgent Local Platform running at: http://127.0.0.1:${PORT}"
echo "      Local Projects folder on disk     : ~/SamAgentProjects"
echo "      Press Ctrl+C to stop."
echo "============================================================"

PYTHONPATH="$REPO_DIR" exec "$VENV_DIR/bin/python" -m samagent.ui_server --host 127.0.0.1 --port "$PORT"
