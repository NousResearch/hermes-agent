#!/usr/bin/env bash
set -euo pipefail

# SamAgent Local Platform & VS Code Bridge Installer (macOS / Linux)
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${SAMAGENT_PORT:-8080}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "============================================================"
echo " Installing SamAgent Local Platform & VS Code Bridge"
echo "============================================================"

PYTHONPATH="$REPO_DIR" "$PYTHON_BIN" - <<PY
from samagent.platform_installer import install_os_desktop_platform
import json
res = install_os_desktop_platform(port=int("${PORT}"))
print(json.dumps(res, indent=2))
PY

echo ""
echo "SamAgent Local Platform installed!"
echo " - Local Platform URL : http://127.0.0.1:${PORT}"
echo " - Local Workspaces   : ~/SamAgentProjects"
echo " - VS Code Extension  : ~/.vscode/extensions/samjuniors.samagent-vscode-0.1.0"
echo " - macOS App Bundle   : ~/Applications/SamAgent.app"
echo " - Linux Desktop App  : ~/.local/share/applications/samagent-platform.desktop"
