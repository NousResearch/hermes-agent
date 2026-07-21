#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HERMES_HOME="${HERMES_HOME:-${HOME}/.hermes}"
PLUGIN_NAME="dreamcycle"
PLUGIN_DST="${HERMES_HOME}/plugins/${PLUGIN_NAME}"
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
PLUGIN_SRC="${REPO_ROOT}/hermes-hermes-integration/dreamcycle-hermes-plugin/__init__.py"

if [[ ! -f "${PLUGIN_SRC}" ]]; then
  echo "[error] Plugin source missing: ${PLUGIN_SRC}" >&2
  exit 1
fi

mkdir -p "${PLUGIN_DST}"
cp "${PLUGIN_SRC}" "${PLUGIN_DST}/__init__.py"
chmod 644 "${PLUGIN_DST}/__init__.py"

echo "Installed DreamCycle Hermes memory provider plugin to: ${PLUGIN_DST}"
echo

echo "Now set these environment variables in ${HERMES_HOME}/.env:"
echo "  DREAMCYCLE_BASE_URL=http://127.0.0.1:8765"
echo "  DREAMCYCLE_API_KEY=<your-sidecar-key>"
echo "  DREAMCYCLE_NAMESPACE=<your-namespace>"
echo "  DREAMCYCLE_USER_ID=<your-user-id>"
echo "  DREAMCYCLE_SOURCE=hermes"
echo "  DREAMCYCLE_HTTP_TIMEOUT=8"
echo
cat <<'EOF'

Then enable:
  hermes config set memory.provider dreamcycle
  hermes memory status
EOF
