#!/usr/bin/env bash
set -euo pipefail

VERSION="${LITELLM_VERSION:-1.94.0}"
HERMES_HOME="${HERMES_HOME:-}"
if [[ -z "$HERMES_HOME" ]]; then
  CONFIG_PATH="$(hermes config path)"
  HERMES_HOME="$(dirname "$CONFIG_PATH")"
fi
ROOT="$HERMES_HOME/integrations/litellm"
VENV="$ROOT/.venv"

command -v uv >/dev/null || { echo "uv is required" >&2; exit 1; }
mkdir -p "$ROOT"
uv venv "$VENV"
uv pip install --python "$VENV/bin/python" "litellm[proxy]==$VERSION" pip-audit

# PYSEC-2026-2447 affected optional diskcache 5.6.3 with no fixed release at
# integration time. LiteLLM's process-local memory cache does not require it.
uv pip uninstall --python "$VENV/bin/python" diskcache >/dev/null 2>&1 || true

SITE="$VENV/lib/python$("$VENV/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')/site-packages"
"$VENV/bin/pip-audit" --path "$SITE" --format json --output "$ROOT/pip-audit.json"
"$VENV/bin/python" -c 'import litellm; print("LiteLLM import OK")'
printf 'Installed LiteLLM %s in %s\n' "$VERSION" "$VENV"
