#!/usr/bin/env bash
# Prove a recorded lazy feature is restored into a replacement venv.
# Uses real ensure() + apply_ledger() with real PyPI packages.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
if [ -n "${UV:-}" ]; then
    UV="$UV"
elif command -v uv >/dev/null 2>&1; then
    UV="$(command -v uv)"
else
    UV="$HOME/.hermes/bin/uv"
fi

WORK=$(mktemp -d)
FEATURE_HOME="$WORK/home"
V1="$WORK/v1"
V2="$WORK/v2"
mkdir -p "$FEATURE_HOME/state"
trap 'chmod -R u+w "$WORK" 2>/dev/null || true; rm -rf "$WORK"' EXIT

"$UV" venv "$V1" >/dev/null
"$UV" venv "$V2" >/dev/null
"$UV" pip install --python "$V1/bin/python" --no-deps -e "$REPO_ROOT" >/dev/null
"$UV" pip install --python "$V2/bin/python" --no-deps -e "$REPO_ROOT" >/dev/null

export HERMES_HOME="$FEATURE_HOME"

# 1. Install a real allowlisted feature into v1.
"$V1/bin/python" - <<'PY'
from tools.lazy_deps import ensure
ensure("tts.edge", prompt=False)
PY

# 2. Verify it landed (fresh process — metadata can be stale on 3.12).
"$V1/bin/python" -c 'import edge_tts; print("v1 has edge_tts", edge_tts.__version__)'

# 3. v2 should not have it yet.
if "$V2/bin/python" -c 'import edge_tts' 2>/dev/null; then
    echo 'replacement venv unexpectedly already contains the feature' >&2
    exit 1
fi

# 4. Replay the ledger into v2 (run from v2 so no re-exec is needed).
PYTHONPATH="$REPO_ROOT" "$V2/bin/python" - <<'PY'
import json
from tools.lazy_deps import apply_ledger
result = apply_ledger()
assert result.get("tts.edge") == "refreshed", json.dumps(result, sort_keys=True)
PY

# 5. Verify v2 now has the feature (fresh process).
"$V2/bin/python" -c 'import edge_tts; print("v2 has edge_tts", edge_tts.__version__)'

printf 'E2E_PASS: lazy feature restored into replacement venv\n'
