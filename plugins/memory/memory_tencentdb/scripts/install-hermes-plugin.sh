#!/usr/bin/env bash
set -euo pipefail

log() { printf '[install-hermes-plugin-v2] %s\n' "$*" >&2; }
fail() { printf '[install-hermes-plugin-v2][ERROR] %s\n' "$*" >&2; exit 1; }
need_cmd() { command -v "$1" >/dev/null 2>&1 || fail "missing command: $1"; }

# Target user/home detection follows the legacy Hermes installer convention:
#   1. INSTALL_AS_USER, 2. SUDO_USER, 3. current user.
USERNAME="${INSTALL_AS_USER:-${SUDO_USER:-$(whoami)}}"
USER_HOME="$(eval echo "~$USERNAME")"

FORCE="${FORCE:-0}"
ALLOW_SYSTEM_PYTHON="${ALLOW_SYSTEM_PYTHON:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROVIDER_SRC="${HERMES_PROVIDER_SRC:-$REPO_ROOT/hermes-plugin/memory/memory_tencentdb}"

HERMES_HOME="${HERMES_HOME:-$USER_HOME/.hermes}"
HERMES_AGENT_DIR="${HERMES_AGENT_DIR:-$HERMES_HOME/hermes-agent}"
HERMES_CONFIG="${HERMES_CONFIG:-$HERMES_HOME/config.yaml}"
HERMES_ENV="${HERMES_ENV:-$HERMES_HOME/.env}"
HERMES_MEMORY_PLUGIN_DIR="${HERMES_MEMORY_PLUGIN_DIR:-$HERMES_AGENT_DIR/plugins/memory}"
PROVIDER_TARGET="$HERMES_MEMORY_PLUGIN_DIR/memory_tencentdb"

TDAI_MEMORY_ENDPOINT="${TDAI_MEMORY_ENDPOINT:-http://127.0.0.1:8420}"
TDAI_MEMORY_API_KEY="${TDAI_MEMORY_API_KEY:-local}"
TDAI_MEMORY_SERVICE_ID="${TDAI_MEMORY_SERVICE_ID:-default}"
WRITE_HERMES_ENV="${WRITE_HERMES_ENV:-1}"
WRITE_HERMES_CONFIG="${WRITE_HERMES_CONFIG:-1}"

if [[ ! -d "$PROVIDER_SRC" ]]; then
  fail "Hermes provider directory not found: $PROVIDER_SRC"
fi

if [[ ! -d "$HERMES_AGENT_DIR" ]]; then
  log "WARN: Hermes agent dir not found: $HERMES_AGENT_DIR"
  log "      Set HERMES_AGENT_DIR if Hermes is installed elsewhere."
fi

log "Installing Hermes provider"
mkdir -p "$HERMES_MEMORY_PLUGIN_DIR"
if [[ -e "$PROVIDER_TARGET" || -L "$PROVIDER_TARGET" ]]; then
  if [[ "$FORCE" == "1" ]]; then
    rm -rf "$PROVIDER_TARGET"
  else
    fail "target already exists: $PROVIDER_TARGET (set FORCE=1 to overwrite)"
  fi
fi
ln -s "$PROVIDER_SRC" "$PROVIDER_TARGET"
log "Provider linked: $PROVIDER_TARGET -> $PROVIDER_SRC"

log "Checking Hermes config"
if [[ "$WRITE_HERMES_CONFIG" == "1" ]]; then
  log "Enabling memory.provider=memory_tencentdb in $HERMES_CONFIG"
  mkdir -p "$(dirname "$HERMES_CONFIG")"
  if [[ -f "$HERMES_CONFIG" ]]; then
    cp "$HERMES_CONFIG" "$HERMES_CONFIG.bak.$(date +%Y%m%d%H%M%S)"
  fi

  HERMES_CONFIG="$HERMES_CONFIG" python3 <<'PY'
import os
import re
from pathlib import Path

path = Path(os.environ["HERMES_CONFIG"])
provider_line = "  provider: memory_tencentdb"


def update_with_pyyaml(text: str) -> str:
    import yaml
    data = yaml.safe_load(text) if text.strip() else {}
    if not isinstance(data, dict):
        data = {}
    memory = data.get("memory")
    if not isinstance(memory, dict):
        memory = {}
    data["memory"] = memory
    memory["provider"] = "memory_tencentdb"
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)


def update_minimal(text: str) -> str:
    lines = text.splitlines()
    memory_start = None
    memory_end = None
    for i, line in enumerate(lines):
        if re.match(r"^memory\s*:\s*(#.*)?$", line):
            memory_start = i
            memory_end = len(lines)
            for j in range(i + 1, len(lines)):
                if lines[j] and not lines[j].startswith((" ", "\t")):
                    memory_end = j
                    break
            break

    if memory_start is None:
        if lines and lines[-1].strip():
            lines.append("")
        lines.extend(["memory:", provider_line])
        return "\n".join(lines) + "\n"

    for i in range(memory_start + 1, memory_end):
        if re.match(r"^\s*provider\s*:", lines[i]):
            indent = re.match(r"^(\s*)", lines[i]).group(1) or "  "
            lines[i] = f"{indent}provider: memory_tencentdb"
            return "\n".join(lines) + "\n"

    insert_at = memory_start + 1
    lines.insert(insert_at, provider_line)
    return "\n".join(lines) + "\n"


text = path.read_text() if path.exists() else ""
try:
    updated = update_with_pyyaml(text)
except Exception:
    updated = update_minimal(text)
path.write_text(updated)
PY
else
  if [[ -f "$HERMES_CONFIG" ]] && sed -n '/^memory:/,/^[[:alpha:]_][[:alnum:]_]*:/p' "$HERMES_CONFIG" | grep -q 'provider: memory_tencentdb'; then
    log "memory.provider already set to memory_tencentdb"
  else
    log "Provider installed but NOT enabled because WRITE_HERMES_CONFIG=$WRITE_HERMES_CONFIG. Add/edit in $HERMES_CONFIG:"
    cat >&2 <<'EOF'

memory:
  provider: memory_tencentdb
EOF
  fi
fi

_update_env() {
  local key="$1"
  local value="$2"
  local file="$3"
  mkdir -p "$(dirname "$file")"
  touch "$file"
  local tmp
  tmp="$(mktemp)"
  grep -v -E "^(# *)?${key}=" "$file" > "$tmp" || true
  local escaped="$value"
  escaped="${escaped//\\/\\\\}"
  escaped="${escaped//\"/\\\"}"
  printf '%s="%s"\n' "$key" "$escaped" >> "$tmp"
  mv "$tmp" "$file"
}

if [[ "$WRITE_HERMES_ENV" == "1" ]]; then
  log "Writing Memory SDK env vars to $HERMES_ENV"
  _update_env "TDAI_MEMORY_ENDPOINT" "$TDAI_MEMORY_ENDPOINT" "$HERMES_ENV"
  _update_env "TDAI_MEMORY_API_KEY" "$TDAI_MEMORY_API_KEY" "$HERMES_ENV"
  _update_env "TDAI_MEMORY_SERVICE_ID" "$TDAI_MEMORY_SERVICE_ID" "$HERMES_ENV"
fi

cat >&2 <<EOF

[install-hermes-plugin-v2] Done.
Provider installed at:
  $PROVIDER_TARGET

Hermes config:
  $HERMES_CONFIG
  memory.provider = memory_tencentdb

Gateway env (written to $HERMES_ENV):
  TDAI_MEMORY_ENDPOINT="$TDAI_MEMORY_ENDPOINT"
  TDAI_MEMORY_API_KEY="$TDAI_MEMORY_API_KEY"
  TDAI_MEMORY_SERVICE_ID="$TDAI_MEMORY_SERVICE_ID"
EOF
