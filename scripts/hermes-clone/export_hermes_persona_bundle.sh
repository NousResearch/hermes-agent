#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Export a reusable Hermes persona bundle from an existing HERMES_HOME.

This script intentionally mirrors Hermes's built-in profile export flow:
- stage a filtered filesystem copy
- archive it as zip

But it extends the built-in default-profile exclusions with additional
persona-bundle exclusions so the result keeps workflows/skills/settings without
carrying over secrets, live state, or bulky host-specific caches.

Included by default:
  - config.yaml (scrubbed of secrets/tokens)
  - SOUL.md
  - skills/
  - skins/
  - scripts/
  - hooks/

Excluded by default:
  - .env
  - auth.json
  - memories/
  - sessions/
  - state.db*
  - logs/
  - gateway routing/runtime files

Usage:
  export_hermes_persona_bundle.sh [options]

Options:
  --source-home PATH        Source Hermes home (default: ${HERMES_HOME:-$HOME/.hermes})
  --output PATH             Output zip path (default: ./hermes-persona-<timestamp>.zip)
  --name NAME               Persona name stored in manifest/README (default: basename of source home)
  --include-memories        Include memories/ directory (off by default)
  --include-cron            Include cron/ directory (off by default)
  --include-hooks           Include hooks/ directory (on by default)
  --exclude-hooks           Exclude hooks/ directory
  --include-scripts         Include scripts/ directory (on by default)
  --exclude-scripts         Exclude scripts/ directory
  -h, --help                Show this help
EOF
}

SOURCE_HOME="${HERMES_HOME:-$HOME/.hermes}"
OUTPUT=""
NAME=""
INCLUDE_MEMORIES=0
INCLUDE_CRON=0
INCLUDE_HOOKS=1
INCLUDE_SCRIPTS=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-home)
      SOURCE_HOME="$2"
      shift 2
      ;;
    --output)
      OUTPUT="$2"
      shift 2
      ;;
    --name)
      NAME="$2"
      shift 2
      ;;
    --include-memories)
      INCLUDE_MEMORIES=1
      shift
      ;;
    --include-cron)
      INCLUDE_CRON=1
      shift
      ;;
    --include-hooks)
      INCLUDE_HOOKS=1
      shift
      ;;
    --exclude-hooks)
      INCLUDE_HOOKS=0
      shift
      ;;
    --include-scripts)
      INCLUDE_SCRIPTS=1
      shift
      ;;
    --exclude-scripts)
      INCLUDE_SCRIPTS=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Missing required command: $1" >&2
    exit 1
  }
}

require_cmd python3
SOURCE_GIT_REF="$(git -C "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)" rev-parse HEAD 2>/dev/null || true)"
SOURCE_HERMES_VERSION="$(hermes --version 2>/dev/null | head -n 1 || true)"

SOURCE_HOME="$(cd "$SOURCE_HOME" && pwd)"
if [[ ! -d "$SOURCE_HOME" ]]; then
  echo "Source home does not exist: $SOURCE_HOME" >&2
  exit 1
fi

if [[ -z "$NAME" ]]; then
  NAME="$(basename "$SOURCE_HOME")"
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
if [[ -z "$OUTPUT" ]]; then
  OUTPUT="$(pwd)/hermes-persona-${NAME}-${TIMESTAMP}.zip"
fi
case "$OUTPUT" in
  *.tar.gz)
    OUTPUT="${OUTPUT%.tar.gz}.zip"
    ;;
  *.tgz)
    OUTPUT="${OUTPUT%.tgz}.zip"
    ;;
  *.tar)
    OUTPUT="${OUTPUT%.tar}.zip"
    ;;
  *.zip)
    ;;
  *)
    OUTPUT="${OUTPUT}.zip"
    ;;
esac
OUTPUT_DIR="$(dirname "$OUTPUT")"
mkdir -p "$OUTPUT_DIR"
OUTPUT="$(cd "$OUTPUT_DIR" && pwd)/$(basename "$OUTPUT")"

TMPDIR="$(mktemp -d)"
BUNDLE_DIR="$TMPDIR/hermes-persona-bundle"
mkdir -p "$BUNDLE_DIR"

cleanup() {
  rm -rf "$TMPDIR"
}
trap cleanup EXIT

# Stage the bundle using the same copytree/ignore architecture as Hermes's
# built-in default-profile export, but with additional root exclusions for a
# clean persona/workflow transplant.
python3 - "$SOURCE_HOME" "$BUNDLE_DIR" <<'PY'
import shutil
import sys
from pathlib import Path

from hermes_cli.profiles import _DEFAULT_EXPORT_EXCLUDE_ROOT, _default_export_ignore

source = Path(sys.argv[1]).resolve()
bundle = Path(sys.argv[2]).resolve()

extra_root_excludes = {
    # Host-local tooling / package trees / caches not useful in persona clones
    'node', 'lsp', 'cache', 'backups', 'state-snapshots', 'm365-tokens',
    'pairing', 'models_dev_cache.json', 'context_length_cache.yaml',
    '.restart_last_processed.json', '.skills_prompt_snapshot.json',
    # Usually source-box-specific or easily regenerated
    'kanban.db', 'channel_directory.json', 'gateway.lock',
}

base_ignore = _default_export_ignore(source)
merged_root_excludes = set(_DEFAULT_EXPORT_EXCLUDE_ROOT) | extra_root_excludes


def ignore(directory: str, contents: list[str]):
    ignored = set(base_ignore(directory, contents))
    if Path(directory).resolve() == source:
        ignored.update(name for name in contents if name in merged_root_excludes)
    return ignored

shutil.copytree(source, bundle, dirs_exist_ok=True, ignore=ignore)
PY

remove_path_if_exists() {
  local rel="$1"
  rm -rf "$BUNDLE_DIR/$rel"
}

# Persona-bundle policy toggles.
if [[ "$INCLUDE_HOOKS" -ne 1 ]]; then
  remove_path_if_exists hooks
fi
if [[ "$INCLUDE_SCRIPTS" -ne 1 ]]; then
  remove_path_if_exists scripts
fi
if [[ "$INCLUDE_CRON" -ne 1 ]]; then
  remove_path_if_exists cron
fi
if [[ "$INCLUDE_MEMORIES" -ne 1 ]]; then
  remove_path_if_exists memories
fi

# Defensive cleanup in case upstream export behavior changes.
for rel in \
  .env auth.json auth.lock .install_method \
  gateway.pid gateway.lock gateway_state.json channel_directory.json processes.json \
  state.db state.db-shm state.db-wal hermes_state.db \
  response_store.db response_store.db-shm response_store.db-wal \
  sessions logs image_cache audio_cache document_cache browser_screenshots checkpoints sandboxes \
  .restart_last_processed.json .skills_prompt_snapshot.json .update_check .hermes_history \
  active_profile node lsp cache backups state-snapshots m365-tokens pairing
  do
  remove_path_if_exists "$rel"
done

find "$BUNDLE_DIR" -maxdepth 1 -type f -name 'config.yaml.bak.*' -delete
find "$BUNDLE_DIR/skills" -maxdepth 1 -type f \( -name '.usage.json' -o -name '.usage.json.lock' -o -name '.curator_state' \) -delete 2>/dev/null || true

if [[ -f "$BUNDLE_DIR/config.yaml" ]]; then
  python3 - "$BUNDLE_DIR/config.yaml" <<'PY'
import re
import sys
from pathlib import Path

import yaml

path = Path(sys.argv[1])
data = yaml.safe_load(path.read_text()) or {}

SENSITIVE_KEY_RE = re.compile(
    r"(api[_-]?key|token|secret|password|passwd|webhook|client_secret|refresh_token|access_token|authorization)",
    re.IGNORECASE,
)
SENSITIVE_VALUE_RE = re.compile(
    r"(sk-[A-Za-z0-9]|AIza|ghp_|gho_|xox[baprs]-|Bearer\s+|-----BEGIN)",
    re.IGNORECASE,
)
IDENTITY_KEY_RE = re.compile(
    r"(chat_id|thread_id|channel_id|user_id|bot_token|phone_number|email|mailbox|home_channel)",
    re.IGNORECASE,
)


def scrub(obj):
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            key = str(k)
            if SENSITIVE_KEY_RE.search(key):
                out[k] = "__SET_ME__"
            elif IDENTITY_KEY_RE.search(key):
                out[k] = "__REVIEW_ME__"
            else:
                out[k] = scrub(v)
        return out
    if isinstance(obj, list):
        return [scrub(v) for v in obj]
    if isinstance(obj, str):
        if SENSITIVE_VALUE_RE.search(obj):
            return "__SET_ME__"
        return obj
    return obj

path.write_text(yaml.safe_dump(scrub(data), sort_keys=False, default_flow_style=False))
PY
fi

if [[ -f "$SOURCE_HOME/.env" ]]; then
  python3 - "$SOURCE_HOME/.env" "$BUNDLE_DIR/env.template" <<'PY'
import sys
from pathlib import Path

src = Path(sys.argv[1])
out = Path(sys.argv[2])
lines = []
for raw in src.read_text().splitlines():
    stripped = raw.strip()
    if not stripped:
        lines.append("")
        continue
    if stripped.startswith("#"):
        lines.append(raw)
        continue
    if "=" not in raw:
        lines.append(f"# REVIEW: {raw}")
        continue
    key, _ = raw.split("=", 1)
    key = key.strip()
    if not key:
        continue
    lines.append(f"{key}=")
out.write_text("\n".join(lines) + "\n")
PY
fi

  cat > "$BUNDLE_DIR/README.md" <<EOF
# Hermes Persona Bundle: ${NAME}

Created: $(date -u +'%Y-%m-%dT%H:%M:%SZ')
Source home: ${SOURCE_HOME}
Source git ref: ${SOURCE_GIT_REF:-unknown}
Source Hermes version: ${SOURCE_HERMES_VERSION:-unknown}
Base snapshot method: Hermes profile-export-style staged copy + zip archive flow

## Included
- config.yaml (scrubbed)
- SOUL.md
- skills/
- skins/
$( [[ "$INCLUDE_HOOKS" -eq 1 ]] && echo '- hooks/' )
$( [[ "$INCLUDE_SCRIPTS" -eq 1 ]] && echo '- scripts/' )
$( [[ "$INCLUDE_CRON" -eq 1 ]] && echo '- cron/' )
$( [[ "$INCLUDE_MEMORIES" -eq 1 ]] && echo '- memories/ (included by request)' )
- env.template (keys only, no values)

## Excluded by default
- .env and live secrets
- auth.json and OAuth tokens
- sessions/, state.db*, logs/
- gateway runtime files and chat routing state
- host-local caches/package trees/backups

## What to customize on the target
1. Fill in .env from env.template with new secrets.
2. Edit SOUL.md for the new bot persona/purpose.
3. Review config.yaml for platform-specific settings and IDs marked __REVIEW_ME__.
4. Start Hermes with a dedicated HERMES_HOME.
EOF

python3 - "$BUNDLE_DIR/manifest.json" <<PY
import json
from pathlib import Path
manifest = {
    "persona_name": ${NAME@Q},
    "created_at_utc": "$(date -u +'%Y-%m-%dT%H:%M:%SZ')",
    "source_home": ${SOURCE_HOME@Q},
    "source_git_ref": ${SOURCE_GIT_REF@Q},
    "source_hermes_version": ${SOURCE_HERMES_VERSION@Q},
    "base_snapshot": "hermes_cli.profiles._default_export_ignore + copytree + zipfile-style flow",
    "included": {
        "config_yaml": True,
        "soul": True,
        "skills": True,
        "skins": True,
        "hooks": bool(${INCLUDE_HOOKS}),
        "scripts": bool(${INCLUDE_SCRIPTS}),
        "cron": bool(${INCLUDE_CRON}),
        "memories": bool(${INCLUDE_MEMORIES}),
        "env_template": True,
    },
    "excluded_defaults": [
        ".env",
        "auth.json",
        "sessions",
        "state.db*",
        "logs",
        "gateway.pid",
        "gateway.lock",
        "gateway_state.json",
        "channel_directory.json",
        "processes.json",
        "node",
        "lsp",
        "cache",
        "backups",
        "state-snapshots",
    ],
}
Path(${BUNDLE_DIR@Q} + "/manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
PY

python3 - "$TMPDIR" "$BUNDLE_DIR" "$OUTPUT" <<'PY'
import os
import sys
import zipfile
from pathlib import Path

tmpdir = Path(sys.argv[1]).resolve()
bundle_dir = Path(sys.argv[2]).resolve()
output = Path(sys.argv[3]).resolve()

with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for path in sorted(bundle_dir.rglob("*")):
        if path.is_dir():
            continue
        zf.write(path, arcname=path.relative_to(tmpdir))
PY

echo "Created bundle: $OUTPUT"
