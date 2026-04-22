#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
KIT_DIR="$ROOT_DIR/starter-kits/agent-launch-closeout-kit"
ARTIFACT_DIR="$KIT_DIR/demo-artifacts"
DEFAULT_DURATION="00:01:19"
SESSION_PATH=""
LOG_PATH=""
RECORDING_PATH=""
EDITED_ASSET_PATH=""
DURATION="$DEFAULT_DURATION"
POSTED_URL=""
NOTES=""

usage() {
  cat <<'EOF'
Usage:
  bash starter-kits/agent-launch-closeout-kit/scripts/demo-capture-headless-finalize.sh
  bash starter-kits/agent-launch-closeout-kit/scripts/demo-capture-headless-finalize.sh \
    --recording-path /absolute/path/to/raw.mov \
    --edited-asset-path /absolute/path/to/final.mp4 \
    [--duration 00:01:19] \
    [--posted-url https://x.com/... ] \
    [--notes "optional note"] \
    [--session-path /absolute/path/to/demo-capture-session.md] \
    [--log-path /absolute/path/to/launch-execution-log.md]

Behavior:
  - resolves the latest demo-capture session packet by default
  - extracts the suggested raw/edit asset paths from that packet
  - requires a real recording file to exist before finalizing
  - calls scripts/demo-capture.sh --finalize without any GUI dependency
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --session-path)
      SESSION_PATH="${2:-}"
      shift 2
      ;;
    --log-path)
      LOG_PATH="${2:-}"
      shift 2
      ;;
    --recording-path)
      RECORDING_PATH="${2:-}"
      shift 2
      ;;
    --edited-asset-path)
      EDITED_ASSET_PATH="${2:-}"
      shift 2
      ;;
    --duration)
      DURATION="${2:-}"
      shift 2
      ;;
    --posted-url)
      POSTED_URL="${2:-}"
      shift 2
      ;;
    --notes)
      NOTES="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown argument: %s\n\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$SESSION_PATH" ]]; then
  SESSION_PATH="$(find "$ARTIFACT_DIR" -maxdepth 1 -name 'demo-capture-session-*.md' -type f | sort | tail -n 1)"
fi
[[ -n "$SESSION_PATH" && -f "$SESSION_PATH" ]] || { printf 'No demo capture session packet found. Run scripts/demo-capture.sh --prepare first.\n' >&2; exit 2; }

if [[ -z "$LOG_PATH" ]]; then
  LOG_PATH="$KIT_DIR/launch-execution-log.md"
fi
[[ -f "$LOG_PATH" ]] || { printf 'Launch log not found: %s\n' "$LOG_PATH" >&2; exit 2; }

extract_field() {
  local regex="$1"
  python3 - <<'PY' "$SESSION_PATH" "$regex"
from pathlib import Path
import re
import sys
text = Path(sys.argv[1]).read_text()
match = re.search(sys.argv[2], text)
print(match.group(1) if match else "")
PY
}

if [[ -z "$RECORDING_PATH" ]]; then
  RECORDING_PATH="$(extract_field 'Suggested raw recording path: `([^`]+)`')"
  [[ -n "$RECORDING_PATH" ]] && RECORDING_PATH="$ROOT_DIR/$RECORDING_PATH"
fi
if [[ -z "$EDITED_ASSET_PATH" ]]; then
  EDITED_ASSET_PATH="$(extract_field 'Suggested edited asset path: `([^`]+)`')"
  [[ -n "$EDITED_ASSET_PATH" ]] && EDITED_ASSET_PATH="$ROOT_DIR/$EDITED_ASSET_PATH"
fi

[[ -n "$RECORDING_PATH" ]] || { printf 'Recording path is required and could not be derived from %s\n' "$SESSION_PATH" >&2; exit 2; }
[[ -n "$EDITED_ASSET_PATH" ]] || { printf 'Edited asset path is required and could not be derived from %s\n' "$SESSION_PATH" >&2; exit 2; }
[[ -n "$DURATION" ]] || { printf 'Duration is required.\n' >&2; exit 2; }
[[ -f "$RECORDING_PATH" ]] || { printf 'Recording file does not exist yet: %s\n' "$RECORDING_PATH" >&2; exit 2; }
[[ -f "$EDITED_ASSET_PATH" ]] || { printf 'Edited asset file does not exist yet: %s\n' "$EDITED_ASSET_PATH" >&2; exit 2; }

printf 'HEADLESS_FINALIZE_READY\n'
printf 'Session packet: %s\n' "$SESSION_PATH"
printf 'Launch log: %s\n' "$LOG_PATH"
printf 'Recording path: %s\n' "$RECORDING_PATH"
printf 'Edited asset path: %s\n' "$EDITED_ASSET_PATH"
printf 'Duration: %s\n' "$DURATION"

bash "$KIT_DIR/scripts/demo-capture.sh" --finalize \
  --recording-path "$RECORDING_PATH" \
  --duration "$DURATION" \
  --edited-asset-path "$EDITED_ASSET_PATH" \
  --posted-url "$POSTED_URL" \
  --notes "$NOTES" \
  --log-path "$LOG_PATH"
