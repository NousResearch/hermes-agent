#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
KIT_DIR="$ROOT_DIR/starter-kits/agent-launch-closeout-kit"
ARTIFACT_DIR="$KIT_DIR/demo-artifacts"
DURATION_SECONDS=79
DISPLAY_ID=1
PREPARE=1
VERIFY=1
SESSION_PATH=""
LOG_PATH="$KIT_DIR/launch-execution-log.md"
RECORDING_PATH=""
EDITED_ASSET_PATH=""
POSTED_URL=""
NOTES="timed screen recording wrapper"
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  bash starter-kits/agent-launch-closeout-kit/scripts/demo-capture-timed-record-wrapper.sh
  bash starter-kits/agent-launch-closeout-kit/scripts/demo-capture-timed-record-wrapper.sh --duration-seconds 90
  bash starter-kits/agent-launch-closeout-kit/scripts/demo-capture-timed-record-wrapper.sh \
    --recording-path /absolute/path/to/raw.mov \
    --edited-asset-path /absolute/path/to/final.mp4 \
    --duration-seconds 79 \
    [--session-path /absolute/path/to/demo-capture-session.md] \
    [--log-path /absolute/path/to/launch-execution-log.md] \
    [--posted-url https://x.com/... ] \
    [--notes "optional note"]
  bash starter-kits/agent-launch-closeout-kit/scripts/demo-capture-timed-record-wrapper.sh --dry-run

Behavior:
  - macOS only
  - optionally refreshes the latest demo-capture session packet via `demo-capture.sh --prepare`
  - records the main display with native `screencapture -v -V...`
  - copies the raw recording to the edited-asset path when they differ so the closeout log has a real attachment file immediately
  - finalizes the launch log through `demo-capture-headless-finalize.sh`
  - verifies the real launch log and asset files with `demo-capture-post-finalize-verify.sh`
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --duration-seconds)
      DURATION_SECONDS="${2:-}"
      shift 2
      ;;
    --display)
      DISPLAY_ID="${2:-}"
      shift 2
      ;;
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
    --posted-url)
      POSTED_URL="${2:-}"
      shift 2
      ;;
    --notes)
      NOTES="${2:-}"
      shift 2
      ;;
    --no-prepare)
      PREPARE=0
      shift
      ;;
    --no-verify)
      VERIFY=0
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
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

[[ "$(uname -s)" == "Darwin" ]] || { printf 'This wrapper currently supports macOS only.\n' >&2; exit 2; }
command -v screencapture >/dev/null 2>&1 || { printf 'screencapture is required on macOS.\n' >&2; exit 2; }
[[ "$DURATION_SECONDS" =~ ^[0-9]+$ ]] || { printf '--duration-seconds must be an integer.\n' >&2; exit 2; }
(( DURATION_SECONDS > 0 )) || { printf '--duration-seconds must be > 0.\n' >&2; exit 2; }
[[ -f "$LOG_PATH" ]] || { printf 'Launch log not found: %s\n' "$LOG_PATH" >&2; exit 2; }
mkdir -p "$ARTIFACT_DIR"

if [[ $PREPARE -eq 1 ]]; then
  bash "$KIT_DIR/scripts/demo-capture.sh" --prepare >/tmp/demo-capture-timed-record.prepare.out
fi

if [[ -z "$SESSION_PATH" ]]; then
  SESSION_PATH="$(find "$ARTIFACT_DIR" -maxdepth 1 -name 'demo-capture-session-*.md' -type f | sort | tail -n 1)"
fi
[[ -n "$SESSION_PATH" && -f "$SESSION_PATH" ]] || { printf 'No demo capture session packet found. Run scripts/demo-capture.sh --prepare first.\n' >&2; exit 2; }

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
mkdir -p "$(dirname "$RECORDING_PATH")" "$(dirname "$EDITED_ASSET_PATH")"

DURATION_LABEL="$(python3 - <<'PY' "$DURATION_SECONDS"
import sys
seconds = int(sys.argv[1])
hours, rem = divmod(seconds, 3600)
minutes, secs = divmod(rem, 60)
print(f"{hours:02d}:{minutes:02d}:{secs:02d}")
PY
)"

print_plan() {
  printf 'TIMED_DEMO_CAPTURE_PLAN\n'
  printf 'Session packet: %s\n' "$SESSION_PATH"
  printf 'Launch log: %s\n' "$LOG_PATH"
  printf 'Display: %s\n' "$DISPLAY_ID"
  printf 'Duration seconds: %s\n' "$DURATION_SECONDS"
  printf 'Duration label: %s\n' "$DURATION_LABEL"
  printf 'Recording path: %s\n' "$RECORDING_PATH"
  printf 'Edited asset path: %s\n' "$EDITED_ASSET_PATH"
  printf 'Verify after finalize: %s\n' "$VERIFY"
}

if [[ $DRY_RUN -eq 1 ]]; then
  print_plan
  exit 0
fi

print_plan
printf 'TIMED_DEMO_CAPTURE_RECORDING\n'
screencapture -v -V"$DURATION_SECONDS" -D"$DISPLAY_ID" "$RECORDING_PATH"
[[ -s "$RECORDING_PATH" ]] || { printf 'Recording file is empty: %s\n' "$RECORDING_PATH" >&2; exit 2; }

if [[ "$EDITED_ASSET_PATH" != "$RECORDING_PATH" ]]; then
  cp "$RECORDING_PATH" "$EDITED_ASSET_PATH"
fi
[[ -s "$EDITED_ASSET_PATH" ]] || { printf 'Edited asset file is empty: %s\n' "$EDITED_ASSET_PATH" >&2; exit 2; }

finalize_notes="$NOTES"
if [[ "$EDITED_ASSET_PATH" != "$RECORDING_PATH" ]]; then
  finalize_notes="$finalize_notes; raw capture copied to edited asset path for immediate attachment readiness"
fi

bash "$KIT_DIR/scripts/demo-capture-headless-finalize.sh" \
  --session-path "$SESSION_PATH" \
  --log-path "$LOG_PATH" \
  --recording-path "$RECORDING_PATH" \
  --edited-asset-path "$EDITED_ASSET_PATH" \
  --duration "$DURATION_LABEL" \
  --posted-url "$POSTED_URL" \
  --notes "$finalize_notes"

if [[ $VERIFY -eq 1 ]]; then
  bash "$KIT_DIR/scripts/demo-capture-post-finalize-verify.sh" --log-path "$LOG_PATH"
fi

printf 'TIMED_DEMO_CAPTURE_OK\n'
printf 'Recording path: %s\n' "$RECORDING_PATH"
printf 'Edited asset path: %s\n' "$EDITED_ASSET_PATH"
printf 'Duration: %s\n' "$DURATION_LABEL"
