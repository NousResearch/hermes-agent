#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
KIT_DIR="$ROOT_DIR/starter-kits/agent-launch-closeout-kit"
ARTIFACT_DIR="$KIT_DIR/demo-artifacts"
LOG_PATH="$KIT_DIR/launch-execution-log.md"
RUNBOOK_PATH="$KIT_DIR/demo-capture-runbook.md"
TRIGGER_PATH="$KIT_DIR/demo-trigger.md"
README_PATH="$KIT_DIR/README.md"
READINESS_PATH="$ARTIFACT_DIR/latest-demo-capture-readiness.md"
TIMESTAMP="$(date +%Y-%m-%dT%H-%M-%S%z)"
STAMP_HUMAN="$(date '+%Y-%m-%d %H:%M %Z')"
CAPTURE_PLAN_PATH="$ARTIFACT_DIR/demo-capture-session-$TIMESTAMP.md"
DEFAULT_RECORDING_PATH="$ARTIFACT_DIR/raw-demo-capture-$TIMESTAMP.mov"
DEFAULT_EDITED_PATH="$ARTIFACT_DIR/edited-demo-capture-$TIMESTAMP.mp4"
MODE="prepare"
RECORDING_PATH=""
EDITED_ASSET_PATH=""
DURATION=""
NOTES=""
POSTED_URL=""
LOG_PATH_OVERRIDE=""

usage() {
  cat <<'EOF'
Usage:
  bash starter-kits/agent-launch-closeout-kit/scripts/demo-capture.sh
  bash starter-kits/agent-launch-closeout-kit/scripts/demo-capture.sh --prepare
  bash starter-kits/agent-launch-closeout-kit/scripts/demo-capture.sh --finalize \
    --recording-path /absolute/path/to/raw.mov \
    --duration 00:01:19 \
    --edited-asset-path /absolute/path/to/final.mp4 \
    [--posted-url https://x.com/... ] \
    [--notes "optional note"] \
    [--log-path /absolute/path/to/launch-execution-log-copy.md]

Behavior:
  --prepare   Run preflight, freeze a timestamped capture-session packet, and print the exact next steps.
  --finalize  Fill launch-execution-log.md with the recorded asset path(s) and flip demo capture to complete.
EOF
}

require_file() {
  local path="$1"
  [[ -f "$path" ]] || { printf 'Required file missing: %s\n' "$path" >&2; exit 2; }
}

escape_python() {
  python3 - <<'PY' "$1"
import json, sys
print(json.dumps(sys.argv[1]))
PY
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prepare)
      MODE="prepare"
      shift
      ;;
    --finalize)
      MODE="finalize"
      shift
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
    --notes)
      NOTES="${2:-}"
      shift 2
      ;;
    --log-path)
      LOG_PATH_OVERRIDE="${2:-}"
      shift 2
      ;;
    --posted-url)
      POSTED_URL="${2:-}"
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

require_file "$RUNBOOK_PATH"
require_file "$TRIGGER_PATH"
require_file "$LOG_PATH"
require_file "$README_PATH"
mkdir -p "$ARTIFACT_DIR"

if [[ -n "$LOG_PATH_OVERRIDE" ]]; then
  LOG_PATH="$LOG_PATH_OVERRIDE"
fi
require_file "$LOG_PATH"

if [[ "$MODE" == "prepare" ]]; then
  preflight_output="$(bash "$KIT_DIR/scripts/demo-capture-preflight.sh" 2>&1)"

  {
    printf '# Demo Capture Session — %s\n\n' "$STAMP_HUMAN"
    printf '## Status\n'
    printf -- '- Session state: ready to record\n'
    printf -- '- Readiness packet: `%s`\n' "${READINESS_PATH#$ROOT_DIR/}"
    printf -- '- Suggested raw recording path: `%s`\n' "${DEFAULT_RECORDING_PATH#$ROOT_DIR/}"
    printf -- '- Suggested edited asset path: `%s`\n\n' "${DEFAULT_EDITED_PATH#$ROOT_DIR/}"

    printf '## Record this exact path\n'
    printf '1. Keep the claim narrow: closeout process only, not broader product proof.\n'
    printf '2. Follow the one-screen trigger card in `%s` (full detail remains in `%s`).\n' "${TRIGGER_PATH#$ROOT_DIR/}" "${RUNBOOK_PATH#$ROOT_DIR/}"
    printf '3. Capture the raw recording to `%s` or replace with your actual asset path.\n' "${DEFAULT_RECORDING_PATH#$ROOT_DIR/}"
    printf '4. After recording/editing, prefer the headless finalize helper:\n\n'
    printf '```bash\n'
    printf 'bash starter-kits/agent-launch-closeout-kit/scripts/demo-capture-headless-finalize.sh \\\n'
    printf '  --recording-path %q \\\n' "$DEFAULT_RECORDING_PATH"
    printf '  --duration 00:01:19 \\\n'
    printf '  --edited-asset-path %q\n' "$DEFAULT_EDITED_PATH"
    printf '```\n\n'
    printf '   If you need the lower-level path, call `scripts/demo-capture.sh --finalize` directly with the same arguments.\n\n'

    printf '## Surfaces to show during capture\n'
    printf -- '- `%s`\n' "${READINESS_PATH#$ROOT_DIR/}"
    printf -- '- `%s`\n' "${TRIGGER_PATH#$ROOT_DIR/}"
    printf -- '- `starter-kits/agentic-cron-orchestration-kit/qa/clean-room-proof-run-2026-04-17.md`\n'
    printf -- '- `%s`\n' "${LOG_PATH#$ROOT_DIR/}"
    printf -- '- `starter-kits/agentic-cron-orchestration-kit/launch/demo-outline.md`\n\n'

    printf '## Raw preflight output\n'
    printf '```\n%s\n```\n' "$preflight_output"
  } > "$CAPTURE_PLAN_PATH"

  printf 'Demo capture session prepared\n'
  printf 'Session packet: %s\n' "${CAPTURE_PLAN_PATH#$ROOT_DIR/}"
  printf 'Suggested raw recording path: %s\n' "${DEFAULT_RECORDING_PATH#$ROOT_DIR/}"
  printf 'Suggested edited asset path: %s\n' "${DEFAULT_EDITED_PATH#$ROOT_DIR/}"
  exit 0
fi

[[ -n "$RECORDING_PATH" ]] || { printf '--recording-path is required in --finalize mode\n' >&2; exit 2; }
[[ -n "$DURATION" ]] || { printf '--duration is required in --finalize mode\n' >&2; exit 2; }
[[ -n "$EDITED_ASSET_PATH" ]] || { printf '--edited-asset-path is required in --finalize mode\n' >&2; exit 2; }

recording_rel="$RECORDING_PATH"
edited_rel="$EDITED_ASSET_PATH"
[[ "$RECORDING_PATH" == "$ROOT_DIR"/* ]] && recording_rel="${RECORDING_PATH#$ROOT_DIR/}"
[[ "$EDITED_ASSET_PATH" == "$ROOT_DIR"/* ]] && edited_rel="${EDITED_ASSET_PATH#$ROOT_DIR/}"

python3 - <<'PY' "$LOG_PATH" "$recording_rel" "$DURATION" "$edited_rel" "$POSTED_URL" "$NOTES"
from pathlib import Path
import re
import sys

log_path = Path(sys.argv[1])
recording_path = sys.argv[2]
duration = sys.argv[3]
edited_asset_path = sys.argv[4]
posted_url = sys.argv[5]
notes = sys.argv[6]
text = log_path.read_text()

# Match the Demo walkthrough section robustly using regex that tolerates
# field-order variation and extra/missing optional fields.
pattern = re.compile(
    r'(## Demo walkthrough\n)'
    r'(- Status: )(pending capture|captured on[^\n]*)\n'
    r'(- Readiness packet: [^\n]+\n)'
    r'(- Capture helper: [^\n]+\n)'
    r'((?:- Headless finalize helper: [^\n]+\n)?)'
    r'(- Trigger card: [^\n]+\n)'
    r'(- Source files:\n)((?:  - [^\n]+\n)+)'
    r'(- Done criteria:\n)((?:  - [^\n]+\n)+)'
    r'(- Record after capture:\n)((?:  - [^\n]*\n)*)',
    re.MULTILINE
)


def replacement(m):
    parts = [
        m.group(1),                           # ## Demo walkthrough
        m.group(2), f"captured on {duration}\n",
        m.group(4),                           # - Readiness packet
        m.group(5),                           # - Capture helper
        m.group(6),                           # optional - Headless finalize helper
        m.group(7),                           # - Trigger card
        m.group(8), m.group(9),               # - Source files: + items
        m.group(10), m.group(11),             # - Done criteria: + items
        "- Recorded asset:\n",
        f"  - Recording path: {recording_path}\n",
        f"  - Duration: {duration}\n",
        f"  - Edited asset path: {edited_asset_path}\n",
        f"  - Posted URL (if published): {posted_url}\n",
        f"  - Notes: {notes}\n",
    ]
    return ''.join(parts)

new_text, count = pattern.subn(replacement, text)
if count == 0:
    raise SystemExit("Demo walkthrough block not found in expected format in launch-execution-log.md")
new_text = new_text.replace("- [ ] Demo walkthrough captured", "- [x] Demo walkthrough captured", 1)
new_text = new_text.replace("- [ ] Final attachment choice recorded here", "- [x] Final attachment choice recorded here", 1)
log_path.write_text(new_text)
PY

printf 'Demo capture finalized\n'
printf 'Launch log updated: %s\n' "${LOG_PATH#$ROOT_DIR/}"
printf 'Recording path: %s\n' "$recording_rel"
printf 'Edited asset path: %s\n' "$edited_rel"
