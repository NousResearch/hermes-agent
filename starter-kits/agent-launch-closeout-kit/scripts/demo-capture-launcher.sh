#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
KIT_DIR="$ROOT_DIR/starter-kits/agent-launch-closeout-kit"
ARTIFACT_DIR="$KIT_DIR/demo-artifacts"
PREPARE=1
DRY_RUN=0
EDITOR_APP="TextEdit"
QUICKTIME_APP="QuickTime Player"

usage() {
  cat <<'EOF'
Usage:
  bash starter-kits/agent-launch-closeout-kit/scripts/demo-capture-launcher.sh
  bash starter-kits/agent-launch-closeout-kit/scripts/demo-capture-launcher.sh --dry-run
  bash starter-kits/agent-launch-closeout-kit/scripts/demo-capture-launcher.sh --no-prepare
  bash starter-kits/agent-launch-closeout-kit/scripts/demo-capture-launcher.sh --editor-app "Visual Studio Code"

Behavior:
  - optionally refreshes the capture session via demo-capture.sh --prepare
  - resolves the latest session packet
  - opens the exact files needed for walkthrough capture in an editor
  - activates QuickTime Player so recording can start without hunting windows

Notes:
  - macOS only (uses `open` / AppleScript activation)
  - does not auto-start screen recording; it primes the workspace and activates QuickTime
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --no-prepare)
      PREPARE=0
      shift
      ;;
    --editor-app)
      EDITOR_APP="${2:-}"
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

[[ "$(uname -s)" == "Darwin" ]] || { printf 'This launcher currently supports macOS only.\n' >&2; exit 2; }
open -Ra "$EDITOR_APP" >/dev/null 2>&1 || { printf 'Editor app not available: %s\n' "$EDITOR_APP" >&2; exit 2; }
open -Ra "$QUICKTIME_APP" >/dev/null 2>&1 || { printf 'QuickTime Player is not available.\n' >&2; exit 2; }

if [[ $PREPARE -eq 1 ]]; then
  bash "$KIT_DIR/scripts/demo-capture.sh" --prepare >/tmp/demo-capture-launcher.prepare.out
fi

LATEST_SESSION="$(find "$ARTIFACT_DIR" -maxdepth 1 -name 'demo-capture-session-*.md' -type f | sort | tail -n 1)"
[[ -n "$LATEST_SESSION" ]] || { printf 'No demo capture session packet found in %s\n' "$ARTIFACT_DIR" >&2; exit 2; }

READINESS_PATH="$ARTIFACT_DIR/latest-demo-capture-readiness.md"
TRIGGER_PATH="$KIT_DIR/demo-trigger.md"
PROOF_PATH="$ROOT_DIR/starter-kits/agentic-cron-orchestration-kit/qa/clean-room-proof-run-2026-04-17.md"
LOG_PATH="$KIT_DIR/launch-execution-log.md"
OUTLINE_PATH="$ROOT_DIR/starter-kits/agentic-cron-orchestration-kit/launch/demo-outline.md"

for path in "$READINESS_PATH" "$TRIGGER_PATH" "$PROOF_PATH" "$LOG_PATH" "$OUTLINE_PATH" "$LATEST_SESSION"; do
  [[ -f "$path" ]] || { printf 'Required file missing: %s\n' "$path" >&2; exit 2; }
done

RAW_PATH="$(python3 -c 'from pathlib import Path; import re, sys; text = Path(sys.argv[1]).read_text(); m = re.search(r"Suggested raw recording path: `([^`]+)`", text); print(m.group(1) if m else "")' "$LATEST_SESSION")"
EDITED_PATH="$(python3 -c 'from pathlib import Path; import re, sys; text = Path(sys.argv[1]).read_text(); m = re.search(r"Suggested edited asset path: `([^`]+)`", text); print(m.group(1) if m else "")' "$LATEST_SESSION")"

print_plan() {
  printf 'DEMO_CAPTURE_LAUNCHER_OK\n'
  printf 'Editor app: %s\n' "$EDITOR_APP"
  printf 'QuickTime app: %s\n' "$QUICKTIME_APP"
  printf 'Latest session packet: %s\n' "${LATEST_SESSION#$ROOT_DIR/}"
  printf 'Readiness packet: %s\n' "${READINESS_PATH#$ROOT_DIR/}"
  printf 'Suggested raw recording path: %s\n' "$RAW_PATH"
  printf 'Suggested edited asset path: %s\n' "$EDITED_PATH"
  printf 'Open order:\n'
  printf '  1. %s\n' "${READINESS_PATH#$ROOT_DIR/}"
  printf '  2. %s\n' "${LATEST_SESSION#$ROOT_DIR/}"
  printf '  3. %s\n' "${TRIGGER_PATH#$ROOT_DIR/}"
  printf '  4. %s\n' "${PROOF_PATH#$ROOT_DIR/}"
  printf '  5. %s\n' "${LOG_PATH#$ROOT_DIR/}"
  printf '  6. %s\n' "${OUTLINE_PATH#$ROOT_DIR/}"
  printf 'Finalize command:\n'
  printf 'bash starter-kits/agent-launch-closeout-kit/scripts/demo-capture.sh --finalize \\\n'
  printf '  --recording-path %q \\\n' "$ROOT_DIR/$RAW_PATH"
  printf '  --duration 00:01:19 \\\n'
  printf '  --edited-asset-path %q\n' "$ROOT_DIR/$EDITED_PATH"
}

if [[ $DRY_RUN -eq 1 ]]; then
  print_plan
  exit 0
fi

open -a "$EDITOR_APP" "$READINESS_PATH"
open -a "$EDITOR_APP" "$LATEST_SESSION"
open -a "$EDITOR_APP" "$TRIGGER_PATH"
open -a "$EDITOR_APP" "$PROOF_PATH"
open -a "$EDITOR_APP" "$LOG_PATH"
open -a "$EDITOR_APP" "$OUTLINE_PATH"
open -a "$QUICKTIME_APP"
osascript -e 'tell application "QuickTime Player" to activate' >/dev/null
print_plan
