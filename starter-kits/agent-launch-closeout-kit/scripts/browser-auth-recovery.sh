#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
X_ACCESS_STATE_FILE="${HERMES_X_ACCESS_STATE_FILE:-$HOME/.hermes/state/x-access.json}"
ARTIFACT_DIR="$ROOT_DIR/starter-kits/agent-launch-closeout-kit/auth-artifacts"
LAUNCH_LOG="$ROOT_DIR/starter-kits/agent-launch-closeout-kit/launch-execution-log.md"
AUDIT_FILE="$ROOT_DIR/starter-kits/agent-launch-closeout-kit/live-browser-auth-audit.md"
MODE="prepare"
SURFACE_URL=""
SCREENSHOT_PATH=""
AUTO_OPEN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prepare)
      MODE="prepare"
      shift
      ;;
    --verified)
      MODE="verified"
      shift
      ;;
    --surface-url)
      SURFACE_URL="$2"
      shift 2
      ;;
    --screenshot-path)
      SCREENSHOT_PATH="$2"
      shift 2
      ;;
    --open)
      AUTO_OPEN=1
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

mkdir -p "$ARTIFACT_DIR"
TIMESTAMP="$(date +%Y-%m-%dT%H-%M-%S%z)"
STAMP_HUMAN="$(date '+%Y-%m-%d %H:%M %Z')"
ARTIFACT_PATH="$ARTIFACT_DIR/browser-auth-recovery-$TIMESTAMP.md"

marker_status="missing"
marker_handle=""
marker_notes=""
if [[ -f "$X_ACCESS_STATE_FILE" ]]; then
  marker_payload="$(python3 - <<'PY' "$X_ACCESS_STATE_FILE"
import json, sys
path = sys.argv[1]
try:
    with open(path) as f:
        data = json.load(f)
except Exception:
    print('unknown|||')
else:
    status = data.get('status', 'unknown')
    handle = data.get('handle', '')
    notes = data.get('notes', '').replace('\n', ' ')
    print(f'{status}|||{handle}|||{notes}')
PY
)"
  marker_status="${marker_payload%%|||*}"
  marker_rest="${marker_payload#*|||}"
  marker_handle="${marker_rest%%|||*}"
  marker_notes="${marker_rest#*|||}"
fi

if [[ "$MODE" == "prepare" ]]; then
  cat > "$ARTIFACT_PATH" <<EOF
# Browser Auth Recovery Packet — $STAMP_HUMAN

## Goal
Recover or prove a real signed-in X browser session in the same Hermes publish environment before attempting launch publish.

## Current marker state
- File: $X_ACCESS_STATE_FILE
- Status: $marker_status
- Handle: ${marker_handle:-unknown}
- Notes: ${marker_notes:-none recorded}

## Required live proof
- Reach https://x.com/home while signed in, or reach https://x.com/compose/post without a login redirect.
- Capture one screenshot proving the signed-in surface.
- Record the screenshot path and exact verified surface.

## Recovery steps
1. Open https://x.com/home in the actual Hermes publish browser session.
2. If X shows a login form, landing page, or "Already have an account?", sign in to the intended account.
3. Re-open https://x.com/compose/post and confirm the composer loads instead of redirecting to /i/flow/login.
4. Save a screenshot of the signed-in surface.
5. Mark the session verified with:
   bash starter-kits/agent-launch-closeout-kit/scripts/browser-auth-recovery.sh --verified --surface-url https://x.com/compose/post --screenshot-path /absolute/path/to/screenshot.png

## Blocking rule
Do not publish from a stale marker alone.
EOF

  printf 'Prepared browser auth recovery packet: %s\n' "$ARTIFACT_PATH"
  if (( AUTO_OPEN == 1 )); then
    if command -v open >/dev/null 2>&1; then
      open 'https://x.com/home' >/dev/null 2>&1 || true
      printf 'Opened https://x.com/home with open\n'
    elif command -v xdg-open >/dev/null 2>&1; then
      xdg-open 'https://x.com/home' >/dev/null 2>&1 || true
      printf 'Opened https://x.com/home with xdg-open\n'
    else
      printf 'No open/xdg-open command available; open the URL manually.\n'
    fi
  fi
  exit 0
fi

if [[ -z "$SURFACE_URL" || -z "$SCREENSHOT_PATH" ]]; then
  echo '--verified requires --surface-url and --screenshot-path' >&2
  exit 1
fi

if [[ ! -f "$LAUNCH_LOG" ]]; then
  echo "Launch execution log missing: $LAUNCH_LOG" >&2
  exit 1
fi

cat > "$ARTIFACT_PATH" <<EOF
# Browser Auth Verified — $STAMP_HUMAN

- Surface URL: $SURFACE_URL
- Screenshot path: $SCREENSHOT_PATH
- Marker file: $X_ACCESS_STATE_FILE
- Marker status before verification: $marker_status
- Marker handle: ${marker_handle:-unknown}

This file records a live browser-auth proof event for the Hermes publish environment.
EOF

python3 - <<'PY' "$LAUNCH_LOG" "$SURFACE_URL" "$SCREENSHOT_PATH"
from pathlib import Path
import re, sys
path = Path(sys.argv[1])
surface = sys.argv[2]
screenshot = sys.argv[3]
text = path.read_text()
text = re.sub(r'- Status: .*', '- Status: verified live in publish session', text, count=1)
if '- Verified surface:' in text:
    text = re.sub(r'- Verified surface:.*', f'- Verified surface: {surface}', text, count=1)
else:
    text = text.replace('- Consequence: do not mark publish unblocked until the actual Hermes publish session reaches a signed-in X surface\n', '- Consequence: do not mark publish unblocked until the actual Hermes publish session reaches a signed-in X surface\n- Verified surface: ' + surface + '\n')
if '- Screenshot path:' in text:
    text = re.sub(r'- Screenshot path:.*', f'- Screenshot path: {screenshot}', text, count=1)
else:
    text = text.replace('- Verified surface: ' + surface + '\n', '- Verified surface: ' + surface + '\n- Screenshot path: ' + screenshot + '\n')
path.write_text(text)
PY

printf 'Recorded browser auth verification artifact: %s\n' "$ARTIFACT_PATH"
printf 'Updated launch log with verified surface and screenshot path.\n'
if [[ -f "$AUDIT_FILE" ]]; then
  printf 'Remember to append the verification outcome to %s\n' "$AUDIT_FILE"
fi
