#!/usr/bin/env bash
# publish-unblock-helper.sh — Agent Launch Closeout Kit
# Usage:
#   bash starter-kits/agent-launch-closeout-kit/scripts/publish-unblock-helper.sh
#   bash starter-kits/agent-launch-closeout-kit/scripts/publish-unblock-helper.sh --execute --screenshot-path /absolute/path/to/signed-in-proof.png
#
# Purpose:
#   1. Run publish preflight and print the real blocker.
#   2. Emit the exact browser-auth verification command needed after a manual sign-in.
#   3. Generate a timestamped handoff artifact so the next sign-in event converts directly into a publish unblock.
#
# Safety rule:
#   --execute requires an explicit screenshot path captured from a signed-in browser surface.
#   The helper must not auto-promote a stale or failed screenshot into a ready marker.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
X_ACCESS_STATE_FILE="${HERMES_X_ACCESS_STATE_FILE:-$HOME/.hermes/state/x-access.json}"
LAUNCH_LOG="$ROOT_DIR/starter-kits/agent-launch-closeout-kit/launch-execution-log.md"
AUDIT_FILE="$ROOT_DIR/starter-kits/agent-launch-closeout-kit/live-browser-auth-audit.md"
ARTIFACT_DIR="$ROOT_DIR/starter-kits/agent-launch-closeout-kit/auth-artifacts"
EXEC_MODE="dry-run"
SCREENSHOT_PATH=""
SURFACE_URL="https://x.com/compose/post"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --execute)
      EXEC_MODE="execute"
      shift
      ;;
    --screenshot-path)
      SCREENSHOT_PATH="$2"
      shift 2
      ;;
    --surface-url)
      SURFACE_URL="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: $0 [--execute] [--surface-url https://x.com/compose/post] [--screenshot-path /absolute/path/to/signed-in-proof.png]" >&2
      exit 1
      ;;
  esac
done

mkdir -p "$ARTIFACT_DIR"

# ── Step 1: Run preflight ─────────────────────────────────────────────────
echo "=== Step 1: Publish preflight ==="
preflight_output=$(bash "$ROOT_DIR/starter-kits/agent-launch-closeout-kit/scripts/publish-preflight.sh" 2>&1) || true
echo "$preflight_output"
echo ""

# ── Step 2: Capture current stale state ───────────────────────────────────
marker_status="unknown"
marker_handle="unknown"
marker_updated="unknown"
if [[ -f "$X_ACCESS_STATE_FILE" ]]; then
  payload=$(python3 - <<'PY' "$X_ACCESS_STATE_FILE"
import json, sys
path = sys.argv[1]
with open(path) as f:
    d = json.load(f)
print(d.get('status','unknown'))
print(d.get('handle','unknown'))
print(d.get('updated_at','unknown'))
PY
)
  marker_status="${payload%%$'\n'*}"
  rest="${payload#*$'\n'}"
  marker_handle="${rest%%$'\n'*}"
  marker_updated="${rest#*$'\n'}"
fi

TIMESTAMP="$(date +%Y-%m-%dT%H-%M-%S%z)"
STAMP_HUMAN="$(date '+%Y-%m-%d %H:%M %Z')"

# ── Step 3: Generate or execute the unblock command ────────────────────────
if [[ -z "$SCREENSHOT_PATH" ]]; then
  shopt -s nullglob
  screenshot_candidates=("$HOME/.hermes/cache/screenshots/"*.png)
  shopt -u nullglob
  if (( ${#screenshot_candidates[@]} > 0 )); then
    LATEST_SCREENSHOT="$(ls -t "${screenshot_candidates[@]}" | head -1)"
  else
    LATEST_SCREENSHOT=""
  fi
else
  LATEST_SCREENSHOT="$SCREENSHOT_PATH"
fi

echo "=== Step 2: Current browser-auth state ==="
echo "  x-access.json status : $marker_status"
echo "  handle               : $marker_handle"
echo "  last updated         : $marker_updated"
echo ""

if [[ "$marker_status" == "ready" ]]; then
  echo "=== Browser session is ready — publish is unblocked ==="
  echo ""
  echo "To publish now:"
  echo "  1. Open $SURFACE_URL in the Hermes publish browser"
  echo "  2. Paste the canonical thread payload (see publish-trigger.md)"
  echo "  3. Attach the highest-priority available asset"
  echo "  4. Post"
  echo "  5. Record URL + timestamp in $LAUNCH_LOG"
  echo ""
else
  echo "=== Browser session is STALE — auth must be restored before publish ==="
  echo ""
  echo "Step A — Manual sign-in required:"
  echo "  1. Open https://x.com/home in the Hermes publish browser"
  echo "  2. Sign in to the KelEvur account if logged out"
  echo "  3. Confirm https://x.com/compose/post loads WITHOUT redirecting to /i/flow/login"
  echo "  4. Save a screenshot of the signed-in composer surface"
  echo ""
  echo "Step B — After sign-in, run this exact command:"
  echo ""
  printf "  %% bash starter-kits/agent-launch-closeout-kit/scripts/browser-auth-recovery.sh \\\\\n"
  printf "       --verified \\\\\n"
  printf "       --surface-url %s \\\\\n" "$SURFACE_URL"
  printf "       --screenshot-path /absolute/path/to/screenshot.png\n"
  echo ""
  if [[ -n "$LATEST_SCREENSHOT" ]]; then
    echo "  (Latest screenshot on disk for reference only: $LATEST_SCREENSHOT)"
    echo "   Do not use it unless it is the new signed-in proof shot from this exact publish session."
  fi
  echo ""
  echo "  OR run in --execute mode after you have a fresh signed-in screenshot:"
  echo "    bash $0 --execute --screenshot-path /absolute/path/to/signed-in-proof.png"
  echo ""
fi

# ── Step 4: Write the handoff artifact ────────────────────────────────────
ARTIFACT_PATH="$ARTIFACT_DIR/publish-unblock-handoff-$TIMESTAMP.md"
PREFLIGHT_HEAD="$(echo "$preflight_output" | head -1)"
cat > "$ARTIFACT_PATH" <<'OUTER_EOF'
# Publish Unblock Handoff — PRE Stamp
OUTER_EOF
# Stamp is added separately so it doesn't get confused with any embedded text
{
  echo ""
  echo "## Current state"
  echo "- x-access.json status : **MARKER_STATUS**"
  echo "- handle               : MARKER_HANDLE"
  echo "- last updated         : MARKER_UPDATED"
  echo "- Preflight result     : PREFLIGHT_HEAD"
  echo ""
  echo "## If status = ready"
  echo "Publish is unblocked. Follow 'publish-trigger.md' and post immediately."
  echo "Record URL + timestamp in 'launch-execution-log.md'."
  echo ""
  echo "## If status = stale"
  echo "**Auth must be restored before publish can proceed.**"
  echo ""
  echo "### Step A — Sign in"
  echo "1. Open https://x.com/home in the Hermes publish browser"
  echo "2. Sign in to KelEvur if logged out"
  echo "3. Confirm https://x.com/compose/post loads without login redirect"
  echo "4. Save a screenshot of the signed-in composer surface"
  echo ""
  echo "### Step B — Record verification"
  echo "Run this command after sign-in (fill in the screenshot path):"
  echo ""
  echo '```bash'
  echo 'bash starter-kits/agent-launch-closeout-kit/scripts/publish-unblock-helper.sh \\'
  echo '     --execute \\'
  echo '     --surface-url https://x.com/compose/post \\'
  echo '     --screenshot-path /absolute/path/to/signed-in-proof.png'
  echo '```'

  echo "### Step C — Publish immediately"
  echo "After Step B completes, post using 'publish-trigger.md'."
  echo ""
  echo "## Evidence files"
  echo "- Audit  : 'live-browser-auth-audit.md'"
  echo "- Log    : 'launch-execution-log.md'"
  echo "- Script : 'scripts/browser-auth-recovery.sh'"
} >> "$ARTIFACT_PATH"

# Now do the substitutions with actual values
python3 - <<PY "$ARTIFACT_PATH" "$marker_status" "$marker_handle" "$marker_updated" "$PREFLIGHT_HEAD" "$STAMP_HUMAN"
import sys
path = sys.argv[1]
marker_status = sys.argv[2]
marker_handle = sys.argv[3]
marker_updated = sys.argv[4]
preflight_head = sys.argv[5]
stamp = sys.argv[6]
content = open(path).read()
content = content.replace('PRE Stamp', stamp)
content = content.replace('MARKER_STATUS', marker_status)
content = content.replace('MARKER_HANDLE', marker_handle)
content = content.replace('MARKER_UPDATED', marker_updated)
content = content.replace('PREFLIGHT_HEAD', preflight_head)
open(path, 'w').write(content)
PY

echo "=== Step 3: Handoff artifact created ==="
echo "  $ARTIFACT_PATH"
echo ""

# ── Step 5: --execute mode ────────────────────────────────────────────────
if [[ "$EXEC_MODE" == "execute" ]]; then
  echo "=== EXECUTE MODE: Running publish unblock ==="
  if [[ "$marker_status" == "ready" ]]; then
    echo "Already ready. No action needed."
  else
    if [[ -z "$SCREENSHOT_PATH" ]]; then
      echo "ERROR: --execute requires --screenshot-path /absolute/path/to/signed-in-proof.png"
      echo "Refusing to auto-promote the latest screenshot on disk because it may be a stale failed-check artifact."
      exit 1
    fi
    if [[ ! -f "$SCREENSHOT_PATH" ]]; then
      echo "ERROR: Screenshot file does not exist: $SCREENSHOT_PATH"
      exit 1
    fi
    echo "Running browser-auth-recovery.sh --verified with:"
    echo "  surface-url    : $SURFACE_URL"
    echo "  screenshot-path: $SCREENSHOT_PATH"
    bash "$ROOT_DIR/starter-kits/agent-launch-closeout-kit/scripts/browser-auth-recovery.sh" \
      --verified \
      --surface-url "$SURFACE_URL" \
      --screenshot-path "$SCREENSHOT_PATH"
    echo ""
    echo "=== Publish is now unblocked ==="
    echo "Next: Follow publish-trigger.md to post the launch thread."
  fi
fi

echo "Done."
