#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
KIT_DIR="$ROOT_DIR/starter-kits/agent-launch-closeout-kit"
SOURCE_LOG="$KIT_DIR/launch-execution-log.md"
TMP_DIR="$(mktemp -d)"
TMP_LOG="$TMP_DIR/launch-execution-log.md"
RAW_ASSET="$TMP_DIR/raw-demo.mov"
EDITED_ASSET="$TMP_DIR/edited-demo.mp4"
POSTED_URL="https://x.com/example/status/1234567890"
NOTES="post-finalize verification smoke"

cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

cp "$SOURCE_LOG" "$TMP_LOG"
printf 'raw demo bytes\n' > "$RAW_ASSET"
printf 'edited demo bytes\n' > "$EDITED_ASSET"

bash "$KIT_DIR/scripts/demo-capture.sh" \
  --finalize \
  --log-path "$TMP_LOG" \
  --recording-path "$RAW_ASSET" \
  --duration 00:01:19 \
  --edited-asset-path "$EDITED_ASSET" \
  --posted-url "$POSTED_URL" \
  --notes "$NOTES" >/dev/null

verify_output="$(bash "$KIT_DIR/scripts/demo-capture-post-finalize-verify.sh" --log-path "$TMP_LOG")"
printf '%s\n' "$verify_output"

printf '%s\n' "$verify_output" | grep -F 'POST_FINALIZE_VERIFY_PASS' >/dev/null
printf '%s\n' "$verify_output" | grep -F "RECORDING_PATH=$RAW_ASSET" >/dev/null
printf '%s\n' "$verify_output" | grep -F "EDITED_PATH=$EDITED_ASSET" >/dev/null
printf '%s\n' "$verify_output" | grep -F 'POST_FINALIZE_VERIFY_RESULT=PASS' >/dev/null

printf 'VERIFY_OK demo-capture post-finalize verification succeeds against real non-empty assets\n'
