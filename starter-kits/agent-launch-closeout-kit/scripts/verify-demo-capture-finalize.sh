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
NOTES="verification smoke"

cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

cp "$SOURCE_LOG" "$TMP_LOG"
: > "$RAW_ASSET"
: > "$EDITED_ASSET"

bash "$KIT_DIR/scripts/demo-capture.sh" \
  --finalize \
  --log-path "$TMP_LOG" \
  --recording-path "$RAW_ASSET" \
  --duration 00:01:19 \
  --edited-asset-path "$EDITED_ASSET" \
  --posted-url "$POSTED_URL" \
  --notes "$NOTES" >/dev/null

python3 - <<'PY' "$TMP_LOG" "$RAW_ASSET" "$EDITED_ASSET" "$POSTED_URL" "$NOTES"
from pathlib import Path
import sys

log_path = Path(sys.argv[1])
raw_asset = sys.argv[2]
edited_asset = sys.argv[3]
posted_url = sys.argv[4]
notes = sys.argv[5]
text = log_path.read_text()
required = [
    "## Demo walkthrough\n- Status: captured on 00:01:19",
    "- Capture helper: `starter-kits/agent-launch-closeout-kit/scripts/demo-capture.sh`",
    "- Trigger card: `starter-kits/agent-launch-closeout-kit/demo-trigger.md`",
    f"  - Recording path: {raw_asset}",
    f"  - Edited asset path: {edited_asset}",
    f"  - Posted URL (if published): {posted_url}",
    f"  - Notes: {notes}",
    "- [x] Demo walkthrough captured",
    "- [x] Final attachment choice recorded here",
]
missing = [item for item in required if item not in text]
if missing:
    raise SystemExit("VERIFY_FAIL missing expected substrings:\n" + "\n".join(missing))
print("VERIFY_OK demo-capture finalize updates current log format without dropping capture-helper/trigger-card fields")
PY
