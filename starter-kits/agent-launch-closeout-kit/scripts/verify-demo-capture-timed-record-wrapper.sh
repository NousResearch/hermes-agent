#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
KIT_DIR="$ROOT_DIR/starter-kits/agent-launch-closeout-kit"
TMP_DIR="$(mktemp -d -t demo-capture-timed-wrapper.XXXXXX)"
trap 'rm -rf "$TMP_DIR"' EXIT

RAW_PATH="$TMP_DIR/raw-demo.mov"
EDITED_PATH="$TMP_DIR/final-demo.mp4"
LOG_PATH="$TMP_DIR/launch-execution-log.md"
SESSION_PATH="$TMP_DIR/demo-capture-session-test.md"
OUTPUT_PATH="$TMP_DIR/wrapper.out"

cp "$KIT_DIR/launch-execution-log.md" "$LOG_PATH"
cat > "$SESSION_PATH" <<EOF
# Demo Capture Session — smoke test

## Status
- Session state: ready to record
- Suggested raw recording path: \
  
EOF
python3 - <<'PY' "$SESSION_PATH" "$RAW_PATH" "$EDITED_PATH"
from pathlib import Path
import sys
session = Path(sys.argv[1])
raw = sys.argv[2]
edited = sys.argv[3]
session.write_text(
    "# Demo Capture Session — smoke test\n\n"
    "## Status\n"
    "- Session state: ready to record\n"
    f"- Suggested raw recording path: `{raw}`\n"
    f"- Suggested edited asset path: `{edited}`\n"
)
PY

bash "$KIT_DIR/scripts/demo-capture-timed-record-wrapper.sh" \
  --no-prepare \
  --session-path "$SESSION_PATH" \
  --log-path "$LOG_PATH" \
  --recording-path "$RAW_PATH" \
  --edited-asset-path "$EDITED_PATH" \
  --duration-seconds 1 \
  --notes "smoke test" \
  >"$OUTPUT_PATH" 2>&1

python3 - <<'PY' "$LOG_PATH" "$RAW_PATH" "$EDITED_PATH" "$OUTPUT_PATH"
from pathlib import Path
import sys
log_path = Path(sys.argv[1])
raw_path = Path(sys.argv[2])
edited_path = Path(sys.argv[3])
output_path = Path(sys.argv[4])
text = log_path.read_text()
output = output_path.read_text()
required = [
    "- Status: captured on 00:00:01",
    f"- Recording path: {raw_path}",
    f"- Edited asset path: {edited_path}",
    "- [x] Demo walkthrough captured",
    "- [x] Final attachment choice recorded here",
]
missing = [item for item in required if item not in text]
if missing:
    raise SystemExit(f"Missing expected log lines: {missing}")
if "TIMED_DEMO_CAPTURE_OK" not in output:
    raise SystemExit("Wrapper output missing TIMED_DEMO_CAPTURE_OK")
if raw_path.stat().st_size <= 0:
    raise SystemExit(f"Raw recording file is empty: {raw_path}")
if edited_path.stat().st_size <= 0:
    raise SystemExit(f"Edited asset file is empty: {edited_path}")
print("VERIFY_OK timed demo wrapper records screen, finalizes log, and verifies asset closeout")
PY
