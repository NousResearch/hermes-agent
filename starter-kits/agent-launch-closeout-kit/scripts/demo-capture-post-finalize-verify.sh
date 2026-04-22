#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
KIT_DIR="$ROOT_DIR/starter-kits/agent-launch-closeout-kit"
LOG_PATH="$KIT_DIR/launch-execution-log.md"

usage() {
  cat <<'EOF'
Usage:
  bash starter-kits/agent-launch-closeout-kit/scripts/demo-capture-post-finalize-verify.sh
  bash starter-kits/agent-launch-closeout-kit/scripts/demo-capture-post-finalize-verify.sh --log-path /absolute/path/to/launch-execution-log.md

Verifies the real post-recording state:
  1. launch-execution-log.md shows `Status: captured on ...`
  2. recording path, duration, and edited asset path are present
  3. recording/edit files exist and are non-empty
  4. cross-note closeout checkboxes are checked
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --log-path)
      LOG_PATH="${2:-}"
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

[[ -f "$LOG_PATH" ]] || { printf 'FAIL: Launch log not found: %s\n' "$LOG_PATH"; exit 1; }

python3 - <<'PY' "$LOG_PATH" "$ROOT_DIR"
from pathlib import Path
import re
import sys

log_path = Path(sys.argv[1])
root_dir = Path(sys.argv[2])
text = log_path.read_text()

section_match = re.search(
    r'^## Demo walkthrough\n(?P<body>.*?)(?=^## |\Z)',
    text,
    re.MULTILINE | re.DOTALL,
)
if not section_match:
    print('FAIL: Could not find the Demo walkthrough section')
    raise SystemExit(1)

body = section_match.group('body')
status_match = re.search(r'^- Status: (?P<value>.+)$', body, re.MULTILINE)
if not status_match:
    print('FAIL: Demo walkthrough section is missing the Status line')
    raise SystemExit(1)

status = status_match.group('value').strip()
print(f'LOG_STATUS={status}')
if status.lower() == 'pending capture':
    print('FAIL: Launch log still shows pending capture')
    raise SystemExit(1)
if not status.lower().startswith('captured on '):
    print('FAIL: Demo walkthrough status does not show captured-on state')
    raise SystemExit(1)


def extract(label: str) -> str:
    match = re.search(rf'^  - {re.escape(label)}: ?(.*)$', body, re.MULTILINE)
    return match.group(1).strip() if match else ''

recording_path = extract('Recording path')
duration = extract('Duration')
edited_path = extract('Edited asset path')
posted_url = extract('Posted URL (if published)')
notes = extract('Notes')

print(f'RECORDING_PATH={recording_path}')
print(f'DURATION={duration}')
print(f'EDITED_PATH={edited_path}')
print(f'POSTED_URL={posted_url}')
print(f'NOTES={notes}')

errors = []
if not recording_path:
    errors.append('FAIL: No Recording path recorded in launch log')
if not duration:
    errors.append('FAIL: No Duration recorded in launch log')
if not edited_path:
    errors.append('FAIL: No Edited asset path recorded in launch log')


def resolve_logged_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root_dir / path

for label, raw_path in [('Recording file', recording_path), ('Edited asset file', edited_path)]:
    if not raw_path:
        continue
    resolved = resolve_logged_path(raw_path)
    if not resolved.exists():
        errors.append(f'FAIL: {label} does not exist: {resolved}')
        continue
    size = resolved.stat().st_size
    if size <= 0:
        errors.append(f'FAIL: {label} is empty (0 bytes): {resolved}')
        continue
    print(f'{label.upper().replace(" ", "_")}_OK={resolved} ({size} bytes)')

for checkbox in [
    '- [x] Demo walkthrough captured',
    '- [x] Final attachment choice recorded here',
]:
    if checkbox not in text:
        errors.append(f'FAIL: Missing checked closeout box: {checkbox}')
    else:
        print(f'CHECKBOX_OK={checkbox}')

if errors:
    for error in errors:
        print(error)
    raise SystemExit(1)

print('POST_FINALIZE_VERIFY_PASS')
PY

printf 'POST_FINALIZE_VERIFY_RESULT=PASS\n'
printf 'Demo capture closeout is consistent with real on-disk assets.\n'
