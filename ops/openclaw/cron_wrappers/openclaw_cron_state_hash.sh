#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE' >&2
Usage:
  openclaw_cron_state_hash.sh --state-file <path> [--input <text> | --stdin] [options]

Options:
  --state-file <path>   JSON state file path (required)
  --input <text>        Input string to hash
  --stdin               Read input from stdin
  --changed-only        Exit 20 if hash unchanged (new/changed => 0)
  --print-hash          Print computed hash
  -h, --help            Show help

State format:
  {
    "hash": "...",
    "updated_epoch": 1700000000,
    "updated_iso": "...Z",
    "bytes": 1234
  }
USAGE
}

STATE_FILE=""
INPUT_MODE="input"
INPUT_TEXT=""
CHANGED_ONLY=0
PRINT_HASH=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --state-file)
      STATE_FILE="${2:-}"
      shift 2
      ;;
    --input)
      INPUT_MODE="input"
      INPUT_TEXT="${2:-}"
      shift 2
      ;;
    --stdin)
      INPUT_MODE="stdin"
      shift
      ;;
    --changed-only)
      CHANGED_ONLY=1
      shift
      ;;
    --print-hash)
      PRINT_HASH=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "$STATE_FILE" ]]; then
  echo "--state-file is required" >&2
  exit 2
fi

if [[ "$INPUT_MODE" == "stdin" ]]; then
  INPUT_TEXT="$(cat)"
fi

new_hash="$(printf '%s' "$INPUT_TEXT" | sha256sum | awk '{print $1}')"
bytes_len="$(printf '%s' "$INPUT_TEXT" | wc -c | awk '{print $1}')"

old_hash=""
if [[ -f "$STATE_FILE" ]]; then
  old_hash="$(python3 - "$STATE_FILE" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
try:
    obj = json.loads(p.read_text(encoding='utf-8'))
    h = obj.get('hash')
    if isinstance(h, str):
        print(h)
except Exception:
    pass
PY
)"
fi

mkdir -p "$(dirname "$STATE_FILE")"
python3 - "$STATE_FILE" "$new_hash" "$bytes_len" <<'PY'
import datetime as dt
import json
import os
import pathlib
import sys

state_file = pathlib.Path(sys.argv[1])
new_hash = sys.argv[2]
bytes_len = int(sys.argv[3])

payload = {
    "hash": new_hash,
    "updated_epoch": int(dt.datetime.now(dt.timezone.utc).timestamp()),
    "updated_iso": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z'),
    "bytes": bytes_len,
}

tmp = state_file.with_suffix(state_file.suffix + '.tmp')
tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
os.replace(tmp, state_file)
PY

if [[ "$PRINT_HASH" -eq 1 ]]; then
  echo "$new_hash"
fi

if [[ "$CHANGED_ONLY" -eq 1 && "$old_hash" == "$new_hash" ]]; then
  exit 20
fi

exit 0
