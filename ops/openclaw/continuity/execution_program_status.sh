#!/usr/bin/env bash
set -euo pipefail

ROOT="${OPENCLAW_ROOT:-/home/yeqiuqiu/clawd-architect}"
JSON_OUT=0
REFRESH=0

usage() {
  cat <<'EOF'
Usage: execution_program_status.sh [options]

Read the canonical execution-program status artifact.

Options:
  --refresh   Recompute continuity/current before reading status
  --json      Print raw JSON
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --refresh)
      REFRESH=1; shift ;;
    --json)
      JSON_OUT=1; shift ;;
    -h|--help)
      usage
      exit 0 ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2 ;;
  esac
done

if [[ "$REFRESH" == "1" ]]; then
  bash "$ROOT/ops/openclaw/continuity/continuity_current.sh" --refresh >/dev/null
fi

python3 - "$ROOT" "$JSON_OUT" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve()
json_out = bool(int(sys.argv[2]))
status_path = root / "state" / "continuity" / "latest" / "execution_program_status.json"

if not status_path.exists():
    payload = {
        "ok": False,
        "error": "execution_program_status_missing",
        "path": str(status_path),
    }
    if json_out:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(
            "EXECUTION PROGRAM STATUS: missing "
            f"(generate via: bash {root}/ops/openclaw/continuity/continuity_current.sh --refresh)"
        )
    raise SystemExit(1)

try:
    payload = json.loads(status_path.read_text(encoding="utf-8"))
except Exception as exc:
    err = {
        "ok": False,
        "error": "execution_program_status_invalid_json",
        "path": str(status_path),
        "detail": str(exc),
    }
    if json_out:
        print(json.dumps(err, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(f"EXECUTION PROGRAM STATUS: invalid_json path={status_path}")
    raise SystemExit(1)

if not isinstance(payload, dict):
    err = {
        "ok": False,
        "error": "execution_program_status_not_object",
        "path": str(status_path),
    }
    if json_out:
        print(json.dumps(err, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(f"EXECUTION PROGRAM STATUS: invalid_object path={status_path}")
    raise SystemExit(1)

if json_out:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
else:
    print(
        "EXECUTION PROGRAM STATUS: "
        f"state={payload.get('program_state') or 'unknown'} "
        f"wave={payload.get('current_wave')} "
        f"workers={payload.get('active_worker_count')} "
        f"focus={payload.get('current_focus') or '-'} "
        f"last_progress_at={payload.get('last_progress_at') or '-'} "
        f"stalled={payload.get('stalled')}"
    )
PY
