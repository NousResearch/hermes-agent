#!/usr/bin/env bash
set -euo pipefail

ROOT="${OPENCLAW_ROOT:-/home/yeqiuqiu/clawd-architect}"
exec python3 "$ROOT/ops/openclaw/continuity/restore_drill_refresh.py" "$@"
