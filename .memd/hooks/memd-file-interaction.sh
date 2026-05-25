#!/usr/bin/env bash
set -euo pipefail

# PostToolUse hook body for Claude Code (Read/Edit/Write/NotebookEdit).
# Reads the tool-call JSON from stdin and forwards it to
# `memd hook file-interaction --stdin` so memd can build a per-session
# ledger of touched files. Non-blocking: failures never break the tool.

load_bundle_env() {
  local bundle_root="${MEMD_BUNDLE_ROOT:-.memd}"
  [ -f "$bundle_root/backend.env" ] && { set -a; . "$bundle_root/backend.env"; set +a; }
  [ -f "$bundle_root/env" ] && { set -a; . "$bundle_root/env"; set +a; }
}
load_bundle_env

BASE_URL="${MEMD_BASE_URL:-http://127.0.0.1:8787}"
OUTPUT="${MEMD_BUNDLE_ROOT:-.memd}"

# Claude Code pipes hook JSON on stdin. Forward to memd.
memd --base-url "$BASE_URL" hook file-interaction \
  --output "$OUTPUT" \
  --stdin || true
