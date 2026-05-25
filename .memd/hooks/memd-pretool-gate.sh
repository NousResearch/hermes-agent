#!/usr/bin/env bash
set -euo pipefail

# PreToolUse hook body for Claude Code (Edit|Write|NotebookEdit).
# Reads the tool-call JSON from stdin and forwards it to
# `memd hook gate --stdin` so memd can enforce continuity policy:
# block/warn on file modifications without a prior Read in this session.
# Non-blocking: failures never break the tool.

load_bundle_env() {
  local bundle_root="${MEMD_BUNDLE_ROOT:-.memd}"
  [ -f "$bundle_root/backend.env" ] && { set -a; . "$bundle_root/backend.env"; set +a; }
  [ -f "$bundle_root/env" ] && { set -a; . "$bundle_root/env"; set +a; }
}
load_bundle_env

OUTPUT="${MEMD_BUNDLE_ROOT:-.memd}"

# Claude Code pipes hook JSON on stdin. Forward to memd.
memd hook gate --output "$OUTPUT" --stdin || true
