#!/usr/bin/env bash
# Generic Codex CLI phase runner for multi-phase orchestrated work.
# Wraps a single `codex exec` invocation and captures:
#   - the full JSONL session stream
#   - the last assistant message (extracted from the JSONL)
#   - a phase-specific prompt file the agent reads as its task spec
#   - a per-phase summary file the agent writes at the end
#
# Use this when you need receipts (per-phase JSONL + summary) for hostile-
# auditor handoff or for re-running a phase after a regression. Pure
# `codex exec "..."` does not preserve the session stream in a parseable
# form by default.
#
# Usage:
#   bash run_codex_phase.sh <phase_id> <phase_spec.md> [repo_root]
#
# Outputs (under docs/codex-runs/<run>/receipts/ by convention):
#   PHASE_<id>_PROMPT.md      - the prompt the agent received
#   PHASE_<id>.codex.jsonl    - the full codex session event stream
#   PHASE_<id>.codex.txt      - the agent's last assistant message
#   PHASE_<id>.codex.exit     - exit code (0 = success, 124 = timeout, etc)
#
# The script is a thin wrapper: it composes the prompt, calls codex,
# polls, extracts the last message, and reports. It does NOT itself
# implement cancellation, retry, or any repair logic.
set -euo pipefail

PHASE_ID="$1"
PHASE_SPEC="$2"
ROOT="${3:-$(pwd)}"
RECEIPT_DIR="${RECEIPT_DIR:-$ROOT/docs/codex-runs/$(basename "$ROOT")/receipts}"
PHASE_PROMPT_FILE="$RECEIPT_DIR/PHASE_${PHASE_ID}_PROMPT.md"
OUT_MSG="$RECEIPT_DIR/PHASE_${PHASE_ID}.codex.txt"
OUT_JSONL="$RECEIPT_DIR/PHASE_${PHASE_ID}.codex.jsonl"
EXIT_FILE="$RECEIPT_DIR/PHASE_${PHASE_ID}.codex.exit"

MAX_WAIT="${MAX_WAIT:-1800}"   # 30 minutes default
SANDBOX="${SANDBOX:-workspace-write}"
MODEL="${MODEL:-}"             # empty = codex default

mkdir -p "$RECEIPT_DIR"

# 1. Write the prompt file. The phase spec is the canonical task spec;
#    the orchestrator's preamble supplies the success contract.
if [[ ! -f "$PHASE_SPEC" ]]; then
  echo "phase spec not found: $PHASE_SPEC" >&2
  exit 1
fi

cat > "$PHASE_PROMPT_FILE" <<EOF
You are Codex working on the repo at $ROOT.

## Authority order
1. Current source files in the checked-out repo.
2. The phase spec at $PHASE_SPEC (YOUR canonical task spec).
3. Active repo docs/validators in the repo root.
4. Prior audit docs.

Source code beats prose. If the prompt and current code disagree,
inspect and patch the code path, then record the divergence.

## Hard requirements
- No silent fallback. Surface typed reasons and receipts for any
  degraded path.
- No event-only truth. Recoverable state must round-trip through
  durable storage.
- No compatibility shim hiding an async/sync runtime boundary bug.
- Receipts or it did not happen. Every claim of "done" needs a
  command output snippet to back it up.

## Phase task spec (CANONICAL)
Read and follow exactly:
  $PHASE_SPEC

Write your deliverable to $RECEIPT_DIR/PHASE_${PHASE_ID}.md
Write a 5-15 line wrap-up to $RECEIPT_DIR/PHASE_${PHASE_ID}.SUMMARY.md

Begin.
EOF

echo "codex exec starting: phase=$PHASE_ID sandbox=$SANDBOX"

# 2. Launch codex in the background. CRITICAL: do NOT pass --output-schema
#    with /dev/null — codex expects a real JSON Schema and exits 1 with
#    "Output schema file /dev/null is not valid JSON" if you do. If you
#    don't need a schema, omit the flag entirely.
CODEX_ARGS=(
  exec
  --sandbox "$SANDBOX"
  -C "$ROOT"
  --ephemeral
  --json
)
if [[ -n "$MODEL" ]]; then
  CODEX_ARGS+=( -m "$MODEL" )
fi

codex "${CODEX_ARGS[@]}" "$(cat "$PHASE_PROMPT_FILE")" \
  > "$OUT_JSONL" 2>&1 &
CODEX_PID=$!

# 3. Poll. Default cap 30 min. The orchestrator should pair this with
#    terminal(background=true, notify_on_complete=true) — do NOT block
#    the controller waiting on this; let notify_on_complete fire and
#    react.
WAITED=0
while kill -0 "$CODEX_PID" 2>/dev/null; do
  sleep 5
  WAITED=$((WAITED+5))
  if [[ "$WAITED" -ge "$MAX_WAIT" ]]; then
    echo "TIMEOUT: codex did not exit within $MAX_WAIT s" >&2
    kill -TERM "$CODEX_PID" 2>/dev/null || true
    sleep 2
    kill -KILL "$CODEX_PID" 2>/dev/null || true
    echo "124" > "$EXIT_FILE"
    exit 124
  fi
done

# 4. Capture exit code.
wait "$CODEX_PID" 2>/dev/null || true
EC=$?
echo "$EC" > "$EXIT_FILE"

# 5. Extract the last assistant message from the JSONL. codex emits
#    item.completed events with type=agent_message and a text field.
python3 - "$OUT_JSONL" "$OUT_MSG" <<'PY'
import json, sys
src, dst = sys.argv[1], sys.argv[2]
last = None
with open(src, errors="ignore") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if obj.get("type") != "item.completed":
            continue
        item = obj.get("item") or {}
        if not isinstance(item, dict):
            continue
        if item.get("type") == "agent_message":
            last = item.get("text") or ""
if last is None:
    with open(dst, "w") as f:
        f.write("(no agent_message in JSONL)\n")
else:
    with open(dst, "w") as f:
        f.write(str(last) + "\n")
PY

echo "codex exec finished: phase=$PHASE_ID pid=$CODEX_PID waited=${WAITED}s exit=$EC"
echo "  prompt:    $PHASE_PROMPT_FILE"
echo "  last msg:  $OUT_MSG"
echo "  jsonl:     $OUT_JSONL"
echo "  exit:      $EXIT_FILE"
exit 0
