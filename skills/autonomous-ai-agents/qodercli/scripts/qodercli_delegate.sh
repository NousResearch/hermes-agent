#!/bin/bash
# qodercli_delegate.sh — print-mode delegation wrapper for the qodercli skill.
#
# One terminal call, one structured result. No interactive monitoring: runs
# qodercli in print mode (-p), classifies the outcome, and emits JSON on stdout.
#
# Usage:
#   bash scripts/qodercli_delegate.sh "<prompt>" [workdir] [timeout_seconds]
#   bash scripts/qodercli_delegate.sh "Migrate src/db/ to SQLAlchemy" ~/project 600
#
# Output: JSON on stdout — exit_code, error_class, files_changed, diff_stat,
#         output_tail, workdir, timeout_used, git_before, git_after.
# Exit:   0 qodercli succeeded | 1 qodercli failed | 2 preflight failed.

set -o pipefail

if [ $# -lt 1 ] || [ -z "${1:-}" ]; then
  echo '{"error_class":"usage_error","exit_code":64,"message":"Usage: qodercli_delegate.sh \"<prompt>\" [workdir] [timeout]"}'
  exit 2
fi

PROMPT="$1"
WORKDIR="${2:-.}"
TIMEOUT="${3:-600}"

# --- Preflight ---

QODERCLI_BIN="${HERMES_QODERCLI_BIN:-$(command -v qodercli 2>/dev/null)}"
if [ -z "$QODERCLI_BIN" ] || [ ! -x "$QODERCLI_BIN" ]; then
  echo '{"error_class":"binary_not_found","exit_code":127,"message":"qodercli not found or not executable. Set HERMES_QODERCLI_BIN, or install: npm install -g @qoder-ai/qodercli"}'
  exit 2
fi

if [ ! -d "$WORKDIR" ]; then
  printf '{"error_class":"workdir_not_found","exit_code":126,"message":"%s does not exist"}\n' "$WORKDIR"
  exit 2
fi

cd "$WORKDIR" || exit 2

# --- Execute ---

# Portable timeout: Linux ships `timeout`, macOS needs coreutils' `gtimeout`.
# An array keeps the expansion quoted-safe; empty means "no timeout wrapper".
TIMEOUT_PREFIX=()
if command -v timeout >/dev/null 2>&1; then
  TIMEOUT_PREFIX=(timeout "$TIMEOUT")
elif command -v gtimeout >/dev/null 2>&1; then
  TIMEOUT_PREFIX=(gtimeout "$TIMEOUT")
fi

GIT_BEFORE=$(git rev-parse HEAD 2>/dev/null || echo "no-git")

OUTPUT_FILE=$(mktemp "${TMPDIR:-/tmp}/qodercli-delegate.XXXXXX") || exit 2

"${TIMEOUT_PREFIX[@]}" "$QODERCLI_BIN" -p "$PROMPT" \
  --permission-mode bypass_permissions \
  > "$OUTPUT_FILE" 2>&1
EXIT_CODE=$?

# --- Classify result ---

# Extended regex (-E) rather than BRE `\|`, which BSD/macOS grep does not
# support reliably and this skill declares macos as a target platform.
ERROR_CLASS="none"
if [ $EXIT_CODE -eq 124 ]; then
  ERROR_CLASS="timeout"
elif [ $EXIT_CODE -ne 0 ]; then
  if grep -qiE "not logged in|please run /login|401|403" "$OUTPUT_FILE" 2>/dev/null; then
    ERROR_CLASS="auth_failure"
  elif grep -qiE "402|credit|quota" "$OUTPUT_FILE" 2>/dev/null; then
    ERROR_CLASS="credit_exhausted"
  elif grep -qiE "permission.*required|confirmation required" "$OUTPUT_FILE" 2>/dev/null; then
    ERROR_CLASS="permission_blocked"
  elif grep -qiE "ECONNREFUSED|ETIMEDOUT|ENOTFOUND|network" "$OUTPUT_FILE" 2>/dev/null; then
    ERROR_CLASS="network_error"
  else
    ERROR_CLASS="unknown_failure"
  fi
fi

# --- Collect evidence ---

GIT_AFTER=$(git rev-parse HEAD 2>/dev/null || echo "no-git")
DIFF_STAT=$(git diff --stat 2>/dev/null | tail -1)
FILES_CHANGED=$(git diff --name-only 2>/dev/null | wc -l | tr -d ' ')
OUTPUT_TAIL=$(tail -3 "$OUTPUT_FILE" 2>/dev/null | tr '\n' ' ' | cut -c1-200)

rm -f "$OUTPUT_FILE"

# --- Structured output ---

# qodercli output routinely carries ANSI escapes, backslashes and quotes. Strip
# C0 control bytes and escape backslash before quote so stdout always parses.
json_escape() {
  printf '%s' "$1" | tr -d '\000-\037' | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g'
}

printf '{
  "exit_code": %s,
  "error_class": "%s",
  "files_changed": %s,
  "diff_stat": "%s",
  "output_tail": "%s",
  "workdir": "%s",
  "timeout_used": %s,
  "git_before": "%s",
  "git_after": "%s"
}\n' \
  "$EXIT_CODE" \
  "$ERROR_CLASS" \
  "${FILES_CHANGED:-0}" \
  "$(json_escape "$DIFF_STAT")" \
  "$(json_escape "$OUTPUT_TAIL")" \
  "$(json_escape "$WORKDIR")" \
  "$TIMEOUT" \
  "$GIT_BEFORE" \
  "$GIT_AFTER"

exit $EXIT_CODE
