#!/usr/bin/env bash
set -euo pipefail

ROOT="${OPENCLAW_ROOT:-/home/yeqiuqiu/clawd-architect}"
TASK="sessions_prune_weekly"
PRUNE_SCRIPT="${OPENCLAW_SESSIONS_PRUNE_SCRIPT:-$ROOT/ops/openclaw/sessions_prune_older_than_14d.sh}"
STDERR_MAX="${OPENCLAW_SESSIONS_PRUNE_STDERR_MAX:-180}"

sanitize_inline() {
  printf '%s' "${1:-}" | tr '\r\n\t' '   ' | sed -e 's/[[:space:]]\+/ /g' -e 's/^ *//' -e 's/ *$//'
}

if ! [[ "$STDERR_MAX" =~ ^[0-9]+$ ]]; then
  STDERR_MAX=180
fi

if [[ ! -f "$PRUNE_SCRIPT" ]]; then
  printf 'BLOCKER: task=%s; reason=prune_script_missing; path=%s\n' "$TASK" "$PRUNE_SCRIPT"
  exit 0
fi

err_file="$(mktemp /tmp/contract_sessions_prune_weekly.XXXXXX.err)"
trap 'rm -f "$err_file"' EXIT

set +e
cmd_out="$(bash "$PRUNE_SCRIPT" 2>"$err_file")"
cmd_rc=$?
set -e

first_line="$(printf '%s\n' "$cmd_out" | awk 'NF{print; exit}')"

if [[ "$cmd_rc" -ne 0 ]]; then
  err_raw="$(cat "$err_file" 2>/dev/null || true)"
  err="$(sanitize_inline "$err_raw")"
  if [[ -z "$err" ]]; then
    err="$(sanitize_inline "$first_line")"
  fi
  if [[ -z "$err" ]]; then
    err="no_stderr"
  fi

  printf 'BLOCKER: task=%s; reason=prune_exec_failed; rc=%s; err=%s\n' "$TASK" "$cmd_rc" "${err:0:STDERR_MAX}"
  exit 0
fi

if [[ "$first_line" == BLOCKER:* ]]; then
  printf '%s\n' "$first_line"
  exit 0
fi

if [[ "$first_line" =~ ^Found[[:space:]]+([0-9]+)[[:space:]]+files[[:space:]]+older[[:space:]]+than[[:space:]]+14d[[:space:]]+\(bytes=([0-9]+)\)\.[[:space:]]+Archiving[[:space:]]+to[[:space:]]+(.+)$ ]]; then
  count="${BASH_REMATCH[1]}"
  bytes="${BASH_REMATCH[2]}"
  arch_path="${BASH_REMATCH[3]}"
  arch_file="${arch_path##*/}"
  done_line="$(printf '%s\n' "$cmd_out" | awk '/^Done\. Archive size=/{print; exit}')"

  archive_bytes="unknown"
  if [[ "$done_line" =~ ^Done\.[[:space:]]+Archive[[:space:]]+size=([0-9]+)[[:space:]]+bytes$ ]]; then
    archive_bytes="${BASH_REMATCH[1]}"
  fi

  printf 'SUMMARY: task=%s; pruned_files=%s; pruned_bytes=%s; archive=%s; archive_bytes=%s\n' \
    "$TASK" "$count" "$bytes" "$arch_file" "$archive_bytes"
  exit 0
fi

printf 'NO_REPLY\n'
exit 0
