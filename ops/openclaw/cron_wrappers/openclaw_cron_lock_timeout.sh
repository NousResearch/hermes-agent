#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE' >&2
Usage:
  openclaw_cron_lock_timeout.sh [options] -- <command> [args...]

Options:
  --lock-name <name>       Logical lock name (required)
  --lock-dir <dir>         Lock directory root (default: /tmp/openclaw_cron_locks)
  --timeout-sec <n>        Hard timeout seconds before SIGTERM (default: 300)
  --grace-sec <n>          Grace seconds between SIGTERM and SIGKILL (default: 30)
  --busy-exit-code <n>     Exit code when lock already held (default: 0)
  --emit-blocker           Emit BLOCKER line on timeout
  --soft-timeout           Return 0 after timeout (instead of timeout exit code)
  -h, --help               Show help

Notes:
- Uses flock when available; falls back to lockdir mutex.
- Timeout path escalates SIGTERM -> SIGKILL.
USAGE
}

LOCK_NAME=""
LOCK_DIR_ROOT="/tmp/openclaw_cron_locks"
TIMEOUT_SEC=300
GRACE_SEC=30
BUSY_EXIT_CODE=0
EMIT_BLOCKER=0
SOFT_TIMEOUT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --lock-name)
      LOCK_NAME="${2:-}"
      shift 2
      ;;
    --lock-dir)
      LOCK_DIR_ROOT="${2:-}"
      shift 2
      ;;
    --timeout-sec)
      TIMEOUT_SEC="${2:-}"
      shift 2
      ;;
    --grace-sec)
      GRACE_SEC="${2:-}"
      shift 2
      ;;
    --busy-exit-code)
      BUSY_EXIT_CODE="${2:-}"
      shift 2
      ;;
    --emit-blocker)
      EMIT_BLOCKER=1
      shift
      ;;
    --soft-timeout)
      SOFT_TIMEOUT=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "$LOCK_NAME" ]]; then
  echo "--lock-name is required" >&2
  exit 2
fi

if [[ $# -eq 0 ]]; then
  echo "missing command after --" >&2
  exit 2
fi

if ! [[ "$TIMEOUT_SEC" =~ ^[0-9]+$ ]] || ! [[ "$GRACE_SEC" =~ ^[0-9]+$ ]] || ! [[ "$BUSY_EXIT_CODE" =~ ^-?[0-9]+$ ]]; then
  echo "timeout/grace/busy-exit-code must be integers" >&2
  exit 2
fi

safe_lock_name="$(printf '%s' "$LOCK_NAME" | tr -cs 'A-Za-z0-9._-' '_')"
[[ -n "$safe_lock_name" ]] || safe_lock_name="cron_task"
mkdir -p "$LOCK_DIR_ROOT"

timeout_happened=0

if command -v flock >/dev/null 2>&1; then
  lock_file="$LOCK_DIR_ROOT/${safe_lock_name}.lock"
  exec 9>"$lock_file"
  if ! flock -n 9; then
    exit "$BUSY_EXIT_CODE"
  fi

  set +e
  if command -v timeout >/dev/null 2>&1; then
    timeout --foreground --signal=TERM --kill-after="${GRACE_SEC}s" "${TIMEOUT_SEC}s" "$@"
    rc=$?
    [[ "$rc" -eq 124 || "$rc" -eq 137 ]] && timeout_happened=1
  else
    "$@" &
    child_pid=$!
    start_epoch=$(date +%s)
    rc=0
    while kill -0 "$child_pid" 2>/dev/null; do
      now_epoch=$(date +%s)
      if (( now_epoch - start_epoch >= TIMEOUT_SEC )); then
        timeout_happened=1
        kill -TERM "$child_pid" 2>/dev/null || true
        sleep "$GRACE_SEC"
        kill -KILL "$child_pid" 2>/dev/null || true
        wait "$child_pid" 2>/dev/null || true
        rc=124
        break
      fi
      sleep 1
    done
    if [[ "$timeout_happened" -eq 0 ]]; then
      wait "$child_pid"
      rc=$?
    fi
  fi
  set -e
else
  lock_dir="$LOCK_DIR_ROOT/${safe_lock_name}.lock.d"
  if ! mkdir "$lock_dir" 2>/dev/null; then
    exit "$BUSY_EXIT_CODE"
  fi
  trap 'rmdir "$lock_dir" 2>/dev/null || true' EXIT

  set +e
  if command -v timeout >/dev/null 2>&1; then
    timeout --foreground --signal=TERM --kill-after="${GRACE_SEC}s" "${TIMEOUT_SEC}s" "$@"
    rc=$?
    [[ "$rc" -eq 124 || "$rc" -eq 137 ]] && timeout_happened=1
  else
    "$@" &
    child_pid=$!
    start_epoch=$(date +%s)
    rc=0
    while kill -0 "$child_pid" 2>/dev/null; do
      now_epoch=$(date +%s)
      if (( now_epoch - start_epoch >= TIMEOUT_SEC )); then
        timeout_happened=1
        kill -TERM "$child_pid" 2>/dev/null || true
        sleep "$GRACE_SEC"
        kill -KILL "$child_pid" 2>/dev/null || true
        wait "$child_pid" 2>/dev/null || true
        rc=124
        break
      fi
      sleep 1
    done
    if [[ "$timeout_happened" -eq 0 ]]; then
      wait "$child_pid"
      rc=$?
    fi
  fi
  set -e
fi

if [[ "$timeout_happened" -eq 1 && "$EMIT_BLOCKER" -eq 1 ]]; then
  echo "BLOCKER: task=${safe_lock_name}; reason=timeout>${TIMEOUT_SEC}s"
fi

if [[ "$timeout_happened" -eq 1 && "$SOFT_TIMEOUT" -eq 1 ]]; then
  exit 0
fi

exit "${rc:-0}"
