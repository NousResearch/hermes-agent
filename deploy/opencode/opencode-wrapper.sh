#!/usr/bin/env bash
# SYNTHWAVE opencode wrapper (installed as ~/.npm-global/bin/opencode).
#
# 2026-08-24 recursion guard: opencode spawned itself 11 levels deep via bash
# and took the VPS down (load 46). A nested `opencode` call from inside an
# opencode run is always the model delegating to itself.
# 2026-10-15 hardening for opencode_task: a global limit on concurrent
# `opencode run` jobs (flock slots), a default model, and the OmniRoute key
# read from a 600 file so no model ever needs to grep ~/.hermes/.env.
# The governed path is tools/opencode_task_runner.py in hermes-agent; this
# wrapper covers direct CLI use.
OPENCODE_REAL_BIN="__OPENCODE_REAL_BIN__"
OPENCODE_SLOT_DIR="__OPENCODE_SLOT_DIR__"
OPENCODE_KEY_FILE="__OPENCODE_KEY_FILE__"
OPENCODE_MAX_SLOTS="${OPENCODE_MAX_SLOTS:-12}"
OPENCODE_DEFAULT_MODEL="omniroute/synthwave-auto"

if [ -n "${OPENCODE_RECURSION_GUARD:-}" ]; then
  echo "opencode: nested invocation blocked by recursion guard. You are already running inside an opencode session: perform the task directly with your own tools instead of delegating to a new 'opencode run'." >&2
  exit 1
fi
export OPENCODE_RECURSION_GUARD=1

if [ -z "${OMNIROUTE_API_KEY:-}" ] && [ -r "$OPENCODE_KEY_FILE" ]; then
  OMNIROUTE_API_KEY="$(head -n 1 "$OPENCODE_KEY_FILE")"
  export OMNIROUTE_API_KEY
fi
# 2026-11-22 PlainBoard bridge token for the MCP server "plainboard" in opencode.jsonc (600 file,
# the host user's own token; opencode_task passes each caller's own token instead).
OPENCODE_PLAINBOARD_TOKEN_FILE="${OPENCODE_PLAINBOARD_TOKEN_FILE:-$(dirname "$OPENCODE_KEY_FILE")/plainboard.token}"
if [ -z "${PLAINBOARD_MCP_TOKEN:-}" ] && [ -r "$OPENCODE_PLAINBOARD_TOKEN_FILE" ]; then
  PLAINBOARD_MCP_TOKEN="$(head -n 1 "$OPENCODE_PLAINBOARD_TOKEN_FILE")"
  export PLAINBOARD_MCP_TOKEN
fi

if [ "${1:-}" = "run" ]; then
  shift
  has_model=0
  has_server=0
  for arg in "$@"; do
    case "$arg" in
      -m|--model|--model=*) has_model=1 ;;
      --standalone|--server|--server=*) has_server=1 ;;
    esac
  done
  if [ "$has_model" = 0 ]; then
    set -- -m "$OPENCODE_DEFAULT_MODEL" "$@"
  fi
  # 2026-11-22 OpenCode 2: `run` talks to a shared background service by default, which keeps the
  # environment it was started with (key, PlainBoard token). A private server per run keeps the
  # wrapper's environment and slots meaningful.
  if [ "$has_server" = 0 ]; then
    set -- --standalone "$@"
  fi
  set -- run "$@"

  mkdir -p "$OPENCODE_SLOT_DIR" 2>/dev/null && chmod 700 "$OPENCODE_SLOT_DIR" 2>/dev/null
  wait_left="${OPENCODE_SLOT_WAIT:-600}"
  case "$wait_left" in ''|*[!0-9]*) wait_left=600 ;; esac
  while :; do
    i=0
    while [ "$i" -lt "$OPENCODE_MAX_SLOTS" ]; do
      exec 9>"$OPENCODE_SLOT_DIR/slot-$i.lock"
      if flock -n 9; then
        # fd 9 stays open (and locked) in the exec'd process until it exits.
        exec "$OPENCODE_REAL_BIN" "$@"
      fi
      exec 9>&-
      i=$((i + 1))
    done
    if [ "$wait_left" -le 0 ]; then
      echo "opencode: all $OPENCODE_MAX_SLOTS slots are busy; retry later or queue the job." >&2
      exit 75
    fi
    sleep 2
    wait_left=$((wait_left - 2))
  done
fi

exec "$OPENCODE_REAL_BIN" "$@"
