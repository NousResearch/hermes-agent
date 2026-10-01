#!/usr/bin/env bash
set -uo pipefail

review_out="$(mktemp "${TMPDIR:-/tmp}/codex-review-final.XXXXXX")"
review_log="$(mktemp "${TMPDIR:-/tmp}/codex-review-log.XXXXXX")"
keep_log=0
review_pid=""
heartbeat_pid=""
heartbeat_status=0
review_codex_home=""
codex_bin=""
review_target=(--uncommitted)
heartbeat_seconds="${CODEX_REVIEW_HEARTBEAT_SECONDS:-60}"
stalled_heartbeat_limit=3
termination_grace_seconds=5

print_usage() {
  cat >&2 <<'EOF'
Usage: ./.codex/scripts/codex-review.sh [--uncommitted|--base-main]

Modes:
  --uncommitted  Review staged, unstaged, and untracked changes. This is the default.
  --base-main    Review changes against origin/main. Use only when explicitly requested.
EOF
}

parse_review_mode() {
  while [[ "$#" -gt 0 ]]; do
    case "$1" in
      --uncommitted)
        review_target=(--uncommitted)
        ;;
      --base-main)
        review_target=(--base origin/main)
        ;;
      -h|--help)
        print_usage
        exit 0
        ;;
      *)
        print_usage
        printf 'Unknown codex review mode: %s\n' "$1" >&2
        exit 2
        ;;
    esac
    shift
  done
}

positive_seconds_or_default() {
  local value="$1"
  local default="$2"

  if [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
    printf '%s' "$value"
  else
    printf '%s' "$default"
  fi
}

format_duration() {
  local total_seconds="$1"
  local hours=$((total_seconds / 3600))
  local minutes=$(((total_seconds % 3600) / 60))
  local seconds=$((total_seconds % 60))

  if ((hours > 0)); then
    printf '%dh%02dm%02ds' "$hours" "$minutes" "$seconds"
  elif ((minutes > 0)); then
    printf '%dm%02ds' "$minutes" "$seconds"
  else
    printf '%ds' "$seconds"
  fi
}

log_size_bytes() {
  wc -c <"$review_log" | tr -d '[:space:]'
}

heartbeat_signal_exit() {
  if [[ -n "${sleep_pid:-}" ]]; then
    kill "$sleep_pid" 2>/dev/null || true
  fi
  if [[ "${stalled:-0}" == "1" ]]; then
    exit 124
  fi
  exit 0
}

copy_codex_config_without_service_tier() {
  local source_file="$1"
  local target_file="$2"

  awk '$0 !~ /^[[:space:]]*service_tier[[:space:]]*=/' "$source_file" >"$target_file"
}

resolve_codex_bin() {
  local candidate

  if [[ -n "${CODEX_REVIEW_CODEX_BIN:-}" ]]; then
    if [[ -x "$CODEX_REVIEW_CODEX_BIN" ]] && "$CODEX_REVIEW_CODEX_BIN" --version >/dev/null 2>&1; then
      printf '%s' "$CODEX_REVIEW_CODEX_BIN"
      return 0
    fi

    printf 'CODEX_REVIEW_CODEX_BIN is not a working executable: %s\n' "$CODEX_REVIEW_CODEX_BIN" >&2
    return 1
  fi

  candidate="$(command -v codex 2>/dev/null || true)"
  if [[ -n "$candidate" ]] && "$candidate" --version >/dev/null 2>&1; then
    printf '%s' "$candidate"
    return 0
  fi

  for candidate in \
    "/Applications/Codex.app/Contents/Resources/codex" \
    "/Applications/ChatGPT.app/Contents/Resources/codex" \
    "$HOME/Applications/Codex.app/Contents/Resources/codex" \
    "$HOME/Applications/ChatGPT.app/Contents/Resources/codex"; do
    if [[ -x "$candidate" ]] && "$candidate" --version >/dev/null 2>&1; then
      printf '%s' "$candidate"
      return 0
    fi
  done

  printf 'Unable to find a working Codex CLI. Set CODEX_REVIEW_CODEX_BIN to a working codex executable.\n' >&2
  return 1
}

prepare_codex_home() {
  local source_home="${CODEX_HOME:-$HOME/.codex}"
  local entry
  local name

  review_codex_home="$(mktemp -d "${TMPDIR:-/tmp}/codex-review-home.XXXXXX")"

  if [[ ! -d "$source_home" ]]; then
    return 0
  fi

  shopt -s nullglob dotglob
  for entry in "$source_home"/*; do
    name="$(basename "$entry")"
    case "$name" in
      config.toml|*.config.toml)
        copy_codex_config_without_service_tier "$entry" "$review_codex_home/$name"
        ;;
      *)
        ln -s "$entry" "$review_codex_home/$name"
        ;;
    esac
  done
  shopt -u nullglob dotglob
}

heartbeat_review() {
  local pid="$1"
  local started_at="$2"
  local interval="$3"
  local stalled_after="$4"
  local termination_grace="$5"
  local last_size
  local last_change_at
  local now
  local current_size
  local elapsed
  local quiet_for
  local log_detail
  local unchanged_heartbeats=0
  local termination_waited=0
  local stalled=0
  local sleep_pid

  trap heartbeat_signal_exit TERM INT
  last_size="$(log_size_bytes)"
  last_change_at="$started_at"

  while kill -0 "$pid" 2>/dev/null; do
    sleep "$interval" &
    sleep_pid=$!
    wait "$sleep_pid" || return 0
    sleep_pid=""
    if ! kill -0 "$pid" 2>/dev/null; then
      return 0
    fi

    now="$(date +%s)"
    current_size="$(log_size_bytes)"
    elapsed=$((now - started_at))

    if [[ "$current_size" != "$last_size" ]]; then
      last_size="$current_size"
      last_change_at="$now"
      unchanged_heartbeats=0
      log_detail="log grew to ${current_size} bytes"
    else
      unchanged_heartbeats=$((unchanged_heartbeats + 1))
      quiet_for=$((now - last_change_at))
      log_detail="log unchanged for $(format_duration "$quiet_for") (${current_size} bytes; heartbeat ${unchanged_heartbeats}/${stalled_after})"
    fi

    if ((unchanged_heartbeats >= stalled_after)); then
      trap '' TERM INT
      if ! kill -TERM -- "-$pid" 2>/dev/null; then
        trap heartbeat_signal_exit TERM INT
        return 0
      fi
      stalled=1
      trap heartbeat_signal_exit TERM INT
      printf '[codex-review] review stalled: elapsed=%s pid=%s %s; terminating its process group so it can be rerun\n' \
        "$(format_duration "$elapsed")" \
        "$pid" \
        "$log_detail" >&2

      while kill -0 -- "-$pid" 2>/dev/null && ((termination_waited < termination_grace)); do
        sleep 1 &
        sleep_pid=$!
        wait "$sleep_pid" || return 0
        sleep_pid=""
        termination_waited=$((termination_waited + 1))
      done

      if kill -0 -- "-$pid" 2>/dev/null; then
        printf '[codex-review] review process group ignored SIGTERM for %ss; forcing termination: pgid=%s\n' \
          "$termination_grace" \
          "$pid" >&2
        kill -KILL -- "-$pid" 2>/dev/null || true
      fi
      return 124
    fi

    printf '[codex-review] still running: elapsed=%s pid=%s %s\n' \
      "$(format_duration "$elapsed")" \
      "$pid" \
      "$log_detail" >&2
  done
}

stop_heartbeat() {
  heartbeat_status=0
  if [[ -n "${heartbeat_pid:-}" ]]; then
    kill "$heartbeat_pid" 2>/dev/null || true
    wait "$heartbeat_pid" 2>/dev/null
    heartbeat_status=$?
    heartbeat_pid=""
  fi
}

cleanup() {
  stop_heartbeat
  rm -f "$review_out"
  if [[ -n "${review_codex_home:-}" ]]; then
    rm -rf "$review_codex_home"
  fi
  if [[ "$keep_log" != "1" ]]; then
    rm -f "$review_log"
  fi
}
trap cleanup EXIT

parse_review_mode "$@"

heartbeat_seconds="$(positive_seconds_or_default "$heartbeat_seconds" 60)"

codex_bin="$(resolve_codex_bin)"
prepare_codex_home

set -m
CODEX_HOME="$review_codex_home" "$codex_bin" exec review -m gpt-6-astra -c features.fast_mode=false -c model_reasoning_effort='"medium"' -c sandbox_mode='"read-only"' --output-last-message "$review_out" "${review_target[@]}" >"$review_log" 2>&1 &
review_pid=$!
set +m
printf '[codex-review] review started: pid=%s heartbeat=%ss stall_limit=%s\n' \
  "$review_pid" \
  "$heartbeat_seconds" \
  "$stalled_heartbeat_limit" >&2
heartbeat_review \
  "$review_pid" \
  "$(date +%s)" \
  "$heartbeat_seconds" \
  "$stalled_heartbeat_limit" \
  "$termination_grace_seconds" &
heartbeat_pid=$!
wait "$review_pid"
status=$?
stop_heartbeat

if [[ "$heartbeat_status" == "124" ]]; then
  status=124
fi

if [[ -s "$review_out" ]]; then
  cat "$review_out"
else
  cat "$review_log"
fi

if [[ "$status" -ne 0 ]]; then
  keep_log=1
  printf '\nFull Codex review log: %s\n' "$review_log" >&2
fi

exit "$status"
