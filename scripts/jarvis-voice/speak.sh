#!/usr/bin/env bash

MEDIA_FILE=""
LOCKDIR=""
LOCK_HELD=0

cleanup() {
  if [ -n "$MEDIA_FILE" ]; then
    rm -f -- "$MEDIA_FILE" >/dev/null 2>&1 || true
  fi
  if [ "$LOCK_HELD" = "1" ] && [ -n "$LOCKDIR" ]; then
    rmdir -- "$LOCKDIR" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT HUP INT TERM

main() {
  if [ "${JARVIS_VOICE_DISABLED:-0}" = "1" ]; then
    return 0
  fi

  local text="$*"
  if [ -z "${text//[[:space:]]/}" ]; then
    return 0
  fi

  local script_dir
  script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)" || return 0

  # v2 ใช้ .venv-gemini (กล่อง .venv เก่าถูกลบไปแล้ว 2026-07-05)
  local edge_tts="$script_dir/.venv-gemini/bin/edge-tts"
  if [ ! -x "$edge_tts" ]; then
    edge_tts="$script_dir/.venv/bin/edge-tts"
  fi
  if [ ! -x "$edge_tts" ]; then
    return 0
  fi

  local tmp_root="${TMPDIR:-/tmp}"
  tmp_root="${tmp_root%/}"
  if [ ! -d "$tmp_root" ]; then
    tmp_root="/tmp"
  fi

  local voice="${JARVIS_VOICE:-th-TH-NiwatNeural}"
  local rate="${JARVIS_RATE:-+10%}"
  local pidfile="$tmp_root/jarvis-voice.pid"
  LOCKDIR="$tmp_root/jarvis-voice.lock"

  MEDIA_FILE="$(mktemp "$tmp_root/jarvis-voice.XXXXXX" 2>/dev/null)" || return 0

  if ! generate_with_timeout "$edge_tts" "$voice" "$rate" "$text" "$MEDIA_FILE"; then
    return 0
  fi

  if [ ! -s "$MEDIA_FILE" ]; then
    return 0
  fi

  if ! acquire_lock "$LOCKDIR"; then
    return 0
  fi
  LOCK_HELD=1

  kill_previous_playback "$pidfile"

  afplay "$MEDIA_FILE" >/dev/null 2>&1 &
  local play_pid=$!
  printf '%s\n' "$play_pid" >"$pidfile" 2>/dev/null || true

  release_lock

  wait "$play_pid" >/dev/null 2>&1 || true

  local current_pid=""
  current_pid="$(cat "$pidfile" 2>/dev/null || true)"
  if [ "$current_pid" = "$play_pid" ]; then
    rm -f -- "$pidfile" >/dev/null 2>&1 || true
  fi

  return 0
}

generate_with_timeout() {
  local edge_tts="$1"
  local voice="$2"
  local rate="$3"
  local text="$4"
  local media_file="$5"
  local timeout_seconds=25

  "$edge_tts" \
    --voice "$voice" \
    --rate "$rate" \
    --text "$text" \
    --write-media "$media_file" \
    >/dev/null 2>&1 &

  local edge_pid=$!
  (
    sleep "$timeout_seconds"
    kill "$edge_pid" >/dev/null 2>&1 || true
  ) &
  local watchdog_pid=$!

  wait "$edge_pid" >/dev/null 2>&1
  local status=$?

  kill "$watchdog_pid" >/dev/null 2>&1 || true
  wait "$watchdog_pid" >/dev/null 2>&1 || true

  return "$status"
}

acquire_lock() {
  local lockdir="$1"
  local i
  for i in {1..50}; do
    if mkdir "$lockdir" >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.05
  done
  return 1
}

release_lock() {
  if [ "$LOCK_HELD" = "1" ] && [ -n "$LOCKDIR" ]; then
    rmdir -- "$LOCKDIR" >/dev/null 2>&1 || true
    LOCK_HELD=0
  fi
}

kill_previous_playback() {
  local pidfile="$1"
  local old_pid=""
  old_pid="$(cat "$pidfile" 2>/dev/null || true)"

  case "$old_pid" in
    ''|*[!0-9]*)
      return 0
      ;;
  esac

  kill "$old_pid" >/dev/null 2>&1 || true
  return 0
}

main "$@" >/dev/null 2>&1 || true
exit 0
