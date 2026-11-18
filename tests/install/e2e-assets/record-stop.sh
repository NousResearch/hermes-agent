#!/usr/bin/env bash
# Stop a recording started by record-start.sh and verify the file is real.
#
# Usage: record-stop.sh OUTPUT.mkv
#
# Graceful stop: the character 'q' on ffmpeg's live stdin (the control
# fifo). Falls back to SIGINT (also a clean finalize for ffmpeg), then
# SIGKILL. Fails if the output is missing or has no decodable duration -
# a zero-frame recording is the classic silent failure.

set -euo pipefail

OUT="${1:?usage: record-stop.sh OUTPUT.mkv}"
STATE="$OUT.state"

[ -f "$STATE" ] || { echo "record-stop: no state at $STATE (was record-start run?)" >&2; exit 1; }
read -r FFMPEG_PID FIFO _SHEPHERD < "$STATE"

if kill -0 "$FFMPEG_PID" 2>/dev/null; then
  # Write the quit key; don't hang if the reader is already gone.
  { printf 'q' > "$FIFO"; } 2>/dev/null &
  WRITER=$!
  for _ in $(seq 1 50); do
    kill -0 "$FFMPEG_PID" 2>/dev/null || break
    sleep 0.2
  done
  kill "$WRITER" 2>/dev/null || true
  if kill -0 "$FFMPEG_PID" 2>/dev/null; then
    echo "record-stop: q did not stop ffmpeg; SIGINT" >&2
    kill -INT "$FFMPEG_PID" 2>/dev/null || true
    for _ in $(seq 1 25); do
      kill -0 "$FFMPEG_PID" 2>/dev/null || break
      sleep 0.2
    done
    kill -9 "$FFMPEG_PID" 2>/dev/null || true
  fi
fi
rm -f "$FIFO" "$STATE"

[ -s "$OUT" ] || { echo "record-stop: $OUT missing or empty" >&2; exit 1; }
if command -v ffprobe >/dev/null 2>&1; then
  dur="$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$OUT" || echo 0)"
  case "$dur" in
    ''|0|0.*) echo "record-stop: $OUT has no duration (zero-frame recording)" >&2; exit 1 ;;
  esac
  echo "record-stop: $OUT finalized (${dur}s)"
else
  echo "record-stop: $OUT finalized (ffprobe absent; size $(wc -c < "$OUT") bytes)"
fi
