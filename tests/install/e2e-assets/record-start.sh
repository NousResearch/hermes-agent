#!/usr/bin/env bash
# Start a continuous ffmpeg screen recording in the background.
#
# Usage: record-start.sh OUTPUT.mkv [INPUT_ARGS...]
#   OUTPUT.mkv   where to record (mkv: stays playable even unfinalized)
#   INPUT_ARGS   optional ffmpeg input override; default picks per-OS:
#                linux  -f x11grab -i "$DISPLAY" (Xvfb or real)
#                macos  -f avfoundation -i <first screen index>
#
# Writes state (pid + control fifo path) next to OUTPUT as OUTPUT.state so
# record-stop.sh can finalize gracefully: ffmpeg stops cleanly on the
# character 'q' on live stdin, so stdin is a fifo we hold open. A missing
# ffmpeg is a HARD error - a graceful skip makes the missing tool invisible
# and the artifact silently loses its recording.

set -euo pipefail

OUT="${1:?usage: record-start.sh OUTPUT.mkv [input args...]}"
shift || true

command -v ffmpeg >/dev/null 2>&1 || {
  echo "record-start: ffmpeg not on PATH (the workflow must install it)" >&2
  exit 1
}

INPUT=("$@")
if [ "${#INPUT[@]}" -eq 0 ]; then
  case "$(uname -s)" in
    Linux)
      : "${DISPLAY:?record-start: DISPLAY not set (start Xvfb first on headless runners)}"
      INPUT=(-f x11grab -framerate 15 -i "$DISPLAY")
      ;;
    Darwin)
      # avfoundation lists devices on stderr; the first "Capture screen"
      # index is the whole display. Parse it rather than hardcoding: the
      # index shifts with attached cameras.
      #
      # `-list_devices true -i ""` always exits non-zero.
      # Capture output/status explicitly (with `|| true`) instead of letting a bare assignment
      # trip `set -e`, which would abort before we ever get to report
      # anything useful.
      probe_out="$(ffmpeg -f avfoundation -list_devices true -i "" 2>&1 || true)"
      screen_idx="$(printf '%s\n' "$probe_out" \
        | sed -n 's/^\[AVFoundation[^]]*\] \[\([0-9]*\)\] Capture screen.*/\1/p' | head -1)"

      if [ -z "$screen_idx" ]; then
        if printf '%s\n' "$probe_out" | grep -qiE 'Input/output error|Unknown input format|errno 5'; then
          echo "record-start: avfoundation could not enumerate devices (I/O error)" >&2
        else
          echo "record-start: no capture screen device found in avfoundation device list:" >&2
        fi
        printf '%s\n' "$probe_out" >&2
        exit 1
      fi
      INPUT=(-f avfoundation -framerate 15 -capture_cursor 1 -i "${screen_idx}:none")
      ;;
    *)
      echo "record-start: unsupported OS $(uname -s) (windows uses record-start.ps1)" >&2
      exit 1
      ;;
  esac
fi

mkdir -p "$(dirname "$OUT")"
FIFO="$OUT.ctl"
STATE="$OUT.state"
rm -f "$FIFO" "$STATE"
mkfifo "$FIFO"

# Hold the fifo's write end open in a shepherd process; ffmpeg reads its
# stdin from the fifo. record-stop.sh writes 'q' into the fifo.
ffmpeg -hide_banner -loglevel error "${INPUT[@]}" \
  -pix_fmt yuv420p -c:v libx264 -preset ultrafast "$OUT" < "$FIFO" &
FFMPEG_PID=$!
# Open a persistent write fd so the fifo doesn't EOF before stop.
exec 9> "$FIFO"
# Hand the fd to a shepherd that outlives this script.
(
  exec 9>&9
  while kill -0 "$FFMPEG_PID" 2>/dev/null; do sleep 1; done
) &
SHEPHERD_PID=$!
disown "$SHEPHERD_PID" 2>/dev/null || true

printf '%s %s %s\n' "$FFMPEG_PID" "$FIFO" "$SHEPHERD_PID" > "$STATE"
echo "record-start: recording to $OUT (ffmpeg pid $FFMPEG_PID)"
