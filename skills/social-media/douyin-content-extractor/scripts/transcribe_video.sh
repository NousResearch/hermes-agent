#!/usr/bin/env bash
# Transcribe a downloaded Douyin video with local MLX Whisper.
#
# Usage:
#   transcribe_video.sh <video.mp4> <output_dir> [output_name]
#
# Env:
#   WHISPER_BIN   path to mlx_whisper binary (default: lookup on PATH, then common user site)
#   WHISPER_MODEL Whisper MLX model id (default: mlx-community/whisper-small-mlx)
#   WHISPER_LANG  Language code (default: zh)

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <video.mp4> <output_dir> [output_name]" >&2
  exit 2
fi

VIDEO="$1"
OUT_DIR="$2"
OUT_NAME="${3:-transcript_small}"
WHISPER_MODEL="${WHISPER_MODEL:-mlx-community/whisper-small-mlx}"
WHISPER_LANG="${WHISPER_LANG:-zh}"

# Locate mlx_whisper: explicit override -> PATH -> common macOS user site
resolve_whisper_bin() {
  if [[ -n "${WHISPER_BIN:-}" ]]; then
    command -v "$WHISPER_BIN" || { echo "WHISPER_BIN not found: $WHISPER_BIN" >&2; exit 3; }
  elif command -v mlx_whisper >/dev/null 2>&1; then
    command -v mlx_whisper
  elif [[ -x "$HOME/Library/Python/3.9/bin/mlx_whisper" ]]; then
    echo "$HOME/Library/Python/3.9/bin/mlx_whisper"
  elif [[ -x "$HOME/Library/Python/3.11/bin/mlx_whisper" ]]; then
    echo "$HOME/Library/Python/3.11/bin/mlx_whisper"
  else
    echo "Could not find mlx_whisper. Install with: pip install mlx-whisper" >&2
    exit 3
  fi
}

WHISPER_BIN_PATH="$(resolve_whisper_bin)"

mkdir -p "$OUT_DIR"
AUDIO="$OUT_DIR/audio.wav"

ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "$VIDEO" > "$OUT_DIR/duration.txt"
ffmpeg -y -i "$VIDEO" -vn -ac 1 -ar 16000 -c:a pcm_s16le "$AUDIO"

"$WHISPER_BIN_PATH" "$AUDIO" \
  --model "$WHISPER_MODEL" \
  --language "$WHISPER_LANG" \
  --output-dir "$OUT_DIR" \
  --output-name "$OUT_NAME" \
  --output-format txt \
  --verbose False

printf 'whisper_bin=%s\nduration_seconds=%s\ntranscript=%s/%s.txt\n' \
  "$WHISPER_BIN_PATH" "$(cat "$OUT_DIR/duration.txt")" "$OUT_DIR" "$OUT_NAME"
