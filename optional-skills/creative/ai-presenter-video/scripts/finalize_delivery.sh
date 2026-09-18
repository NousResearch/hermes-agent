#!/usr/bin/env bash

set -euo pipefail

usage() {
  printf 'Usage: %s rendered.mp4 output-dir output-stem\n' "$(basename "$0")" >&2
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

[[ "$#" -eq 3 ]] || { usage; exit 64; }

INPUT="$1"
OUTPUT_DIR="$2"
STEM="$3"

[[ -f "$INPUT" && -s "$INPUT" ]] || die "input render is missing or empty: $INPUT"
[[ "$STEM" =~ ^[A-Za-z0-9._-]+$ ]] || die "output stem contains unsupported characters: $STEM"

for command in ffmpeg ffprobe jq mktemp awk sed; do
  command -v "$command" >/dev/null 2>&1 || die "required command unavailable: $command"
done

mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd -P)"
INPUT="$(cd "$(dirname "$INPUT")" && pwd -P)/$(basename "$INPUT")"

MASTER="$OUTPUT_DIR/${STEM}-master.mp4"
SHARE="$OUTPUT_DIR/${STEM}-share.mp4"
REPORT="$OUTPUT_DIR/${STEM}-delivery-report.json"
CONTACT="$OUTPUT_DIR/${STEM}-contact-sheet.png"

# Keep shared reports portable and avoid exposing the developer machine path.
INPUT_REPORT="$(basename "$INPUT")"
MASTER_REPORT="$(basename "$MASTER")"
SHARE_REPORT="$(basename "$SHARE")"
CONTACT_REPORT="$(basename "$CONTACT")"

for output in "$MASTER" "$SHARE" "$REPORT" "$CONTACT"; do
  [[ ! -e "$output" ]] || die "refusing to overwrite existing output: $output"
done

TMP_ROOT="${TMPDIR:-/tmp}"
TMP_DIR="$(mktemp -d "${TMP_ROOT%/}/presenter-finalize.XXXXXX")"
trap 'rm -rf -- "$TMP_DIR"' EXIT INT TERM

STREAM_JSON="$TMP_DIR/source-probe.json"
ffprobe -v error -show_streams -show_format -of json "$INPUT" >"$STREAM_JSON"
jq -e '.streams | any(.codec_type == "video") and any(.codec_type == "audio")' "$STREAM_JSON" >/dev/null \
  || die "input must contain decodable video and audio streams"

WIDTH="$(jq -r '[.streams[] | select(.codec_type == "video")][0].width' "$STREAM_JSON")"
HEIGHT="$(jq -r '[.streams[] | select(.codec_type == "video")][0].height' "$STREAM_JSON")"
FPS="$(jq -r '[.streams[] | select(.codec_type == "video")][0].avg_frame_rate' "$STREAM_JSON")"
[[ "$FPS" != "0/0" && -n "$FPS" ]] || FPS="30"

MEASURE_LOG="$TMP_DIR/loudnorm-pass1.log"
ffmpeg -hide_banner -nostdin -nostats -i "$INPUT" \
  -map 0:a:0 -vn \
  -af 'loudnorm=I=-16:TP=-1.5:LRA=9:print_format=json' \
  -f null - >/dev/null 2>"$MEASURE_LOG"

MEASURE_JSON="$TMP_DIR/measure.json"
sed -n '/^{/,/^}/p' "$MEASURE_LOG" >"$MEASURE_JSON"
jq -e . "$MEASURE_JSON" >/dev/null || die "could not parse loudness measurement"

MEASURED_I="$(jq -r .input_i "$MEASURE_JSON")"
MEASURED_TP="$(jq -r .input_tp "$MEASURE_JSON")"
MEASURED_LRA="$(jq -r .input_lra "$MEASURE_JSON")"
MEASURED_THRESH="$(jq -r .input_thresh "$MEASURE_JSON")"
OFFSET="$(jq -r .target_offset "$MEASURE_JSON")"

LOUDNORM="loudnorm=I=-16:TP=-1.5:LRA=9:measured_I=${MEASURED_I}:measured_TP=${MEASURED_TP}:measured_LRA=${MEASURED_LRA}:measured_thresh=${MEASURED_THRESH}:offset=${OFFSET}:linear=true:print_format=summary"
VIDEO_FILTER="scale=trunc(iw/2)*2:trunc(ih/2)*2:flags=lanczos,setsar=1,fps=${FPS},format=yuv420p"

encode() {
  local crf="$1"
  local preset="$2"
  local audio_bitrate="$3"
  local destination="$4"
  local temporary="$5"

  ffmpeg -hide_banner -nostdin -loglevel warning -stats -i "$INPUT" \
    -map 0:v:0 -map 0:a:0 -sn -dn \
    -vf "$VIDEO_FILTER" -af "$LOUDNORM" \
    -c:v libx264 -preset "$preset" -crf "$crf" -profile:v high \
    -pix_fmt yuv420p -tag:v avc1 -fps_mode cfr \
    -color_primaries bt709 -color_trc bt709 -colorspace bt709 \
    -c:a aac -b:a "$audio_bitrate" -ar 48000 -ac 2 \
    -map_metadata -1 -map_chapters -1 -movflags +faststart \
    "$temporary"
  [[ -s "$temporary" ]] || die "encoder produced an empty output"
  mv -- "$temporary" "$destination"
}

encode 16 slow 256k "$MASTER" "$TMP_DIR/master.mp4"
encode 24 medium 160k "$SHARE" "$TMP_DIR/share.mp4"

for file in "$MASTER" "$SHARE"; do
  ffmpeg -hide_banner -nostdin -v error -xerror -i "$file" \
    -map 0:v:0 -map 0:a:0 -f null - >/dev/null
done

ffprobe -v error -show_streams -show_format -of json "$MASTER" >"$TMP_DIR/master-probe.json"
ffprobe -v error -show_streams -show_format -of json "$SHARE" >"$TMP_DIR/share-probe.json"

for probe_file in "$STREAM_JSON" "$TMP_DIR/master-probe.json" "$TMP_DIR/share-probe.json"; do
  jq 'if .format.filename? then .format.filename = (.format.filename | split("/") | last) else . end' \
    "$probe_file" >"${probe_file}.portable"
  mv -- "${probe_file}.portable" "$probe_file"
done

ffmpeg -hide_banner -nostdin -i "$MASTER" \
  -vf 'blackdetect=d=0.10:pix_th=0.02' -an -f null - \
  >/dev/null 2>"$TMP_DIR/blackdetect.log"
BLACK_EVENTS="$(grep -c 'black_start:' "$TMP_DIR/blackdetect.log" || true)"

DURATION="$(ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "$MASTER")"
awk -v duration="$DURATION" 'BEGIN {
  p[1]=0.2; p[2]=duration*0.125; p[3]=duration*0.25; p[4]=duration*0.375;
  p[5]=duration*0.5; p[6]=duration*0.625; p[7]=duration*0.75;
  p[8]=duration*0.875; p[9]=duration-0.2;
  for(i=1;i<=9;i++) printf "%.6f\n", p[i]
}' >"$TMP_DIR/timestamps.txt"

if (( WIDTH > HEIGHT )); then
  CONTACT_FRAME_FILTER='scale=480:270:force_original_aspect_ratio=decrease:flags=lanczos,pad=480:270:(ow-iw)/2:(oh-ih)/2:color=black'
elif (( HEIGHT > WIDTH )); then
  CONTACT_FRAME_FILTER='scale=270:480:force_original_aspect_ratio=decrease:flags=lanczos,pad=270:480:(ow-iw)/2:(oh-ih)/2:color=black'
else
  CONTACT_FRAME_FILTER='scale=360:360:force_original_aspect_ratio=decrease:flags=lanczos,pad=360:360:(ow-iw)/2:(oh-ih)/2:color=black'
fi

index=0
while IFS= read -r timestamp; do
  index=$((index + 1))
  printf -v frame '%s/frame-%02d.png' "$TMP_DIR" "$index"
  ffmpeg -hide_banner -nostdin -loglevel error -ss "$timestamp" -i "$MASTER" \
    -frames:v 1 -vf "$CONTACT_FRAME_FILTER" -update 1 "$frame"
done <"$TMP_DIR/timestamps.txt"

ffmpeg -hide_banner -nostdin -loglevel error -framerate 1 -start_number 1 \
  -i "$TMP_DIR/frame-%02d.png" -frames:v 1 \
  -vf 'tile=3x3:padding=12:margin=12:color=0x101218' -update 1 "$TMP_DIR/contact.png"
mv -- "$TMP_DIR/contact.png" "$CONTACT"

jq -n \
  --arg status verified \
  --arg input "$INPUT_REPORT" \
  --arg master "$MASTER_REPORT" \
  --arg share "$SHARE_REPORT" \
  --arg contact "$CONTACT_REPORT" \
  --arg duration "$DURATION" \
  --argjson black_events "$BLACK_EVENTS" \
  --slurpfile measured "$MEASURE_JSON" \
  --slurpfile source_probe "$STREAM_JSON" \
  --slurpfile master_probe "$TMP_DIR/master-probe.json" \
  --slurpfile share_probe "$TMP_DIR/share-probe.json" \
  '{status:$status,input:$input,master:$master,share:$share,contact_sheet:$contact,duration_s:($duration|tonumber),source_loudness:$measured[0],source_probe:$source_probe[0],master_probe:$master_probe[0],share_probe:$share_probe[0],full_decode_passed:true,black_frame_events:$black_events}' \
  >"$TMP_DIR/report.json"
mv -- "$TMP_DIR/report.json" "$REPORT"

printf 'Master: %s\nShare: %s\nReport: %s\nContact: %s\n' "$MASTER" "$SHARE" "$REPORT" "$CONTACT"
