#!/usr/bin/env python3
"""Canonical transcription script: faster-whisper large-v3 (int8, CPU), with progress.

Usage: python3 transcribe.py <audio_path> <output_json_path> <duration_seconds>

This is the ONLY transcription script workers should use. Never generate ad-hoc
scripts in /tmp/ — they will use the wrong model (small/medium/base/tiny).
"""
import json, sys, time
from faster_whisper import WhisperModel

audio_path = sys.argv[1]
output_path = sys.argv[2]
duration_hint = int(sys.argv[3]) if len(sys.argv) > 3 else 0

print(f"[{time.strftime('%H:%M:%S')}] Loading large-v3 (int8, CPU)...", flush=True)
model = WhisperModel("large-v3", device="cpu", compute_type="int8", cpu_threads=6)

dur_str = f" (~{round(duration_hint/60)}min estimated)" if duration_hint else ""
print(f"[{time.strftime('%H:%M:%S')}] Transcribing {audio_path}{dur_str}...", flush=True)

start = time.time()
segments, info = model.transcribe(
    audio_path,
    language=None,
    vad_filter=True,
    word_timestamps=False,
    beam_size=5,
)

transcript_segments = []
last_report = 0
for seg in segments:
    transcript_segments.append({
        "start": round(seg.start, 2),
        "end": round(seg.end, 2),
        "text": seg.text.strip()
    })
    # Progress every 5 minutes of audio
    if seg.end - last_report > 300:
        elapsed = time.time() - start
        print(f"[{time.strftime('%H:%M:%S')}] Progress: {round(seg.end/60)}min / ~{round(info.duration/60)}min "
              f"({round(elapsed/60)}min elapsed, {round(elapsed/(seg.end or 1), 1)}x realtime)", flush=True)
        last_report = seg.end

with open(output_path, "w") as f:
    json.dump({
        "language": info.language,
        "duration": info.duration,
        "segments": transcript_segments
    }, f, indent=2, ensure_ascii=False)

elapsed = time.time() - start
print(f"[{time.strftime('%H:%M:%S')}] DONE: {len(transcript_segments)} segments, lang={info.language}, "
      f"took {round(elapsed/60)}min ({round(elapsed/max(info.duration,1), 1)}x realtime)", flush=True)
