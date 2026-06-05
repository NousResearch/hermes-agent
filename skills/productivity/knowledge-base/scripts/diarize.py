#!/usr/bin/env python3
"""Canonical diarization script: pyannote.audio >=4.0 speaker-diarization-3.1, with progress.

Usage: python3 diarize.py <audio_path_8khz_wav> <output_json_path>

This is the ONLY diarization script workers should use. Never generate ad-hoc
scripts in /tmp/ — they will get the API wrong or use the wrong model.

Always used — diarization is mandatory for all video content (YouTube, Instagram,
Mega). No exceptions for single-speaker / monologue content.

Input must be 8kHz WAV mono (not MP3 — pyannote rejects MP3 with sample count mismatch).
Output: JSON with {"segments": [{"start": float, "end": float, "speaker": str}, ...]}
"""

import json
import os
import sys
import time

from pyannote.audio import Pipeline

audio_path = sys.argv[1]
output_path = sys.argv[2]

if "HF_TOKEN" not in os.environ:
    print("ERROR: HF_TOKEN not set — load from researcher-videos profile:", file=sys.stderr)
    print("  export HF_TOKEN=$(grep -oP 'HF_TOKEN=\\K[^#\\n]+' /root/.hermes/profiles/researcher-videos/.env | head -1)", file=sys.stderr)
    sys.exit(1)

print(f"[{time.strftime('%H:%M:%S')}] Loading pyannote/speaker-diarization-3.1...", flush=True)
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    token=os.environ["HF_TOKEN"],
)

print(f"[{time.strftime('%H:%M:%S')}] Diarizing {audio_path}...", flush=True)
start = time.time()

# pyannote >=4.0: returns DiarizeOutput, use .speaker_diarization.itertracks()
diarization = pipeline(audio_path)

segments = []
for turn, _, speaker in diarization.speaker_diarization.itertracks(yield_label=True):
    segments.append({
        "start": round(turn.start, 2),
        "end": round(turn.end, 2),
        "speaker": speaker,
    })
    # Progress every 300 segments (~60s of speech at typical turn length)
    if len(segments) % 300 == 0:
        elapsed = time.time() - start
        print(f"[{time.strftime('%H:%M:%S')}] Progress: {len(segments)} segments, "
              f"{round(elapsed/60)}min elapsed", flush=True)

unique_speakers = len(set(s["speaker"] for s in segments))

with open(output_path, "w") as f:
    json.dump({"segments": segments}, f, indent=2)

elapsed = time.time() - start
print(f"[{time.strftime('%H:%M:%S')}] DONE: {len(segments)} segments, "
      f"{unique_speakers} speakers, took {round(elapsed/60)}min "
      f"({round(elapsed/60, 1)}min)", flush=True)
