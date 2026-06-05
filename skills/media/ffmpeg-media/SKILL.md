---
name: ffmpeg-media
description: "Convert, trim, transcode, mux, watermark, or extract frames from audio/video using FFmpeg. Use whenever the user asks to manipulate media files (video, audio, GIFs, HLS, WebRTC dumps, screen recordings)."
version: 1.0.0
platforms: [linux, macos]
metadata:
  hermes:
    tags: [media, video, audio, ffmpeg]
---

# FFmpeg

Installed (static GPL build) at ``ffmpeg`/`ffprobe` on PATH`. Latest master
build with libx264, libx265, libvpx, libopus, libmp3lame, libass.

## Quick recipes

```bash
# 1. Convert MP4 → WebM
ffmpeg -i in.mp4 -c:v libvpx-vp9 -b:v 1M -c:a libopus out.webm

# 2. Trim 30s starting at 00:01:15 (no re-encode)
ffmpeg -ss 00:01:15 -i in.mp4 -t 30 -c copy out.mp4

# 3. Extract 1 frame per second as PNG
ffmpeg -i in.mp4 -vf fps=1 frame_%04d.png

# 4. Mux external audio onto video
ffmpeg -i video.mp4 -i audio.aac -c copy -map 0:v -map 1:a out.mp4

# 5. Probe a file
ffprobe -v error -show_format -show_streams in.mp4

# 6. Hardware-free real-time transcode for HLS
ffmpeg -i in.mp4 -c:v libx264 -preset veryfast -hls_time 4 -hls_playlist_type vod out.m3u8
```

## When to invoke
- Any media manipulation step in a pipeline (used by `vimax-video`,
  `hyperframes-render`, voice-memo transcription, screen recordings).
- Generating evidence GIFs/MP4s for QA reports.

## Tips
- For deterministic output use `-map_metadata -1 -fflags +bitexact`.
- For batch jobs prefer `-loglevel error -nostats` to keep logs short.
- In `scale`, do **not** use `force_original_aspect_ratio=cover` (invalid). Use `force_original_aspect_ratio=increase` + `crop` for vertical/horizontal target framing.
- If source avatar is square-ish (≈1:1), avoid aggressive crop/stretch to 9:16. Prefer **blurred vertical background + centered foreground** to preserve face proportions.
- Avoid `-f concat -c copy` for user-facing final exports when segments may differ in timestamps/audio streams; prefer filter concat with re-encode (`[0:v][0:a][1:v][1:a]concat=...`) for stable endings.
- If CTA clip has no audio stream, synthesize silent audio (`anullsrc`) before concat to prevent tail truncation or concat failure.
- Before concat (`-f concat -c copy`), normalize all segments to same resolution/fps/codecs to avoid non-obvious mux failures.
- For vertical reels with fixed overlay text, force safe-area wrapping explicitly (manual line breaks in `textfile=` and conservative `fontsize`), then verify on exported frame(s). If any edge clips on 1080x1920, reduce font size and reflow lines before final render.
- When user requests continuous soundtrack, DO NOT mix per-clip production audio. Build final audio from a single long/looped music bed + VO only (`amix`), trim/fade music to final duration, and omit clip-native audio from the map.
- Do **not** trust container/stream duration alone (`ffprobe`) as proof of audio continuity. Add a tail QA pass (e.g., `-ss <late_time> -t <window> -af silencedetect`) to catch silent tails or abrupt cutoffs that still report full duration.
- For complex ducking/mix graphs, prefer a 2-step master flow when debugging: (1) render audio master WAV first, (2) mux that master into final MP4. This avoids filtergraph label regressions and makes tail-audio verification deterministic.
- Delivery QA for chat workflows: after render, always send the final file via `MEDIA:/absolute/path` in the same handoff message so the user can review immediately.
- Watermark removal workflow + QA gate: `references/watermark-removal-quality-gate.md`.
- **Deterministic tail pattern:** if there is any tail risk, pre-render a target-length mixed WAV (e.g. 35s) and then mux that WAV into final MP4; alternatively use `amix=duration=first` with a guaranteed target-length bed. Avoid relying on fragile `amix`+`apad`+`atrim` chains without tail QA.
- When user requests continuous soundtrack, DO NOT mix per-clip production audio. Build final audio from a single long/looped music bed + VO only (`amix`), trim/fade music to final duration, and omit clip-native audio from the map.
- Do **not** trust container/stream duration alone (`ffprobe`) as proof of audio continuity. Add a tail QA pass (e.g., `-ss <late_time> -t <window> -af silencedetect`) to catch silent tails or abrupt cutoffs that still report full duration.
- For complex ducking/mix graphs, prefer a 2-step master flow when debugging: (1) render audio master WAV first, (2) mux that master into final MP4. This avoids filtergraph label regressions and makes tail-audio verification deterministic.
- Delivery QA for chat workflows: after render, always send the final file via `MEDIA:/absolute/path` in the same handoff message so the user can review immediately.
- Watermark removal workflow + QA gate: `references/watermark-removal-quality-gate.md`.
- Audio tail continuity troubleshooting: `references/audio-tail-continuity.md`.
- UGC vertical compose troubleshooting note: `references/ugc-vertical-compose-troubleshooting.md`.
- Reel remix audio continuity (reference music + ElevenLabs VO + ducking): `references/reel-remix-audio-continuity.md`.

## Lip-sync QA/Tuning (field pattern)
Use this when generated talking-head clips look unnatural (head motion, mouth artifacts).

1. **Shorten test window first**
   - Work in 6–10s segments before running full-length takes.
   - Long clips hide config issues and increase cost/latency.

2. **Normalize and trim audio**
   - Use loudness normalization and short duration for cleaner sync tests:
   - `ffmpeg -y -i in.ogg -t 8.2 -af "loudnorm=I=-16:TP=-1.5:LRA=11" out_8s.ogg`

3. **Prepare stable source variants**
   - Generate at least 2-3 source variants (neutral, smooth, tight crop) and compare.
   - Keep output vertical 9:16 and consistent FPS (25).

4. **Create side-by-side or 2x2 comparison before deciding**
   - Label each panel with model/config.
   - Prefer one artifact with shared audio for human review.

5. **Decision rule**
   - Pick winner on *facial naturality first* (micro-motion + mouth coherence), then sharpness.
   - If quality is still below paid-model baseline, use open-source outputs only for low-cost ideation/AB tests, not final hero creatives.
