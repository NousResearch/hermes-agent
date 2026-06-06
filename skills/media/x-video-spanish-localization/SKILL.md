---
name: x-video-spanish-localization
description: Use when localizing X/Twitter videos into Spanish with dubbing, subtitles, or both. Preserves original format/framing, uses the highest practical X MP4 master, verifies audio timing visually/technically, and sends Telegram-safe deliverables.
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [macos, linux]
metadata:
  hermes:
    tags: [x, twitter, video, dubbing, subtitles, ffmpeg, telegram]
    related_skills: [youtube-content]
---

# X Video Spanish Localization

## Overview

Use this workflow when a user sends an X/Twitter video and asks to translate, dub, subtitle, or make it understandable in Spanish. The job is done only when the output preserves source framing, uses source timing, verifies sync, and survives Telegram delivery.

## When to Use

- "Traduce/dobla este vídeo de X al español"
- "Ponle subtítulos en español"
- "Audio incluido"
- "Mantén el formato original"
- User says Telegram changed preview or the video looks stretched/cropped.

## Workflow

1. Resolve X media and collect MP4 variants. Prefer the highest practical non-HLS MP4 master, especially for talks, UI, code, slides, or legibility.
2. Probe source geometry/timing with `ffprobe`: width, height, DAR/SAR, fps, duration, audio duration, rotation.
3. Build transcript/subtitles with segment start/end times. Do not replace segment timing with one long narration track unless requested.
4. Choose output mode: segment-aligned Spanish TTS, original audio + Spanish subtitles, or both. If dubbing drifts after one fix attempt, fall back to original audio + subtitles.
5. Preserve geometry: no square/portrait defaults. For 16:9 X videos use 1280x720, 960x540, or 854x480 unless source says otherwise.
6. Verify before delivery: ffprobe dimensions/durations, visual source-vs-final frame comparison, sync checks at start/middle/end.
7. Deliver from `~/.hermes/document_cache/`; use `[[as_document]]` when Telegram inline rendering may alter playback.

## ffmpeg Patterns

Soft Spanish subtitles with original audio:

```bash
ffmpeg -y -i source.mp4 -i spanish.srt \
  -map 0:v -map 0:a? -map 1:0 \
  -c:v libx264 -preset slow -crf 28 \
  -c:a aac -b:a 96k -c:s mov_text \
  -metadata:s:s:0 language=spa \
  output_original_audio_subtitles_es.mp4
```

Segment-aligned dubbed audio:

```bash
ffmpeg -y -i source.mp4 -i spanish_aligned.wav \
  -map 0:v -map 1:a -c:v libx264 -preset slow -crf 28 \
  -c:a aac -b:a 96k -shortest output_dubbed_es.mp4
```

## User Simulation Tests

- User opens in Telegram: file is not distorted.
- User compares to X original: same aspect/framing.
- User listens at 0:00, 5:00, 15:00, final minute: no cumulative drift.
- User asks "as file, no zip": send MP4 document.
- User rejects dubbing quality: fall back to original audio + subtitles.

## Common Pitfalls

1. Using the lowest 640x360 variant as master. It may pass `ffprobe` but look wrong.
2. Trusting Telegram preview. Send as document when quality matters.
3. One long TTS file drifting. Segment alignment beats total-duration stretching.
4. Only checking duration. Equal duration does not mean synchronized speech.
5. Forgetting to send the `.srt` separately when subtitles matter.

## Verification Checklist

- [ ] Source media URL/variant recorded.
- [ ] Output DAR matches source DAR.
- [ ] Source and final frame visually compared.
- [ ] Audio/subtitle timestamps checked at start/middle/end.
- [ ] Telegram delivery mode selected deliberately.
- [ ] Filename summarizes the video.
