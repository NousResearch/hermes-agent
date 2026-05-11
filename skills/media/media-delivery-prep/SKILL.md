---
name: media-delivery-prep
description: Use when a user wants to send, convert, compress, inspect, trim, or prepare media attachments for Hermes messaging platforms.
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [media, attachments, ffmpeg, messaging, conversion, compression]
    related_skills: [songsee, youtube-content]
prerequisites:
  commands: [ffmpeg, ffprobe]
---

# Media Delivery Prep

## Overview

Hermes messaging adapters can deliver files through `MEDIA:/absolute/path`
tags. This skill prepares local media so those attachments work reliably across
Telegram, Discord, Matrix, Signal, Feishu, Weixin, Yuanbao, email, and other
gateway platforms.

Use FFmpeg and ffprobe for format inspection, conversion, compression, trimming,
thumbnailing, audio extraction, and subtitle handling. Then send the final file
with `send_message` or by returning a `MEDIA:` tag.

## When to Use

- The user asks to send a local image, audio file, video, archive, PDF, or other
  generated artifact through a Hermes messaging platform.
- A platform rejects a media attachment, sends it as the wrong type, or reports
  that the file is too large.
- The user asks to convert media before sharing it, such as MOV to MP4, WAV to
  MP3, WEBM to MP4, or a video clip to GIF.
- The user wants to extract audio from a video for transcription, podcast notes,
  meeting summaries, or follow-up processing.
- The user wants a short clip, thumbnail, compressed copy, burned subtitles, or
  a platform-friendly version of a generated media file.

Do not use this skill for deep visual/audio analysis by itself. Use
`vision_analyze`, `video_analyze`, `songsee`, or `youtube-content` when the
task is primarily understanding content rather than preparing files for
delivery.

## Core Rule: Deliver Host-Visible Paths

`MEDIA:` paths are read by the gateway process. Always emit a path that exists
on the host running Hermes, not only inside a Docker container, SSH machine, or
temporary remote sandbox.

Good:

```text
MEDIA:/home/user/.hermes/cache/media/final.mp4
```

Bad:

```text
MEDIA:/workspace/final.mp4
```

If using a Docker-backed terminal, write the output to a mounted directory and
emit the host-side path.

## Inspect First

Before converting or compressing, inspect the file:

```bash
ffprobe -hide_banner -i input.mp4

ffprobe -v error -show_entries format=duration,size,format_name \
  -show_streams -of json input.mp4
```

Check:

- container format
- video codec, resolution, frame rate, duration, bitrate
- audio codec, channels, sample rate, bitrate
- file size
- subtitle streams

## Common Delivery Formats

| Need | Preferred output |
|---|---|
| Broad video compatibility | MP4 container, H.264 video, AAC audio |
| Small video preview | MP4, H.264, lower resolution/CRF |
| Audio attachment | MP3 or M4A |
| Voice-message audio | OGG container, Opus audio |
| Animated preview | GIF for short loops, MP4 for smaller size |
| Image preview | JPG for photos, PNG for screenshots/diagrams |
| Documents | Leave as original document type unless platform rejects it |

## Recipes

### Convert Video to Messaging-Friendly MP4

```bash
ffmpeg -y -i input.mov \
  -c:v libx264 -preset medium -crf 23 -pix_fmt yuv420p \
  -c:a aac -b:a 128k \
  output.mp4
```

Use this when the input is MOV, MKV, AVI, WEBM, or a generated file that may not
play inline on mobile clients.

### Compress a Video

```bash
ffmpeg -y -i input.mp4 \
  -vf "scale='min(1280,iw)':-2" \
  -c:v libx264 -preset medium -crf 28 -pix_fmt yuv420p \
  -c:a aac -b:a 96k \
  output-small.mp4
```

Increase CRF for smaller files and lower quality. Typical useful range:

| CRF | Use |
|---|---|
| 23 | Good quality |
| 28 | Smaller sharing copy |
| 32 | Very small preview |

### Trim Without Re-Encoding

```bash
ffmpeg -y -ss 00:01:20 -to 00:02:05 -i input.mp4 -c copy clip.mp4
```

This is fast but cuts only near keyframes. If the cut must be exact, re-encode:

```bash
ffmpeg -y -ss 00:01:20 -to 00:02:05 -i input.mp4 \
  -c:v libx264 -crf 23 -preset fast -c:a aac clip.mp4
```

### Extract Audio for Transcription

```bash
ffmpeg -y -i input.mp4 -vn -ac 1 -ar 16000 -c:a pcm_s16le audio.wav
```

Use the resulting WAV with speech-to-text workflows. For compact sharing:

```bash
ffmpeg -y -i input.mp4 -vn -c:a libmp3lame -b:a 128k audio.mp3
```

### Normalize Audio Loudness

```bash
ffmpeg -y -i input.mp3 \
  -af loudnorm=I=-16:TP=-1.5:LRA=11 \
  -c:a libmp3lame -b:a 128k \
  normalized.mp3
```

Use for voice notes, podcasts, meeting clips, and generated TTS that is too loud
or quiet.

### Convert WAV/MP3 to OGG Opus for Voice Messages

```bash
ffmpeg -y -i input.wav -c:a libopus -b:a 32k -vbr on output.ogg
```

Use this for spoken voice notes and Telegram-style voice bubbles. For music,
prefer MP3/M4A or use a higher Opus bitrate.

### Create a Thumbnail

```bash
ffmpeg -y -ss 00:00:03 -i input.mp4 -frames:v 1 thumbnail.jpg
```

For a contact sheet:

```bash
ffmpeg -y -i input.mp4 \
  -vf "fps=1/10,scale=320:-1,tile=3x3" \
  contact-sheet.jpg
```

### Make a Short GIF

```bash
ffmpeg -y -ss 00:00:05 -t 3 -i input.mp4 \
  -vf "fps=12,scale=480:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" \
  preview.gif
```

Prefer MP4 for anything longer than a few seconds because GIF files grow
quickly.

### Burn Subtitles Into Video

```bash
ffmpeg -y -i input.mp4 -vf "subtitles=captions.srt" \
  -c:v libx264 -crf 23 -preset medium -c:a copy \
  subtitled.mp4
```

If the subtitle path has spaces, quote it carefully.

### Extract Embedded Subtitles

```bash
ffmpeg -y -i input.mkv -map 0:s:0 captions.srt
```

## Sending With Hermes

After preparing the file, send it through the messaging tool:

```text
send_message(action="send", platform="telegram", target="...", message="Here is the clip.\nMEDIA:/absolute/path/output.mp4")
```

If the current conversation is already on a gateway platform, returning the
`MEDIA:` tag in the final response is often enough:

```text
Here is the compressed version:
MEDIA:/absolute/path/output-small.mp4
```

Use `[[as_document]]` when Telegram should send an image/video/audio file as a
downloadable document instead of trying native media rendering:

```text
[[as_document]]
MEDIA:/absolute/path/archive.zip
```

## Platform Notes

| Platform | Notes |
|---|---|
| Telegram | `MEDIA:` attachments work well. Use MP4/H.264/AAC for video. OGG/Opus is best for voice bubbles. |
| Discord | MP4 and common image/audio/document files upload as native attachments. Compress large files before sending. |
| Matrix | Supports images, audio, video, and files through the homeserver media repository. |
| Signal | Sends media as attachments; large batches may be throttled. |
| Feishu/Lark | Sends images, audio, video, and files through adapter upload APIs; GIFs may be treated as files. |
| Weixin/Yuanbao | Prefer common formats and small files; use MP4/JPG/PNG/MP3 when unsure. |
| Email | Attachments work, but SMTP servers often enforce message-size limits. Compress or link large files. |

## Safety and Robustness

- Never overwrite the only copy of a user file. Write to a new output path.
- Use `-y` only when the output is a new generated path or intentionally
  replaceable.
- Prefer absolute paths in final `MEDIA:` tags.
- Verify output exists and is non-empty before sending:

```bash
ls -lh output.mp4
ffprobe -v error -show_entries format=duration,size -of default=nw=1 output.mp4
```

- If FFmpeg fails with a codec error, retry with broad compatibility settings:
  H.264 video, AAC audio, `-pix_fmt yuv420p`.
- If a platform still rejects the file, send a smaller version or force document
  delivery where supported.

## Verification Checklist

- [ ] Input file exists and was inspected with `ffprobe`.
- [ ] Output file uses a platform-friendly extension and codec.
- [ ] Output file exists, is non-empty, and plays or probes successfully.
- [ ] Final `MEDIA:` path is absolute and visible to the gateway host.
- [ ] User-facing message explains any quality/size tradeoff made.
