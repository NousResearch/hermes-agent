---
name: streaming-content
description: "Transcribe clips and VODs from Twitch, Kick, Rumble, and X."
version: 0.1.0
author: "PunchTaylor (@punchtaylor)"
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    contributor: "PunchTaylor (@punchtaylor)"
    tags: ["media", "transcript", "streaming", "twitch", "kick", "rumble", "x"]
---

# Streaming Content

Transcribe clips and VODs from live-streaming platforms and convert them into useful formats. Unlike YouTube, these platforms don't serve caption tracks, so this skill downloads the audio and transcribes it.

## When to use

Use when the user shares a Twitch / Kick / Rumble / X (Twitter) clip, VOD, video post, or broadcast link, asks to summarize a stream, or wants a transcript from a live-streaming platform. For YouTube, use the youtube-content skill instead — it reads served captions and is cheaper.

## Setup

Use `uv` so the dependency is installed into the same Hermes-managed environment that runs the helper script:

```bash
uv pip install yt-dlp
```

ffmpeg must also be on PATH (`brew install ffmpeg` / `apt install ffmpeg`).

A transcription backend is required (faster-whisper is default and runs locally; otherwise GROQ_API_KEY, VOICE_TOOLS_OPENAI_KEY, etc.). With no backend, audio downloads but transcription fails.

## Helper Script

`SKILL_DIR` is the directory containing this SKILL.md file.

```bash
# JSON with metadata + transcript
uv run python3 SKILL_DIR/scripts/fetch_transcript.py "https://www.twitch.tv/<channel>/clip/<slug>"

# Plain text (good for piping into further processing)
uv run python3 SKILL_DIR/scripts/fetch_transcript.py "URL" --text-only
```

Accepts any single clip or VOD URL yt-dlp supports.

## Supported platforms

- **Twitch** — clips and VODs.
- **Kick** — clips and VODs.
- **Rumble** — standard video URLs.
- **X (Twitter)** — native video posts and broadcasts (no login). Image/text/link-only posts return a clean error.
- Any other yt-dlp-supported site.

## Output Formats

After fetching the transcript, format it based on what the user asks for:

- **Summary**: Concise overview.
- **Thread**: Numbered posts under 280 chars.
- **Blog post**: Full article.
- **Quotes / Chapters**: Topic-based breakdown.

## Workflow

1. Fetch the transcript with the helper script.
2. Validate the output. A `no playable video in this post` error means an image/text/link-only X post (expected). `audio download failed` usually means private/sub-only/expired.
3. Transform into the requested format.

## Error Handling

- **Audio download failed**: private, sub-only, expired, or wrong URL.
- **No playable video in this post**: valid X post with no video content (image, text, or link-only). Not a failure.
- **Transcription failed**: confirm ffmpeg and a transcription backend are configured.
- **Dependency missing**: `uv pip install yt-dlp` and ffmpeg on PATH.
