---
name: youtube-playback
description: Play YouTube videos and music with ultra-low latency.
version: 1.0.0
author: Ibrahim Abdelsattar, Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [YouTube, Video, Music, Audio, Media, Playback]
    related_skills: [youtube-content]
---

# YouTube Playback Skill

Search, open, and play YouTube videos or music tracks autonomously with ultra-low latency. It launches videos in the user's default browser or extracts direct stream metadata, but does not download files or transcode video.

## When to Use

Use when the user asks to play a video, song, soundtrack, or audio from YouTube (e.g., "play Iron Man music from YouTube", "open YouTube and play X", "شغل موسيقى أيرون مان من يوتيوب", "شغل فيديو كذا من يوتيوب").

## Prerequisites

Python 3.10+ standard library. Uses the bundled helper script without requiring external API keys or browser automation overhead.

## How to Run

Execute the helper script using the `terminal` tool:

```bash
# Play top matching YouTube video immediately in default browser
python skills/media/youtube-playback/scripts/play_youtube.py "Iron Man music"

# Direct YouTube URL or video ID playback
python skills/media/youtube-playback/scripts/play_youtube.py "https://www.youtube.com/watch?v=IyR25B-IGyg"

# Search and return JSON metadata without opening browser
python skills/media/youtube-playback/scripts/play_youtube.py "Iron Man music" --no-open --json

# Retrieve top 5 matching videos
python skills/media/youtube-playback/scripts/play_youtube.py "Iron Man music" --no-open --json --max-results 5
```

## Quick Reference

| Command | Purpose |
|---------|---------|
| `play_youtube.py "<query>"` | Search and launch playback with autoplay in default browser |
| `play_youtube.py "<query>" --no-open --json` | Resolve title, duration, and URL without launching browser |
| `play_youtube.py "<url>"` | Immediately launch URL in default browser with autoplay |

## Procedure

1. **Extract query**: Identify the search term from the user's message (strip polite conversational openers like "please", "can you", "شغل من يوتيوب").
2. **Execute playback**: Run `python skills/media/youtube-playback/scripts/play_youtube.py "<query>"` via the `terminal` tool.
3. **Confirm concisely**: Reply to the user in 1-2 spoken sentences confirming that playback was launched, noting the song or video title and URL.

## Pitfalls

- **Avoid GUI computer-use**: Do not use screen capture or slow visual browser clicking tools when playing YouTube media; direct protocol launch is 30x faster.
- **Handling ambient noise**: If the user query is very short or ambiguous, search for the most relevant official soundtrack or music video.

## Verification

Run the helper script with `--no-open --json`:

```bash
python skills/media/youtube-playback/scripts/play_youtube.py "Iron Man music" --no-open --json
```

Verify that `"success": true` and `"top"` contains a valid `url`, `title`, and `duration`.
