---
name: youtube-download
description: "Download YouTube videos as MP4 files using yt-dlp. Accepts any YouTube URL format (watch, youtu.be, shorts, embeds)."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [youtube, download, video, yt-dlp]
    related_skills: [youtube-content]
---

# YouTube Download

Download a single YouTube video as an MP4 file using `yt-dlp`.

## When to Use

Use when the user shares a YouTube URL and wants the video file downloaded.

- User says "download this video", "save this video", "grab this video"
- User wants to archive or clip a YouTube video locally

**Don't use for:**
- Summarizing or transcribing video content → use `youtube-content`
- Downloading playlists or batch URLs (this skill is single-video only)

## Prerequisites

```bash
# yt-dlp must be installed
which yt-dlp || pip install yt-dlp

# ffmpeg needed for MP4 muxing (video + audio merge)
which ffmpeg || sudo apt install ffmpeg
```

## Download

```bash
# Default: best quality MP4, saved to ~/Downloads/
yt-dlp -f "bv*+ba/b" --merge-output-format mp4 \
  -o "~/Downloads/%(title)s.%(ext)s" \
  "URL"
```

### Parameters

| Part | What it does |
|------|-------------|
| `-f "bv*+ba/b"` | Best video stream + best audio stream, fallback to single best file |
| `--merge-output-format mp4` | Force MP4 container (muxes with ffmpeg) |
| `-o "~/Downloads/%(title)s.%(ext)s"` | Save as `<video title>.mp4` in ~/Downloads/ |

### Custom output directory

```bash
yt-dlp -f "bv*+ba/b" --merge-output-format mp4 \
  -o "<DIR>/%(title)s.%(ext)s" \
  "URL"
```

## Workflow

1. Parse the URL from the user's message. Accepts `youtube.com/watch?v=`, `youtu.be/`, `youtube.com/shorts/`, and `youtube.com/embed/` formats.
2. Run the download command.
3. Report the saved filename and file size to the user.

## Common Pitfalls

1. **Age-restricted or members-only videos**: needs `--cookies-from-browser` or a cookies file. Ask the user for credentials if needed:
   ```bash
   yt-dlp --cookies-from-browser chrome -f "bv*+ba/b" --merge-output-format mp4 "URL"
   ```
2. **HTTP 429 rate limiting**: yt-dlp auto-retries, but persistent throttling may require waiting. Updating yt-dlp often fixes extractor breakage:
   ```bash
   pip install -U yt-dlp
   ```
3. **No ffmpeg installed**: the merge step fails with "ffmpeg not found". Install it first.
4. **Disk space on constrained devices**: check available space before downloading long videos — a 30-min 1080p video can be 500MB+.
