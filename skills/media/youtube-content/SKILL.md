---
name: youtube-content
description: "Use when the user shares a YouTube URL or asks to summarize a video — transcript extraction, video search, comment extraction, and audio transcription for unsubtitled videos. Output formats: summaries, chapters, threads, blog posts."
version: 1.1.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [youtube, video, transcript, subtitles, search]
    related_skills: [bilibili]
---

# YouTube Content Tool

## When to use

Use when the user shares a YouTube URL or video link, asks to summarize a video, requests a transcript, or wants to extract and reformat content from any YouTube video. Transforms transcripts into structured content (chapters, summaries, threads, blog posts).

Extract transcripts from YouTube videos and convert them into useful formats.

## Setup

Use `uv` so the dependency is installed into the same Hermes-managed environment
that runs the helper script:

```bash
uv pip install youtube-transcript-api
```

## Helper Script

`SKILL_DIR` is the directory containing this SKILL.md file. The script accepts any standard YouTube URL format, short links (youtu.be), shorts, embeds, live links, or a raw 11-character video ID.

```bash
# JSON output with metadata
uv run python3 SKILL_DIR/scripts/fetch_transcript.py "https://youtube.com/watch?v=VIDEO_ID"

# Plain text (good for piping into further processing)
uv run python3 SKILL_DIR/scripts/fetch_transcript.py "URL" --text-only

# With timestamps
uv run python3 SKILL_DIR/scripts/fetch_transcript.py "URL" --timestamps

# Specific language with fallback chain
uv run python3 SKILL_DIR/scripts/fetch_transcript.py "URL" --language tr,en
```

## Output Formats

After fetching the transcript, format it based on what the user asks for:

- **Chapters**: Group by topic shifts, output timestamped chapter list
- **Summary**: Concise 5-10 sentence overview of the entire video
- **Chapter summaries**: Chapters with a short paragraph summary for each
- **Thread**: Twitter/X thread format — numbered posts, each under 280 chars
- **Blog post**: Full article with title, sections, and key takeaways
- **Quotes**: Notable quotes with timestamps

### Example — Chapters Output

```
00:00 Introduction — host opens with the problem statement
03:45 Background — prior work and why existing solutions fall short
12:20 Core method — walkthrough of the proposed approach
24:10 Results — benchmark comparisons and key takeaways
31:55 Q&A — audience questions on scalability and next steps
```

## Workflow

1. **Fetch** the transcript using the helper script with `--text-only --timestamps` via `uv run python3`.
2. **Validate**: confirm the output is non-empty and in the expected language. If empty, retry without `--language` to get any available transcript. If still empty, tell the user the video likely has transcripts disabled.
3. **Chunk if needed**: if the transcript exceeds ~50K characters, split into overlapping chunks (~40K with 2K overlap) and summarize each chunk before merging.
4. **Transform** into the requested output format. If the user did not specify a format, default to a summary.
5. **Verify**: re-read the transformed output to check for coherence, correct timestamps, and completeness before presenting.

## Error Handling

- **Transcript disabled**: tell the user; suggest they check if subtitles are
  available on the video page. Also try the yt-dlp fallback (see Extended
  Capabilities below).
- **Private/unavailable video**: relay the error and ask the user to verify the URL.
- **No matching language**: retry without `--language` to fetch any available
  transcript, then note the actual language to the user.
- **Dependency missing**: run `uv pip install youtube-transcript-api` and retry.

## Extended Capabilities (via yt-dlp)

The `youtube-transcript-api` helper above handles simple transcript fetching.
For video search, comment extraction, metadata, and transcription of
unsubtitled videos, use yt-dlp as a more powerful backend.

### Setup

```bash
pip install yt-dlp
# yt-dlp needs a JS runtime for some features.
# deno is preferred (works out of box), or node.js with config.
```

### Video Search

```bash
# Search YouTube and get video metadata
yt-dlp --dump-json "ytsearch5:query"

# Just titles and URLs
yt-dlp --print "%(title)s | %(webpage_url)s" "ytsearch5:query"
```

### Video Metadata

```bash
# Full metadata dump (title, duration, uploader, views, likes, etc.)
yt-dlp --dump-json "URL"
```

### Comment Extraction

```bash
# Extract comments (best-effort, uses webpage scraping)
yt-dlp --write-comments --skip-download --write-info-json \
  --extractor-args "youtube:max_comments=20" \
  -o "/tmp/%(id)s" "URL"
# Comments are in the .info.json file under the "comments" field
```

> Note: `--write-comments` uses webpage scraping, not the official Data API.
> Some comments may be missing.

### yt-dlp Transcript Extraction (fallback)

When `youtube-transcript-api` fails, use yt-dlp:

```bash
# Auto-generated subtitles
yt-dlp --write-auto-sub --sub-lang "en" --skip-download -o "/tmp/%(id)s" "URL"
# Manually uploaded
yt-dlp --write-sub --sub-lang "en" --skip-download -o "/tmp/%(id)s" "URL"
# Read the .vtt file
cat /tmp/VIDEO_ID.*.vtt
```

### Audio Transcription (for unsubtitled videos)

When a video has no subtitles at all:

```bash
# Download audio
yt-dlp -f "bestaudio" --extract-audio --audio-format mp3 \
  -o "/tmp/%(id)s.%(ext)s" "URL"

# Transcribe with Groq Whisper (free key from console.groq.com)
curl -X POST "https://api.groq.com/openai/v1/audio/transcriptions" \
  -H "Authorization: Bearer *** \  -F "file=@/tmp/VIDEO_ID.mp3" \
  -F "model=whisper-large-v3-turbo" \
  -F "response_format=verbose_json"

# For local transcription (privacy-preserving), use whisper.cpp:
ffmpeg -i /tmp/VIDEO_ID.mp3 -ar 16000 -ac 1 /tmp/VIDEO_ID.wav
whisper-cli -m /path/to/ggml-model.bin -f /tmp/VIDEO_ID.wav
```

## Backend Selection

| Task | Primary | Fallback |
|------|---------|----------|
| Transcript | youtube-transcript-api (Python) | yt-dlp --write-auto-sub |
| Search | yt-dlp ytsearch | web_search site:youtube.com |
| Comments | yt-dlp --write-comments | Not available otherwise |
| Metadata | yt-dlp --dump-json | web_extract on watch page |
| Unsubtitled audio | yt-dlp audio + Whisper | — |

## Common Pitfalls

1. **Transcripts disabled.** Some videos have no subtitles at all. Use the
   audio transcription fallback.
2. **youtube-transcript-api ratelimit.** Frequent requests may trigger IP
   blocks. Space requests or use yt-dlp as alternative.
3. **yt-dlp JS runtime.** yt-dlp needs deno or node.js for some features.
   Install deno for simpler setup.

## Verification Checklist

- [ ] `uv pip install youtube-transcript-api` succeeds
- [ ] Helper script fetches a transcript from a known public video
- [ ] yt-dlp is installed and `yt-dlp --version` works
