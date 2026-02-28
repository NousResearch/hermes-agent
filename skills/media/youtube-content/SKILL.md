---
name: youtube-content
description: Fetch YouTube video transcripts and transform them into structured content (chapters, summaries, threads, blog posts).
---

# YouTube Content Tool

Extract transcripts from YouTube videos and convert them into useful formats.

## Setup

```bash
pip install youtube-transcript-api
```

## Helper script

This skill includes `fetch_transcript.py` — use it to fetch transcripts quickly:

```bash
# JSON output with metadata
python3 SKILL_DIR/scripts/fetch_transcript.py "https://youtube.com/watch?v=VIDEO_ID"

# With timestamps
python3 SKILL_DIR/scripts/fetch_transcript.py "https://youtube.com/watch?v=VIDEO_ID" --timestamps

# Plain text output (good for piping into further processing)
python3 SKILL_DIR/scripts/fetch_transcript.py "https://youtube.com/watch?v=VIDEO_ID" --text-only

# Specific language with fallback
python3 SKILL_DIR/scripts/fetch_transcript.py "https://youtube.com/watch?v=VIDEO_ID" --language tr,en

# Timestamped plain text
python3 SKILL_DIR/scripts/fetch_transcript.py "https://youtube.com/watch?v=VIDEO_ID" --text-only --timestamps
```

`SKILL_DIR` is the directory containing this SKILL.md file.

## URL formats supported

The script accepts any of these formats (or a raw 11-character video ID):

- `https://www.youtube.com/watch?v=VIDEO_ID`
- `https://youtu.be/VIDEO_ID`
- `https://youtube.com/shorts/VIDEO_ID`
- `https://youtube.com/embed/VIDEO_ID`
- `https://youtube.com/live/VIDEO_ID`

## Output formats

After fetching the transcript, format it based on what the user asks for:

- **Chapters**: Group by topic shifts, output timestamped chapter list (`00:00 Introduction`, `03:45 Main Topic`, etc.)
- **Summary**: Concise 5-10 sentence overview of the entire video
- **Chapter summaries**: Chapters with a short paragraph summary for each
- **Thread**: Twitter/X thread format — numbered posts, each under 280 chars
- **Blog post**: Full article with title, sections, and key takeaways
- **Quotes**: Notable quotes with timestamps

## Workflow

1. Fetch the transcript using the helper script
2. If the transcript is very long (>50K chars), summarize in chunks
3. Transform into the requested output format using your own reasoning

## Error handling

- **Transcript disabled**: Some videos have transcripts turned off — tell the user
- **Private/unavailable**: The API will raise an error — relay it clearly
- **No matching language**: Try without specifying a language to get whatever's available
- **Dependency missing**: Run `pip install youtube-transcript-api` first
