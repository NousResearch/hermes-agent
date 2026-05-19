---
name: youtube-content
description: "YouTube transcripts to summaries, threads, blogs."
platforms: [linux, macos, windows]
---

# YouTube Content Tool

## When to use

Use when the user shares a YouTube URL or video link, asks to summarize a video, requests a transcript, or wants to extract and reformat content from any YouTube video. Transforms transcripts into structured content (chapters, summaries, threads, blog posts).

Extract transcripts from YouTube videos and convert them into useful formats.

## Setup

```bash
pip install youtube-transcript-api
```

## Helper Script

`SKILL_DIR` is the directory containing this SKILL.md file. Before running terminal commands, extract only the 11-character YouTube video ID from the user's link (characters `A-Z`, `a-z`, `0-9`, `_`, `-`). Do not paste a raw user-provided URL into a shell command; shell command substitution such as `$()` and backticks can execute even inside double quotes.

```bash
# JSON output with metadata
python3 SKILL_DIR/scripts/fetch_transcript.py 'VIDEO_ID'

# Plain text (good for piping into further processing)
python3 SKILL_DIR/scripts/fetch_transcript.py 'VIDEO_ID' --text-only

# With timestamps
python3 SKILL_DIR/scripts/fetch_transcript.py 'VIDEO_ID' --timestamps

# Specific language with fallback chain
python3 SKILL_DIR/scripts/fetch_transcript.py 'VIDEO_ID' --language tr,en
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

1. **Extract** the 11-character video ID from the user's link. If you cannot identify exactly one valid ID, ask the user to provide a standard YouTube URL or the video ID.
2. **Fetch** the transcript by passing only that video ID to the helper script with `--text-only --timestamps`.
3. **Validate**: confirm the output is non-empty and in the expected language. If empty, retry without `--language` to get any available transcript. If still empty, tell the user the video likely has transcripts disabled.
4. **Chunk if needed**: if the transcript exceeds ~50K characters, split into overlapping chunks (~40K with 2K overlap) and summarize each chunk before merging.
5. **Transform** into the requested output format. If the user did not specify a format, default to a summary.
6. **Verify**: re-read the transformed output to check for coherence, correct timestamps, and completeness before presenting.

## Error Handling

- **Transcript disabled**: tell the user; suggest they check if subtitles are available on the video page.
- **Private/unavailable video**: relay the error and ask the user to verify the URL.
- **No matching language**: retry without `--language` to fetch any available transcript, then note the actual language to the user.
- **Dependency missing**: run `pip install youtube-transcript-api` and retry.
