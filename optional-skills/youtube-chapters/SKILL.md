---
name: youtube-chapters
description: Generate paste-ready, validated YouTube chapter markers from a YouTube URL, video ID, or timestamped transcript. Use when asked to create YouTube chapters, timestamps, description markers, or a chapter-based summary of a YouTube video.
---

# YouTube Chapters

Generate YouTube-compatible chapters from transcript-supported topic transitions. Prefer
deterministic helpers in `utils/` for parsing, normalization, chunking, formatting, and
validation.

## Workflow

1. Extract and normalize a YouTube video ID from the user message with
   `utils.youtube.extract_video_id`.
2. Run `python scripts/fetch_transcript.py "<youtube-url-or-video-id>"` from this skill
   directory when the live transcript dependency is available.
3. Read the normalized timestamped transcript from the script's JSON output.
4. If the script returns a structured error, explain that captions could not be retrieved,
   audio transcription fallback is not implemented yet, and the user may provide a
   timestamped transcript or enable captions.
5. Normalize segments with `utils.transcript.normalize_segments` into:

   ```json
   [{"start": 0.0, "end": 4.2, "text": "..."}]
   ```

6. Group segments chronologically with `utils.transcript.group_segments`. Preserve the
   earliest source timestamp for each chunk.
7. Generate candidate chapters using the prompt in `references/prompts.md`.
8. Parse and validate candidates with `utils.validation.parse_chapter_lines` and
   `utils.validation.validate_chapters`.
9. If validation fails, apply the repair prompt once, then validate again.
10. Return only the paste-ready chapter list unless the user asks for explanation.

## Chapter Rules

- Start with `00:00`.
- Use at least 3 chapters when transcript length and content support them.
- Keep timestamps strictly increasing and within the known duration.
- Use `MM:SS`, or `HH:MM:SS` when needed for long videos.
- Use concise, meaningful, transcript-supported titles.
- Avoid generic titles such as `Part 1`, `Section 2`, or `Topic`.
- Prefer major topic transitions over sentence-level changes.
- Prefer roughly 5-12 chapters for a normal 10-30 minute video.
- Do not invent content or precise timestamps.
- Say reliable chapter generation is not possible when the transcript is too short,
  incomplete, untimestamped, or unusable.

## Provider Decisions

- Use the isolated `youtube-transcript-api` provider for public YouTube captions. It requires
  neither a browser nor YouTube API key.
- Do not use the official YouTube Data API as the sole arbitrary-public-video transcript
  provider; caption download generally requires OAuth permission and quota.
- Treat official caption download as optional for videos the user owns.
- Do not attempt audio download or ASR fallback in this MVP.
- Do not bypass access restrictions or download private videos.

Read `references/provider-architecture.md` before implementing or selecting a provider.

## Output

Return plain text without bullets or fences:

```text
00:00 Introduction
01:42 Project setup
04:18 Hermes Agent architecture
08:35 Transcript processing
12:10 Final result
```

For examples and failure behavior, read the relevant file in `examples/`.
