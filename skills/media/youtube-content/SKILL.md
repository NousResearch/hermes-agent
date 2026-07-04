---
name: youtube-content
description: "YouTube transcripts to summaries, threads, blogs."
platforms: [linux, macos, windows]
---

# YouTube Content Tool

## When to use

Use when the user shares a YouTube URL or video link, asks to summarize a video, requests a transcript, or wants to extract and reformat content from a YouTube video. Transforms transcripts into structured content (chapters, summaries, threads, blog posts).

Extract transcripts from single YouTube videos and convert them into useful formats.

## Scope and boundaries

Default to a single-video workflow:

- Accept exactly one YouTube video URL or one raw 11-character video ID.
- Reject playlists, channels, search URLs, and bulk URL lists unless the user explicitly approves a separate workflow and that workflow is supported by the current upstream tooling/docs.
- Prefer transcript/caption retrieval first.
- Do not download audio or video by default.
- Do not require API keys by default.
- Do not auto-install missing dependencies by default; report the missing dependency and the install command.
- Preserve the existing upstream access model. Do not introduce or tighten cookies, proxies, OAuth, API keys, age/geo bypass, or circumvention behavior beyond what the helper/upstream docs already support.

## Setup

Use `uv` so the dependency is installed into the same Hermes-managed environment
that runs the helper script:

```bash
uv pip install youtube-transcript-api
```

## Helper Script

`SKILL_DIR` is the directory containing this SKILL.md file. The script accepts standard single-video YouTube URL formats, short links (youtu.be), shorts, embeds, live links, or a raw 11-character video ID.

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

1. **Scope**: accept one YouTube video URL or 11-character video ID. Reject playlists, channels, search URLs, and bulk URL lists unless the user explicitly approves a separate supported workflow.
2. **Fetch** the transcript using the helper script with `--text-only --timestamps` via `uv run python3` when the dependency is already available.
3. **Validate**: confirm the output is non-empty and in the expected language. If empty, retry without `--language` to get any available transcript. If still empty, tell the user the video likely has transcripts disabled.
4. **Chunk if needed**: if the transcript exceeds ~50K characters, split into overlapping chunks (~40K with 2K overlap) and summarize each chunk before merging.
5. **Transform** into the requested output format. If the user did not specify a format, default to a structured video summary.
6. **Verify**: re-read the transformed output to check for coherence, correct timestamps, transcript-source caveats, and completeness before presenting.

## Structured Video Summary Defaults

When summarizing a video from a URL, include:

- **Metadata**: title, channel/uploader, date, duration, URL, and video ID when available.
- **Transcript source**: manual captions, auto captions, user-provided transcript, or unavailable/unknown. Prefer manual captions over auto captions; use ASR only after explicit approval.
- **Concise summary**: short, transcript-grounded overview.
- **Timestamped sections**: chapters or topic shifts with timestamps when available.
- **Key claims**: claims made in the video, with supporting timestamps where possible. Do not present them as externally verified unless you separately verified them.
- **Takeaways/action items**: practical points the user can use.
- **Quality caveats**: manual captions are usually better than auto captions, and auto captions/ASR can misrecognize names, numbers, speaker turns, or timestamps.

For privacy/compliance-heavy or implementation-design tasks, consult `references/video-summarizer-architecture.md`. It captures the transcript-first architecture, fallback ASR gates, artifact layout, metadata schema, routing/privacy rules, and YouTube ToS/API caveats.

## Privacy and data routing

- Default to the current private-safe Hermes route for summarization.
- Treat unknown sensitivity as private/sensitive.
- Use public/free/trial routes only when the content is explicitly marked `public_non_sensitive` and the user separately approves that route.
- Do not send transcript, audio, or video to cloud ASR/summarization routes unless the route and data movement are explicitly approved.
- Keep local artifacts under the Hermes home directory by default, not inside a project repo, unless the user asks for project artifacts.

## Approval Gates

Do not silently escalate from transcript retrieval to heavier or riskier operations. Ask for separate approval before adding or using any behavior not already supported/documented upstream, including:

- package installs or dependency/lockfile changes;
- audio/video download or extraction;
- local ASR such as `whisper.cpp` or OpenAI Whisper;
- cloud ASR or cloud summarization of transcript/audio/video;
- cookies, OAuth, API keys, proxies, age/geo bypass, or YouTube Data API setup;
- playlist, channel, search, or bulk extraction.

Use transcript/caption sources first. Label manual vs generated captions when known, and frame “key claims” as claims made in the video unless independently verified.

## ToS, API, and compliance caveats

- YouTube Data API captions are not a general public-caption solution: listing/downloading captions requires OAuth, and downloads require permission to edit the video.
- YouTube Terms restrict downloading/reproducing content except as authorized and restrict automated access such as scrapers except in limited cases or with permission.
- Avoid media download, playlists/channels/bulk extraction, cookies, proxies, OAuth, API keys, age/geo bypass, and other circumvention-related behavior by default.
- This is operational guidance, not legal advice.

## Error Handling

- **Transcript disabled**: tell the user; suggest they check if subtitles are available on the video page.
- **Private/unavailable video**: relay the error and ask the user to verify the URL.
- **No matching language**: retry without `--language` to fetch any available transcript, then note the actual language to the user.
- **Dependency missing**: do not auto-install unless package installation is in scope or separately approved; report the missing dependency and the install command (`uv pip install youtube-transcript-api`).
