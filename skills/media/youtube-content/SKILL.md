---
name: youtube-content
description: "YouTube transcripts to summaries, threads, blogs."
platforms: [linux, macos, windows]
---

# YouTube Content Tool

## When to use

Use when the user shares a YouTube video URL, asks for a transcript, or wants to turn one video's transcript into a summary, chapters, claims, a short thread, or a blog outline.

Keep the bundled skill focused on single-video, transcript-first work. Do not create a separate YouTube summarizer skill for this flow.

## Scope and boundaries

- Accept one YouTube video URL or one raw 11-character video ID.
- Reject playlists, channels, search URLs, and bulk URL lists unless the user separately approves a supported bulk workflow.
- Prefer available captions/transcripts before any heavier fallback.
- Preserve supported upstream behavior around cookies, proxies, OAuth, API keys, access handling, and helper-script capabilities; do not introduce new bypass or circumvention behavior.
- Do not download media, run ASR, install packages, change lockfiles, or send content to external or alternate providers unless the user approves that step and the behavior is supported by the current configuration/docs.

## Dependency handling

The helper requires `youtube-transcript-api` in the environment that runs the skill. If the dependency is missing, report the missing dependency and stop unless package installation is in scope for the current task or separately approved by the user.

## Helper script

`SKILL_DIR` is the directory containing this `SKILL.md`. The script accepts standard single-video YouTube URL formats, short links (`youtu.be`), shorts, embeds, live links, or a raw 11-character video ID.

```bash
# JSON output with metadata
uv run python3 SKILL_DIR/scripts/fetch_transcript.py "https://youtube.com/watch?v=VIDEO_ID"

# Plain text
uv run python3 SKILL_DIR/scripts/fetch_transcript.py "URL" --text-only

# With timestamps
uv run python3 SKILL_DIR/scripts/fetch_transcript.py "URL" --timestamps

# Specific language with fallback chain
uv run python3 SKILL_DIR/scripts/fetch_transcript.py "URL" --language tr,en
```

## Workflow

1. **Scope**: confirm the request is for one video. If it is a playlist, channel, search result, or bulk list, ask for approval before using any supported bulk workflow.
2. **Fetch**: use the helper script with `--text-only --timestamps` via `uv run python3` when the dependency is already available.
3. **Validate**: confirm the transcript is non-empty and in the expected language. If a requested language is unavailable, retry without `--language` to fetch any available transcript and note the actual language.
4. **Transform**: produce the requested format. If unspecified, default to a concise structured summary.
5. **Caveat**: identify transcript source when known and state that claims are from the video unless externally verified.
6. **Verify**: check that timestamps, caveats, and requested output format are coherent before replying.

## User-facing examples

- “Summarize this YouTube video in 8 bullets with timestamps.”
- “Create timestamped chapters for this talk.”
- “Extract the key claims and caveats from this video.”
- “Turn this transcript into a short thread or blog outline.”

## Output formats

- **Summary**: concise transcript-grounded overview.
- **Chapters**: topic shifts with timestamps.
- **Chapter summaries**: timestamped sections with short explanations.
- **Key claims**: claims made in the video, with timestamps and caveats.
- **Thread/blog outline**: short posts or article sections derived from the transcript.
- **Quotes**: notable lines with timestamps when available.

## Structured summary defaults

When summarizing a video from a URL, include:

- metadata when available: title, channel/uploader, date, duration, URL, video ID;
- transcript source: manual captions, auto captions, user-provided transcript, ASR, unavailable, or unknown;
- concise summary;
- timestamped sections or chapters when timestamps are available;
- key claims with supporting timestamps where possible;
- takeaways/action items when useful;
- quality caveats: manual captions are usually better than auto captions; auto captions and ASR can misrecognize names, numbers, speaker turns, or timestamps.

Consult `references/video-summarizer-architecture.md` for transcript-first architecture, approval gates, artifact/metadata guidance, provider-routing language, and YouTube ToS/API caveats.

## Provider and privacy handling

- Use the configured default model/provider for summarization.
- Follow the configured privacy policy for data movement and retention.
- When content sensitivity is unclear, avoid sending transcripts, audio, or video to alternate or external providers without user approval.
- Use external or alternate providers only when enabled by the user/configuration and approved for the specific content.
- If writing artifacts, keep paths scoped to the user's requested workspace or configured artifact location; do not imply unsupported workflow state.

## Approval gates

Do not silently escalate from transcript retrieval to heavier or unsupported behavior. Ask for separate approval before:

- package installs or dependency/lockfile changes;
- media/audio/video download or extraction;
- ASR, including local ASR;
- cloud transcription or cloud ASR;
- sending transcript/audio/video to external or alternate providers when not already approved by the user/configuration;
- cookies, proxies, OAuth, API keys, age/geo bypass, or YouTube Data API setup when not already supported by upstream behavior;
- playlist, channel, search, or bulk extraction.

## ToS, API, and quality caveats

- YouTube Data API captions are not a general caption-access solution: listing/downloading captions requires OAuth, and downloads require permission to edit the video.
- YouTube Terms restrict downloading/reproducing content except as authorized and restrict automated access such as scrapers except in limited cases or with permission.
- Prefer manual captions over auto captions; use ASR only after approval.
- Claims are from the video unless externally verified.
- This is operational guidance, not legal advice.

## Error handling

- **Transcript disabled**: tell the user and suggest checking whether subtitles are available on the video page.
- **Private/unavailable video**: relay the error and ask the user to verify access to the URL.
- **No matching language**: retry without `--language`, then note the language actually retrieved.
- **Dependency missing**: report the missing dependency and stop unless package installation is in scope for the current task or separately approved by the user.
