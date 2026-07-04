# YouTube/video summarizer architecture notes

Use these notes when a user asks for a structured summary from a YouTube URL or asks to harden/extend the bundled `youtube-content` skill for video summarization.

## Recommended shape

- Extend the existing `youtube-content` skill instead of creating a duplicate YouTube video summarizer skill.
- Keep the default workflow single-video and transcript-first.
- Do not add a core model tool for this class by default. Terminal/file helpers plus skill instructions avoid extra model-tool schema footprint.
- Add a worker/helper only later for long videos, chunked summaries, claim-verification passes, or repeatable artifact generation, and only after separate approval.

## Transcript-first flow

1. Accept exactly one YouTube video URL or 11-character video ID.
2. Reject playlists, channel URLs, search URLs, and bulk URL lists unless the user explicitly approved a separate workflow and the behavior is already supported/documented upstream.
3. Prefer available captions/transcripts:
   - `youtube-transcript-api` when installed: useful first probe for transcript retrieval. When track metadata is available, prefer manual captions over generated captions and capture language/source details.
   - `yt-dlp` or any downloader-style tool is not part of the bundled helper's default path. If a future upstream workflow documents it, constrain it to subtitle/caption paths by default and do not download media by default.
4. Normalize to timestamped segments with source metadata when possible.
5. Summarize only from the transcript. Preserve timestamps and label key claims as claims made in the video unless independently verified.

## Fallback ASR flow

Fallback ASR is not the default because it generally requires audio extraction/download and model files.

- Ask explicit approval before any audio/video download or extraction.
- Prefer local/offline ASR such as `whisper.cpp` only when already installed and separately approved.
- OpenAI Whisper local package is a viable heavier local fallback only when already installed and separately approved.
- Cloud ASR or public/free/trial summarization routes require the user to mark the input `public_non_sensitive` and separately approve the route.

## Approval gates

Require separate approval before adding or using any behavior not already supported/documented upstream, including:

- package installs or dependency/lockfile changes;
- audio/video download or extraction;
- local ASR;
- cloud ASR or cloud summarization of transcript/audio/video;
- cookies, OAuth, API keys, proxies, age/geo bypass, or YouTube Data API setup;
- playlist/channel/search/bulk extraction.

Usually allowed without extra approval when the user provided the URL for summarization:

- single-video transcript/caption retrieval with already-installed supported tools;
- metadata-only inspection that uses no cookies/proxies and no media download, when supported by upstream docs/tooling.

## Privacy and routing rules

- Default to the current private-safe Hermes route for summarization.
- Treat unknown sensitivity as private/sensitive.
- Treat URLs supplied in private chats as potentially sensitive even if the video is public.
- Do not send transcript/audio/video to public/free/trial auxiliary models unless the user labels the input `public_non_sensitive` and separately approves that route.
- Never print, copy, or store API keys/cookies. Do not use browser cookies by default.

## Metadata and artifacts

Default artifact base:

```text
${HERMES_HOME:-~/.hermes}/artifacts/youtube-video-summaries/<video_id>/<YYYYMMDD-HHMMSS>/
```

Recommended files when the user asks for saved artifacts:

- `metadata.json` — URL, title, channel, date, duration, transcript source, tool versions if known.
- `transcript.raw.<ext>` — original caption/transcript artifact when available.
- `transcript.normalized.json` — canonical timestamped segments.
- `transcript.timestamped.md` — readable transcript.
- `summary.md` — user-facing structured summary.
- `claims.json` — claims with supporting timestamps and caveats.
- `run.json` — command classes run, approvals, routing, timings, errors.

## Metadata schema

```json
{
  "schema_version": "youtube_video_summary.v1",
  "video": {
    "id": "string",
    "url": "string",
    "title": "string|null",
    "channel": "string|null",
    "upload_date": "YYYY-MM-DD|null",
    "duration_seconds": 0,
    "language_hint": "string|null"
  },
  "transcript": {
    "source": "manual_captions|auto_captions|user_provided|local_asr|unavailable|unknown",
    "language": "string|null",
    "is_generated": true,
    "is_translated": false,
    "format": "json|vtt|srt|plain|null",
    "segments_count": 0,
    "quality_caveats": ["string"]
  },
  "routing": {
    "summarization_route": "private_safe_default|approved_public_non_sensitive|local_only",
    "asr_route": "none|local_whisper_cpp|local_openai_whisper|approved_cloud_asr",
    "approvals": ["string"]
  },
  "artifacts": {
    "base_dir": "string",
    "metadata": "metadata.json",
    "normalized_transcript": "transcript.normalized.json",
    "summary": "summary.md"
  }
}
```

## Summary template checklist

A structured video summary should include:

- title/channel/date/duration when available;
- URL/video ID;
- transcript source and quality caveats;
- concise summary;
- timestamped sections/chapters when available;
- key claims with evidence timestamps and caveats;
- action items/useful takeaways;
- optional follow-up questions;
- local artifact paths when files were written.

## Transcript quality caveats

- Prefer manual captions over auto captions.
- Treat ASR as a fallback only after approval.
- Auto captions and ASR can omit punctuation, misrecognize names/numbers, lose speaker attribution, and have approximate timestamps.
- Claims are claims from the video unless externally verified.

## Compliance notes

- YouTube Data API captions are not a general public-caption solution: listing/downloading captions requires OAuth, and downloads require permission to edit the video.
- YouTube Terms restrict downloading/reproducing content except as authorized and restrict automated access such as scrapers except in limited cases or with permission.
- Avoid media download, playlists/channels/bulk extraction, cookies, proxies, OAuth, API keys, age/geo bypass, and other circumvention-related behavior by default.
- This is operational guidance, not legal advice.
