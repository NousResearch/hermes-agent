# YouTube/video summarizer architecture notes

Use these notes when hardening the bundled `youtube-content` skill for single-video summaries or when a user asks for a structured summary from one YouTube URL.

## Recommended shape

- Extend the existing `youtube-content` skill; do not create a duplicate video summarizer skill.
- Keep the normal path single-video and transcript-first.
- Do not add a core model tool for this workflow. Skill instructions plus the existing helper keep the core tool schema unchanged.
- Add new helpers only after separate approval, and only for supported upstream behavior such as long-video chunking, repeatable artifact generation, or claim-review passes.

## Transcript-first flow

1. Accept one YouTube video URL or 11-character video ID.
2. Pause before playlists, channels, search URLs, or bulk URL lists unless the user approves a supported bulk workflow.
3. Prefer available captions/transcripts. If transcript metadata is available, prefer manual captions over generated captions and capture language/source details.
4. Keep downloader-style tooling outside the default path. If future upstream guidance documents it, constrain it to caption/subtitle handling by default and do not download media by default.
5. Normalize to timestamped segments with source metadata when possible.
6. Summarize from the transcript using the configured default model/provider.
7. Preserve timestamps and label key claims as claims made in the video unless independently verified.

## Approval gates

Ask for separate approval before adding or using heavier or unsupported behavior, including:

- package installs or dependency/lockfile changes;
- media/audio/video download or extraction;
- ASR, including local ASR;
- cloud transcription or cloud ASR;
- sending transcript/audio/video to external or alternate providers when not already approved by the user/configuration;
- cookies, proxies, OAuth, API keys, age/geo bypass, or YouTube Data API setup when not already supported by upstream behavior;
- playlist, channel, search, or bulk extraction.

Usually allowed without extra approval when the user provided one video URL for summarization:

- single-video transcript/caption retrieval with already-installed supported tools;
- metadata-only inspection that requires no new access-handling setup and no media download, when supported by upstream docs/tooling;
- summarization through the configured default model/provider under the configured privacy policy.

## Provider and privacy handling

- Use the configured default model/provider for summarization.
- Follow the configured privacy policy for data movement and retention.
- When content sensitivity is unclear, avoid sending transcripts/audio/video to alternate or external providers without user approval.
- Use external or alternate providers only when enabled by the user/configuration and approved for the specific content.
- Never print, copy, or store API keys/cookies. Do not use browser cookies unless supported by upstream behavior and approved for the specific task.

## Dependency handling

If a transcript dependency is missing, report the missing dependency and stop unless package installation is in scope for the current task or separately approved by the user. Do not change dependency files as part of transcript summarization guidance.

## Metadata and artifacts

Write artifacts only when the user asks for saved output or the workflow needs durable intermediate files. Keep artifact paths scoped to the requested workspace or configured artifact location, and avoid implying unsupported workflow state.

Suggested files when artifacts are useful:

- `metadata.json` — URL, title, channel, date, duration, transcript source, tool versions if known.
- `transcript.raw.<ext>` — original caption/transcript artifact when available.
- `transcript.normalized.json` — canonical timestamped segments.
- `transcript.timestamped.md` — readable transcript.
- `summary.md` — user-facing structured summary.
- `claims.json` — claims with supporting timestamps and caveats.
- `run.json` — command classes run, approvals, provider choices, timings, errors.

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
    "source": "manual_captions|auto_captions|user_provided|asr|unavailable|unknown",
    "language": "string|null",
    "is_generated": true,
    "is_translated": false,
    "format": "json|vtt|srt|plain|null",
    "segments_count": 0,
    "quality_caveats": ["string"]
  },
  "providers": {
    "summarization": "configured_default|approved_external_or_alternate",
    "transcription": "none|approved_asr|approved_cloud_transcription",
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
- artifact paths only when files were written.

## Transcript quality caveats

- Manual captions are preferred over auto captions.
- Auto captions are preferred over ASR when both are available.
- ASR is a fallback only after approval.
- Auto captions and ASR can omit punctuation, misrecognize names/numbers, lose speaker attribution, and have approximate timestamps.
- Claims are claims from the video unless externally verified.

## Compliance notes

- YouTube Data API captions are not a general caption-access solution: listing/downloading captions requires OAuth, and downloads require permission to edit the video.
- YouTube Terms restrict downloading/reproducing content except as authorized and restrict automated access such as scrapers except in limited cases or with permission.
- Preserve supported upstream behavior around access handling; do not add cookies, proxies, OAuth, API keys, or bypass behavior as a side effect of summarization guidance.
- This is operational guidance, not legal advice.
