---
name: streaming-content
description: >
  Fetch transcripts from clips and VODs on live-streaming platforms (Twitch, Kick, Rumble,
  X/Twitter, and other yt-dlp-supported sites) and transform them into structured content
  (summaries, threads, blog posts, quotes). Use when the user shares a Twitch / Kick / Rumble /
  X (Twitter) clip, VOD, video post, or broadcast link, asks to summarize a stream, or wants
  a transcript from a live-streaming platform. For YouTube, use the youtube-content skill
  instead — it reads served captions and is cheaper.
---

# Streaming Content

Transcribe clips and VODs from live-streaming platforms and convert them into useful
formats. Unlike YouTube, these platforms don't serve caption tracks, so this skill
downloads the audio and transcribes it.

## How it works

1. If the source already serves a caption track (some Rumble videos do), `yt-dlp` reads it directly - no download, no transcription.
2. Otherwise (Twitch, Kick), `yt-dlp` downloads the audio and the shared `transcribe_audio()` tool transcribes it (local faster-whisper, Groq, or
   OpenAI — whichever the environment is configured for).
3. The transcript is returned as JSON, ready to reshape.

## Setup

```bash
pip install yt-dlp faster-whisper   # ffmpeg must also be on PATH (brew install ffmpeg / apt install ffmpeg)
```

Requires Python 3.10+ (older interpreters silently fail Twitch's GraphQL — run inside the
Hermes venv).

A transcription backend is required: faster-whisper (installed above) runs locally with no key and is the default. Otherwise set one of GROQ_API_KEY (free tier), VOICE_TOOLS_OPENAI_KEY, MISTRAL_API_KEY, XAI_API_KEY, or ELEVENLABS_API_KEY. With no backend, audio downloads fine but every transcript fails with: No STT provider available.

## Helper Script

`SKILL_DIR` is the directory containing this SKILL.md.

```bash
# JSON with metadata + transcript
python3 SKILL_DIR/scripts/fetch_transcript.py "https://www.twitch.tv/<channel>/clip/<slug>"

# Plain text (good for piping into further processing)
python3 SKILL_DIR/scripts/fetch_transcript.py "URL" --text-only
```

Accepts any single clip or VOD URL yt-dlp supports.

## Supported platforms

- **Twitch** — clips and VODs. Note: VODs auto-expire after ~2 weeks unless saved as
  Highlights; clips are permanent.
- **Kick** — clips and VODs (Cloudflare can occasionally require cookies; pass them via
  yt-dlp's `--cookies` if a fetch is blocked).
- **Rumble** — standard video URLs (`rumble.com/v…-….html`, via yt-dlp's RumbleEmbed extractor).
  Note: `rumble.com/shorts/` URLs are not yet supported by yt-dlp — they fall back to unreliable
  generic extraction, so pass the standard video URL instead.
- **X (Twitter)** — native video posts (`x.com/<user>/status/<id>`, `twitter.com/...`) and
  broadcasts (`x.com/i/broadcasts/<id>`) transcribe without login via yt-dlp's guest token.
  Image / text / link-only posts return `no playable video in this post` (expected — there's
  nothing to transcribe, not a bad link). Spaces and protected / age-gated / NSFW posts need a
  logged-in session, which isn't wired in. X rotates its private endpoints often, so keep
  yt-dlp current.
- Any other yt-dlp-supported site is handled by the same path.

## Output Formats

After fetching the transcript, format it based on what the user asks for:

- **Summary**: Concise 5-10 sentence overview.
- **Thread**: Twitter/X thread — numbered posts, each under 280 chars.
- **Blog post**: Full article with title, sections, and key takeaways.
- **Quotes**: Notable lines from the stream.
- **Chapters**: Topic-shift breakdown. Stream transcripts are un-timestamped, so chapters
  are grouped by topic rather than timecode.

## Workflow

1. **Fetch** the transcript with the helper script.
2. **Validate**: confirm the output is non-empty. A `no playable video in this post` error
   means an X/Twitter (or similar) post with no video — image, text, or link-only; that's
   expected, not a bad link. An `audio download failed` error usually means the URL is
   private, sub-only, or an expired VOD — ask the user to verify it.
3. **Chunk if needed**: if the transcript exceeds ~50K characters, split into overlapping
   ~40K chunks (2K overlap) and summarize each before merging.
4. **Transform** into the requested format. Default to a summary if unspecified.
5. **Verify**: re-read the output for coherence before presenting.

## Error Handling

- **Audio download failed**: the clip/VOD is private, sub-only, expired (Twitch VODs), or
  the URL is wrong. Relay and ask the user to verify the link.
- **No playable video in this post**: an X/Twitter (or similar) post that has no video —
  image, text, or link-only. The URL is fine; there's just nothing to transcribe. Not a
  failure to retry.
- **Transcription failed**: confirm `ffmpeg` is installed and a transcription backend is
  configured (`transcribe_audio` falls back across local / Groq / OpenAI).
- **Dependency missing**: `pip install yt-dlp` and make sure `ffmpeg` is on PATH.
